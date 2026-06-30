// phylotreegpu.cpp — Phase G.2 in-tree GPU log-likelihood integration (host side).
//
// G.2.0a — gpuLnLCrossCheckOnce(): a one-shot, read-only diagnostic that proved the harness↔in-tree BRIDGE
//   (eigen convention, tip-ambiguity fold, ptn_freq pattern weights, pi-fold, NORM_LH) by rebuilding the
//   validated K1 eigen-space postorder sweep clean-room from the LIVE IQ-TREE objects and comparing its GPU
//   total lnL against IQ-TREE's own curScore. PASSED rel=1.235e-16 (job 170203514). Zero coupling to the
//   fn-pointer seam / host partial buffers / TraversalInfo.
//
// G.2.0b — wire the seam (lnL-only, -blfix). setLikelihoodKernelGPU() overrides ONLY computeLikelihoodBranch-
//   Pointer at the setLikelihoodKernel funnel so IQ-TREE's own computeLikelihood routes through the GPU.
//   Verified (adversarial workflow over the dev tree): under `--gpu -te TREE -m LG+G4 -blfix`, branch-length
//   Newton-Raphson is unreachable (fixed_branch_length==BRLEN_FIX gates out optimizeAllBranches at
//   modelfactory.cpp:1628; -te zeroes min_iterations so no tree search), so computeLikelihoodDerv /
//   computeLikelihoodFromBuffer NEVER fire; +G4 alpha is tuned by derivative-free Brent (rategamma.cpp:240)
//   which only calls computeLikelihood->Branch. The ONE host consumer of per-pattern data on the final path is
//   the UNCONDITIONAL computeLogLVariance() (phyloanalysis.cpp:3946) -> computePatternLikelihood() which reads
//   host _pattern_lh[] (phylotree.cpp:1515-1528). So the Branch override ALSO mirrors the per-pattern
//   log|lh_ptn| into _pattern_lh[] (the launcher already computes them) and zeroes the branch lh_scale_factor
//   (NORM_LH no-scaling path) -> the existing CPU variance/report code produces the correct s.e. with no edits
//   to phyloanalysis.cpp.
//
// Scope gate (NORM_LH / unscaled, reversible, bifurcating, single mixture, num_states in {4,20}, no +I/+ASC,
// no -wsl/-wpl/-alrt/-abayes/-b/-bb/-asr/dating/pll, fixed_branch_length==BRLEN_FIX): anything else -> the
// helper returns NaN and the Branch override delegates to the saved CPU pointer; the funnel installer simply
// does not install. Whole TU compiled only when IQTREE_GPU=ON (CMake); guarded anyway.
#include "iqtree_config.h"
#ifdef IQTREE_GPU

#include "phylotree.h"
#include "phylonode.h"
#include <cstring>   // G.6: memcpy in the free-Q decompose callback
#include "model/modelsubst.h"
#include "model/modelmixture.h"   // G.8.0: ModelMixture per-class component accessors (at(m)->getEigenvalues() etc.)
#include "model/rateheterogeneity.h"
#include "model/rategamma.h"   // G.4.3b: GAMMA_CUT_MEAN — the robust mean-gamma discriminator (isGammaRate())
#include "alignment/alignment.h"
#include "tree/gpu/gpu_iqtree.h"
#include <vector>
#include <map>
#include <functional>
#include <cmath>
#include <cstdio>
#include <mutex>     // G.8.2.2: serialize the unguarded mix clean-room launchers across ModelFinder's per-model OpenMP threads
using namespace std;

// G.8.2.2 — process-wide lock for optimizeParametersJOLTMix. The mixture clean-room launchers
// (gpu_lnl_crosscheck_mix / the all-branch-derv-mix launcher) are NOT internally mutexed (unlike gpu_jolt_optimize,
// which holds its own jolt_gpu_mtx), and ModelFinder scores candidate models across-model OpenMP-parallel
// (phylotesting.cpp). Without this, concurrent JOLTMix calls would race the single GPU's constant memory. JOLTMix
// therefore serializes on the one GPU (the G.4.2b decision); ineligible candidates still run N-parallel on the CPU.
static std::mutex gpu_mixjolt_mtx;

// ============================================================================================================
// Reusable clean-room GPU whole-tree log-likelihood (extracted from the G.2.0a cross-check). SILENT (no prints
// per call — it is invoked once per Branch evaluation): returns NaN on any unsupported regime / CUDA error so
// callers fall back. If out_patlh != nullptr it is filled with the per-pattern log|lh_ptn| (aln->size()
// entries, pattern order) = exactly what _pattern_lh[] holds under NORM_LH (no scaling).
// ============================================================================================================
double PhyloTree::gpuComputeTreeLnLCleanRoom(double *out_patlh) {
    // ---- regime gate (silent) ----
    if (!model || !site_rate || !aln) return (double)NAN;
    int ns = aln->num_states;
    if (ns != 4 && ns != 20) return (double)NAN;
    if (!model->isReversible()) return (double)NAN;
    if (model->getNMixtures() != 1) return (double)NAN;
    if (model->isSiteSpecificModel()) return (double)NAN;
    if (site_rate->getPInvar() > 0.0) return (double)NAN;   // +I omits ptn_invar in the clean-room sweep -> CPU

    int ncat = site_rate->getNRate();
    int nptn = (int)aln->size();
    int ntax = (int)aln->getNSeq();
    if (ncat < 1 || ncat > 64) return (double)NAN;

    // ---- model eigen factors (IQ-TREE convention: U=evec, Uinv=inv_evec, P(t)=U exp(Lambda t) Uinv) ----
    double *eval = model->getEigenvalues();
    double *U    = model->getEigenvectors();
    double *Uinv = model->getInverseEigenvectors();
    if (!eval || !U || !Uinv) return (double)NAN;
    vector<double> freq(ns, 0.0);
    model->getStateFrequency(freq.data(), 0);
    vector<double> UinvRowSum(ns, 0.0);
    for (int i = 0; i < ns; i++) { double s = 0; for (int j = 0; j < ns; j++) s += Uinv[i*ns+j]; UinvRowSum[i] = s; }
    vector<double> catRate(ncat), catProp(ncat);
    for (int c = 0; c < ncat; c++) { catRate[c] = site_rate->getRate(c); catProp[c] = site_rate->getProp(c); }

    // ---- topology rooted at an internal node R (IQ-TREE roots at a leaf; lnL is reversible-invariant) ----
    if (!root || !root->isLeaf() || root->neighbors.empty()) return (double)NAN;
    Node *R = root->neighbors[0]->node;     // internal node adjacent to the root leaf
    if (R->isLeaf()) return (double)NAN;

    map<Node*,int> nid;
    vector<Node*> nodes;
    vector<double> parentLen;               // edge length to parent (R's unused)
    vector<int>    isLeafV, leafTax;
    function<void(Node*,Node*,double)> indexDfs = [&](Node *n, Node *dad, double lenToDad) {
        int myi = (int)nodes.size(); nid[n] = myi; nodes.push_back(n);
        parentLen.push_back(lenToDad);
        int lf = n->isLeaf() ? 1 : 0; isLeafV.push_back(lf);
        leafTax.push_back(lf ? aln->getSeqID(n->name) : -1);
        for (auto nb : n->neighbors) { if (nb->node == dad) continue; indexDfs(nb->node, n, nb->length); }
    };
    indexDfs(R, nullptr, 0.0);
    int nNodes = (int)nodes.size();

    vector<int> postInternal;               // node indices, post-order (children precede parents)
    vector<int> slot(nNodes, -1);
    function<void(Node*,Node*)> postDfs = [&](Node *n, Node *dad) {
        for (auto nb : n->neighbors) { if (nb->node == dad) continue; postDfs(nb->node, n); }
        if (!n->isLeaf()) { slot[nid[n]] = (int)postInternal.size(); postInternal.push_back(nid[n]); }
    };
    postDfs(R, nullptr);
    int nInternal = (int)postInternal.size();

    // ---- echild[child_node][cat][x][i] = U[x][i] * exp(eval[i]*rate_c*parentLen[child]) ----
    size_t ecStride = (size_t)ncat*ns*ns;
    vector<double> echild((size_t)nNodes*ecStride, 0.0);
    for (int v = 0; v < nNodes; v++) {
        if (v == nid[R]) continue;          // R has no parent edge
        double len_v = parentLen[v];
        for (int c = 0; c < ncat; c++) {
            double l = len_v * catRate[c];
            double ex[20]; for (int i = 0; i < ns; i++) ex[i] = exp(eval[i]*l);
            double *e = &echild[(size_t)v*ecStride + (size_t)c*ns*ns];
            for (int x = 0; x < ns; x++) for (int i = 0; i < ns; i++) e[x*ns+i] = U[x*ns+i]*ex[i];
        }
    }

    // ---- compact tip states[taxon][ptn] ; pattern frequencies ----
    vector<unsigned char> tip((size_t)ntax*nptn);
    for (int v = 0; v < nNodes; v++) {
        if (!isLeafV[v]) continue;
        int tax = leafTax[v];
        if (tax < 0 || tax >= ntax) return (double)NAN;
        for (int p = 0; p < nptn; p++) { int st = (int)aln->at(p)[tax]; tip[(size_t)tax*nptn+p] = (unsigned char)((st < ns) ? st : ns); }
    }
    vector<double> ptnFreq(nptn);
    for (int p = 0; p < nptn; p++) ptnFreq[p] = (double)aln->at(p).frequency;

    // ---- per-internal-node descriptors (postorder) ----
    vector<int> dRoot(nInternal), dNch(nInternal), dOut(nInternal);
    vector<int> dChildNode(nInternal*3, -1), dChildIsLeaf(nInternal*3, 0), dChildLeaf(nInternal*3, -1), dChildSlot(nInternal*3, -1);
    for (int idx = 0; idx < nInternal; idx++) {
        int vi = postInternal[idx]; Node *n = nodes[vi];
        Node *dad = nullptr;
        if (n != R) { for (auto nb : n->neighbors) { if (nid[nb->node] < vi) { dad = nb->node; break; } } }
        dRoot[idx] = (n == R) ? 1 : 0;
        dOut[idx]  = (n == R) ? -1 : slot[vi];
        int k = 0;
        for (auto nb : n->neighbors) {
            if (nb->node == dad) continue;
            if (k >= 3) return (double)NAN;
            int cv = nid[nb->node];
            dChildNode[idx*3+k] = cv;
            if (isLeafV[cv]) { dChildIsLeaf[idx*3+k] = 1; dChildLeaf[idx*3+k] = leafTax[cv]; }
            else             { dChildIsLeaf[idx*3+k] = 0; dChildSlot[idx*3+k] = slot[cv]; }
            k++;
        }
        dNch[idx] = k;
    }

    return gpu_lnl_crosscheck(ns, nptn, ncat, ntax, nNodes, nInternal,
        Uinv, UinvRowSum.data(), freq.data(), catProp.data(), echild.data(), tip.data(), ptnFreq.data(),
        dRoot.data(), dNch.data(), dOut.data(),
        dChildNode.data(), dChildIsLeaf.data(), dChildLeaf.data(), dChildSlot.data(),
        out_patlh);
}

// ============================================================================================================
// G.8.0 — clean-room whole-tree lnL for a PROFILE MIXTURE (C20/C60/MEOW80). Reads the LIVE ModelMixture's per-class
// eigen via the component accessors mix[m]->getEigenvalues()/...() (already offset into the AVX-padded packed array,
// so NO manual m*480 stride math — sidesteps the #1 silent-bug risk), per-class freq via getStateFrequency(.,m),
// weights via getMixtureWeight(m). Regime r = m*ncat + c; builds echild[node][r] and per-class Uinv/freq, then calls
// gpu_lnl_crosscheck_mix. SILENT, returns NaN on any unsupported regime. CPU path byte-unchanged (additive).
// ============================================================================================================
double PhyloTree::gpuComputeTreeLnLCleanRoomMix(double *out_patlh, double *out_lhcat, const double *w_override,
                                                const double *parentLenOverride, double alphaOverride, double pinvOverride) {
    if (!model || !site_rate || !aln) return (double)NAN;
    int ns = aln->num_states;
    if (ns != 20 && ns != 4) return (double)NAN;
    if (!model->isReversible()) return (double)NAN;
    int N = model->getNMixtures();
    if (N <= 1) return (double)NAN;                       // single-model -> the non-mix path
    if (model->isSiteSpecificModel()) return (double)NAN; // PMSF stays on CPU (per-site pi, no class sum)
    // A1 (+I): +I is now handled (the per-class invariant clsinv + the (1-pinv) bridge below). pinvOverride>=0 selects a
    // trial pinv (the LM's pinv-FD/step); pinvOverride<0 uses the stored site_rate->getPInvar(). pinv_eff<=0 => no +I work.

    int ncat = site_rate->getNRate();
    int nptn = (int)aln->size();
    int ntax = (int)aln->getNSeq();
    if (ncat < 1) return (double)NAN;
    int R = N*ncat;

    ModelMixture *mix = dynamic_cast<ModelMixture*>(model);
    if (!mix || (int)mix->size() != N) return (double)NAN;
    if (mix->isFused()) return (double)NAN;   // LG4M/LG4X = 1:1 class<->rate pairing, NOT the N*ncat cross-product
                                              // this builds (weight from site_rate->getProp, not getMixtureWeight) -> CPU

    // ---- per-class eigen (stride-safe via component pointers), freq, weight; per-regime weight = w_m*catProp_c ----
    std::vector<double> Uinv((size_t)N*ns*ns), UinvRowSum((size_t)N*ns), freqC((size_t)N*ns);
    std::vector<double> evalC((size_t)N*ns), Uc((size_t)N*ns*ns);
    std::vector<double> catRate(ncat), catProp(ncat), wreg((size_t)R);
    for (int c = 0; c < ncat; c++) { catRate[c] = site_rate->getRate(c); catProp[c] = site_rate->getProp(c); }
    // G.8.2.1b: optional iterate-alpha override — recompute mean-1 discrete-gamma catRate[] at alphaOverride (catProp
    // stays fixed 1/ncat). Used by the joint-optimiser's alpha FD-Newton; warm-run reproduces the live rates exactly.
    bool alphaOv = (alphaOverride > 0.0 && ncat > 1);
    if (alphaOv) gpu_discrete_gamma_mean(alphaOverride, ncat, catRate.data());   // catRate now mean-1 ρ_c (NO pinv rescale)
    // A1 (+I) the (1-pinv) bridge — make the VARIABLE part use pinv_eff: catRate=ρ/(1-pinv_eff), catProp=(1-pinv_eff)·w_c.
    // RateGammaInvar's getRate=ρ/(1-pinv0), getProp=(1-pinv0)/ncat carry the STORED pinv0; rebase to pinv_eff. With an
    // alpha override catRate is already the mean-1 ρ_c (no pinv), so un-rescale getRate's 1/(1-pinv0) ONLY when !alphaOv.
    // At pinv_eff<=0 (no +I) this whole block is skipped => the validated +G path is byte-identical.
    double pinv0 = site_rate->getPInvar();
    double pinv_eff = (pinvOverride >= 0.0) ? pinvOverride : pinv0;
    if (pinv_eff > 0.0) {
        double f0 = 1.0 - pinv0, fe = 1.0 - pinv_eff;
        for (int c = 0; c < ncat; c++) {
            double rho = alphaOv ? catRate[c] : catRate[c] * f0;   // -> mean-1 ρ_c
            catRate[c] = rho / fe;                                 // ρ/(1-pinv_eff)
            catProp[c] = catProp[c] * fe / f0;                    // (1-pinv0)·w_c -> (1-pinv_eff)·w_c
        }
    }
    std::vector<double> wM(N);   // A1 (+I): the per-class weight actually used (for clsinv) — MUST match wreg's wm
    for (int m = 0; m < N; m++) {
        ModelMarkov *cm = (ModelMarkov*)(*mix)[m];
        double *ev = cm->getEigenvalues(), *U = cm->getEigenvectors(), *Ui = cm->getInverseEigenvectors();
        if (!ev || !U || !Ui) return (double)NAN;
        for (int i = 0; i < ns; i++) evalC[(size_t)m*ns+i] = ev[i];
        for (int x = 0; x < ns*ns; x++) { Uc[(size_t)m*ns*ns+x] = U[x]; Uinv[(size_t)m*ns*ns+x] = Ui[x]; }
        for (int i = 0; i < ns; i++) { double s=0; for (int j=0;j<ns;j++) s += Ui[i*ns+j]; UinvRowSum[(size_t)m*ns+i]=s; }
        double wf[64]; model->getStateFrequency(wf, m);   // ns<=20
        for (int x = 0; x < ns; x++) freqC[(size_t)m*ns+x] = wf[x];
        double wm = w_override ? w_override[m] : model->getMixtureWeight(m);   // G.8.2: optional EM-iterate weights
        wM[m] = wm;
        for (int c = 0; c < ncat; c++) wreg[(size_t)m*ncat+c] = wm * catProp[c];
    }
    // A1 (+I): per-class invariant clsinv[m][p] = w_m · pinv_eff · base_invar_m[p]. base_invar_m = const-site freq under
    // class m's π (lift of the single-matrix logic at ~:2034; const_char/STATE_UNKNOWN/DNA|PROTEIN ambiguity), but with
    // freqC[m] in place of the one model freq. Σ_m clsinv[m][p] = pinv·Σ_m w_m·π_{m,const} = the root invariant term.
    std::vector<double> clsinv;
    if (pinv_eff > 0.0) {
        clsinv.assign((size_t)N*nptn, 0.0);
        const int ambi_aa[] = {4+8, 32+64, 512+1024};   // B=N|D, Z=Q|E, U=I|L
        int SU = (int)aln->STATE_UNKNOWN;
        for (int m = 0; m < N; m++) {
            const double *sf = &freqC[(size_t)m*ns]; double scal = wM[m] * pinv_eff;
            for (int p = 0; p < nptn; p++) {
                int cc = (int)aln->at(p).const_char; double bi = 0.0;
                if (cc > SU)                            bi = 0.0;
                else if (cc == SU)                      bi = 1.0;
                else if (cc < ns)                       bi = sf[cc];
                else if (aln->seq_type == SEQ_DNA)     { double s=0; int cs=cc-ns+1; for (int x=0;x<ns;x++) if (cs & (1<<x)) s+=sf[x]; bi=s; }
                else if (aln->seq_type == SEQ_PROTEIN) { double s=0; int cs=cc-ns;   if (cs>=0 && cs<3) for (int x=0;x<11;x++) if (ambi_aa[cs] & (1<<x)) s+=sf[x]; bi=s; }
                clsinv[(size_t)m*nptn + p] = scal * bi;
            }
        }
    }

    // ---- topology rooted at an internal node R (reversible-invariant) — identical to the single-model sweep ----
    if (!root || !root->isLeaf() || root->neighbors.empty()) return (double)NAN;
    Node *Rt = root->neighbors[0]->node;
    if (Rt->isLeaf()) return (double)NAN;
    map<Node*,int> nid; vector<Node*> nodes; vector<double> parentLen; vector<int> isLeafV, leafTax;
    function<void(Node*,Node*,double)> indexDfs = [&](Node *n, Node *dad, double lenToDad) {
        int myi=(int)nodes.size(); nid[n]=myi; nodes.push_back(n); parentLen.push_back(lenToDad);
        int lf=n->isLeaf()?1:0; isLeafV.push_back(lf); leafTax.push_back(lf?aln->getSeqID(n->name):-1);
        for (auto nb:n->neighbors){ if(nb->node==dad) continue; indexDfs(nb->node,n,nb->length); }
    };
    indexDfs(Rt, nullptr, 0.0);
    int nNodes=(int)nodes.size();
    // G.8.2.1b: optional iterate-branch override — replace the live edge lengths with parentLenOverride[v] (indexed by
    // this function's DFS nid; root entry is 0.0). The echild build below consumes it transparently. nullptr => live tree.
    if (parentLenOverride) for (int v = 0; v < nNodes; v++) parentLen[v] = parentLenOverride[v];
    vector<int> postInternal; vector<int> slot(nNodes,-1);
    function<void(Node*,Node*)> postDfs = [&](Node *n, Node *dad){
        for (auto nb:n->neighbors){ if(nb->node==dad) continue; postDfs(nb->node,n); }
        if (!n->isLeaf()){ slot[nid[n]]=(int)postInternal.size(); postInternal.push_back(nid[n]); }
    };
    postDfs(Rt, nullptr);
    int nInternal=(int)postInternal.size();

    // ---- echild[child][r=m*ncat+c][x][i] = U_m[x][i]*exp(eval_m[i]*rate_c*parentLen) ----
    size_t ecStride = (size_t)R*ns*ns;
    vector<double> echild((size_t)nNodes*ecStride, 0.0);
    for (int v = 0; v < nNodes; v++) {
        if (v == nid[Rt]) continue;
        double len_v = parentLen[v];
        for (int m = 0; m < N; m++) {
            const double *ev = &evalC[(size_t)m*ns]; const double *U = &Uc[(size_t)m*ns*ns];
            for (int c = 0; c < ncat; c++) {
                double l = len_v * catRate[c]; int r = m*ncat + c;
                double ex[20]; for (int i=0;i<ns;i++) ex[i]=exp(ev[i]*l);
                double *e = &echild[(size_t)v*ecStride + (size_t)r*ns*ns];
                for (int x=0;x<ns;x++) for (int i=0;i<ns;i++) e[x*ns+i] = U[x*ns+i]*ex[i];
            }
        }
    }

    // ---- compact tip states ; pattern frequencies (identical to single-model) ----
    vector<unsigned char> tip((size_t)ntax*nptn);
    for (int v = 0; v < nNodes; v++) {
        if (!isLeafV[v]) continue;
        int tax = leafTax[v]; if (tax<0 || tax>=ntax) return (double)NAN;
        for (int p=0;p<nptn;p++){ int st=(int)aln->at(p)[tax]; tip[(size_t)tax*nptn+p]=(unsigned char)((st<ns)?st:ns); }
    }
    vector<double> ptnFreq(nptn);
    for (int p=0;p<nptn;p++) ptnFreq[p]=(double)aln->at(p).frequency;

    // ---- per-internal-node descriptors (postorder) — identical to single-model ----
    vector<int> dRoot(nInternal), dNch(nInternal), dOut(nInternal);
    vector<int> dChildNode(nInternal*3,-1), dChildIsLeaf(nInternal*3,0), dChildLeaf(nInternal*3,-1), dChildSlot(nInternal*3,-1);
    for (int idx=0; idx<nInternal; idx++){
        int vi=postInternal[idx]; Node *n=nodes[vi]; Node *dad=nullptr;
        if (n!=Rt){ for(auto nb:n->neighbors){ if(nid[nb->node]<vi){ dad=nb->node; break; } } }
        dRoot[idx]=(n==Rt)?1:0; dOut[idx]=(n==Rt)?-1:slot[vi];
        int k=0;
        for (auto nb:n->neighbors){
            if (nb->node==dad) continue; if (k>=3) return (double)NAN;
            int cv=nid[nb->node]; dChildNode[idx*3+k]=cv;
            if (isLeafV[cv]){ dChildIsLeaf[idx*3+k]=1; dChildLeaf[idx*3+k]=leafTax[cv]; }
            else            { dChildIsLeaf[idx*3+k]=0; dChildSlot[idx*3+k]=slot[cv]; }
            k++;
        }
        dNch[idx]=k;
    }

    return gpu_lnl_crosscheck_mix(ns, nptn, ncat, N, ntax, nNodes, nInternal,
        Uinv.data(), UinvRowSum.data(), freqC.data(), wreg.data(), echild.data(), tip.data(), ptnFreq.data(),
        dRoot.data(), dNch.data(), dOut.data(),
        dChildNode.data(), dChildIsLeaf.data(), dChildLeaf.data(), dChildSlot.data(),
        out_patlh, out_lhcat,
        pinv_eff, (pinv_eff > 0.0 ? clsinv.data() : nullptr));   // A1 (+I)
}

// ============================================================================================================
// G.2.0b — GPU override for computeLikelihoodBranchPointer (byte-matches ComputeLikelihoodBranchType). For a
// reversible model the whole-tree lnL is independent of the rooting branch, so (dad_branch,dad) are ignored
// except to zero their lh_scale_factor (NORM_LH no-scaling path, so computePatternLikelihood's memmove yields
// the correct per-pattern values for logl_variance/s.e.). Mirrors the per-pattern log|lh_ptn| into _pattern_lh.
// ============================================================================================================
double PhyloTree::computeLikelihoodBranchGPU(PhyloNeighbor *dad_branch, PhyloNode *dad, bool save_log_value) {
    double lnL = gpuComputeTreeLnLCleanRoom(_pattern_lh);   // _pattern_lh may be null -> launcher skips the mirror
    if (std::isnan(lnL)) {
        static bool warned = false;
        if (!warned) { warned = true;
            printf("[GPU-BRANCH] unsupported/CUDA-error this call -> CPU fallback (computeLikelihoodBranchGenericSIMD)\n"); }
        if (cpuComputeLikelihoodBranchPointer)
            return (this->*cpuComputeLikelihoodBranchPointer)(dad_branch, dad, save_log_value);
        return lnL;   // no CPU fallback available (should not happen — installer always saves one)
    }
    // NORM_LH: zero the branch scale factors so computePatternLikelihood takes the no-scaling memmove path
    // (ptn_lh = _pattern_lh = the GPU per-pattern log-lh) -> correct logl_variance / reported s.e.
    if (dad_branch) dad_branch->lh_scale_factor = 0.0;
    if (dad && dad_branch) {
        PhyloNeighbor *back = (PhyloNeighbor*)dad->findNeighbor(dad_branch->node);
        if (back) back->lh_scale_factor = 0.0;
    }
    static bool announced = false;
    if (!announced) { announced = true;
        printf("[GPU-BRANCH] computeLikelihoodBranchGPU active (clean-room full sweep; _pattern_lh mirrored)\n"); }
    return lnL;
}

// ============================================================================================================
// G.2.1a — clean-room single-edge branch-length derivative df/ddf for the edge (dad_branch->node, dad).
// STATELESS (no device-resident state — like the lnL helper): builds TWO directed clean-room sub-sweeps split
// by the central edge (sub-roots = the two endpoints, each excluding the central neighbour), so node_eig /
// dad_eig are the eigen-space endpoint partials EXCLUDING the central transition; the derivative kernel applies
// exp(eval·r·t) for the central branch length t = dad_branch->length. Returns df = d(lnL)/dt (un-negated; the
// CPU computeFuncDerv negates), *out_ddf the 2nd derivative, *out_lnL the tree lnL at t. NaN if unsupported.
// G.2.1a scope: BOTH endpoints must be internal (leaf-endpoint support is G.2.1b).
double PhyloTree::gpuComputeEdgeDervCleanRoom(PhyloNeighbor *dad_branch, PhyloNode *dad, double *out_ddf, double *out_lnL) {
    if (!model || !site_rate || !aln) return (double)NAN;
    int ns = aln->num_states;
    if (ns != 4 && ns != 20) return (double)NAN;
    if (!model->isReversible() || model->getNMixtures() != 1 || model->isSiteSpecificModel()) return (double)NAN;
    if (site_rate->getPInvar() > 0.0) return (double)NAN;   // +I omits ptn_invar in the clean-room sweep -> CPU
    Node *node = dad_branch->node;   // one endpoint (the "node" side)
    Node *dadN = dad;                // other endpoint (the "dad" side)
    if (!node || !dadN) return (double)NAN;   // G.2.1b: leaf endpoints OK (≤1 leaf per edge; tip eigen synthesized)

    int ncat = site_rate->getNRate();
    int nptn = (int)aln->size();
    int ntax = (int)aln->getNSeq();
    if (ncat < 1 || ncat > 64) return (double)NAN;
    double *eval = model->getEigenvalues();
    double *U    = model->getEigenvectors();
    double *Uinv = model->getInverseEigenvectors();
    if (!eval || !U || !Uinv) return (double)NAN;
    vector<double> freq(ns, 0.0); model->getStateFrequency(freq.data(), 0);
    vector<double> UinvRowSum(ns, 0.0);
    for (int i = 0; i < ns; i++) { double s = 0; for (int j = 0; j < ns; j++) s += Uinv[i*ns+j]; UinvRowSum[i] = s; }
    vector<double> catRate(ncat), catProp(ncat);
    for (int c = 0; c < ncat; c++) { catRate[c] = site_rate->getRate(c); catProp[c] = site_rate->getProp(c); }
    double t = dad_branch->length;

    // two-sub-root DFS, central edge (node<->dadN) excluded from both
    map<Node*,int> nid; vector<Node*> nodes; vector<double> parentLen; vector<int> isLeafV, leafTax;
    function<void(Node*,Node*,double)> indexDfs = [&](Node *n, Node *par, double lenToPar) {
        int myi = (int)nodes.size(); nid[n] = myi; nodes.push_back(n);
        parentLen.push_back(lenToPar);
        int lf = n->isLeaf() ? 1 : 0; isLeafV.push_back(lf);
        leafTax.push_back(lf ? aln->getSeqID(n->name) : -1);
        for (auto nb : n->neighbors) { if (nb->node == par) continue; indexDfs(nb->node, n, nb->length); }
    };
    indexDfs(node, dadN, 0.0);    // node-side subtree (parent dir = dadN, excluded)
    indexDfs(dadN, node, 0.0);    // dad-side subtree (parent dir = node, excluded)
    int nNodes = (int)nodes.size();

    vector<int> postInternal; vector<int> slot(nNodes, -1);
    function<void(Node*,Node*)> postDfs = [&](Node *n, Node *par) {
        for (auto nb : n->neighbors) { if (nb->node == par) continue; postDfs(nb->node, n); }
        if (!n->isLeaf()) { slot[nid[n]] = (int)postInternal.size(); postInternal.push_back(nid[n]); }
    };
    postDfs(node, dadN);
    postDfs(dadN, node);
    int nInternal = (int)postInternal.size();
    // endpoint eigen partial: internal -> its postorder slot; leaf -> synthesized tip eigen (slot=-1, pass taxon)
    int nodeSlot = node->isLeaf() ? -1 : slot[nid[node]];
    int nodeLeafTax = node->isLeaf() ? leafTax[nid[node]] : -1;
    int dadSlot  = dadN->isLeaf() ? -1 : slot[nid[dadN]];
    int dadLeafTax = dadN->isLeaf() ? leafTax[nid[dadN]] : -1;

    // echild[v] = U·exp(eval·rate·parentLen[v]) for every node except the two sub-roots (no parent edge)
    size_t ecStride = (size_t)ncat*ns*ns;
    vector<double> echild((size_t)nNodes*ecStride, 0.0);
    for (int v = 0; v < nNodes; v++) {
        if (v == nid[node] || v == nid[dadN]) continue;
        double len_v = parentLen[v];
        for (int c = 0; c < ncat; c++) {
            double l = len_v * catRate[c];
            double ex[20]; for (int i = 0; i < ns; i++) ex[i] = exp(eval[i]*l);
            double *e = &echild[(size_t)v*ecStride + (size_t)c*ns*ns];
            for (int x = 0; x < ns; x++) for (int i = 0; i < ns; i++) e[x*ns+i] = U[x*ns+i]*ex[i];
        }
    }

    vector<unsigned char> tip((size_t)ntax*nptn);
    for (int v = 0; v < nNodes; v++) {
        if (!isLeafV[v]) continue;
        int tax = leafTax[v];
        if (tax < 0 || tax >= ntax) return (double)NAN;
        for (int p = 0; p < nptn; p++) { int st = (int)aln->at(p)[tax]; tip[(size_t)tax*nptn+p] = (unsigned char)((st < ns) ? st : ns); }
    }
    vector<double> ptnFreq(nptn);
    for (int p = 0; p < nptn; p++) ptnFreq[p] = (double)aln->at(p).frequency;

    // descriptors: ALL internal (isRoot=0); each (incl. node & dadN sub-roots) writes its eigen partial to slot
    vector<int> dRoot(nInternal, 0), dNch(nInternal), dOut(nInternal);
    vector<int> dChildNode(nInternal*3, -1), dChildIsLeaf(nInternal*3, 0), dChildLeaf(nInternal*3, -1), dChildSlot(nInternal*3, -1);
    for (int idx = 0; idx < nInternal; idx++) {
        int vi = postInternal[idx]; Node *n = nodes[vi];
        // parent = the unique neighbour with a smaller nid within the same subtree (pre-order). The two
        // sub-roots have none (their only smaller-nid neighbour would be across the excluded central edge).
        Node *par = nullptr;
        if (n != node && n != dadN) {
            for (auto nb : n->neighbors) { auto it = nid.find(nb->node); if (it != nid.end() && it->second < vi) { par = nb->node; break; } }
        }
        dOut[idx] = slot[vi];
        int k = 0;
        for (auto nb : n->neighbors) {
            if (nb->node == par) continue;
            if (n == node && nb->node == dadN) continue;   // exclude central edge at sub-root node
            if (n == dadN && nb->node == node) continue;   // exclude central edge at sub-root dadN
            if (k >= 3) return (double)NAN;
            int cv = nid[nb->node];
            dChildNode[idx*3+k] = cv;
            if (isLeafV[cv]) { dChildIsLeaf[idx*3+k] = 1; dChildLeaf[idx*3+k] = leafTax[cv]; }
            else             { dChildIsLeaf[idx*3+k] = 0; dChildSlot[idx*3+k] = slot[cv]; }
            k++;
        }
        dNch[idx] = k;
    }

    return gpu_derv_crosscheck(ns, nptn, ncat, ntax, nNodes, nInternal,
        Uinv, UinvRowSum.data(), freq.data(), catProp.data(), echild.data(), tip.data(), ptnFreq.data(),
        dRoot.data(), dNch.data(), dOut.data(),
        dChildNode.data(), dChildIsLeaf.data(), dChildLeaf.data(), dChildSlot.data(),
        nodeSlot, nodeLeafTax, dadSlot, dadLeafTax, eval, catRate.data(), t, out_ddf, out_lnL);
}

// ============================================================================================================
// TS.2 Increment 2 — gpuScreenNNICleanRoom: NON-MUTATING NNI screener. Scores the swapped topology
// (node1_nei <-> node2_nei exchanged across the central node1-node2 edge) at the OLD branch lengths from the
// UNMUTATED tree — the L2 batched use-case (score every candidate without a physical doNNI). Modeled on
// gpuComputeEdgeDervCleanRoom (two-sub-root central-edge split + gpu_derv_crosscheck), but builds EXPLICIT
// recorded adjacency (parentIdx/parentLen/childrenIdx) during a swap-aware DFS rather than reading
// n->neighbors downstream — because the swap is virtual (the physical tree still has node1->S1, node2->S2).
// KEY (design-review BUG 1): a moved-in subtree root (S2 under node1, S1 under node2) carries its MOVED edge
// length (L2 = node2_nei->length, L1 = node1_nei->length) explicitly; it is NEVER a neighbour lookup (no
// node1<->S2 edge exists physically). Recursing INTO a moved-in subtree skips that subtree's REAL physical
// parent (node2 for S2, node1 for S1) since the subtree internals are byte-identical to the original tree.
// *out_lnL = whole-tree lnL of the swapped topology @ central old length == the CPU tsr_pre oracle.
// Same eligibility gate as gpuComputeEdgeDervCleanRoom; NaN -> CPU.
// ============================================================================================================
double PhyloTree::gpuScreenNNICleanRoom(PhyloNode *node1, PhyloNode *node2,
                                        PhyloNeighbor *node1_nei, PhyloNeighbor *node2_nei,
                                        double *out_ddf, double *out_lnL) {
    if (!model || !site_rate || !aln) return (double)NAN;
    int ns = aln->num_states;
    if (ns != 4 && ns != 20) return (double)NAN;
    if (!model->isReversible() || model->getNMixtures() != 1 || model->isSiteSpecificModel()) return (double)NAN;
    if (site_rate->getPInvar() > 0.0) return (double)NAN;   // +I omits ptn_invar in the clean-room sweep -> CPU
    if (!node1 || !node2 || !node1_nei || !node2_nei) return (double)NAN;
    if (node1->isLeaf() || node2->isLeaf()) return (double)NAN;   // central endpoints internal (ASSERT in caller)

    int ncat = site_rate->getNRate();
    int nptn = (int)aln->size();
    int ntax = (int)aln->getNSeq();
    if (ncat < 1 || ncat > 64) return (double)NAN;
    double *eval = model->getEigenvalues();
    double *U    = model->getEigenvectors();
    double *Uinv = model->getInverseEigenvectors();
    if (!eval || !U || !Uinv) return (double)NAN;
    vector<double> freq(ns, 0.0); model->getStateFrequency(freq.data(), 0);
    vector<double> UinvRowSum(ns, 0.0);
    for (int i = 0; i < ns; i++) { double s = 0; for (int j = 0; j < ns; j++) s += Uinv[i*ns+j]; UinvRowSum[i] = s; }
    vector<double> catRate(ncat), catProp(ncat);
    for (int c = 0; c < ncat; c++) { catRate[c] = site_rate->getRate(c); catProp[c] = site_rate->getProp(c); }

    // central edge length (UNCHANGED by the swap)
    PhyloNeighbor *node12 = (PhyloNeighbor*) node1->findNeighbor(node2);
    if (!node12) return (double)NAN;
    double t = node12->length;

    // resolve swap operands on the UNMUTATED tree
    Node *S1 = node1_nei->node; double L1 = node1_nei->length;   // node1-side moved subtree
    Node *S2 = node2_nei->node; double L2 = node2_nei->length;   // node2-side moved subtree
    if (!S1 || !S2 || S1 == S2) return (double)NAN;
    Node *Bn = nullptr; double Lb = 0.0;   // node1's OTHER non-central neighbour (stays)
    for (auto nb : node1->neighbors) { if (nb->node == node2 || nb->node == S1) continue; Bn = nb->node; Lb = nb->length; }
    Node *Dn = nullptr; double Ld = 0.0;   // node2's OTHER non-central neighbour (stays)
    for (auto nb : node2->neighbors) { if (nb->node == node1 || nb->node == S2) continue; Dn = nb->node; Ld = nb->length; }
    if (!Bn || !Dn) return (double)NAN;    // degree != 3

    // ---- swap-aware DFS building EXPLICIT recorded adjacency (parentIdx/parentLen/childrenIdx) ----
    map<Node*,int> nid; vector<Node*> nodes;
    vector<int> parentIdx; vector<double> parentLen; vector<int> isLeafV, leafTax;
    vector<vector<int> > childrenIdx;
    function<void(Node*,Node*,int,double)> dfs = [&](Node *n, Node *skipPhysical, int parI, double lenToPar) {
        int myi = (int)nodes.size(); nid[n] = myi; nodes.push_back(n);
        parentIdx.push_back(parI); parentLen.push_back(lenToPar);
        int lf = n->isLeaf() ? 1 : 0; isLeafV.push_back(lf);
        leafTax.push_back(lf ? aln->getSeqID(n->name) : -1);
        childrenIdx.push_back(vector<int>());
        if (lf) return;   // leaf: no children (degree 1)
        // virtual child list (node, length, physical-parent-to-skip-on-recursion)
        vector<Node*> cN; vector<double> cL; vector<Node*> cSkip;
        if (n == node1) {
            cN.push_back(S2); cL.push_back(L2); cSkip.push_back(node2);   // moved-in: skip S2's real parent node2
            cN.push_back(Bn); cL.push_back(Lb); cSkip.push_back(node1);
        } else if (n == node2) {
            cN.push_back(S1); cL.push_back(L1); cSkip.push_back(node1);   // moved-in: skip S1's real parent node1
            cN.push_back(Dn); cL.push_back(Ld); cSkip.push_back(node2);
        } else {
            for (auto nb : n->neighbors) { if (nb->node == skipPhysical) continue; cN.push_back(nb->node); cL.push_back(nb->length); cSkip.push_back(n); }
        }
        for (size_t k = 0; k < cN.size(); k++) {
            int ci = (int)nodes.size();             // child's index (set as myi on entry)
            childrenIdx[myi].push_back(ci);         // re-index each iter (outer vector may realloc in recursion)
            dfs(cN[k], cSkip[k], myi, cL[k]);
        }
    };
    dfs(node1, node2, -1, 0.0);   // node1-side sub-root (parent dir = node2, excluded)
    dfs(node2, node1, -1, 0.0);   // node2-side sub-root (parent dir = node1, excluded)
    int nNodes = (int)nodes.size();

    // build-time invariants (the riskiest thing per design review): moved roots carry the MOVED length;
    // each sub-root has exactly 2 recorded children; the central edge is excluded from both sub-roots.
    if (childrenIdx[nid[node1]].size() != 2 || childrenIdx[nid[node2]].size() != 2) return (double)NAN;
    if (parentLen[nid[S2]] != L2 || parentLen[nid[S1]] != L1) return (double)NAN;

    // ---- postorder slots over the RECORDED children ----
    vector<int> postInternal; vector<int> slot(nNodes, -1);
    function<void(int)> postDfs = [&](int v) {
        for (size_t j = 0; j < childrenIdx[v].size(); j++) postDfs(childrenIdx[v][j]);
        if (!isLeafV[v]) { slot[v] = (int)postInternal.size(); postInternal.push_back(v); }
    };
    postDfs(nid[node1]);
    postDfs(nid[node2]);
    int nInternal = (int)postInternal.size();

    // endpoint eigen partial: node1/node2 internal => their postorder slots (leaf-endpoint branch is dead here)
    int nodeSlot = isLeafV[nid[node1]] ? -1 : slot[nid[node1]];
    int nodeLeafTax = -1;
    int dadSlot  = isLeafV[nid[node2]] ? -1 : slot[nid[node2]];
    int dadLeafTax = -1;

    // echild[v] = U·exp(eval·rate·parentLen[v]) for every node except the two sub-roots (no parent edge)
    int r1 = nid[node1], r2 = nid[node2];
    size_t ecStride = (size_t)ncat*ns*ns;
    vector<double> echild((size_t)nNodes*ecStride, 0.0);
    for (int v = 0; v < nNodes; v++) {
        if (v == r1 || v == r2) continue;
        double len_v = parentLen[v];
        for (int c = 0; c < ncat; c++) {
            double l = len_v * catRate[c];
            double ex[20]; for (int i = 0; i < ns; i++) ex[i] = exp(eval[i]*l);
            double *e = &echild[(size_t)v*ecStride + (size_t)c*ns*ns];
            for (int x = 0; x < ns; x++) for (int i = 0; i < ns; i++) e[x*ns+i] = U[x*ns+i]*ex[i];
        }
    }

    vector<unsigned char> tip((size_t)ntax*nptn);
    for (int v = 0; v < nNodes; v++) {
        if (!isLeafV[v]) continue;
        int tax = leafTax[v];
        if (tax < 0 || tax >= ntax) return (double)NAN;
        for (int p = 0; p < nptn; p++) { int st = (int)aln->at(p)[tax]; tip[(size_t)tax*nptn+p] = (unsigned char)((st < ns) ? st : ns); }
    }
    vector<double> ptnFreq(nptn);
    for (int p = 0; p < nptn; p++) ptnFreq[p] = (double)aln->at(p).frequency;

    // descriptors: ALL internal (isRoot=0); children/parent come from the RECORDED adjacency (NOT n->neighbors)
    vector<int> dRoot(nInternal, 0), dNch(nInternal), dOut(nInternal);
    vector<int> dChildNode(nInternal*3, -1), dChildIsLeaf(nInternal*3, 0), dChildLeaf(nInternal*3, -1), dChildSlot(nInternal*3, -1);
    for (int idx = 0; idx < nInternal; idx++) {
        int vi = postInternal[idx];
        dOut[idx] = slot[vi];
        int k = 0;
        for (size_t j = 0; j < childrenIdx[vi].size(); j++) {
            int cv = childrenIdx[vi][j];
            if (k >= 3) return (double)NAN;
            dChildNode[idx*3+k] = cv;
            if (isLeafV[cv]) { dChildIsLeaf[idx*3+k] = 1; dChildLeaf[idx*3+k] = leafTax[cv]; }
            else             { dChildIsLeaf[idx*3+k] = 0; dChildSlot[idx*3+k] = slot[cv]; }
            k++;
        }
        dNch[idx] = k;
    }

    return gpu_derv_crosscheck(ns, nptn, ncat, ntax, nNodes, nInternal,
        Uinv, UinvRowSum.data(), freq.data(), catProp.data(), echild.data(), tip.data(), ptnFreq.data(),
        dRoot.data(), dNch.data(), dOut.data(),
        dChildNode.data(), dChildIsLeaf.data(), dChildLeaf.data(), dChildSlot.data(),
        nodeSlot, nodeLeafTax, dadSlot, dadLeafTax, eval, catRate.data(), t, out_ddf, out_lnL);
}

// ============================================================================================================
// TS.2 Increment 3a — gpuScreenNNIFoldCleanRoom: host driver for the RESIDENT-POSTORDER + RE-PAIRING-FOLD
// screener. Builds the SAME physical two-sub-root descriptors as gpuComputeEdgeDervCleanRoom (central edge
// node1<->node2 excluded; node1's subtree contains {S1,Bn}, node2's {S2,Dn} as resident lower partials), then
// resolves the four surrounding subtrees' (echild-node, slot|leaf) and calls gpu_screen_nni_fold_crosscheck,
// which re-pairs them (node1<-{S2,Bn}, node2<-{S1,Dn}) via two k1_node folds + k2_derv at the UNCHANGED central
// length. The swap is purely in the fold GROUPING — each subtree keeps its own physical echild (its length is
// unchanged; a moved Neighbor keeps its length). *out_lnL = swapped-topology lnL, == gpuScreenNNICleanRoom (the
// I2 oracle) to 1e-9. NO new kernel, NO swap-aware DFS, ONE resident postorder. Same eligibility gate; NaN -> CPU.
// ============================================================================================================
double PhyloTree::gpuScreenNNIFoldCleanRoom(PhyloNode *node1, PhyloNode *node2,
                                            PhyloNeighbor *node1_nei, PhyloNeighbor *node2_nei,
                                            double *out_ddf, double *out_lnL) {
    if (!model || !site_rate || !aln) return (double)NAN;
    int ns = aln->num_states;
    if (ns != 4 && ns != 20) return (double)NAN;
    if (!model->isReversible() || model->getNMixtures() != 1 || model->isSiteSpecificModel()) return (double)NAN;
    if (site_rate->getPInvar() > 0.0) return (double)NAN;
    if (!node1 || !node2 || !node1_nei || !node2_nei) return (double)NAN;
    if (node1->isLeaf() || node2->isLeaf()) return (double)NAN;

    int ncat = site_rate->getNRate();
    int nptn = (int)aln->size();
    int ntax = (int)aln->getNSeq();
    if (ncat < 1 || ncat > 64) return (double)NAN;
    double *eval = model->getEigenvalues();
    double *U    = model->getEigenvectors();
    double *Uinv = model->getInverseEigenvectors();
    if (!eval || !U || !Uinv) return (double)NAN;
    vector<double> freq(ns, 0.0); model->getStateFrequency(freq.data(), 0);
    vector<double> UinvRowSum(ns, 0.0);
    for (int i = 0; i < ns; i++) { double s = 0; for (int j = 0; j < ns; j++) s += Uinv[i*ns+j]; UinvRowSum[i] = s; }
    vector<double> catRate(ncat), catProp(ncat);
    for (int c = 0; c < ncat; c++) { catRate[c] = site_rate->getRate(c); catProp[c] = site_rate->getProp(c); }

    // central edge (node1<->node2), length UNCHANGED by the swap
    PhyloNeighbor *node12 = (PhyloNeighbor*) node1->findNeighbor(node2);
    if (!node12) return (double)NAN;
    double t = node12->length;

    // swap operands (physical tree): S1/Bn on node1's side, S2/Dn on node2's side
    Node *S1 = node1_nei->node; Node *S2 = node2_nei->node;
    if (!S1 || !S2 || S1 == S2) return (double)NAN;
    Node *Bn = nullptr; for (auto nb : node1->neighbors) { if (nb->node == node2 || nb->node == S1) continue; Bn = nb->node; }
    Node *Dn = nullptr; for (auto nb : node2->neighbors) { if (nb->node == node1 || nb->node == S2) continue; Dn = nb->node; }
    if (!Bn || !Dn) return (double)NAN;

    Node *node = node1, *dadN = node2;   // physical two-sub-root central-edge split (NO swap applied)

    // ---- PHYSICAL two-sub-root DFS (identical to gpuComputeEdgeDervCleanRoom) ----
    map<Node*,int> nid; vector<Node*> nodes; vector<double> parentLen; vector<int> isLeafV, leafTax;
    function<void(Node*,Node*,double)> indexDfs = [&](Node *n, Node *par, double lenToPar) {
        int myi = (int)nodes.size(); nid[n] = myi; nodes.push_back(n);
        parentLen.push_back(lenToPar);
        int lf = n->isLeaf() ? 1 : 0; isLeafV.push_back(lf);
        leafTax.push_back(lf ? aln->getSeqID(n->name) : -1);
        for (auto nb : n->neighbors) { if (nb->node == par) continue; indexDfs(nb->node, n, nb->length); }
    };
    indexDfs(node, dadN, 0.0);
    indexDfs(dadN, node, 0.0);
    int nNodes = (int)nodes.size();

    vector<int> postInternal; vector<int> slot(nNodes, -1);
    function<void(Node*,Node*)> postDfs = [&](Node *n, Node *par) {
        for (auto nb : n->neighbors) { if (nb->node == par) continue; postDfs(nb->node, n); }
        if (!n->isLeaf()) { slot[nid[n]] = (int)postInternal.size(); postInternal.push_back(nid[n]); }
    };
    postDfs(node, dadN);
    postDfs(dadN, node);
    int nInternal = (int)postInternal.size();

    size_t ecStride = (size_t)ncat*ns*ns;
    vector<double> echild((size_t)nNodes*ecStride, 0.0);
    for (int v = 0; v < nNodes; v++) {
        if (v == nid[node] || v == nid[dadN]) continue;
        double len_v = parentLen[v];
        for (int c = 0; c < ncat; c++) {
            double l = len_v * catRate[c];
            double ex[20]; for (int i = 0; i < ns; i++) ex[i] = exp(eval[i]*l);
            double *e = &echild[(size_t)v*ecStride + (size_t)c*ns*ns];
            for (int x = 0; x < ns; x++) for (int i = 0; i < ns; i++) e[x*ns+i] = U[x*ns+i]*ex[i];
        }
    }

    vector<unsigned char> tip((size_t)ntax*nptn);
    for (int v = 0; v < nNodes; v++) {
        if (!isLeafV[v]) continue;
        int tax = leafTax[v];
        if (tax < 0 || tax >= ntax) return (double)NAN;
        for (int p = 0; p < nptn; p++) { int st = (int)aln->at(p)[tax]; tip[(size_t)tax*nptn+p] = (unsigned char)((st < ns) ? st : ns); }
    }
    vector<double> ptnFreq(nptn);
    for (int p = 0; p < nptn; p++) ptnFreq[p] = (double)aln->at(p).frequency;

    vector<int> dRoot(nInternal, 0), dNch(nInternal), dOut(nInternal);
    vector<int> dChildNode(nInternal*3, -1), dChildIsLeaf(nInternal*3, 0), dChildLeaf(nInternal*3, -1), dChildSlot(nInternal*3, -1);
    for (int idx = 0; idx < nInternal; idx++) {
        int vi = postInternal[idx]; Node *n = nodes[vi];
        Node *par = nullptr;
        if (n != node && n != dadN) {
            for (auto nb : n->neighbors) { auto it = nid.find(nb->node); if (it != nid.end() && it->second < vi) { par = nb->node; break; } }
        }
        dOut[idx] = slot[vi];
        int k = 0;
        for (auto nb : n->neighbors) {
            if (nb->node == par) continue;
            if (n == node && nb->node == dadN) continue;   // exclude central edge at sub-root node
            if (n == dadN && nb->node == node) continue;   // exclude central edge at sub-root dadN
            if (k >= 3) return (double)NAN;
            int cv = nid[nb->node];
            dChildNode[idx*3+k] = cv;
            if (isLeafV[cv]) { dChildIsLeaf[idx*3+k] = 1; dChildLeaf[idx*3+k] = leafTax[cv]; }
            else             { dChildIsLeaf[idx*3+k] = 0; dChildSlot[idx*3+k] = slot[cv]; }
            k++;
        }
        dNch[idx] = k;
    }

    // ---- resolve the 4 re-paired children: ec = node index (its echild carries the UNCHANGED length); slot|leaf ----
    if (nid.find(S1)==nid.end() || nid.find(S2)==nid.end() || nid.find(Bn)==nid.end() || nid.find(Dn)==nid.end())
        return (double)NAN;
    auto childDesc = [&](Node *X, int &ec, int &sl, int &lf) {
        int v = nid[X]; ec = v;
        if (isLeafV[v]) { sl = -1; lf = leafTax[v]; } else { sl = slot[v]; lf = -1; }
    };
    int n1a_ec,n1a_sl,n1a_lf, n1b_ec,n1b_sl,n1b_lf, n2a_ec,n2a_sl,n2a_lf, n2b_ec,n2b_sl,n2b_lf;
    childDesc(S2, n1a_ec,n1a_sl,n1a_lf);   // node1 <- S2 (moved in,  keeps L2)
    childDesc(Bn, n1b_ec,n1b_sl,n1b_lf);   // node1 <- Bn (stays,     keeps Lb)
    childDesc(S1, n2a_ec,n2a_sl,n2a_lf);   // node2 <- S1 (moved in,  keeps L1)
    childDesc(Dn, n2b_ec,n2b_sl,n2b_lf);   // node2 <- Dn (stays,     keeps Ld)

    return gpu_screen_nni_fold_crosscheck(ns, nptn, ncat, ntax, nNodes, nInternal,
        Uinv, UinvRowSum.data(), freq.data(), catProp.data(), echild.data(), tip.data(), ptnFreq.data(),
        dRoot.data(), dNch.data(), dOut.data(),
        dChildNode.data(), dChildIsLeaf.data(), dChildLeaf.data(), dChildSlot.data(),
        n1a_ec,n1a_sl,n1a_lf, n1b_ec,n1b_sl,n1b_lf, n2a_ec,n2a_sl,n2a_lf, n2b_ec,n2b_sl,n2b_lf,
        eval, catRate.data(), t, out_ddf, out_lnL);
}

// ============================================================================================================
// TS.2 Increment 3b-i — gpuAllBranchUpperCheckCleanRoom: host driver for the PERSISTENT-UPPER preorder validator.
// Builds a fixed-root whole-tree sweep (clone of gpuComputeTreeLnLCleanRoom: root at R = internal node adjacent to
// the root leaf), adds the per-node expfac (parent-branch factor for kj_pre) + flat child/leaf/slot/parentLen
// arrays, and calls gpu_allbranch_upper_check. It builds ONE resident postorder + ONE preorder with a PERSISTENT
// per-node upper buffer, computes every edge's lnL = k2_derv(lower_v, pre_v, b_v), and the whole-tree lnL. The
// INVARIANT (every edge lnL == tree lnL, reversible model) validates the persistent-upper machinery — the substrate
// 3b-ii's re-pairing reuses — WITHOUT re-pairing. Returns false (ineligible / CUDA error) on NaN. Out: max rel-err
// over edges, #edges, #pass@1e-9, #bit-exact, tree lnL.
// ============================================================================================================
bool PhyloTree::gpuAllBranchUpperCheckCleanRoom(double *out_max_rel, long long *out_nedge, long long *out_npass,
                                                long long *out_nbitexact, double *out_tree_lnL) {
    if (!model || !site_rate || !aln) return false;
    int ns = aln->num_states;
    if (ns != 4 && ns != 20) return false;
    if (!model->isReversible() || model->getNMixtures() != 1 || model->isSiteSpecificModel()) return false;
    if (site_rate->getPInvar() > 0.0) return false;
    int ncat = site_rate->getNRate();
    int nptn = (int)aln->size();
    int ntax = (int)aln->getNSeq();
    if (ncat < 1 || ncat > 64) return false;
    double *eval = model->getEigenvalues();
    double *U    = model->getEigenvectors();
    double *Uinv = model->getInverseEigenvectors();
    if (!eval || !U || !Uinv) return false;
    vector<double> freq(ns, 0.0); model->getStateFrequency(freq.data(), 0);
    vector<double> UinvRowSum(ns, 0.0);
    for (int i = 0; i < ns; i++) { double s = 0; for (int j = 0; j < ns; j++) s += Uinv[i*ns+j]; UinvRowSum[i] = s; }
    vector<double> catRate(ncat), catProp(ncat);
    for (int c = 0; c < ncat; c++) { catRate[c] = site_rate->getRate(c); catProp[c] = site_rate->getProp(c); }

    if (!root || !root->isLeaf() || root->neighbors.empty()) return false;
    Node *R = root->neighbors[0]->node;
    if (R->isLeaf()) return false;

    map<Node*,int> nid; vector<Node*> nodes; vector<double> parentLen; vector<int> isLeafV, leafTax;
    function<void(Node*,Node*,double)> indexDfs = [&](Node *n, Node *dad, double lenToDad) {
        int myi = (int)nodes.size(); nid[n] = myi; nodes.push_back(n);
        parentLen.push_back(lenToDad);
        int lf = n->isLeaf() ? 1 : 0; isLeafV.push_back(lf);
        leafTax.push_back(lf ? aln->getSeqID(n->name) : -1);
        for (auto nb : n->neighbors) { if (nb->node == dad) continue; indexDfs(nb->node, n, nb->length); }
    };
    indexDfs(R, nullptr, 0.0);
    int nNodes = (int)nodes.size();

    vector<int> postInternal; vector<int> slot(nNodes, -1);
    function<void(Node*,Node*)> postDfs = [&](Node *n, Node *dad) {
        for (auto nb : n->neighbors) { if (nb->node == dad) continue; postDfs(nb->node, n); }
        if (!n->isLeaf()) { slot[nid[n]] = (int)postInternal.size(); postInternal.push_back(nid[n]); }
    };
    postDfs(R, nullptr);
    int nInternal = (int)postInternal.size();

    // echild[v] = U·exp(eval·rate·b_v); expfac[v] = exp(eval·rate·b_v) (no U; kj_pre applies g_U). Root: zeroed.
    size_t ecStride = (size_t)ncat*ns*ns, exStride = (size_t)ncat*ns;
    vector<double> echild((size_t)nNodes*ecStride, 0.0);
    vector<double> expfac((size_t)nNodes*exStride, 0.0);
    for (int v = 0; v < nNodes; v++) {
        if (v == nid[R]) continue;
        double len_v = parentLen[v];
        for (int c = 0; c < ncat; c++) {
            double l = len_v * catRate[c];
            double ex[20]; for (int i = 0; i < ns; i++) ex[i] = exp(eval[i]*l);
            double *e = &echild[(size_t)v*ecStride + (size_t)c*ns*ns];
            for (int x = 0; x < ns; x++) for (int i = 0; i < ns; i++) e[x*ns+i] = U[x*ns+i]*ex[i];
            double *ef = &expfac[(size_t)v*exStride + (size_t)c*ns];
            for (int i = 0; i < ns; i++) ef[i] = ex[i];
        }
    }

    vector<unsigned char> tip((size_t)ntax*nptn);
    for (int v = 0; v < nNodes; v++) {
        if (!isLeafV[v]) continue;
        int tax = leafTax[v];
        if (tax < 0 || tax >= ntax) return false;
        for (int p = 0; p < nptn; p++) { int st = (int)aln->at(p)[tax]; tip[(size_t)tax*nptn+p] = (unsigned char)((st < ns) ? st : ns); }
    }
    vector<double> ptnFreq(nptn);
    for (int p = 0; p < nptn; p++) ptnFreq[p] = (double)aln->at(p).frequency;

    // per-node children / leaf / slot / parentLen (parent = the unique neighbour with a smaller nid)
    vector<int> node_nchild(nNodes, 0), node_child(nNodes*3, -1), node_leaf(nNodes, -1), node_slot(nNodes, -1);
    vector<double> node_parentLen(nNodes, 0.0);
    for (int v = 0; v < nNodes; v++) {
        Node *n = nodes[v];
        node_leaf[v] = isLeafV[v] ? leafTax[v] : -1;
        node_slot[v] = slot[v];
        node_parentLen[v] = parentLen[v];
        Node *dad = nullptr;
        for (auto nb : n->neighbors) { auto it = nid.find(nb->node); if (it != nid.end() && it->second < v) { dad = nb->node; break; } }
        int k = 0;
        for (auto nb : n->neighbors) {
            if (nb->node == dad) continue;
            if (k >= 3) return false;
            node_child[v*3+k] = nid[nb->node];
            k++;
        }
        node_nchild[v] = k;
    }

    vector<double> edgeLnL(nNodes, (double)NAN);
    double treeLnL = (double)NAN;
    double rc = gpu_allbranch_upper_check(ns, nptn, ncat, ntax, nNodes, nInternal, nid[R],
        Uinv, U, UinvRowSum.data(), freq.data(), catProp.data(), eval, catRate.data(),
        echild.data(), expfac.data(), tip.data(), ptnFreq.data(),
        node_nchild.data(), node_child.data(), node_leaf.data(), node_slot.data(),
        node_parentLen.data(), postInternal.data(),
        edgeLnL.data(), &treeLnL);
    if (rc != rc) return false;   // NaN -> ineligible / CUDA error

    long long nedge=0, npass=0, nbit=0; double maxrel=0.0;
    for (int v = 0; v < nNodes; v++) {
        if (edgeLnL[v] != edgeLnL[v]) continue;   // NaN = root (no edge) or unwritten
        nedge++;
        double rel = (treeLnL != 0.0) ? fabs((edgeLnL[v] - treeLnL)/treeLnL) : fabs(edgeLnL[v] - treeLnL);
        if (rel > maxrel) maxrel = rel;
        if (rel <= 1e-9) npass++;
        if (edgeLnL[v] == treeLnL) nbit++;
    }
    if (out_max_rel) *out_max_rel = maxrel;
    if (out_nedge) *out_nedge = nedge;
    if (out_npass) *out_npass = npass;
    if (out_nbitexact) *out_nbitexact = nbit;
    if (out_tree_lnL) *out_tree_lnL = treeLnL;
    return true;
}

// ============================================================================================================
// TS.2 Increment 3b-ii — gpuScreenNNIBatchCleanRoom: host driver for the BATCHED re-pairing NNI screener (perf
// core). Builds the I3b-i fixed-root sweep (resident lowers + persistent uppers built ONCE), enumerates every
// inner branch's 2 NNI moves, scores them all in one gpu_screen_nni_batch_crosscheck call (cheap folds off the
// resident state), and cross-checks EACH move vs gpuScreenNNIFoldCleanRoom (the 3a oracle, which re-roots a full
// postorder per move). Times the batched path vs the M oracle calls (the first perf number — note: launch-bound
// at example scale; the scale-free claim is the 1:M postorder-count ratio). Out: max rel-err vs 3a, #moves,
// #pass@1e-9, tree lnL, and the two wall times. Returns false if ineligible / no moves / CUDA error.
// ============================================================================================================
bool PhyloTree::gpuScreenNNIBatchCleanRoom(double *out_max_rel, long long *out_nmove, long long *out_npass,
                                           double *out_tree_lnL, double *out_wall_batched, double *out_wall_oracle) {
    if (!model || !site_rate || !aln) return false;
    int ns = aln->num_states;
    if (ns != 4 && ns != 20) return false;
    if (!model->isReversible() || model->getNMixtures() != 1 || model->isSiteSpecificModel()) return false;
    if (site_rate->getPInvar() > 0.0) return false;
    int ncat = site_rate->getNRate();
    int nptn = (int)aln->size();
    int ntax = (int)aln->getNSeq();
    if (ncat < 1 || ncat > 64) return false;
    double *eval = model->getEigenvalues();
    double *U    = model->getEigenvectors();
    double *Uinv = model->getInverseEigenvectors();
    if (!eval || !U || !Uinv) return false;
    vector<double> freq(ns, 0.0); model->getStateFrequency(freq.data(), 0);
    vector<double> UinvRowSum(ns, 0.0);
    for (int i = 0; i < ns; i++) { double s = 0; for (int j = 0; j < ns; j++) s += Uinv[i*ns+j]; UinvRowSum[i] = s; }
    vector<double> catRate(ncat), catProp(ncat);
    for (int c = 0; c < ncat; c++) { catRate[c] = site_rate->getRate(c); catProp[c] = site_rate->getProp(c); }

    if (!root || !root->isLeaf() || root->neighbors.empty()) return false;
    Node *R = root->neighbors[0]->node;
    if (R->isLeaf()) return false;

    map<Node*,int> nid; vector<Node*> nodes; vector<double> parentLen; vector<int> isLeafV, leafTax;
    function<void(Node*,Node*,double)> indexDfs = [&](Node *n, Node *dad, double lenToDad) {
        int myi = (int)nodes.size(); nid[n] = myi; nodes.push_back(n);
        parentLen.push_back(lenToDad);
        int lf = n->isLeaf() ? 1 : 0; isLeafV.push_back(lf);
        leafTax.push_back(lf ? aln->getSeqID(n->name) : -1);
        for (auto nb : n->neighbors) { if (nb->node == dad) continue; indexDfs(nb->node, n, nb->length); }
    };
    indexDfs(R, nullptr, 0.0);
    int nNodes = (int)nodes.size();

    vector<int> postInternal; vector<int> slot(nNodes, -1);
    function<void(Node*,Node*)> postDfs = [&](Node *n, Node *dad) {
        for (auto nb : n->neighbors) { if (nb->node == dad) continue; postDfs(nb->node, n); }
        if (!n->isLeaf()) { slot[nid[n]] = (int)postInternal.size(); postInternal.push_back(nid[n]); }
    };
    postDfs(R, nullptr);
    int nInternal = (int)postInternal.size();

    size_t ecStride = (size_t)ncat*ns*ns, exStride = (size_t)ncat*ns;
    vector<double> echild((size_t)nNodes*ecStride, 0.0);
    vector<double> expfac((size_t)nNodes*exStride, 0.0);
    for (int v = 0; v < nNodes; v++) {
        if (v == nid[R]) continue;
        double len_v = parentLen[v];
        for (int c = 0; c < ncat; c++) {
            double l = len_v * catRate[c];
            double ex[20]; for (int i = 0; i < ns; i++) ex[i] = exp(eval[i]*l);
            double *e = &echild[(size_t)v*ecStride + (size_t)c*ns*ns];
            for (int x = 0; x < ns; x++) for (int i = 0; i < ns; i++) e[x*ns+i] = U[x*ns+i]*ex[i];
            double *ef = &expfac[(size_t)v*exStride + (size_t)c*ns];
            for (int i = 0; i < ns; i++) ef[i] = ex[i];
        }
    }

    vector<unsigned char> tip((size_t)ntax*nptn);
    for (int v = 0; v < nNodes; v++) {
        if (!isLeafV[v]) continue;
        int tax = leafTax[v];
        if (tax < 0 || tax >= ntax) return false;
        for (int p = 0; p < nptn; p++) { int st = (int)aln->at(p)[tax]; tip[(size_t)tax*nptn+p] = (unsigned char)((st < ns) ? st : ns); }
    }
    vector<double> ptnFreq(nptn);
    for (int p = 0; p < nptn; p++) ptnFreq[p] = (double)aln->at(p).frequency;

    vector<int> node_nchild(nNodes, 0), node_child(nNodes*3, -1), node_leaf(nNodes, -1), node_slot(nNodes, -1);
    vector<double> node_parentLen(nNodes, 0.0);
    vector<int> parentNid(nNodes, -1);
    for (int v = 0; v < nNodes; v++) {
        Node *n = nodes[v];
        node_leaf[v] = isLeafV[v] ? leafTax[v] : -1;
        node_slot[v] = slot[v];
        node_parentLen[v] = parentLen[v];
        Node *dad = nullptr;
        for (auto nb : n->neighbors) { auto it = nid.find(nb->node); if (it != nid.end() && it->second < v) { dad = nb->node; parentNid[v] = it->second; break; } }
        int k = 0;
        for (auto nb : n->neighbors) {
            if (nb->node == dad) continue;
            if (k >= 3) return false;
            node_child[v*3+k] = nid[nb->node];
            k++;
        }
        node_nchild[v] = k;
    }

    int Rnid = nid[R];
    auto rit = nid.find(root); if (rit == nid.end()) return false;
    int rootLeafNid = rit->second;
    auto childDesc = [&](int xnid, int &ec, int &sl, int &lf) {
        ec = xnid;
        if (isLeafV[xnid]) { sl = -1; lf = leafTax[xnid]; } else { sl = slot[xnid]; lf = -1; }
    };

    // ---- enumerate every inner branch's 2 NNI moves (fixed-root orientation): for edge (u=parent, v=child),
    //      w = u's other child; swap(w, v1) and swap(w, v2). u==root keeps the root-leaf on u's side. ----
    vector<int> mv_u, mv_uIsRoot; vector<double> mv_bv;
    vector<int> n1a_ec,n1a_sl,n1a_lf, n1b_ec,n1b_sl,n1b_lf, n2a_ec,n2a_sl,n2a_lf, n2b_ec,n2b_sl,n2b_lf;
    vector<int> orc_n1, orc_n2, orc_w, orc_vswap;   // 3a-oracle operands (nids)
    for (int v = 0; v < nNodes; v++) {
        if (v == Rnid || isLeafV[v]) continue;       // v internal, non-root
        if (node_nchild[v] != 2) continue;
        int u = parentNid[v];
        if (u < 0) continue;
        int v1 = node_child[v*3+0], v2 = node_child[v*3+1];
        int w = -1, stayR = -1;
        if (u == Rnid) {
            for (int k = 0; k < node_nchild[u]; k++) { int c = node_child[u*3+k];
                if (c == v) continue;
                if (c == rootLeafNid) { stayR = c; continue; }
                w = c; }
            if (w < 0 || stayR < 0) continue;        // R not the expected (root-leaf + 2) trifurcation
        } else {
            for (int k = 0; k < node_nchild[u]; k++) { int c = node_child[u*3+k]; if (c != v) { w = c; break; } }
            if (w < 0) continue;
        }
        for (int mi = 0; mi < 2; mi++) {
            int vk_swap = (mi==0) ? v1 : v2;
            int vk_stay = (mi==0) ? v2 : v1;
            int e,s,l;
            mv_u.push_back(u); mv_uIsRoot.push_back(u==Rnid?1:0); mv_bv.push_back(node_parentLen[v]);
            childDesc(vk_swap, e,s,l); n1a_ec.push_back(e); n1a_sl.push_back(s); n1a_lf.push_back(l);
            if (u==Rnid) childDesc(stayR, e,s,l); else { e=-1; s=-1; l=-1; }
            n1b_ec.push_back(e); n1b_sl.push_back(s); n1b_lf.push_back(l);
            childDesc(w, e,s,l);       n2a_ec.push_back(e); n2a_sl.push_back(s); n2a_lf.push_back(l);
            childDesc(vk_stay, e,s,l); n2b_ec.push_back(e); n2b_sl.push_back(s); n2b_lf.push_back(l);
            orc_n1.push_back(u); orc_n2.push_back(v); orc_w.push_back(w); orc_vswap.push_back(vk_swap);
        }
    }
    int M = (int)mv_u.size();
    if (M == 0) return false;

    // ---- batched launcher (TIMED): 1 postorder + 1 preorder + M cheap folds ----
    vector<double> moveLnL(M, (double)NAN);
    double treeLnL = (double)NAN;
    double t0 = getRealTime();
    double rc = gpu_screen_nni_batch_crosscheck(ns, nptn, ncat, ntax, nNodes, nInternal, Rnid,
        Uinv, U, UinvRowSum.data(), freq.data(), catProp.data(), eval, catRate.data(),
        echild.data(), expfac.data(), tip.data(), ptnFreq.data(),
        node_nchild.data(), node_child.data(), node_leaf.data(), node_slot.data(),
        node_parentLen.data(), postInternal.data(),
        M, mv_u.data(), mv_uIsRoot.data(), mv_bv.data(),
        n1a_ec.data(),n1a_sl.data(),n1a_lf.data(), n1b_ec.data(),n1b_sl.data(),n1b_lf.data(),
        n2a_ec.data(),n2a_sl.data(),n2a_lf.data(), n2b_ec.data(),n2b_sl.data(),n2b_lf.data(),
        moveLnL.data(), &treeLnL);
    double wall_batched = getRealTime() - t0;
    if (rc != rc) return false;

    // ---- per-move 3a oracle cross-check (TIMED): M× full-postorder gpuScreenNNIFoldCleanRoom ----
    long long nmove=0, npass=0; double maxrel=0.0;
    double t1 = getRealTime();
    for (int m = 0; m < M; m++) {
        PhyloNode *n1 = (PhyloNode*) nodes[orc_n1[m]];
        PhyloNode *n2 = (PhyloNode*) nodes[orc_n2[m]];
        PhyloNeighbor *nei1 = (PhyloNeighbor*) n1->findNeighbor(nodes[orc_w[m]]);
        PhyloNeighbor *nei2 = (PhyloNeighbor*) n2->findNeighbor(nodes[orc_vswap[m]]);
        double oddf=0.0, olnL=(double)NAN;
        if (nei1 && nei2) gpuScreenNNIFoldCleanRoom(n1, n2, nei1, nei2, &oddf, &olnL);
        if (olnL != olnL) continue;          // 3a oracle ineligible for this move -> no reference, skip
        nmove++;
        if (moveLnL[m] != moveLnL[m]) {      // batched NaN while the oracle is finite = a real FAILURE (don't mask it)
            maxrel = HUGE_VAL;
            continue;                        // counted in nmove, NOT in npass
        }
        double rel = (olnL != 0.0) ? fabs((moveLnL[m]-olnL)/olnL) : fabs(moveLnL[m]-olnL);
        if (rel > maxrel) maxrel = rel;
        if (rel <= 1e-9) npass++;
    }
    double wall_oracle = getRealTime() - t1;

    if (out_max_rel) *out_max_rel = maxrel;
    if (out_nmove) *out_nmove = nmove;
    if (out_npass) *out_npass = npass;
    if (out_tree_lnL) *out_tree_lnL = treeLnL;
    if (out_wall_batched) *out_wall_batched = wall_batched;
    if (out_wall_oracle) *out_wall_oracle = wall_oracle;
    return true;
}

// ============================================================================================================
// TS.2 Increment 3c — gpuScreenNNITileCleanRoom: host driver for the PATTERN-TILED batched NNI screener. Same
// build + move-enumeration as gpuScreenNNIBatchCleanRoom (3b-ii), but the launcher (gpu_screen_nni_tile_crosscheck)
// tiles nptn so the persistent per-node upper fits at AA-1M. Gates: (THE GATE) every tiled move == the untiled 3a
// oracle (gpuScreenNNIFoldCleanRoom) to 1e-9 — works at all scales (the oracle uses one full postorder, ~59GB,
// which fits on an H200). (BONUS, example scale only, when nTile=1 fits) the auto-tiled per-move lnLs are
// BIT-IDENTICAL to forced nTile∈{3,7} AND to the frozen 3b-ii batch launcher — proving tiling-invariance + that the
// new code's nTile=1 path matches the validated 3b-ii. Reports the auto-picked nTile + the bit-identity count.
// ============================================================================================================
bool PhyloTree::gpuScreenNNITileCleanRoom(double *out_max_rel, long long *out_nmove, long long *out_npass,
                                          double *out_tree_lnL, double *out_wall_tiled, double *out_wall_oracle,
                                          int *out_ntile, long long *out_bitexact, long long *out_nmoves_total) {
    if (!model || !site_rate || !aln) return false;
    int ns = aln->num_states;
    if (ns != 4 && ns != 20) return false;
    if (!model->isReversible() || model->getNMixtures() != 1 || model->isSiteSpecificModel()) return false;
    if (site_rate->getPInvar() > 0.0) return false;
    int ncat = site_rate->getNRate();
    int nptn = (int)aln->size();
    int ntax = (int)aln->getNSeq();
    if (ncat < 1 || ncat > 64) return false;
    double *eval = model->getEigenvalues();
    double *U    = model->getEigenvectors();
    double *Uinv = model->getInverseEigenvectors();
    if (!eval || !U || !Uinv) return false;
    vector<double> freq(ns, 0.0); model->getStateFrequency(freq.data(), 0);
    vector<double> UinvRowSum(ns, 0.0);
    for (int i = 0; i < ns; i++) { double s = 0; for (int j = 0; j < ns; j++) s += Uinv[i*ns+j]; UinvRowSum[i] = s; }
    vector<double> catRate(ncat), catProp(ncat);
    for (int c = 0; c < ncat; c++) { catRate[c] = site_rate->getRate(c); catProp[c] = site_rate->getProp(c); }

    if (!root || !root->isLeaf() || root->neighbors.empty()) return false;
    Node *R = root->neighbors[0]->node;
    if (R->isLeaf()) return false;

    map<Node*,int> nid; vector<Node*> nodes; vector<double> parentLen; vector<int> isLeafV, leafTax;
    function<void(Node*,Node*,double)> indexDfs = [&](Node *n, Node *dad, double lenToDad) {
        int myi = (int)nodes.size(); nid[n] = myi; nodes.push_back(n);
        parentLen.push_back(lenToDad);
        int lf = n->isLeaf() ? 1 : 0; isLeafV.push_back(lf);
        leafTax.push_back(lf ? aln->getSeqID(n->name) : -1);
        for (auto nb : n->neighbors) { if (nb->node == dad) continue; indexDfs(nb->node, n, nb->length); }
    };
    indexDfs(R, nullptr, 0.0);
    int nNodes = (int)nodes.size();

    vector<int> postInternal; vector<int> slot(nNodes, -1);
    function<void(Node*,Node*)> postDfs = [&](Node *n, Node *dad) {
        for (auto nb : n->neighbors) { if (nb->node == dad) continue; postDfs(nb->node, n); }
        if (!n->isLeaf()) { slot[nid[n]] = (int)postInternal.size(); postInternal.push_back(nid[n]); }
    };
    postDfs(R, nullptr);
    int nInternal = (int)postInternal.size();

    size_t ecStride = (size_t)ncat*ns*ns, exStride = (size_t)ncat*ns;
    vector<double> echild((size_t)nNodes*ecStride, 0.0);
    vector<double> expfac((size_t)nNodes*exStride, 0.0);
    for (int v = 0; v < nNodes; v++) {
        if (v == nid[R]) continue;
        double len_v = parentLen[v];
        for (int c = 0; c < ncat; c++) {
            double l = len_v * catRate[c];
            double ex[20]; for (int i = 0; i < ns; i++) ex[i] = exp(eval[i]*l);
            double *e = &echild[(size_t)v*ecStride + (size_t)c*ns*ns];
            for (int x = 0; x < ns; x++) for (int i = 0; i < ns; i++) e[x*ns+i] = U[x*ns+i]*ex[i];
            double *ef = &expfac[(size_t)v*exStride + (size_t)c*ns];
            for (int i = 0; i < ns; i++) ef[i] = ex[i];
        }
    }

    vector<unsigned char> tip((size_t)ntax*nptn);
    for (int v = 0; v < nNodes; v++) {
        if (!isLeafV[v]) continue;
        int tax = leafTax[v];
        if (tax < 0 || tax >= ntax) return false;
        for (int p = 0; p < nptn; p++) { int st = (int)aln->at(p)[tax]; tip[(size_t)tax*nptn+p] = (unsigned char)((st < ns) ? st : ns); }
    }
    vector<double> ptnFreq(nptn);
    for (int p = 0; p < nptn; p++) ptnFreq[p] = (double)aln->at(p).frequency;

    vector<int> node_nchild(nNodes, 0), node_child(nNodes*3, -1), node_leaf(nNodes, -1), node_slot(nNodes, -1);
    vector<double> node_parentLen(nNodes, 0.0);
    vector<int> parentNid(nNodes, -1);
    for (int v = 0; v < nNodes; v++) {
        Node *n = nodes[v];
        node_leaf[v] = isLeafV[v] ? leafTax[v] : -1;
        node_slot[v] = slot[v];
        node_parentLen[v] = parentLen[v];
        Node *dad = nullptr;
        for (auto nb : n->neighbors) { auto it = nid.find(nb->node); if (it != nid.end() && it->second < v) { dad = nb->node; parentNid[v] = it->second; break; } }
        int k = 0;
        for (auto nb : n->neighbors) {
            if (nb->node == dad) continue;
            if (k >= 3) return false;
            node_child[v*3+k] = nid[nb->node];
            k++;
        }
        node_nchild[v] = k;
    }

    int Rnid = nid[R];
    auto rit = nid.find(root); if (rit == nid.end()) return false;
    int rootLeafNid = rit->second;
    auto childDesc = [&](int xnid, int &ec, int &sl, int &lf) {
        ec = xnid;
        if (isLeafV[xnid]) { sl = -1; lf = leafTax[xnid]; } else { sl = slot[xnid]; lf = -1; }
    };

    vector<int> mv_u, mv_uIsRoot; vector<double> mv_bv;
    vector<int> n1a_ec,n1a_sl,n1a_lf, n1b_ec,n1b_sl,n1b_lf, n2a_ec,n2a_sl,n2a_lf, n2b_ec,n2b_sl,n2b_lf;
    vector<int> orc_n1, orc_n2, orc_w, orc_vswap;
    for (int v = 0; v < nNodes; v++) {
        if (v == Rnid || isLeafV[v]) continue;
        if (node_nchild[v] != 2) continue;
        int u = parentNid[v];
        if (u < 0) continue;
        int v1 = node_child[v*3+0], v2 = node_child[v*3+1];
        int w = -1, stayR = -1;
        if (u == Rnid) {
            for (int k = 0; k < node_nchild[u]; k++) { int c = node_child[u*3+k];
                if (c == v) continue;
                if (c == rootLeafNid) { stayR = c; continue; }
                w = c; }
            if (w < 0 || stayR < 0) continue;
        } else {
            for (int k = 0; k < node_nchild[u]; k++) { int c = node_child[u*3+k]; if (c != v) { w = c; break; } }
            if (w < 0) continue;
        }
        for (int mi = 0; mi < 2; mi++) {
            int vk_swap = (mi==0) ? v1 : v2;
            int vk_stay = (mi==0) ? v2 : v1;
            int e,s,l;
            mv_u.push_back(u); mv_uIsRoot.push_back(u==Rnid?1:0); mv_bv.push_back(node_parentLen[v]);
            childDesc(vk_swap, e,s,l); n1a_ec.push_back(e); n1a_sl.push_back(s); n1a_lf.push_back(l);
            if (u==Rnid) childDesc(stayR, e,s,l); else { e=-1; s=-1; l=-1; }
            n1b_ec.push_back(e); n1b_sl.push_back(s); n1b_lf.push_back(l);
            childDesc(w, e,s,l);       n2a_ec.push_back(e); n2a_sl.push_back(s); n2a_lf.push_back(l);
            childDesc(vk_stay, e,s,l); n2b_ec.push_back(e); n2b_sl.push_back(s); n2b_lf.push_back(l);
            orc_n1.push_back(u); orc_n2.push_back(v); orc_w.push_back(w); orc_vswap.push_back(vk_swap);
        }
    }
    int M = (int)mv_u.size();
    if (M == 0) return false;

    // a single tiled launch with a chosen forced_ntile (0 = auto)
    auto tileCall = [&](int forced, vector<double> &out, int *ntile) -> double {
        return gpu_screen_nni_tile_crosscheck(ns, nptn, ncat, ntax, nNodes, nInternal, Rnid,
            Uinv, U, UinvRowSum.data(), freq.data(), catProp.data(), eval, catRate.data(),
            echild.data(), expfac.data(), tip.data(), ptnFreq.data(),
            node_nchild.data(), node_child.data(), node_leaf.data(), node_slot.data(),
            node_parentLen.data(), postInternal.data(),
            M, mv_u.data(), mv_uIsRoot.data(), mv_bv.data(),
            n1a_ec.data(),n1a_sl.data(),n1a_lf.data(), n1b_ec.data(),n1b_sl.data(),n1b_lf.data(),
            n2a_ec.data(),n2a_sl.data(),n2a_lf.data(), n2b_ec.data(),n2b_sl.data(),n2b_lf.data(),
            forced, out.data(), nullptr, ntile);
    };

    // ---- AUTO tiled launcher (TIMED, the reported result): nTile auto-picked from free VRAM ----
    vector<double> moveLnL(M, (double)NAN);
    int ntileAuto = 1;
    double treeLnL = (double)NAN;
    double t0 = getRealTime();
    double rc = gpu_screen_nni_tile_crosscheck(ns, nptn, ncat, ntax, nNodes, nInternal, Rnid,
        Uinv, U, UinvRowSum.data(), freq.data(), catProp.data(), eval, catRate.data(),
        echild.data(), expfac.data(), tip.data(), ptnFreq.data(),
        node_nchild.data(), node_child.data(), node_leaf.data(), node_slot.data(),
        node_parentLen.data(), postInternal.data(),
        M, mv_u.data(), mv_uIsRoot.data(), mv_bv.data(),
        n1a_ec.data(),n1a_sl.data(),n1a_lf.data(), n1b_ec.data(),n1b_sl.data(),n1b_lf.data(),
        n2a_ec.data(),n2a_sl.data(),n2a_lf.data(), n2b_ec.data(),n2b_sl.data(),n2b_lf.data(),
        /*forced_ntile=*/0, moveLnL.data(), &treeLnL, &ntileAuto);
    double wall_tiled = getRealTime() - t0;
    if (rc != rc) return false;

    // ---- BONUS bit-identity (example scale only): tiling-invariance (forced 3,7) + new-vs-frozen-3b-ii ----
    // nTile=1 footprint (the upper + lowers + scratch); if it fits a comfortable budget run the bit-exact checks.
    long long bitexact = -1;   // -1 = SKIPPED (nTile=1 would OOM at this scale)
    size_t nt1Bytes = ((size_t)nNodes + (size_t)nInternal + 5) * (size_t)ncat*ns * (size_t)nptn * sizeof(double);
    if (nt1Bytes < (size_t)90 * 1000000000ULL) {
        bitexact = M;
        auto bitcmp = [&](const vector<double>&a, const vector<double>&b){
            for (int m=0;m<M;m++) if (memcmp(&a[m],&b[m],sizeof(double))!=0) return false; return true; };
        // forced nTile ∈ {3,7} (likely non-divisors of nptn -> exercise the ragged tail)
        for (int fk : {3, 7}) {
            if (fk > nptn) continue;
            vector<double> mv(M, (double)NAN); int nt=0; double r = tileCall(fk, mv, &nt);
            if (r != r || !bitcmp(mv, moveLnL)) bitexact = 0;
        }
        // new tiled nTile=1 path vs the frozen 3b-ii batch launcher (new-vs-old regression)
        vector<double> mvBatch(M, (double)NAN); double tlB = (double)NAN;
        double rb = gpu_screen_nni_batch_crosscheck(ns, nptn, ncat, ntax, nNodes, nInternal, Rnid,
            Uinv, U, UinvRowSum.data(), freq.data(), catProp.data(), eval, catRate.data(),
            echild.data(), expfac.data(), tip.data(), ptnFreq.data(),
            node_nchild.data(), node_child.data(), node_leaf.data(), node_slot.data(),
            node_parentLen.data(), postInternal.data(),
            M, mv_u.data(), mv_uIsRoot.data(), mv_bv.data(),
            n1a_ec.data(),n1a_sl.data(),n1a_lf.data(), n1b_ec.data(),n1b_sl.data(),n1b_lf.data(),
            n2a_ec.data(),n2a_sl.data(),n2a_lf.data(), n2b_ec.data(),n2b_sl.data(),n2b_lf.data(),
            mvBatch.data(), &tlB);
        if (rb != rb || !bitcmp(mvBatch, moveLnL)) bitexact = 0;
    }

    // ---- per-move 3a oracle cross-check (TIMED, THE GATE): M× full-postorder gpuScreenNNIFoldCleanRoom ----
    long long nmove=0, npass=0; double maxrel=0.0;
    double t1 = getRealTime();
    for (int m = 0; m < M; m++) {
        PhyloNode *n1 = (PhyloNode*) nodes[orc_n1[m]];
        PhyloNode *n2 = (PhyloNode*) nodes[orc_n2[m]];
        PhyloNeighbor *nei1 = (PhyloNeighbor*) n1->findNeighbor(nodes[orc_w[m]]);
        PhyloNeighbor *nei2 = (PhyloNeighbor*) n2->findNeighbor(nodes[orc_vswap[m]]);
        double oddf=0.0, olnL=(double)NAN;
        if (nei1 && nei2) gpuScreenNNIFoldCleanRoom(n1, n2, nei1, nei2, &oddf, &olnL);
        if (olnL != olnL) continue;
        nmove++;
        if (moveLnL[m] != moveLnL[m]) { maxrel = HUGE_VAL; continue; }
        double rel = (olnL != 0.0) ? fabs((moveLnL[m]-olnL)/olnL) : fabs(moveLnL[m]-olnL);
        if (rel > maxrel) maxrel = rel;
        if (rel <= 1e-9) npass++;
    }
    double wall_oracle = getRealTime() - t1;

    if (out_max_rel) *out_max_rel = maxrel;
    if (out_nmove) *out_nmove = nmove;
    if (out_npass) *out_npass = npass;
    if (out_tree_lnL) *out_tree_lnL = treeLnL;
    if (out_wall_tiled) *out_wall_tiled = wall_tiled;
    if (out_wall_oracle) *out_wall_oracle = wall_oracle;
    if (out_ntile) *out_ntile = ntileAuto;
    if (out_bitexact) *out_bitexact = bitexact;
    if (out_nmoves_total) *out_nmoves_total = M;   // total enumerated moves (bit-identity is over all M; nmove = oracle-eligible subset)
    return true;
}

// ============================================================================================================
// TS.2 Integration Step 1 — gpuScreenNNIRank: the LEAN per-round screener for the NNI search front-end. Same
// build + 2-move enumeration as gpuScreenNNITileCleanRoom (3c), then ONE auto-tiled launch — NO oracle loop, NO
// bit-identity re-launches. Returns, per inner branch, the GPU fixed-length (pre-reopt) lnL of its 2 NNI swaps,
// keyed by pairInteger(parentNode->id, childNode->id) — the SAME key the CPU nniBranches use (pairInteger is
// symmetric; the screener's (u=parent,v=child) orientation == getBestNNIForBran's TOWARD_ROOT reorientation).
// Each move's lnL == the CPU tsr_pre/preloglh (== computeLikelihoodBranch at OLD lengths) — bit-exact-validated
// by the whole 3a→3c chain. branchBest[id] = max(swap0_lnL, swap1_lnL); branchBoth[id] = (swap0_lnL, swap1_lnL).
// Caller (evaluateNNIsScreened) uses this to validate the GPU per round (Step 1) / rank top-k (Step 2). Returns
// false on ineligibility / no moves / CUDA error => caller falls back to pure CPU evaluateNNIs (byte-identical).
// Rebuilt every call (reads LIVE branch lengths + current eigendecomposition) — NEVER cache across rounds.
// ============================================================================================================
bool PhyloTree::gpuScreenNNIRank(std::map<int,double> &branchBest,
                                 std::map<int,std::pair<double,double> > *branchBoth,
                                 int *out_ntile, double *out_wall_screen) {
    branchBest.clear(); if (branchBoth) branchBoth->clear();
    // TS_SCREEN_SPLIT (2026-06-26): host-rebuild vs GPU-launch split of the screener's per-round wall. The
    // out_wall_screen timer (:1462) covers ONLY the gpu_screen_nni_tile_crosscheck launch; the DFS reindex +
    // echild/expfac exp-table + tip gather BELOW (the per-round stateless rebuild, P2) is untimed. env-gated => off=byte-identical.
    static bool g_scrsplit_init=false; static bool g_scrsplit=false; static double g_scr_hostbuild=0.0,g_scr_launch=0.0; static long g_scr_calls=0;
    if(!g_scrsplit_init){ g_scrsplit=(getenv("TS_SCREEN_SPLIT")!=nullptr); g_scrsplit_init=true; }
    double _scr_entry = g_scrsplit ? getRealTime() : 0.0;
    if (!model || !site_rate || !aln) return false;
    int ns = aln->num_states;
    if (ns != 4 && ns != 20) return false;
    if (!model->isReversible() || model->getNMixtures() != 1 || model->isSiteSpecificModel()) return false;
    // A3 (FIX-B-PROPER screener half, 2026-06-26): +I is now SUPPORTED — the per-move kernel k2_derv_mix_inv adds the
    // branch-independent invariant term pinv*baseinvar[ptn], and catRate/catProp already carry the 1/(1-pinv) rescale
    // (getRate/getProp for RateGammaInvar). base_invar is computed below (the computePtnInvar replica) and passed to
    // gpu_screen_nni_tile_crosscheck with pinv. (Was: `if (getPInvar()>0) return false` => CPU fallback.) Gate:
    // --ts-fused-check pass@1e-9 on +I vs CPU preloglh. Tree-lnL (out_tree_lnL) stays +I-incomplete (diagnostic only;
    // unused for ranking). FIXED-pinvar +I still optimises fine here (it's only the screener score, no pinv opt).
    int ncat = site_rate->getNRate();
    int nptn = (int)aln->size();
    int ntax = (int)aln->getNSeq();
    if (ncat < 1 || ncat > 64) return false;
    double *eval = model->getEigenvalues();
    double *U    = model->getEigenvectors();
    double *Uinv = model->getInverseEigenvectors();
    if (!eval || !U || !Uinv) return false;
    vector<double> freq(ns, 0.0); model->getStateFrequency(freq.data(), 0);
    vector<double> UinvRowSum(ns, 0.0);
    for (int i = 0; i < ns; i++) { double s = 0; for (int j = 0; j < ns; j++) s += Uinv[i*ns+j]; UinvRowSum[i] = s; }
    vector<double> catRate(ncat), catProp(ncat);
    for (int c = 0; c < ncat; c++) { catRate[c] = site_rate->getRate(c); catProp[c] = site_rate->getProp(c); }

    if (!root || !root->isLeaf() || root->neighbors.empty()) return false;
    Node *R = root->neighbors[0]->node;
    if (R->isLeaf()) return false;

    map<Node*,int> nid; vector<Node*> nodes; vector<double> parentLen; vector<int> isLeafV, leafTax;
    function<void(Node*,Node*,double)> indexDfs = [&](Node *n, Node *dad, double lenToDad) {
        int myi = (int)nodes.size(); nid[n] = myi; nodes.push_back(n);
        parentLen.push_back(lenToDad);
        int lf = n->isLeaf() ? 1 : 0; isLeafV.push_back(lf);
        leafTax.push_back(lf ? aln->getSeqID(n->name) : -1);
        for (auto nb : n->neighbors) { if (nb->node == dad) continue; indexDfs(nb->node, n, nb->length); }
    };
    indexDfs(R, nullptr, 0.0);
    int nNodes = (int)nodes.size();

    vector<int> postInternal; vector<int> slot(nNodes, -1);
    function<void(Node*,Node*)> postDfs = [&](Node *n, Node *dad) {
        for (auto nb : n->neighbors) { if (nb->node == dad) continue; postDfs(nb->node, n); }
        if (!n->isLeaf()) { slot[nid[n]] = (int)postInternal.size(); postInternal.push_back(nid[n]); }
    };
    postDfs(R, nullptr);
    int nInternal = (int)postInternal.size();

    size_t ecStride = (size_t)ncat*ns*ns, exStride = (size_t)ncat*ns;
    vector<double> echild((size_t)nNodes*ecStride, 0.0);
    vector<double> expfac((size_t)nNodes*exStride, 0.0);
    for (int v = 0; v < nNodes; v++) {
        if (v == nid[R]) continue;
        double len_v = parentLen[v];
        for (int c = 0; c < ncat; c++) {
            double l = len_v * catRate[c];
            double ex[20]; for (int i = 0; i < ns; i++) ex[i] = exp(eval[i]*l);
            double *e = &echild[(size_t)v*ecStride + (size_t)c*ns*ns];
            for (int x = 0; x < ns; x++) for (int i = 0; i < ns; i++) e[x*ns+i] = U[x*ns+i]*ex[i];
            double *ef = &expfac[(size_t)v*exStride + (size_t)c*ns];
            for (int i = 0; i < ns; i++) ef[i] = ex[i];
        }
    }

    vector<unsigned char> tip((size_t)ntax*nptn);
    for (int v = 0; v < nNodes; v++) {
        if (!isLeafV[v]) continue;
        int tax = leafTax[v];
        if (tax < 0 || tax >= ntax) return false;
        for (int p = 0; p < nptn; p++) { int st = (int)aln->at(p)[tax]; tip[(size_t)tax*nptn+p] = (unsigned char)((st < ns) ? st : ns); }
    }
    vector<double> ptnFreq(nptn);
    for (int p = 0; p < nptn; p++) ptnFreq[p] = (double)aln->at(p).frequency;

    vector<int> node_nchild(nNodes, 0), node_child(nNodes*3, -1), node_leaf(nNodes, -1), node_slot(nNodes, -1);
    vector<double> node_parentLen(nNodes, 0.0);
    vector<int> parentNid(nNodes, -1);
    for (int v = 0; v < nNodes; v++) {
        Node *n = nodes[v];
        node_leaf[v] = isLeafV[v] ? leafTax[v] : -1;
        node_slot[v] = slot[v];
        node_parentLen[v] = parentLen[v];
        Node *dad = nullptr;
        for (auto nb : n->neighbors) { auto it = nid.find(nb->node); if (it != nid.end() && it->second < v) { dad = nb->node; parentNid[v] = it->second; break; } }
        int k = 0;
        for (auto nb : n->neighbors) {
            if (nb->node == dad) continue;
            if (k >= 3) return false;
            node_child[v*3+k] = nid[nb->node];
            k++;
        }
        node_nchild[v] = k;
    }

    int Rnid = nid[R];
    auto rit = nid.find(root); if (rit == nid.end()) return false;
    int rootLeafNid = rit->second;
    auto childDesc = [&](int xnid, int &ec, int &sl, int &lf) {
        ec = xnid;
        if (isLeafV[xnid]) { sl = -1; lf = leafTax[xnid]; } else { sl = slot[xnid]; lf = -1; }
    };

    vector<int> mv_u, mv_uIsRoot; vector<double> mv_bv;
    vector<int> n1a_ec,n1a_sl,n1a_lf, n1b_ec,n1b_sl,n1b_lf, n2a_ec,n2a_sl,n2a_lf, n2b_ec,n2b_sl,n2b_lf;
    vector<int> orc_n1, orc_n2;   // CPU-side (parent,child) nids per move (for the branchID map)
    for (int v = 0; v < nNodes; v++) {
        if (v == Rnid || isLeafV[v]) continue;
        if (node_nchild[v] != 2) continue;
        int u = parentNid[v];
        if (u < 0) continue;
        int v1 = node_child[v*3+0], v2 = node_child[v*3+1];
        int w = -1, stayR = -1;
        if (u == Rnid) {
            for (int k = 0; k < node_nchild[u]; k++) { int c = node_child[u*3+k];
                if (c == v) continue;
                if (c == rootLeafNid) { stayR = c; continue; }
                w = c; }
            if (w < 0 || stayR < 0) continue;
        } else {
            for (int k = 0; k < node_nchild[u]; k++) { int c = node_child[u*3+k]; if (c != v) { w = c; break; } }
            if (w < 0) continue;
        }
        for (int mi = 0; mi < 2; mi++) {
            int vk_swap = (mi==0) ? v1 : v2;
            int vk_stay = (mi==0) ? v2 : v1;
            int e,s,l;
            mv_u.push_back(u); mv_uIsRoot.push_back(u==Rnid?1:0); mv_bv.push_back(node_parentLen[v]);
            childDesc(vk_swap, e,s,l); n1a_ec.push_back(e); n1a_sl.push_back(s); n1a_lf.push_back(l);
            if (u==Rnid) childDesc(stayR, e,s,l); else { e=-1; s=-1; l=-1; }
            n1b_ec.push_back(e); n1b_sl.push_back(s); n1b_lf.push_back(l);
            childDesc(w, e,s,l);       n2a_ec.push_back(e); n2a_sl.push_back(s); n2a_lf.push_back(l);
            childDesc(vk_stay, e,s,l); n2b_ec.push_back(e); n2b_sl.push_back(s); n2b_lf.push_back(l);
            orc_n1.push_back(u); orc_n2.push_back(v);
        }
    }
    int M = (int)mv_u.size();
    if (M == 0) return false;

    // A3 (+I): per-pattern invariant base = Σ_{const states s} freq[s] (replica of computePtnInvar, == ptn_invar[p]/pinv).
    // The move kernel adds pinv*base_invar[ptn] so the GPU per-move lnL == the CPU +I preloglh. Only built when +I.
    double pinvScreen = site_rate->getPInvar();
    vector<double> base_invar(nptn, 0.0);
    if (pinvScreen > 0.0) {
        const int ambi_aa[] = {4+8, 32+64, 512+1024};   // B=N|D, Z=Q|E, U=I|L (mirror optimizeParametersJOLT)
        int SU = (int)aln->STATE_UNKNOWN;
        for (int p = 0; p < nptn; p++) {
            int cc = (int)aln->at(p).const_char;
            if (cc > SU)                          base_invar[p] = 0.0;
            else if (cc == SU)                    base_invar[p] = 1.0;
            else if (cc < ns)                     base_invar[p] = freq[cc];
            else if (aln->seq_type == SEQ_DNA)   { double s=0; int cs=cc-ns+1; for (int x=0;x<ns;x++) if (cs & (1<<x)) s+=freq[x]; base_invar[p]=s; }
            else if (aln->seq_type == SEQ_PROTEIN){ double s=0; int cs=cc-ns;   if (cs>=0 && cs<3) for (int x=0;x<11;x++) if (ambi_aa[cs] & (1<<x)) s+=freq[x]; base_invar[p]=s; }
        }
    }

    // ---- ONE auto-tiled launch (lean: no oracle, no bit-identity) ----
    vector<double> moveLnL(M, (double)NAN);
    int ntileAuto = 1; double treeLnL = (double)NAN;
    double t0 = getRealTime();
    double rc = gpu_screen_nni_tile_crosscheck(ns, nptn, ncat, ntax, nNodes, nInternal, Rnid,
        Uinv, U, UinvRowSum.data(), freq.data(), catProp.data(), eval, catRate.data(),
        echild.data(), expfac.data(), tip.data(), ptnFreq.data(),
        node_nchild.data(), node_child.data(), node_leaf.data(), node_slot.data(),
        node_parentLen.data(), postInternal.data(),
        M, mv_u.data(), mv_uIsRoot.data(), mv_bv.data(),
        n1a_ec.data(),n1a_sl.data(),n1a_lf.data(), n1b_ec.data(),n1b_sl.data(),n1b_lf.data(),
        n2a_ec.data(),n2a_sl.data(),n2a_lf.data(), n2b_ec.data(),n2b_sl.data(),n2b_lf.data(),
        /*forced_ntile=*/0, moveLnL.data(), &treeLnL, &ntileAuto,
        base_invar.data(), pinvScreen);   // A3 (+I): pinv<=0 -> non-+I (bit-identical)
    double _scr_launch_s = getRealTime() - t0;
    if (out_wall_screen) *out_wall_screen = _scr_launch_s;
    if (rc != rc) return false;   // CUDA error
    if (g_scrsplit) {   // TS_SCREEN_SPLIT: per-round host-rebuild vs GPU-launch (host_build = the untimed P2 stateless rebuild)
        g_scr_hostbuild += (t0 - _scr_entry); g_scr_launch += _scr_launch_s; g_scr_calls++;
        printf("TS-SCRSPLIT call %ld host_build_s %.4f gpu_launch_s %.4f M %d cum_host %.3f cum_launch %.3f\n",
               g_scr_calls, t0-_scr_entry, _scr_launch_s, M, g_scr_hostbuild, g_scr_launch); fflush(stdout);
    }

    // GROUND-TRUTH DUMP (TS_SCREEN_DUMP=1): per move, the ACTUAL swapped/stayed subtree PhyloNode ids + lengths +
    // leaf-flags + the GPU move lnL, keyed by branch id. Joined offline with the CPU per-move dump (val[].node?Nei_it)
    // to see EXACTLY which subtrees/lengths the broken move uses. Localises the alternating one-swap-wrong bug.
    static bool ts_gdumped = false;
    if (getenv("TS_SCREEN_DUMP") && !ts_gdumped) {
        ts_gdumped = true;
        for (int m = 0; m < M && m < 48; m++) {
            int id = pairInteger(nodes[orc_n1[m]]->id, nodes[orc_n2[m]]->id);
            printf("TS-GDUMP m=%d mi=%d bid=%d u=%d v=%d w=%d swapc=%d stayc=%d "
                   "swapLeaf=%d stayLeaf=%d bv=%.6f bswap=%.6f bstay=%.6f bw=%.6f g=%.6f\n",
                m, m&1, id, nodes[orc_n1[m]]->id, nodes[orc_n2[m]]->id, nodes[n2a_ec[m]]->id,
                nodes[n1a_ec[m]]->id, nodes[n2b_ec[m]]->id, n1a_lf[m], n2b_lf[m],
                mv_bv[m], node_parentLen[n1a_ec[m]], node_parentLen[n2b_ec[m]], node_parentLen[n2a_ec[m]],
                moveLnL[m]);
        }
        fflush(stdout);
    }

    // ---- reduce the 2 moves/branch -> per-branch maps keyed by pairInteger(parent->id, child->id) ----
    // The 2 swaps of a branch are pushed consecutively (mi=0,1) with the same (orc_n1,orc_n2); fill .first then
    // .second in enumeration order. branchBest = max of the FINITE swaps; a branch with both swaps NaN is left
    // out of branchBest (caller falls back to CPU for it). Key == the CPU nniBranches key (pairInteger symmetric).
    map<int,bool> firstFilled;
    for (int m = 0; m < M; m++) {
        int id = pairInteger(nodes[orc_n1[m]]->id, nodes[orc_n2[m]]->id);
        double l = moveLnL[m];
        if (l == l) {   // finite -> contribute to the per-branch best
            auto bit = branchBest.find(id);
            if (bit == branchBest.end()) branchBest[id] = l; else if (l > bit->second) bit->second = l;
        }
        if (branchBoth) {
            if (!firstFilled[id]) { (*branchBoth)[id] = std::make_pair(l, l); firstFilled[id] = true; }
            else (*branchBoth)[id].second = l;
        }
    }
    if (out_ntile) *out_ntile = ntileAuto;
    return true;
}

// ============================================================================================================
// G.8.1b — clean-room single-edge branch derivative for PROFILE MIXTURES: df/ddf SUMMED over the N*ncat regimes,
// validated vs CPU computeLikelihoodDerv. Combines the two-sub-root central-edge split (gpuComputeEdgeDervCleanRoom)
// with the per-class eigen assembly (gpuComputeTreeLnLCleanRoomMix). Returns df = d(lnL)/dt (un-negated, like the
// single-model path; computeFuncDerv negates downstream); *out_ddf the 2nd derivative; *out_lnL the tree lnL at t.
// Eligibility gate IDENTICAL to the mixture lnL path (+I/fused/PMSF/nonrev/single-model -> NaN -> CPU) so an edge
// never gets a GPU lnL but a CPU derivative or vice versa.
// ============================================================================================================
double PhyloTree::gpuComputeEdgeDervCleanRoomMix(PhyloNeighbor *dad_branch, PhyloNode *dad, double *out_ddf, double *out_lnL) {
    if (!model || !site_rate || !aln) return (double)NAN;
    int ns = aln->num_states;
    if (ns != 20 && ns != 4) return (double)NAN;
    if (!model->isReversible()) return (double)NAN;
    int N = model->getNMixtures();
    if (N <= 1) return (double)NAN;                       // single-model -> the non-mix derivative path
    if (model->isSiteSpecificModel()) return (double)NAN; // PMSF stays on CPU
    if (site_rate->getPInvar() > 0.0) return (double)NAN;  // +I omitted in the clean-room sweep -> CPU
    int ncat = site_rate->getNRate();
    int nptn = (int)aln->size();
    int ntax = (int)aln->getNSeq();
    if (ncat < 1) return (double)NAN;
    int R = N*ncat;
    ModelMixture *mix = dynamic_cast<ModelMixture*>(model);
    if (!mix || (int)mix->size() != N) return (double)NAN;
    if (mix->isFused()) return (double)NAN;               // LG4M/LG4X = 1:1 class<->rate, not the N*ncat product
    Node *node = dad_branch->node;
    Node *dadN = dad;
    if (!node || !dadN) return (double)NAN;

    // ---- per-class eigen (component pointers, AVX-padding intrinsic), freq, per-regime weight = w_m*catProp_c ----
    std::vector<double> Uinv((size_t)N*ns*ns), UinvRowSum((size_t)N*ns), freqC((size_t)N*ns);
    std::vector<double> evalC((size_t)N*ns), Uc((size_t)N*ns*ns);
    std::vector<double> catRate(ncat), catProp(ncat), wreg((size_t)R);
    for (int c = 0; c < ncat; c++) { catRate[c] = site_rate->getRate(c); catProp[c] = site_rate->getProp(c); }
    for (int m = 0; m < N; m++) {
        ModelMarkov *cm = (ModelMarkov*)(*mix)[m];
        double *ev = cm->getEigenvalues(), *U = cm->getEigenvectors(), *Ui = cm->getInverseEigenvectors();
        if (!ev || !U || !Ui) return (double)NAN;
        for (int i = 0; i < ns; i++) evalC[(size_t)m*ns+i] = ev[i];
        for (int x = 0; x < ns*ns; x++) { Uc[(size_t)m*ns*ns+x] = U[x]; Uinv[(size_t)m*ns*ns+x] = Ui[x]; }
        for (int i = 0; i < ns; i++) { double s=0; for (int j=0;j<ns;j++) s += Ui[i*ns+j]; UinvRowSum[(size_t)m*ns+i]=s; }
        double wf[64]; model->getStateFrequency(wf, m);   // ns<=20
        for (int x = 0; x < ns; x++) freqC[(size_t)m*ns+x] = wf[x];
        double wm = model->getMixtureWeight(m);
        for (int c = 0; c < ncat; c++) wreg[(size_t)m*ncat+c] = wm * catProp[c];
    }
    double t = dad_branch->length;

    // ---- two-sub-root DFS, central edge (node<->dadN) excluded from both (identical to single-model derivative) ----
    map<Node*,int> nid; vector<Node*> nodes; vector<double> parentLen; vector<int> isLeafV, leafTax;
    function<void(Node*,Node*,double)> indexDfs = [&](Node *n, Node *par, double lenToPar) {
        int myi = (int)nodes.size(); nid[n]=myi; nodes.push_back(n); parentLen.push_back(lenToPar);
        int lf = n->isLeaf()?1:0; isLeafV.push_back(lf); leafTax.push_back(lf?aln->getSeqID(n->name):-1);
        for (auto nb : n->neighbors) { if (nb->node==par) continue; indexDfs(nb->node, n, nb->length); }
    };
    indexDfs(node, dadN, 0.0);
    indexDfs(dadN, node, 0.0);
    int nNodes = (int)nodes.size();

    vector<int> postInternal; vector<int> slot(nNodes, -1);
    function<void(Node*,Node*)> postDfs = [&](Node *n, Node *par) {
        for (auto nb : n->neighbors) { if (nb->node==par) continue; postDfs(nb->node, n); }
        if (!n->isLeaf()) { slot[nid[n]]=(int)postInternal.size(); postInternal.push_back(nid[n]); }
    };
    postDfs(node, dadN);
    postDfs(dadN, node);
    int nInternal = (int)postInternal.size();
    int nodeSlot = node->isLeaf() ? -1 : slot[nid[node]];
    int nodeLeafTax = node->isLeaf() ? leafTax[nid[node]] : -1;
    int dadSlot  = dadN->isLeaf() ? -1 : slot[nid[dadN]];
    int dadLeafTax = dadN->isLeaf() ? leafTax[nid[dadN]] : -1;

    // ---- per-regime echild[v][r=m*ncat+c][x][i] = U_m[x][i]*exp(eval_m[i]*rate_c*parentLen[v]); skip sub-roots ----
    size_t ecStride = (size_t)R*ns*ns;
    vector<double> echild((size_t)nNodes*ecStride, 0.0);
    for (int v = 0; v < nNodes; v++) {
        if (v == nid[node] || v == nid[dadN]) continue;
        double len_v = parentLen[v];
        for (int m = 0; m < N; m++) {
            const double *ev = &evalC[(size_t)m*ns]; const double *U = &Uc[(size_t)m*ns*ns];
            for (int c = 0; c < ncat; c++) {
                double l = len_v * catRate[c]; int r = m*ncat + c;
                double ex[20]; for (int i=0;i<ns;i++) ex[i]=exp(ev[i]*l);
                double *e = &echild[(size_t)v*ecStride + (size_t)r*ns*ns];
                for (int x=0;x<ns;x++) for (int i=0;i<ns;i++) e[x*ns+i] = U[x*ns+i]*ex[i];
            }
        }
    }

    vector<unsigned char> tip((size_t)ntax*nptn);
    for (int v = 0; v < nNodes; v++) {
        if (!isLeafV[v]) continue;
        int tax = leafTax[v]; if (tax<0 || tax>=ntax) return (double)NAN;
        for (int p=0;p<nptn;p++){ int st=(int)aln->at(p)[tax]; tip[(size_t)tax*nptn+p]=(unsigned char)((st<ns)?st:ns); }
    }
    vector<double> ptnFreq(nptn);
    for (int p=0;p<nptn;p++) ptnFreq[p]=(double)aln->at(p).frequency;

    // ---- descriptors: ALL internal (isRoot=0); central edge excluded at the two sub-roots ----
    vector<int> dRoot(nInternal,0), dNch(nInternal), dOut(nInternal);
    vector<int> dChildNode(nInternal*3,-1), dChildIsLeaf(nInternal*3,0), dChildLeaf(nInternal*3,-1), dChildSlot(nInternal*3,-1);
    for (int idx=0; idx<nInternal; idx++){
        int vi=postInternal[idx]; Node *n=nodes[vi];
        Node *par=nullptr;
        if (n!=node && n!=dadN) {
            for (auto nb:n->neighbors){ auto it=nid.find(nb->node); if(it!=nid.end() && it->second<vi){ par=nb->node; break; } }
        }
        dOut[idx]=slot[vi];
        int k=0;
        for (auto nb:n->neighbors){
            if (nb->node==par) continue;
            if (n==node && nb->node==dadN) continue;   // exclude central edge at sub-root node
            if (n==dadN && nb->node==node) continue;   // exclude central edge at sub-root dadN
            if (k>=3) return (double)NAN;
            int cv=nid[nb->node]; dChildNode[idx*3+k]=cv;
            if (isLeafV[cv]){ dChildIsLeaf[idx*3+k]=1; dChildLeaf[idx*3+k]=leafTax[cv]; }
            else            { dChildIsLeaf[idx*3+k]=0; dChildSlot[idx*3+k]=slot[cv]; }
            k++;
        }
        dNch[idx]=k;
    }

    return gpu_derv_crosscheck_mix(ns, nptn, ncat, N, ntax, nNodes, nInternal,
        Uinv.data(), UinvRowSum.data(), freqC.data(), wreg.data(), echild.data(), tip.data(), ptnFreq.data(),
        dRoot.data(), dNch.data(), dOut.data(),
        dChildNode.data(), dChildIsLeaf.data(), dChildLeaf.data(), dChildSlot.data(),
        nodeSlot, nodeLeafTax, dadSlot, dadLeafTax, evalC.data(), catRate.data(), t, out_ddf, out_lnL);
}

// ============================================================================================================
// G.8.2.1a — clean-room ALL-BRANCH derivative for PROFILE MIXTURES: df/ddf for EVERY edge in ONE postorder + ONE
// preorder sweep (Ji-2020 linear-time), rooted at an internal node, vs the single-edge gpuComputeEdgeDervCleanRoomMix
// (two-sub-root split, ONE edge). Fills four parallel out-vectors (one entry per NON-ROOT node v): the edge v->parent
// gets childNodes[k]=v, parentNodes[k]=parent, dfOut[k]=d(lnL)/db_v, ddfOut[k]=d²(lnL)/db_v². Mirrors the eigen/echild
// gather of gpuComputeEdgeDervCleanRoomMix and the single-root topology of optimizeParametersJOLT (additionally builds
// the per-node expfac = exp(eval_m·rate_c·b_parent) the preorder kernel needs). Returns false on ineligibility/CUDA
// error (same +I/fused/PMSF/nonrev/single-model gate as the lnL mix path). Read-only (no host/device state persists).
// ============================================================================================================
bool PhyloTree::gpuComputeAllBranchDervCleanRoomMix(std::vector<Node*>& childNodes, std::vector<Node*>& parentNodes,
                                                    std::vector<double>& dfOut, std::vector<double>& ddfOut,
                                                    const double *parentLenOverride, double alphaOverride, double pinvOverride) {
    childNodes.clear(); parentNodes.clear(); dfOut.clear(); ddfOut.clear();
    if (!model || !site_rate || !aln) return false;
    int ns = aln->num_states;
    if (ns != 20 && ns != 4) return false;
    if (!model->isReversible()) return false;
    int N = model->getNMixtures();
    if (N <= 1) return false;
    if (model->isSiteSpecificModel()) return false;
    // A1 (+I): +I handled — the invariant is branch-independent (enters only the 1/Lp denominator via base_invar_comb).
    int ncat = site_rate->getNRate();
    int nptn = (int)aln->size();
    int ntax = (int)aln->getNSeq();
    if (ncat < 1) return false;
    int R = N*ncat;
    ModelMixture *mix = dynamic_cast<ModelMixture*>(model);
    if (!mix || (int)mix->size() != N || mix->isFused()) return false;
    if (!root || !root->isLeaf() || root->neighbors.empty()) return false;
    Node *Rt = root->neighbors[0]->node;   // internal root (IQ-TREE roots at a leaf; lnL is reversible-invariant)
    if (!Rt || Rt->isLeaf()) return false;

    // ---- per-class eigen (Uinv down-map, Uc up-map, evalC) + freq + per-regime weight (== gpuComputeEdgeDervCleanRoomMix) ----
    std::vector<double> Uinv((size_t)N*ns*ns), UinvRowSum((size_t)N*ns), freqC((size_t)N*ns);
    std::vector<double> evalC((size_t)N*ns), Uc((size_t)N*ns*ns);
    std::vector<double> catRate(ncat), catProp(ncat), wreg((size_t)R);
    for (int c = 0; c < ncat; c++) { catRate[c] = site_rate->getRate(c); catProp[c] = site_rate->getProp(c); }
    // G.8.2.1b: optional iterate-alpha override (joint-optimiser gradient at an FD-perturbed alpha) — see the lnL path.
    bool alphaOv = (alphaOverride > 0.0 && ncat > 1);
    if (alphaOv) gpu_discrete_gamma_mean(alphaOverride, ncat, catRate.data());   // catRate now mean-1 ρ_c (NO pinv rescale)
    // A1 (+I) the (1-pinv) bridge (== the lnL path): rebase the variable-part rates/props from the stored pinv0 to
    // pinv_eff. Skipped at pinv_eff<=0 => the validated +G derivative path is byte-identical.
    double pinv0 = site_rate->getPInvar();
    double pinv_eff = (pinvOverride >= 0.0) ? pinvOverride : pinv0;
    if (pinv_eff > 0.0) {
        double f0 = 1.0 - pinv0, fe = 1.0 - pinv_eff;
        for (int c = 0; c < ncat; c++) { double rho = alphaOv ? catRate[c] : catRate[c] * f0; catRate[c] = rho / fe; catProp[c] = catProp[c] * fe / f0; }
    }
    std::vector<double> wM(N);
    for (int m = 0; m < N; m++) {
        ModelMarkov *cm = (ModelMarkov*)(*mix)[m];
        double *ev = cm->getEigenvalues(), *U = cm->getEigenvectors(), *Ui = cm->getInverseEigenvectors();
        if (!ev || !U || !Ui) return false;
        for (int i = 0; i < ns; i++) evalC[(size_t)m*ns+i] = ev[i];
        for (int x = 0; x < ns*ns; x++) { Uc[(size_t)m*ns*ns+x] = U[x]; Uinv[(size_t)m*ns*ns+x] = Ui[x]; }
        for (int i = 0; i < ns; i++) { double s=0; for (int j=0;j<ns;j++) s += Ui[i*ns+j]; UinvRowSum[(size_t)m*ns+i]=s; }
        double wf[64]; model->getStateFrequency(wf, m);
        for (int x = 0; x < ns; x++) freqC[(size_t)m*ns+x] = wf[x];
        double wm = model->getMixtureWeight(m); wM[m] = wm;
        for (int c = 0; c < ncat; c++) wreg[(size_t)m*ncat+c] = wm * catProp[c];
    }
    // A1 (+I): COMBINED invariant base_invar_comb[p] = Σ_m w_m·base_invar_m[p] (pinv applied in-kernel by k2_derv_mix_inv).
    // base_invar_m uses class m's π (freqC[m]); same const-site logic as the lnL path. Live weights wM (synced per outer).
    std::vector<double> base_invar_comb;
    if (pinv_eff > 0.0) {
        base_invar_comb.assign(nptn, 0.0);
        const int ambi_aa[] = {4+8, 32+64, 512+1024};
        int SU = (int)aln->STATE_UNKNOWN;
        for (int p = 0; p < nptn; p++) {
            int cc = (int)aln->at(p).const_char; double acc = 0.0;
            for (int m = 0; m < N; m++) {
                const double *sf = &freqC[(size_t)m*ns]; double bi = 0.0;
                if (cc > SU)                            bi = 0.0;
                else if (cc == SU)                      bi = 1.0;
                else if (cc < ns)                       bi = sf[cc];
                else if (aln->seq_type == SEQ_DNA)     { double s=0; int cs=cc-ns+1; for (int x=0;x<ns;x++) if (cs & (1<<x)) s+=sf[x]; bi=s; }
                else if (aln->seq_type == SEQ_PROTEIN) { double s=0; int cs=cc-ns;   if (cs>=0 && cs<3) for (int x=0;x<11;x++) if (ambi_aa[cs] & (1<<x)) s+=sf[x]; bi=s; }
                acc += wM[m] * bi;
            }
            base_invar_comb[p] = acc;
        }
    }

    // ---- single-root topology rooted at Rt (indexDfs), flat arrays (mirrors optimizeParametersJOLT) ----
    std::map<Node*,int> nid; std::vector<Node*> nodes, parentNode; std::vector<double> parentLen; std::vector<int> leafTax;
    std::vector<std::vector<int>> childList;
    std::function<void(Node*,Node*,double)> indexDfs = [&](Node *n, Node *dad, double lenToDad) {
        int myi=(int)nodes.size(); nid[n]=myi; nodes.push_back(n); parentNode.push_back(dad); parentLen.push_back(lenToDad);
        leafTax.push_back(n->isLeaf()?aln->getSeqID(n->name):-1); childList.push_back(std::vector<int>());
        for (auto nb:n->neighbors){ if(nb->node==dad) continue; indexDfs(nb->node,n,nb->length); } };
    indexDfs(Rt, nullptr, 0.0);
    int nNodes = (int)nodes.size();
    // G.8.2.1b: optional iterate-branch override (joint-optimiser gradient at iterate b) — indexed by this function's
    // DFS nid; the echild + nodeParentLen (preorder expfac) builds below consume it transparently. nullptr => live tree.
    if (parentLenOverride) for (int v = 0; v < nNodes; v++) parentLen[v] = parentLenOverride[v];
    for (int i=0;i<nNodes;i++){ Node *n=nodes[i], *dad=parentNode[i];
        for (auto nb:n->neighbors){ if(nb->node==dad) continue; childList[i].push_back(nid[nb->node]); }
        if ((int)childList[i].size()>3) return false; }

    // ---- tip states + pattern frequencies ----
    std::vector<unsigned char> tip((size_t)ntax*nptn);
    for (int i=0;i<nNodes;i++){ if(leafTax[i]<0) continue; int tax=leafTax[i]; if(tax<0||tax>=ntax) return false;
        for (int p=0;p<nptn;p++){ int st=(int)aln->at(p)[tax]; tip[(size_t)tax*nptn+p]=(unsigned char)((st<ns)?st:ns); } }
    std::vector<double> ptnFreq(nptn); for (int p=0;p<nptn;p++) ptnFreq[p]=(double)aln->at(p).frequency;

    // ---- echild[v][r][x][i]=U_m[x][i]*exp(eval_m[i]*rate_c*parentLen[v]) + expfac[v][r][i]=exp(...) (root: no parent edge) ----
    size_t ecStride=(size_t)R*ns*ns, exStride=(size_t)R*ns;
    std::vector<double> echild((size_t)nNodes*ecStride,0.0), expfac((size_t)nNodes*exStride,0.0);
    for (int v=0;v<nNodes;v++){ if(parentNode[v]==nullptr) continue;
        double len_v=parentLen[v];
        for (int m=0;m<N;m++){ const double *ev=&evalC[(size_t)m*ns]; const double *U=&Uc[(size_t)m*ns*ns];
            for (int c=0;c<ncat;c++){ double l=len_v*catRate[c]; int r=m*ncat+c;
                double ex[20]; for(int i=0;i<ns;i++) ex[i]=exp(ev[i]*l);
                double *e=&echild[(size_t)v*ecStride+(size_t)r*ns*ns];
                for(int x=0;x<ns;x++) for(int i=0;i<ns;i++) e[x*ns+i]=U[x*ns+i]*ex[i];
                double *ef=&expfac[(size_t)v*exStride+(size_t)r*ns]; for(int i=0;i<ns;i++) ef[i]=ex[i]; } } }

    // ---- flat topology arrays for the launcher ----
    std::vector<int> nodeNch(nNodes), nodeChild((size_t)nNodes*3,-1), nodeLeaf(nNodes); std::vector<double> nodeParentLen(nNodes);
    for (int i=0;i<nNodes;i++){ nodeNch[i]=(int)childList[i].size(); nodeLeaf[i]=leafTax[i]; nodeParentLen[i]=parentLen[i];
        for (int k=0;k<(int)childList[i].size()&&k<3;k++) nodeChild[(size_t)i*3+k]=childList[i][k]; }

    std::vector<double> dfV(nNodes,0.0), ddfV(nNodes,0.0);
    double rc = gpu_allbranch_derv_crosscheck_mix(ns,nptn,ncat,N,ntax,nNodes,/*root=*/nid[Rt],
        Uinv.data(), Uc.data(), UinvRowSum.data(), freqC.data(), wreg.data(), evalC.data(), catRate.data(),
        echild.data(), expfac.data(), tip.data(), ptnFreq.data(),
        nodeNch.data(), nodeChild.data(), nodeLeaf.data(), nodeParentLen.data(),
        dfV.data(), ddfV.data(),
        pinv_eff, (pinv_eff > 0.0 ? base_invar_comb.data() : nullptr));   // A1 (+I)
    if (std::isnan(rc)) return false;

    for (int v=0;v<nNodes;v++){ if(parentNode[v]==nullptr) continue;
        childNodes.push_back(nodes[v]); parentNodes.push_back(parentNode[v]); dfOut.push_back(dfV[v]); ddfOut.push_back(ddfV[v]); }
    return true;
}

// ============================================================================================================
// G.2.1b — GPU override for computeLikelihoodDervPointer (byte-matches ComputeLikelihoodDervType). STATELESS:
// the single-edge df/ddf is recomputed clean-room from the live tree each call (no device-resident theta /
// partials -> no coherence hole, per the verified contract). Writes UN-negated df/ddf (computeFuncDerv negates).
// Delegates to the saved CPU Derv pointer if the regime is unsupported (NaN).
void PhyloTree::computeLikelihoodDervGPU(PhyloNeighbor *dad_branch, PhyloNode *dad, double *df, double *ddf) {
    double gddf = 0.0, glnL = 0.0;
    double gdf = gpuComputeEdgeDervCleanRoom(dad_branch, dad, &gddf, &glnL);
    if (std::isnan(gdf)) {
        if (cpuComputeLikelihoodDervPointer) { (this->*cpuComputeLikelihoodDervPointer)(dad_branch, dad, df, ddf); return; }
        *df = gdf; *ddf = gddf; return;
    }
    *df = gdf; *ddf = gddf;   // un-negated: d(lnL)/dt, d2(lnL)/dt2 (computeFuncDerv negates downstream)
    static bool announced = false;
    if (!announced) { announced = true;
        printf("[GPU-DERV] computeLikelihoodDervGPU active (clean-room single-edge df/ddf, stateless)\n"); }
}

// G.2.1b — GPU override for computeLikelihoodFromBufferPointer (byte-matches ComputeLikelihoodFromBufferType,
// no args). STATELESS: the from-buffer lnL at the current branch lengths == the whole-tree lnL (reversible), so
// recompute it clean-room from the live tree (which already reflects current_it->length set by computeFuncDerv).
double PhyloTree::computeLikelihoodFromBufferGPU() {
    double l = gpuComputeTreeLnLCleanRoom(nullptr);
    if (std::isnan(l)) {
        if (cpuComputeLikelihoodFromBufferPointer) return (this->*cpuComputeLikelihoodFromBufferPointer)();
        return l;
    }
    static bool announced = false;
    if (!announced) { announced = true;
        printf("[GPU-FROMBUF] computeLikelihoodFromBufferGPU active (clean-room whole-tree lnL, stateless)\n"); }
    return l;
}

// ============================================================================================================
// G.2.0b/G.2.1b — gated funnel hook (called LAST in PhyloTree::setLikelihoodKernel). Saves the ISA-set CPU
// Branch/Derv/FromBuffer pointers and installs the GPU overrides. Idempotent + re-applied on every funnel
// re-invocation (the ISA setter resets the pointers to CPU each call; this re-installs GPU). INSTALL gate is
// params-level (known even before the model is built); the per-call helpers re-check model-level conditions
// (reversible / single-mixture / non-site-specific / no +I / 4-or-20-state) and fall back to CPU.
// ============================================================================================================
void PhyloTree::setLikelihoodKernelGPU() {
    if (!params || !params->gpu) return;
    // G.4.2: under --jolt, the stateless GPU Branch/Derv/FromBuffer overrides must NOT install. JOLT is the ONLY
    // GPU path — it replaces ModelFactory::optimizeParameters wholesale for eligible candidates (+G/base), while
    // INELIGIBLE candidates (+I, +R, +FO, mixture) must fall back to the PURE CPU likelihood (normal speed), NOT
    // the slow stateless GPU clean-room sweep (G.2.2a measured ~25 min/model = 50-100x). Keeping both active would
    // (a) make the +I/+R tail run on the slow stateless path (timeout) and (b) reduce optimizeParametersJOLT's
    // self-check to GPU-vs-GPU; with this no-op the self-check's computeLikelihood() is a genuine CPU recompute.
    if (params->jolt) return;
    if (!aln || (aln->num_states != 4 && aln->num_states != 20)) return;
    if (isSuperTree()) return;
    // reject regimes whose output reads per-pattern/-category buffers the GPU overrides do not populate
    if (params->print_site_lh != WSL_NONE) return;
    if (params->print_partition_lh) return;
    if (params->print_site_rate) return;
    if (params->print_ancestral_sequence != AST_NONE) return;
    if (params->aLRT_replicates > 0 || params->localbp_replicates > 0 || params->aLRT_test || params->aBayes_test) return;
    if (params->gbo_replicates > 0 || params->num_bootstrap_samples > 0) return;
    if (!params->dating_method.empty()) return;
    if (params->pll) return;

    // save the genuine ISA-set CPU pointers (guard against re-saving our own GPU overrides), then install GPU.
    if (computeLikelihoodBranchPointer != &PhyloTree::computeLikelihoodBranchGPU)
        cpuComputeLikelihoodBranchPointer = computeLikelihoodBranchPointer;
    if (computeLikelihoodDervPointer != &PhyloTree::computeLikelihoodDervGPU)
        cpuComputeLikelihoodDervPointer = computeLikelihoodDervPointer;
    if (computeLikelihoodFromBufferPointer != &PhyloTree::computeLikelihoodFromBufferGPU)
        cpuComputeLikelihoodFromBufferPointer = computeLikelihoodFromBufferPointer;
    computeLikelihoodBranchPointer     = &PhyloTree::computeLikelihoodBranchGPU;       // G.2.0b lnL
    computeLikelihoodDervPointer       = &PhyloTree::computeLikelihoodDervGPU;         // G.2.1b single-edge df/ddf
    computeLikelihoodFromBufferPointer = &PhyloTree::computeLikelihoodFromBufferGPU;   // G.2.1b from-buffer lnL

    static bool announced = false;
    if (!announced) { announced = true;
        printf("[GPU-KERNEL] setLikelihoodKernelGPU: Branch+Derv+FromBuffer -> GPU (stateless clean-room); "
               "fixed_branch_length=%d num_states=%d (branch-opt %s)\n",
               params->fixed_branch_length, aln->num_states,
               params->fixed_branch_length == BRLEN_FIX ? "fixed/-blfix" : "GPU"); }
}

// ============================================================================================================
// Phase G.6 — host callback handed to gpu_jolt_optimize for DNA free-Q models (the eigensystem MOVES during the
// optimise). It applies a trial free-Q vector q[nFreeQ] to the LIVE model (gpuSetFreeParamsDecompose -> param_spec
// rate-class mapping + the G-T=1 gauge + decomposeRateMatrix), then copies the fresh eigensystem back to the
// launcher's host buffers. extern "C" to match the jolt_qdecompose_fn C ABI. ctx = the model + ns. The launcher is
// mutex-serialized and the model is thread-local, so this mutates only the calling thread's own model; the final
// optimised Q is written back deterministically after gpu_jolt_optimize returns (do NOT rely on the launcher's
// internal Q thrashing leaving the model in any particular state).
// ============================================================================================================
namespace { struct JoltQCtx { ModelSubst* model; int ns; }; }
extern "C" void jolt_qdecompose_intree(void* vctx, const double* q, double* eval, double* U, double* Uinv) {
    JoltQCtx* c = reinterpret_cast<JoltQCtx*>(vctx);
    c->model->gpuSetFreeParamsDecompose(q);
    int ns = c->ns;
    memcpy(eval, c->model->getEigenvalues(),         sizeof(double) * ns);
    memcpy(U,    c->model->getEigenvectors(),         sizeof(double) * (size_t)ns * ns);
    memcpy(Uinv, c->model->getInverseEigenvectors(),  sizeof(double) * (size_t)ns * ns);
}

// ============================================================================================================
// Phase G.4.2 — GPU JOLT joint-gradient optimiser for ONE candidate model. Builds the clean-room inputs from the
// LIVE objects (mirroring gpuComputeTreeLnLCleanRoom), runs the validated G.4.1b joint LM driver on the GPU,
// writes the optimised (197 branches + alpha) back through the cache-invalidating setters, and self-checks that a
// FRESH CPU computeLikelihood() reproduces the JOLT lnL. Returns NaN if JOLT-ineligible / CUDA error -> caller
// falls back to the standard CPU path.
// ============================================================================================================
double PhyloTree::optimizeParametersJOLT(int fixed_len, bool brlenOnly, bool leanTail, int brlenMaxIter) {
    // ---- eligibility gate (the validated G.4.1/G.4.1b scope: fixed-Q reversible, ns in {4,20}, no +I, gamma-or-uniform) ----
    // G.4.3a diagnostic: JOLT_DEBUG=1 logs the gate decision per candidate (which decline reason, or engage), so we can
    // tell whether an ineligible family (e.g. +F) REACHES this hook and is declined by a specific gate, vs never arrives
    // (staged-search dispatches it elsewhere). Env-gated => zero cost in production; touches no CPU-path behaviour.
    static const bool JOLT_DBG = (getenv("JOLT_DEBUG") != nullptr);
    if (JOLT_DBG) {
        string mn = model ? model->getName() : string("(nullmodel)");
        // freqtype: 1=USER_DEFINED 2=EQUAL 3=EMPIRICAL(+F) 4=ESTIMATE(+FO) (tools.h StateFreqType)
        fprintf(stderr, "[JOLT-GATE] reached hook model=%s freqtype=%d ns=%d rev=%d nmix=%d ssm=%d ndim=%d pinv=%.4g ncat=%d alpha=%.4g fixedlen=%d\n",
                mn.c_str(), model ? (int)model->getFreqType() : -1, aln ? aln->num_states : -1,
                model ? (int)model->isReversible() : -1, model ? model->getNMixtures() : -1,
                model ? (int)model->isSiteSpecificModel() : -1, model ? model->getNDim() : -999,
                site_rate ? site_rate->getPInvar() : -1.0, site_rate ? site_rate->getNRate() : -1,
                (site_rate && site_rate->getNRate() > 1) ? site_rate->getGammaShape() : -1.0, fixed_len);
        fflush(stderr);
    }
    #define JOLT_DECLINE(why) do { if (JOLT_DBG) { fprintf(stderr, "[JOLT-GATE] decline reason=%s\n", why); fflush(stderr); } return (double)NAN; } while (0)
    if (!model || !site_rate || !aln) JOLT_DECLINE("null-ptr");
    if (fixed_len != BRLEN_OPTIMIZE) JOLT_DECLINE("brlen-mode");   // JOLT optimises branches; other brlen modes -> CPU
    int ns = aln->num_states;
    if (ns != 4 && ns != 20) JOLT_DECLINE("num-states");
    if (!model->isReversible() || model->getNMixtures() != 1 || model->isSiteSpecificModel()) JOLT_DECLINE("nonrev/mixture/ssm");
    // TS.1 pre-mortem latent gap (red-team #3): the kernel's nptn == aln->size() EXCLUDES model_factory->unobserved_ptns,
    // so an ascertainment-bias model (+ASC) would receive an UN-corrected lnL. The full tail's rel<=1e-6 gate catches this
    // (-> NaN -> CPU fallback); the lean tail dropped that catch, so decline +ASC explicitly here. Inert for non-ASC models
    // (model_factory ptr + ASCType/ASC_NONE already in scope via phylotree.h -> modelfactory.h + utils/tools.h).
    if (model_factory && model_factory->getASC() != ASC_NONE) JOLT_DECLINE("ascertainment-bias");
    // G.6.1: free substitution params (DNA HKY..GTR) — the eigensystem MOVES, so JOLT FD-optimises them via the
    // decompose callback. ON BY DEFAULT (validated job 170795329: DNA -m MF 70 engage / 9 decline [8 +R + 1 pure-+I],
    // best-by-BIC == CPU F81+F+G4, worst write-back rel 6.2e-12, ZERO mismatch). JOLT_NO_FREEQ disables it (debug/A-B).
    // Restricted to ns==4 reversible, getNDim()<=5, FIXED freqs (exclude +FO, FREQ_ESTIMATE, whose free freq dims are
    // NOT yet handled). AA fixed-Q (getNDim()==0) is unaffected.
    static const bool JOLT_FREEQ_EN = (getenv("JOLT_NO_FREEQ") == nullptr);
    int nFreeQ = 0;
    {
        int ndim = model->getNDim();
        // audit RISK-1 hardening: tied-frequency DNA types (+FRY/+F1112/... = FREQ_DNA_*) contribute 1-3 FREE freq
        // params to getNDim(), which gpuGetFreeParams packs into the Q-vector tail; the launcher then mis-clamps them
        // as exchangeabilities ([MINQ,MAXQ]=[1e-4,100] vs the correct ~[0,1]) -> a coherent-but-SUBOPTIMAL lnL that
        // still passes the write-back gate (which checks GPU/CPU coherence, not optimality). Require nFreqParams==0 so
        // the entire getVariables() tail is exchangeabilities. +FQ/+F (0 freq dims) still engage; tied-freq is not in
        // the default -m MF DNA set, so this declines such explicit user models to CPU (defensive, no live regression).
        bool freeQok = JOLT_FREEQ_EN && ndim > 0 && ndim <= 5 && ns == 4 &&
                       model->getFreqType() != FREQ_ESTIMATE && model->isReversible() &&
                       nFreqParams(model->getFreqType()) == 0;
        if (ndim != 0 && !freeQok) JOLT_DECLINE("free-subst-params");  // +FO / tied-freq / AA-GTR / production free-Q -> CPU
        nFreeQ = freeQok ? ndim : 0;
    }
    if (brlenOnly) nFreeQ = 0;   // TS.1: brlen-only reopt holds Q FIXED (no free-Q optimisation; same eligibility gate)
    int ncat = site_rate->getNRate();
    if (ncat < 1 || ncat > 64) JOLT_DECLINE("ncat-range");
    // G.4.3b audit fix: discriminate the rate model by isGammaRate() (robust) NOT getGammaShape() (which is a
    // POSITIVE inherited value for RateFree/+R -> the old check let +R / +R+I wrongly engage JOLT with uniform
    // proportions + mean-gamma rates, silently wrong since writeback precedes the self-check). JOLT only implements
    // the MEAN discrete-gamma (Yang-1994) discretisation, so require exactly GAMMA_CUT_MEAN: this declines +R
    // (isGammaRate()==0), +R+I, and the MEDIAN gamma variant +Gm/+I+Gm (isGammaRate()==GAMMA_CUT_MEDIAN).
    // G.5.1a: let PURE +R (FreeRate, no +I) through to the launcher ONLY under JOLT_RGRADCHECK — it runs the
    // weight-gradient FD self-check then declines to CPU (the +R optimiser branch is G.5.1b, not yet wired).
    // G.5.1b / G.5.1c (ladder 2a): ENGAGE the in-tree +R joint LM. Regime: +R with NO +I yet (pinv<=0; +I+R is ladder
    // 2b), Q either FIXED (AA / JC,F81: nFreeQ==0) OR FREE (2a — HKY..GTR+R: the diagonal-LM optimises the free-Q axis
    // (gradQ/ddQ, :2094) JOINTLY with the rate/weight axes (g_y/g_z), which are independent — no pinv coupling when
    // optPinv==0, so applyPinv(0) is identity and the +R seeding/gauge are unchanged). Both rates+weights FREE
    // (getNDim()==2*ncat-2; a user-fixed +R{...} or a mid-EM substep -> CPU), ncat<=JOLT_FREERATE_MAXCAT (harness: R4
    // reproducible / R6 multimodal => conservative <=4, R5 unvalidated), FULL model-param path only (brlenOnly/lean
    // holds +R FIXED). (Was: `&& nFreeQ == 0` — that XOR was a gate restriction only; the LM never enforced it.)
    static const int JOLT_FREERATE_MAXCAT = 4;
    // G.5.1d (ladder 2b): +I+R now engages. RateFree must be FULLY free (rates+weights = 2K-2 dims); RateFreeInvar adds
    // one free dim (pinv) when present (getNDim == 2K-1), so strip it before the 2K-2 test. Fixed-pinv keeps getNDim==
    // 2K-2 -> freeRateOK true but the +I guard (isFixPInvar, :1946) then declines it; a user-fixed +R{...} -> rfDim<2K-2
    // -> CPU. The +I (1-pinv) bridge through the kernel's meanR/bprop basis is byte-identical at pinv=0 (pure +R / 2a).
    int rfDim = site_rate->getNDim() - ((site_rate->getPInvar() > 0.0 && !site_rate->isFixPInvar()) ? 1 : 0);
    bool freeRateOK = (ncat > 1 && site_rate->isFreeRate()
                       && rfDim == 2*ncat - 2 && ncat <= JOLT_FREERATE_MAXCAT && !brlenOnly);
    bool rgcheck = (ncat > 1 && site_rate->isFreeRate() && site_rate->getPInvar() <= 0.0 && getenv("JOLT_RGRADCHECK") != nullptr);
    if (ncat > 1 && site_rate->isGammaRate() != GAMMA_CUT_MEAN && !freeRateOK && !rgcheck) JOLT_DECLINE("non-mean-gamma");
    // G.4.3b — +I (proportion of invariant sites) is now JOINTLY optimised by JOLT, but ONLY for +I+G
    // (RateGammaInvar: getProp(c)=(1-pinv)/K, standard mean-1 discrete-gamma rates). Pure +I (RateInvar, ncat==1)
    // rescales getRate=1/(1-pinv) -> out of JOLT scope -> CPU. A user-FIXED pinv, or no constant sites (pinvMax->0
    // degenerate), also fall to CPU. The invariant term L_p += pinv*base_invar[p] is added in the kernel; the joint
    // LM step moves pinv alongside the branches + alpha (same machinery that absorbed alpha in G.4.1b).
    static const double JOLT_MIN_PINVAR = 1e-6;          // == MIN_PINVAR (model/rateinvar.h)
    double pinv0 = site_rate->getPInvar();
    int optPinv = 0;
    if (pinv0 > 0.0) {
        if (site_rate->isFixPInvar())                                JOLT_DECLINE("fixed-pinvar");
        if (ncat <= 1)                                               JOLT_DECLINE("pure-pinvar-no-gamma");  // RateInvar getRate=1/(1-pinv) -> CPU (ncat>1 already => mean-gamma per the check above)
        if (params && params->no_rescale_gamma_invar)                JOLT_DECLINE("no-rescale-gamma-invar"); // GPU unconditionally rescales rates by 1/(1-pinv); this flag disables IQ-TREE's rescale -> mismatch -> CPU
        if (aln->frac_const_sites <= 2.0*JOLT_MIN_PINVAR)            JOLT_DECLINE("no-const-sites");
        optPinv = 1;
    }
    if (brlenOnly) optPinv = 0;   // TS.1: brlen-only reopt holds p_invar FIXED
    // AUDIT FIX (job 172316260, GTR+I+G4): setting optPinv=0 for brlen-only does NOT merely HOLD pinv fixed -- it also
    // zeroes the additive invariant term (base_invar is computed only `if (optPinv)`, :1971) AND drops the 1/(1-pinv)
    // gamma-rate rescale (applyPinv(0) => rates=meanR un-rescaled). So on +I the device computes lnL AND the LM
    // gradients (pdf/pddf) against the WRONG objective -- it optimises branches incorrectly, not just mis-reports:
    // empirically a SYSTEMATIC ~21-nat bias, 837/837 JOLT_AUDIT DRIFT>1e-6 on GTR+I+G4 vs 8.8e-12 on +I-free LG+G4.
    // The final tree only survived because the subsequent exact-CPU pass recovers the MLE (invariant patterns are
    // branch-insensitive), but the in-loop curScore + move ranking were silently +I-wrong. DECLINE +I in the lean/
    // brlen-only path -> CPU optimizeAllBranches(1) (exact), mirroring the ASC decline (:1858). SCOPED to brlenOnly so
    // the already-correct FULL +I+G joint path (optPinv=1, base_invar populated, rel<=1e-6 self-check :2068) is
    // UNTOUCHED; the screener likewise declines +I at :1309. FIX-B-PROPER = plumb pinv0 through the lean kernels +
    // screener to re-enable +I on the GPU lean path (kernel work, logged follow-up; needs the rel<=1e-6 discipline).
    if (brlenOnly && pinv0 > 0.0) JOLT_DECLINE("invar-sites-brlenonly");

    // ---- model eigen factors (alpha-independent; same convention as the clean-room lnL) ----
    double *eval = model->getEigenvalues();
    double *U    = model->getEigenvectors();
    double *Uinv = model->getInverseEigenvectors();
    if (!eval || !U || !Uinv) return (double)NAN;
    vector<double> UinvRowSum(ns, 0.0);
    for (int i = 0; i < ns; i++) { double s = 0; for (int j = 0; j < ns; j++) s += Uinv[i*ns+j]; UinvRowSum[i] = s; }
    vector<double> catProp(ncat), catRate0(ncat);
    for (int c = 0; c < ncat; c++) { catProp[c] = site_rate->getProp(c); catRate0[c] = site_rate->getRate(c); }   // G.5.1: +R rates/weights
    // G.6 free-Q: the initial free exchangeabilities (model->getVariables()[1..nFreeQ], raw rates), the ctx the
    // decompose callback binds to, and the output Q buffer. All empty/no-op for fixed-Q (nFreeQ==0).
    vector<double> q0vec(nFreeQ > 0 ? nFreeQ : 0), outQ(nFreeQ > 0 ? nFreeQ : 0);
    if (nFreeQ > 0) model->gpuGetFreeParams(q0vec.data());
    JoltQCtx qctx{ model, ns };

    // --jolt-diag (A3): H1 = the once-per-call HOST rebuild (DFS reindex + O(ntax*nptn) tip[] recompaction + flat arrays)
    double _jd_h1_t0 = params->jolt_diag ? getRealTime() : 0.0;

    // ---- topology rooted at internal node R (IQ-TREE roots at a leaf; lnL is reversible-invariant) ----
    if (!root || !root->isLeaf() || root->neighbors.empty()) return (double)NAN;
    Node *R = root->neighbors[0]->node;
    if (R->isLeaf()) return (double)NAN;

    map<Node*,int> nid;
    vector<Node*> nodes, parentNode;
    vector<double> parentLen;
    vector<int> leafTax;
    vector<vector<int>> childList;
    function<void(Node*,Node*,double)> indexDfs = [&](Node *n, Node *dad, double lenToDad) {
        int myi = (int)nodes.size(); nid[n] = myi; nodes.push_back(n);
        parentNode.push_back(dad); parentLen.push_back(lenToDad);
        leafTax.push_back(n->isLeaf() ? aln->getSeqID(n->name) : -1);
        childList.push_back(vector<int>());
        for (auto nb : n->neighbors) { if (nb->node == dad) continue; indexDfs(nb->node, n, nb->length); }
    };
    indexDfs(R, nullptr, 0.0);
    int nNodes = (int)nodes.size();
    // children must be recorded AFTER all indices assigned (a node's child indices are known once its subtree is visited)
    for (int i = 0; i < nNodes; i++) {
        Node *n = nodes[i], *dad = parentNode[i];
        for (auto nb : n->neighbors) { if (nb->node == dad) continue; childList[i].push_back(nid[nb->node]); }
        if ((int)childList[i].size() > 3) return (double)NAN;   // >3 children (only R has 3) -> unsupported
    }

    // ---- compact tip states + pattern frequencies + flat topology arrays ----
    int nptn = (int)aln->size(), ntax = (int)aln->getNSeq();
    vector<unsigned char> tip((size_t)ntax*nptn);
    for (int i = 0; i < nNodes; i++) {
        if (leafTax[i] < 0) continue; int tax = leafTax[i];
        if (tax < 0 || tax >= ntax) return (double)NAN;
        for (int p = 0; p < nptn; p++) { int st = (int)aln->at(p)[tax]; tip[(size_t)tax*nptn+p] = (unsigned char)((st < ns) ? st : ns); }
    }
    vector<double> ptnFreq(nptn);
    for (int p = 0; p < nptn; p++) ptnFreq[p] = (double)aln->at(p).frequency;

    // G.4.3b — pinv-independent invariant base per pattern (== ptn_invar[p]/pinv; replicates the constant-site
    // logic of PhyloTree::computePtnInvar, phylotreesse.cpp:560). base_invar[p] = Σ_{states s with which EVERY
    // taxon is compatible as a constant site} freq[s]: const_char==STATE_UNKNOWN -> 1 ; <ns -> freq[const_char] ;
    // DNA/PROTEIN ambiguous -> sum over compatible states ; >STATE_UNKNOWN -> 0 (a variable pattern). Multiplying
    // by pinv reproduces IQ-TREE's own ptn_invar exactly, so the final CPU self-check is a genuine parity gate.
    vector<double> base_invar(nptn, 0.0);
    double pinvMax = aln->frac_const_sites;
    if (optPinv) {
        vector<double> sf(ns, 0.0); model->getStateFrequency(sf.data(), 0);
        const int ambi_aa[] = {4+8, 32+64, 512+1024};   // B=N|D, Z=Q|E, U=I|L
        int SU = (int)aln->STATE_UNKNOWN;
        for (int p = 0; p < nptn; p++) {
            int cc = (int)aln->at(p).const_char;
            if (cc > SU)                            base_invar[p] = 0.0;
            else if (cc == SU)                      base_invar[p] = 1.0;
            else if (cc < ns)                       base_invar[p] = sf[cc];
            else if (aln->seq_type == SEQ_DNA)     { double s=0; int cs=cc-ns+1; for (int x=0;x<ns;x++) if (cs & (1<<x)) s+=sf[x]; base_invar[p]=s; }
            else if (aln->seq_type == SEQ_PROTEIN) { double s=0; int cs=cc-ns;   if (cs>=0 && cs<3) for (int x=0;x<11;x++) if (ambi_aa[cs] & (1<<x)) s+=sf[x]; base_invar[p]=s; }
        }
    }

    vector<int> nodeNch(nNodes), nodeChild(nNodes*3, -1), nodeLeaf(nNodes);
    vector<double> nodeParentLen(nNodes);
    for (int i = 0; i < nNodes; i++) {
        nodeNch[i] = (int)childList[i].size(); nodeLeaf[i] = leafTax[i]; nodeParentLen[i] = parentLen[i];
        for (int k = 0; k < (int)childList[i].size() && k < 3; k++) nodeChild[i*3+k] = childList[i][k];
    }

    double jd_h1 = params->jolt_diag ? getRealTime() - _jd_h1_t0 : 0.0;   // --jolt-diag: end H1 (host rebuild)

    double alpha0 = (ncat > 1) ? site_rate->getGammaShape() : 1.0;
    int optAlpha = (!brlenOnly && ncat > 1 && !site_rate->isFixGammaShape()) ? 1 : 0;   // TS.1: brlen-only holds alpha FIXED
    if (freeRateOK) optAlpha = 0;   // G.5.1b / red-team CRITICAL-1: +R has no alpha; setGammaShape(outAlpha) would recompute (clobber) the FreeRate rates

    // ---- run the JOLT optimiser on the GPU ----
    vector<double> outBrlen(nNodes, 0.0); double outAlpha = alpha0; double outPinv = pinv0; int outIters = 0;
    vector<double> outRates(freeRateOK ? ncat : 0), outProps(freeRateOK ? ncat : 0);   // G.5.1b: +R optimised rates/weights (writeback below)
    double _jd_dev_t0 = params->jolt_diag ? getRealTime() : 0.0;   // --jolt-diag: device-call wall start
    double joltLnL = gpu_jolt_optimize(ns, nptn, ncat, ntax, nNodes, /*root=*/nid[R],
        Uinv, UinvRowSum.data(), U, eval, catProp.data(), tip.data(), ptnFreq.data(),
        nodeNch.data(), nodeChild.data(), nodeLeaf.data(), nodeParentLen.data(),
        alpha0, optAlpha, /*maxiter=*/brlenMaxIter,
        base_invar.data(), pinv0, optPinv, JOLT_MIN_PINVAR, pinvMax,
        catRate0.data(), (freeRateOK ? 1 : (rgcheck ? 2 : 0)),   // G.5.1b: 1=ENGAGE the +R joint LM; 2=RGRADCHECK-only then decline to CPU
        nFreeQ, (nFreeQ > 0 ? q0vec.data() : nullptr), jolt_qdecompose_intree, &qctx,   // G.6: DNA free-Q
        (nFreeQ > 0 ? outQ.data() : nullptr),
        outBrlen.data(), &outAlpha, &outPinv, &outIters,
        (freeRateOK ? outRates.data() : nullptr), (freeRateOK ? outProps.data() : nullptr));   // G.5.1b: optimised +R rates/weights
    if (params->jolt_diag) {   // --jolt-diag (A3): per-call H1 (host rebuild) vs device wall; echild tax printed by the CU TU
        double jd_dev = getRealTime() - _jd_dev_t0;
        printf("JOLT-DIAG-HOST H1=%.6f device=%.6f iters=%d ntax=%d nptn=%d\n", jd_h1, jd_dev, outIters, ntax, nptn);
    }
    if (std::isnan(joltLnL)) {
        static bool warned = false;
        if (!warned) { warned = true; printf("[JOLT] gpu_jolt_optimize returned NaN -> CPU fallback (optimizeParameters)\n"); }
        return (double)NAN;
    }

    // ---- write the optimised branch lengths back (both directed neighbours of each edge v -> parent) ----
    for (int v = 0; v < nNodes; v++) {
        Node *child = nodes[v], *par = parentNode[v];
        if (!par) continue;                                     // R: no parent edge (covered as some node's child edge)
        Neighbor *fwd = par->findNeighbor(child); Neighbor *bwd = child->findNeighbor(par);
        if (fwd) fwd->length = outBrlen[v];
        if (bwd) bwd->length = outBrlen[v];
    }
    if (leanTail) {
        // TS.1 (reborn / L1) lean in-loop entry: brlen-only reopt (optAlpha=optPinv=nFreeQ=0 => model params UNCHANGED,
        // no setters needed). Reproduce the CPU optimizeAllBranches(1) coherence contract: invalidate partials
        // (clearAllPartialLH = the same flag-flip dirty set the all-branch CPU sweep produces, recomputed lazily by
        // the next NNI round), trust the device-returned lnL for curScore, and SKIP the full CPU computeLikelihood()
        // self-check (the ModelFinder-only D4 gain-eraser, plan §17.1). NaN was already handled above -> CPU fallback.
        clearAllPartialLH();
        // TS.1 coherence pre-mortem AUDIT (JOLT_AUDIT=1; OFF by default => the production lean path is byte-identical).
        // Both red/blue teams' single residual: JOLT lnL fidelity (rel<=1e-6) was validated ONLY on FITTED ModelFinder
        // trees, never on the INTERMEDIATE far-from-optimum trees this NNI loop feeds JOLT; the lean tail dropped the
        // full-tail rel<=1e-6 catch (see :2065 below) for a finite-but-WRONG device lnL. When set, recompute the CPU lnL
        // at the written-back lengths and LOG rel WITHOUT gating (still return joltLnL) -> ONE search measures max(rel)
        // over all intermediate-tree calls, converting "unverified assumption" into a measured fact.
        if (getenv("JOLT_AUDIT")) {
            double cpuLnL = computeLikelihood();   // fresh CPU postorder at the JOLT-written lengths (partials just cleared)
            double arel = (cpuLnL != 0.0) ? fabs((joltLnL - cpuLnL) / cpuLnL) : fabs(joltLnL - cpuLnL);
            double aabs = fabs(joltLnL - cpuLnL);
            static int    audit_n = 0;
            static double audit_max_rel = 0.0, audit_max_abs = 0.0;
            audit_n++;
            if (arel > audit_max_rel) audit_max_rel = arel;
            if (aabs > audit_max_abs) audit_max_abs = aabs;
            printf("[JOLT-AUDIT] call=%d joltLnL=%.6f cpuLnL=%.6f rel=%.3e abs=%.6f %s | run_max_rel=%.3e run_max_abs=%.6f\n",
                   audit_n, joltLnL, cpuLnL, arel, aabs,
                   (arel <= 1e-6 ? "OK" : "DRIFT>1e-6"), audit_max_rel, audit_max_abs);
            fflush(stdout);
            clearAllPartialLH();   // computeLikelihood left partials VALID; restore the production dirty-set so state matches
        }
        setCurScore(joltLnL);
        return joltLnL;
    }
    // ---- write Q + alpha + pinv back through the setters, then invalidate ALL partial-LH + transition caches ----
    // G.6: set the model to the OPTIMISED free-Q deterministically (the launcher's internal Q thrashing leaves the
    // model in an indeterminate state) — gpuSetFreeParamsDecompose applies param_spec + re-decomposes, so the
    // self-check below recomputes the CPU lnL at exactly the JOLT optimum (a genuine GPU-vs-CPU write-back gate).
    if (nFreeQ > 0) model->gpuSetFreeParamsDecompose(outQ.data());
    if (optPinv) site_rate->setPInvar(outPinv);                 // G.4.3b: sets p_invar + recomputes rates (RateGammaInvar::setPInvar)
    if (optAlpha) site_rate->setGammaShape(outAlpha);           // sets gamma_shape + recomputes the discrete rates
    if (freeRateOK) {   // G.5.1b: write the JOLT-optimised FreeRate rates + weights (gauged Σ w·r=1 == RateFree's
        for (int c = 0; c < ncat; c++) {   // meanRates()==1 convention) via the public setters. optAlpha forced 0 above => setGammaShape did NOT run.
            site_rate->setRate(c, outRates[c]); site_rate->setProp(c, outProps[c]); }
    }
    clearAllPartialLH();                                        // brlen + alpha + pinv + Q + (R) changed -> partials, theta & ptn_invar stale

    // ---- self-check: a FRESH CPU computeLikelihood() must reproduce the JOLT lnL (the load-bearing G.4.2a gate) ----
    double cpuLnL = computeLikelihood();
    double rel = (cpuLnL != 0.0) ? fabs((joltLnL - cpuLnL) / cpuLnL) : fabs(joltLnL - cpuLnL);
    static int report_count = 0;
    // G.4.3a: use model->getName() (includes the +F/+FO freq suffix) not model->name (matrix only) — the old print
    // dropped +F, mislabelling LG+F+G4 as "LG+G4" and making +F JOLT-coverage invisible/uncountable. Cap raised so a
    // full -m TESTONLY logs every engagement (coverage is now measurable).
    string joltModelName = model->getName() + (ncat > 1 ? ((freeRateOK ? "+R" : "+G") + std::to_string(ncat)) : string(""));
    // The per-model GPU-vs-CPU validation line is dev diagnostics (fires once per
    // candidate -> ~60-90 lines on an AA coarse pass). Gate behind JOLT_DEBUG so a
    // production --jolt/--ctf run shows only the standard ModelFinder output + the
    // JOLT banner. The CPU recompute + safety gate below are NOT gated — they are the
    // load-bearing write-back coherence check that falls back to CPU on a bad result.
    if (getenv("JOLT_DEBUG") && report_count < 1000) { report_count++;
        printf("[JOLT] model=%s ns=%d ncat=%d: %d joint iters | GPU lnL=%.6f  CPU lnL=%.6f  rel=%.3e %s | alpha %.6f->%.6f | pinv %.6f->%.6f%s\n",
               joltModelName.c_str(), ns, ncat, outIters,
               joltLnL, cpuLnL, rel, (rel <= 1e-9 ? "PASS" : (rel <= 1e-6 ? "OK(gamma-resid)" : "MISMATCH")),
               alpha0, (ncat>1?site_rate->getGammaShape():0.0),
               pinv0, (optPinv?site_rate->getPInvar():0.0), (optPinv?" +I":"")); }

    // G.6.1 safety gate: if the fresh CPU recompute disagrees with the JOLT lnL at the SAME written-back params by
    // more than the gamma-residual band, the GPU result is untrustworthy (a kernel/regime failure, not a convergence
    // gap — write-back coherence is otherwise ~1e-12 universally). Return NaN so the caller re-optimises on the CPU
    // from scratch. (Convergence-to-the-CPU-MLE is validated separately: G.6.0b JOLT>=CPU on HKY..GTR; +G/+I rel ~1e-12.)
    if (!(rel <= 1e-6)) {   // audit RISK-3: NOT(<=) so a NaN/inf rel (cpuLnL underflowed to NaN) ALSO trips the
                            // fallback and returns NaN BEFORE the setCurScore(cpuLnL) below could poison _cur_score
        static bool warned_mismatch = false;
        if (!warned_mismatch) { warned_mismatch = true;
            printf("[JOLT] write-back MISMATCH rel=%.3e > 1e-6 -> CPU fallback (model=%s)\n", rel, joltModelName.c_str()); }
        return (double)NAN;
    }

    setCurScore(cpuLnL);
    return cpuLnL;
}

// TS.1 (reborn / L1) — lean in-loop JOLT all-branch reopt: the GPU replacement for optimizeAllBranches(1) in the NNI
// search loop (optallbranches 19.5% surface, plan §17). brlenOnly=true holds the model params fixed (only branch
// lengths move); leanTail=true writes back brlens + clearAllPartialLH (flag-flip) + trusts the device lnL, skipping
// the ModelFinder-only clearAllPartialLH+CPU computeLikelihood() self-check that erases the gain in-loop (the D4
// hazard). maxiter is low (warm-started near the optimum after doNNIs). Returns the device lnL, or NaN (ineligible
// regime / CUDA error / write-back mismatch) -> the caller falls back to the exact CPU optimizeAllBranches(1).
double PhyloTree::optimizeAllBranchesJOLT(int maxiter) {
    // PER-ROUND LM CAP. Default is now 2 (phylotree.h:2130; was 12 — see that doc for the full validation + caveats).
    // 12 over-converged each INTERMEDIATE topology (changes next round; the final tree is CPU-reconverged on modelEps-
    // improving rounds only, NOT JOLT — so the cap can shift the final lnL O(1e-3) at FIXED topology, RF==0). maxiter=2
    // holds the tight gate (RF==0 + dlnL<=1e-3) DIRECTLY on AA-100K + DNA-200tx, INFERRED on AA-200tx, at ~2.5× wall.
    // JOLT_BRLEN_MAXITER env: >0 caps the LM iters; <0 skips the GPU reopt entirely (CPU fallback) for pure-screener nsys.
    static const int env = []{ const char* e = getenv("JOLT_BRLEN_MAXITER"); return e ? atoi(e) : 0; }();
    if (env < 0) return (double)NAN;   // PROFILE-ONLY: JOLT_BRLEN_MAXITER<0 => skip the GPU reopt entirely => CPU
                                       // optimizeAllBranches(1) fallback => the nsys GPU timeline is PURE SCREENER (no
                                       // kj_pre/k1_node from gpu_jolt_optimize colliding with the screener kernels). UNSET
                                       // => env=0 => this branch never taken => byte-identical production.
    if (env > 0) maxiter = env;
    return optimizeParametersJOLT(BRLEN_OPTIMIZE, /*brlenOnly=*/true, /*leanTail=*/true, /*brlenMaxIter=*/maxiter);
}

// ============================================================================================================
// Phase G.8.2.2 — PRODUCTION GPU JOLT optimiser for NON-FUSED PROFILE-MIXTURE models (C20/C30/C60/MEOW...).
// The mixture analogue of optimizeParametersJOLT: dispatched from ModelFactory::optimizeParameters under --jolt
// when getNMixtures()>1. Optimises BRANCHES + the gamma shape alpha on the GPU (validated diagonal-LM joint step
// over the regime axis r=m*ncat+c), holding the CLASS WEIGHTS FIXED, then writes back + self-checks vs a fresh CPU
// computeLikelihood (rel<=1e-6 gate -> NaN/CPU fallback). It reuses the VALIDATED block-optimiser core of the
// gpuMixJointOptimizeCrossCheckOnce kill-switch (PASS@1e-8, cold-vs-warm 8.85e-9) — the warm path with the EM weight
// block REMOVED, because the eligibility gate restricts to weight-fixed mixtures.
//
// ELIGIBILITY — gate on model->getNDim()==0 (red-teamed, the necessary-AND-sufficient "branches+alpha only" test):
// ModelMixture::getNDim() = (fix_prop?0:size-1) + Sum_m at(m)->getNDim(). ==0 therefore implies BOTH fix_prop==true
// (fixed published class weights, the C-series default; else +size-1 weight dims) AND every per-class getNDim()==0
// (no free per-class freq/Q dims). The latter matters: with -mfopt each class becomes FREQ_ESTIMATE adding ns-1 free
// freq params per class that the CPU optimises and the GPU would SILENTLY DROP — and the write-back self-check would
// NOT catch it (it recomputes the CPU lnL at the SAME un-optimised freqs the GPU used, so they agree). This mirrors
// the single-model gate (optimizeParametersJOLT: if (ndim!=0 && !freeQok) DECLINE). +I and free weights -> CPU.
// ============================================================================================================
double PhyloTree::optimizeParametersJOLTMix(int fixed_len) {
    static const bool JOLT_DBG = (getenv("JOLT_DEBUG") != nullptr);
    #define JMIX_DECLINE(why) do { if (JOLT_DBG) { fprintf(stderr, "[JOLTMIX-GATE] decline reason=%s\n", why); fflush(stderr); } return (double)NAN; } while (0)
    if (!model || !site_rate || !aln) JMIX_DECLINE("null-ptr");
    if (fixed_len != BRLEN_OPTIMIZE) JMIX_DECLINE("brlen-mode");
    int ns = aln->num_states;
    if (ns != 4 && ns != 20) JMIX_DECLINE("num-states");
    if (!model->isReversible() || model->getNMixtures() <= 1 || model->isSiteSpecificModel()) JMIX_DECLINE("nonrev/single/ssm");
    ModelMixture *mix = dynamic_cast<ModelMixture*>(model);
    if (!mix || mix->isFused()) JMIX_DECLINE("not-nonfused-mixture");   // LG4M/LG4X 1:1 class<->rate pairing -> CPU
    int N = model->getNMixtures();
    int ncat = site_rate->getNRate();
    if (ncat < 1 || ncat > 64) JMIX_DECLINE("ncat-range");
    // A1 (+I): engage profile-mixture +I+G (ncat>1, RateGammaInvar). Pure +I-alone (ncat==1, no gamma) is a SEPARATE
    // gap (the invariant-only optimiser) -> keep declined. The +I term enters via the per-class clsinv / base_invar_comb
    // built in the clean-room launchers (pinvOverride threaded below); at pinv==0 those paths are byte-identical to +G.
    if (ncat <= 1 && site_rate->getPInvar() > 0.0) JMIX_DECLINE("pure-plusI");
    if (ncat > 1 && site_rate->isGammaRate() != GAMMA_CUT_MEAN) JMIX_DECLINE("non-mean-gamma");  // only the Yang-1994 mean discretisation
    if (!root || !root->isLeaf() || root->neighbors.empty()) JMIX_DECLINE("bad-root");
    Node *Rt = root->neighbors[0]->node;
    if (!Rt || Rt->isLeaf()) JMIX_DECLINE("Rt-leaf");
    // G.8.2.4 — ELIGIBILITY (red-teamed). model->getNDim()==0 => FIXED published weights (C20/C60), branches+alpha
    // only (the validated G.8.2.2 path). Otherwise the model has free params: engage ONLY if they are the class
    // WEIGHTS ALONE (fix_prop=false AND every per-class getNDim()==0 AND no linked-GTR), in which case we add the EM
    // weight block (MEOW80 / ESmodel: getNDim()==N-1). ANY free per-class freq/Q dim (-mfopt) or linked-GTR -> CPU:
    // the GPU would silently drop those and the write-back self-check (recomputes CPU lnL at the SAME un-optimised
    // freqs the GPU used) would NOT catch it. getNDim()==N-1 alone is unsafe (could be per-class dims), so the test
    // is the compound !isFixMixtureWeight() && Sum_m component->getNDim()==0.
    int optWeights;
    { int nd = model->getNDim();
      if (nd == 0) optWeights = 0;
      else { int scd = 0; for (int m = 0; m < N; m++) scd += (*mix)[m]->getNDim();
             if (mix->isFixMixtureWeight() || scd != 0 || (params && params->optimize_linked_gtr))
                 JMIX_DECLINE("free-per-class-or-linked-gtr");   // -mfopt / linked-GTR -> CPU
             optWeights = 1; } }
    #undef JMIX_DECLINE

    // canonical node ordering: REPLICATE the clean-room indexDfs (root at Rt, preorder, skip dad) so b[] indexed by
    // this nid is a valid parentLenOverride for BOTH clean-room functions (identical indexDfs => identical nid). The
    // self-check below proves the index match (a divergent nid would corrupt the branch step and fail the gate).
    std::map<Node*,int> nidMap; std::vector<Node*> nodes; std::vector<Node*> parentOf; std::vector<double> bLive;
    std::function<void(Node*,Node*,double)> indexDfs = [&](Node *n, Node *dad, double lenToDad) {
        nidMap[n] = (int)nodes.size(); nodes.push_back(n); parentOf.push_back(dad); bLive.push_back(lenToDad);
        for (auto nb : n->neighbors) { if (nb->node == dad) continue; indexDfs(nb->node, n, nb->length); }
    };
    indexDfs(Rt, nullptr, 0.0);
    int nNodes = (int)nodes.size();
    int rootId = nidMap[Rt];

    double alpha0 = (ncat > 1) ? site_rate->getGammaShape() : 1.0;
    int optAlpha = (ncat > 1 && !site_rate->isFixGammaShape()) ? 1 : 0;
    auto alphaArg = [&](double a){ return (ncat > 1) ? a : -1.0; };   // -1 => clean-room keeps the live cat rates (ncat==1 path)
    // A1 (+I): pinv iterate + bounds (only ncat>1 +I+G reaches here; pure-+I declined above). optPinv=1 optimises pinv
    // (free, non-fixed) — else a FIXED-+I model still COMPUTES the invariant (pinvArg(-1) => launcher uses the stored
    // pinv) but holds it constant. pinvArg(pv): optPinv ? trial pv : -1 (use the model's stored pinv).
    double pinv0 = site_rate->getPInvar();
    double pinvMin = 1e-6, pinvMax = aln->frac_const_sites;
    int optPinv = (pinv0 > 0.0 && !site_rate->isFixPInvar() && pinvMax > pinvMin) ? 1 : 0;
    double pinv = pinv0;
    auto pinvArg = [&](double pv){ return optPinv ? pv : -1.0; };

    // G.8.2.4 — estimated-weight support. f[]/Ftot: pattern frequencies for the EM M-step. w: the weight iterate
    // (init from the live model). wp drives the lnL clean-room calls — w.data() for the EM case (stable: w is never
    // resized) or nullptr => live published weights (byte-identical to the fixed-weight G.8.2.2 path). The all-branch
    // derivative reads the LIVE model weights and has no w_override, so each outer sets the live weights to the
    // current w (optWeights only) to keep the gradient consistent with the lnL the backtracking evaluates.
    int nptn = (int)aln->size();
    std::vector<double> f(nptn); double Ftot = 0.0;
    if (optWeights) { for (int p = 0; p < nptn; p++) { f[p] = (double)aln->at(p).frequency; Ftot += f[p]; } }
    std::vector<double> w(N); for (int m = 0; m < N; m++) w[m] = model->getMixtureWeight(m);
    const double* wp = optWeights ? w.data() : nullptr;

    // ---- GPU optimiser loop: joint diagonal-LM over (all branches + alpha), weights FIXED (w_override=nullptr => the
    // clean-room reads the live model weights, identical to what the derivative uses). Validated diagonal-LM core from
    // gpuMixJointOptimizeCrossCheckOnce::runOpt (warm path, EM weight block removed). Held under the process-wide mutex
    // because the mix clean-room launchers are not internally locked and ModelFinder runs candidates OpenMP-parallel. ----
    std::vector<double> b = bLive; double alpha = (ncat > 1 ? alpha0 : 1.0);
    double finalLnL = (double)NAN; int outIters = 0;
    {
        std::lock_guard<std::mutex> lk(gpu_mixjolt_mtx);
        double mu = 1.0; int stall = 0;
        double lnL = gpuComputeTreeLnLCleanRoomMix(nullptr, nullptr, wp, b.data(), alphaArg(alpha), pinvArg(pinv));
        if (!std::isnan(lnL)) {
            const int maxOuter = 400;
            for (int outer = 0; outer < maxOuter; outer++) {
                double lnL0 = lnL;
                // optWeights: sync the LIVE model weights to the current iterate so the all-branch derivative (which
                // reads live weights, no w_override) is consistent with the lnL backtracking (which uses wp=w.data()).
                if (optWeights) for (int m = 0; m < N; m++) model->setMixtureWeight(m, w[m]);
                std::vector<Node*> cN, pN; std::vector<double> df, ddf;
                if (!gpuComputeAllBranchDervCleanRoomMix(cN, pN, df, ddf, b.data(), alphaArg(alpha), pinvArg(pinv))) { lnL = (double)NAN; break; }
                std::vector<double> gdf(nNodes, 0.0), gddf(nNodes, 0.0);
                for (size_t i = 0; i < cN.size(); i++) { int v = nidMap.at(cN[i]); gdf[v] = df[i]; gddf[v] = ddf[i]; }
                double ga = 0.0, curvA = 1e-12, eps = 0.0;
                if (optAlpha) {   // alpha gradient/curvature by central FD on the clean-room lnL (no new kernel)
                    eps = 1e-3 * std::max(alpha, 1.0);
                    double lp = gpuComputeTreeLnLCleanRoomMix(nullptr, nullptr, wp, b.data(), alpha+eps, pinvArg(pinv));
                    double lm = gpuComputeTreeLnLCleanRoomMix(nullptr, nullptr, wp, b.data(), alpha-eps, pinvArg(pinv));
                    if (std::isnan(lp) || std::isnan(lm)) { lnL = (double)NAN; break; }
                    ga = (lp - lm) / (2.0*eps); curvA = std::fabs((lp - 2.0*lnL + lm) / (eps*eps)); if (curvA < 1e-12) curvA = 1e-12;
                }
                // A1 (+I): pinv gradient/curvature by FD on the clean-room lnL (mirrors the alpha arm). One-sided near a
                // bound (keeps both points in [pinvMin,pinvMax]); curvP is a damping estimate only (the accept gate enforces
                // correctness). No new kernel — pinvArg threads the trial pinv into the existing lnL launcher.
                double gp = 0.0, curvP = 1e-12, ep = 0.0;
                if (optPinv) {
                    ep = 1e-4; double php, plo;
                    if (pinv + ep > pinvMax)      { php = pinv;      plo = pinv - ep; }   // backward FD near the upper bound
                    else if (pinv - ep < pinvMin) { php = pinv + ep; plo = pinv;      }   // forward FD near the lower bound
                    else                          { php = pinv + ep; plo = pinv - ep; }   // central
                    double lp = gpuComputeTreeLnLCleanRoomMix(nullptr, nullptr, wp, b.data(), alphaArg(alpha), pinvArg(php));
                    double lm = gpuComputeTreeLnLCleanRoomMix(nullptr, nullptr, wp, b.data(), alphaArg(alpha), pinvArg(plo));
                    if (std::isnan(lp) || std::isnan(lm)) { lnL = (double)NAN; break; }
                    gp = (lp - lm) / (php - plo); curvP = std::fabs((lp - 2.0*lnL + lm) / (ep*ep)); if (curvP < 1e-12) curvP = 1e-12;
                }
                for (int bt = 0; bt < 16; bt++) {   // shared-mu LM backtracking over ALL branches + the scalar alpha together
                    std::vector<double> bc = b;
                    for (int v = 0; v < nNodes; v++) { if (v == rootId) continue;
                        double stepv = gdf[v] / (std::fabs(gddf[v]) + mu); bc[v] = b[v] + stepv;
                        if (bc[v] < 1e-6) bc[v] = 1e-6; if (bc[v] > 20.0) bc[v] = 20.0; }
                    double ac = alpha;
                    if (optAlpha) { ac = alpha + ga / (curvA + mu); if (ac < 0.02) ac = 0.02; if (ac > 50.0) ac = 50.0; }
                    double pc = pinv;   // A1 (+I): shared-mu diagonal-Newton pinv step, clamped to (pinvMin,pinvMax)
                    if (optPinv) { pc = pinv + gp / (curvP + mu); if (pc < pinvMin) pc = pinvMin; if (pc > pinvMax) pc = pinvMax; }
                    double ln = gpuComputeTreeLnLCleanRoomMix(nullptr, nullptr, wp, bc.data(), alphaArg(ac), pinvArg(pc));
                    if (std::isnan(ln)) { lnL = (double)NAN; break; }
                    if (ln > lnL) { b = bc; alpha = ac; pinv = pc; lnL = ln; mu = std::max(mu*0.5, 1e-9); break; }   // accept any strict improvement
                    else mu = std::min(mu*4.0, 1e12);                                                     // cap mu (else it runs to +inf and freezes)
                }
                if (std::isnan(lnL)) break;
                // G.8.2.4 — EM WEIGHT BLOCK (estimated weights only; mirrors ModelMixture::optimizeWeights + the
                // validated kill-switch EM). a_{p,m} is WEIGHT-INDEPENDENT, so ONE GPU sweep at uniform w yields
                // lhc[m][p]=a_{p,m}/N (the 1/N cancels in the posterior); the full EM M-step then runs on the HOST:
                // gamma_{p,m}=w_m*lhc / Sum_m w_m*lhc ; w_m = Sum_p gamma_{p,m}*freq_p / Sum freq. One final GPU sweep
                // gives the exact lnL at the new weights. Branches/alpha are fixed in this block (block-coordinate).
                if (optWeights) {
                    std::vector<double> lhc((size_t)N*nptn), wunif(N, 1.0/N), wn(N);
                    double l1 = gpuComputeTreeLnLCleanRoomMix(nullptr, lhc.data(), wunif.data(), b.data(), alphaArg(alpha), pinvArg(pinv));
                    if (std::isnan(l1)) { lnL = (double)NAN; break; }
                    for (int em = 0; em < 1000; em++) {
                        std::fill(wn.begin(), wn.end(), 0.0);
                        for (int p = 0; p < nptn; p++) { double s = 0.0; for (int m = 0; m < N; m++) s += w[m]*lhc[(size_t)m*nptn+p];
                            if (s <= 0.0) continue; double fp = f[p];
                            for (int m = 0; m < N; m++) wn[m] += fp * (w[m]*lhc[(size_t)m*nptn+p] / s); }
                        double stepw = 0.0; for (int m = 0; m < N; m++) { wn[m] /= Ftot; if (wn[m] < 1e-10) wn[m] = 1e-10;
                            double d = std::fabs(wn[m]-w[m]); if (d > stepw) stepw = d; }
                        w = wn;
                        if (em > 0 && stepw < 1e-12) break;
                    }
                    lnL = gpuComputeTreeLnLCleanRoomMix(nullptr, nullptr, w.data(), b.data(), alphaArg(alpha), pinvArg(pinv));
                    if (std::isnan(lnL)) break;
                }
                outIters = outer + 1;
                if (JOLT_DBG) fprintf(stderr, "[JOLTMIX-DBG] outer=%d lnL=%.6f mu=%.2e alpha=%.4f\n", outer, lnL, mu, alpha);
                // ridge-recognizing termination: 3 consecutive outers improving by <1e-7 => the diagonal-LM has reached
                // its ~1e-8 accuracy floor on the ill-conditioned mixture ridge (validated in the kill-switch). The
                // 1e-6 write-back gate is comfortably above this floor, so coherence holds; the lnL may sit marginally
                // below the true MLE but the gap (<<1) is inconsequential for BIC selection (exact penalty term).
                if (std::fabs(lnL - lnL0) < 1e-7) stall++; else stall = 0;
                if (outer > 0 && stall >= 3) break;
            }
            finalLnL = lnL;
        }
    }
    if (std::isnan(finalLnL)) {
        static bool warned = false;
        if (!warned) { warned = true; printf("[JOLTMIX] gpu mixture optimise returned NaN -> CPU fallback (optimizeParameters)\n"); }
        return (double)NAN;
    }

    // ---- write-back. optWeights: set the FINAL EM weights live (the last EM updated w AFTER the per-outer sync, so
    // the live weights are one EM-step stale).
    // G.8.2.5 — RATE-1 SCALE GUARD (red-teamed; resolves the "rate-1 rescale write-back" question). IQ-TREE writes
    // branch lengths in the Sum_m prop_m*total_num_subst_m = 1 convention. For PROFILE mixtures (C20/C60/MEOW80)
    // every class is a pure frequency profile sharing the LG exchangeabilities, individually normalised to
    // total_num_subst = 1, so rho = Sum_m w_m*tns_m = Sum_m w_m = 1 IDENTICALLY for any weights (MEASURED job
    // 171734557: C20 fixed rho=1.0000000000, MEOW80 EM rho=1.0000000005, tns[min=max=1]). The branches are therefore
    // ALREADY in convention => NO rescale is needed (the earlier "off-convention" worry only applies to RATE-varying
    // mixtures, which eligibility already excludes). The ONLY way rho != 1 is a class with tns != 1 (a non-profile /
    // rate mixture) slipping past the gate — we have NOT validated a branch rescale for that, so DECLINE to CPU
    // rather than silently write globally mis-scaled branch lengths. computeTransMatrix uses time/total_num_subst.
    {
        double rho = 0.0, tmin = 1e300, tmax = -1e300;
        for (int m = 0; m < N; m++) {
            double tns = ((ModelMarkov*)((*mix)[m]))->total_num_subst;
            double wm = optWeights ? w[m] : model->getMixtureWeight(m);
            rho += wm * tns; if (tns < tmin) tmin = tns; if (tns > tmax) tmax = tns;
        }
        if (JOLT_DBG) fprintf(stderr, "[JOLTMIX-RATE1] rho=Sum w*tns=%.10f  tns[min=%.6f max=%.6f]  -> %s\n",
                rho, tmin, tmax, (std::fabs(rho-1.0) <= 1e-6 ? "in-convention (no rescale)" : "OFF-CONVENTION -> DECLINE"));
        if (std::fabs(rho - 1.0) > 1e-6) {
            static bool warned_r1 = false;
            if (!warned_r1) { warned_r1 = true;
                printf("[JOLTMIX] rate-1 guard: overall rate rho=%.6f != 1 (tns[min=%.4f max=%.4f]) -> branch scale unvalidated -> CPU fallback\n", rho, tmin, tmax); }
            return (double)NAN;
        }
    }
    if (optWeights) for (int m = 0; m < N; m++) model->setMixtureWeight(m, w[m]);
    // ---- write the optimised branch lengths (both directed neighbours) + alpha back ----
    for (int v = 0; v < nNodes; v++) {
        Node *child = nodes[v], *par = parentOf[v];
        if (!par) continue;                                     // Rt: no parent edge (covered as some node's child edge)
        Neighbor *fwd = par->findNeighbor(child); Neighbor *bwd = child->findNeighbor(par);
        if (fwd) fwd->length = b[v];
        if (bwd) bwd->length = b[v];
    }
    if (optPinv) site_rate->setPInvar(pinv);                    // A1 (+I): p_invar + recomputes the (1-pinv)-scaled rates (RateGammaInvar::setPInvar). MUST precede setGammaShape (single-matrix order, :2200-2201).
    if (optAlpha) site_rate->setGammaShape(alpha);              // sets gamma_shape + recomputes the discrete rates
    clearAllPartialLH();                                        // brlen + alpha + pinv + weights changed -> partials/theta stale

    // ---- self-check: a FRESH CPU computeLikelihood() must reproduce the JOLT lnL at the written-back params ----
    double cpuLnL = computeLikelihood();
    double rel = (cpuLnL != 0.0) ? std::fabs((finalLnL - cpuLnL) / cpuLnL) : std::fabs(finalLnL - cpuLnL);
    static int report_count = 0;
    string joltModelName = model->getName() + (optPinv || pinv0 > 0.0 ? string("+I") : string("")) + (ncat > 1 ? ("+G" + std::to_string(ncat)) : string(""));
    // dev diagnostics — gate behind JOLT_DEBUG (the CPU recompute + safety gate below stay)
    if (getenv("JOLT_DEBUG") && report_count < 1000) { report_count++;
        printf("[JOLTMIX] model=%s N=%d ns=%d ncat=%d weights=%s: %d iters | GPU lnL=%.6f  CPU lnL=%.6f  rel=%.3e %s | alpha %.6f->%.6f | pinv %.6f->%.6f%s\n",
               joltModelName.c_str(), N, ns, ncat, (optWeights ? "EM" : "fixed"), outIters, finalLnL, cpuLnL, rel,
               (rel <= 1e-6 ? "OK" : "MISMATCH"), alpha0, (ncat > 1 ? site_rate->getGammaShape() : 0.0),
               pinv0, (optPinv ? site_rate->getPInvar() : pinv0), (optPinv ? " +I" : "")); }

    if (!(rel <= 1e-6)) {   // NOT(<=) so a NaN/inf rel also trips the fallback before setCurScore poisons _cur_score
        static bool warned_mm = false;
        if (!warned_mm) { warned_mm = true;
            printf("[JOLTMIX] write-back MISMATCH rel=%.3e > 1e-6 -> CPU fallback (model=%s)\n", rel, joltModelName.c_str()); }
        return (double)NAN;
    }
    setCurScore(cpuLnL);
    return cpuLnL;
}

#endif // IQTREE_GPU
