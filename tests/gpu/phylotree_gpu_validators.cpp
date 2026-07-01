// tests/gpu/phylotree_gpu_validators.cpp — GPU tree-search diagnostic/validation methods.
//
// PhyloTree methods that cross-check the GPU tree-search screener against CPU/GPU reference
// values. Every entry point here is reachable ONLY via an explicit, default-off --ts-*-check
// CLI flag (see utils/tools.h/.cpp): gpuScreenNNICleanRoom (--ts-screen2-check),
// gpuScreenNNIFoldCleanRoom (--ts-screen3-check; also the internal oracle used by the batch/tile
// validators below), gpuAllBranchUpperCheckCleanRoom (--ts-upper-check), gpuScreenNNIBatchCleanRoom
// (--ts-batch-check), gpuScreenNNITileCleanRoom (--ts-tile-check). None of these run during a
// normal search and none influence its result -- they are the correctness proofs documented in
// docs/validation.md: compiled into the shipped binary and invoked on Gadi via PBS jobs, not a
// hermetic unit-test binary (GitHub-hosted CI has no GPU and can only prove this file compiles).
//
// Compiled into the same iqtree_gpu CMake target as tree/gpu/ (only when IQTREE_GPU=ON).
// Declarations stay in tree/phylotree.h alongside the rest of PhyloTree.
#include "iqtree_config.h"
#ifdef IQTREE_GPU

#include "tree/phylotree.h"
#include "tree/phylonode.h"
#include "model/modelsubst.h"
#include "model/modelmixture.h"
#include "model/rateheterogeneity.h"
#include "model/rategamma.h"
#include "alignment/alignment.h"
#include "tree/gpu/gpu_iqtree.h"
#include <vector>
#include <map>
#include <functional>
#include <cmath>
#include <cstdio>
using namespace std;

// ============================================================================================================
// gpuScreenNNICleanRoom: non-mutating NNI screener. Scores the swapped topology (node1_nei <-> node2_nei
// exchanged across the central node1-node2 edge) at the old branch lengths of the unmutated tree, i.e. scores a
// candidate without a physical doNNI. Modeled on gpuComputeEdgeDervCleanRoom (two-sub-root central-edge split +
// gpu_derv_crosscheck), but builds explicit recorded adjacency (parentIdx/parentLen/childrenIdx) during a
// swap-aware DFS rather than reading n->neighbors downstream, because the swap is virtual (the physical tree
// still has node1->S1, node2->S2).
// Ordering requirement: a moved-in subtree root (S2 under node1, S1 under node2) carries its MOVED edge length
// (L2 = node2_nei->length, L1 = node1_nei->length) explicitly; it is never a neighbour lookup (no node1<->S2
// edge exists physically). Recursing into a moved-in subtree skips that subtree's real physical parent (node2
// for S2, node1 for S1), since the subtree internals are unchanged from the original tree.
// *out_lnL = whole-tree lnL of the swapped topology at the central old length, == the CPU reference.
// Same eligibility gate as gpuComputeEdgeDervCleanRoom; NaN -> CPU.
// ============================================================================================================
double PhyloTree::gpuScreenNNICleanRoom(PhyloNode *node1, PhyloNode *node2,
                                        PhyloNeighbor *node1_nei, PhyloNeighbor *node2_nei,
                                        double *out_ddf, double *out_lnL) {
    if (!model || !site_rate || !aln) return (double)NAN;
    int ns = aln->num_states;
    if (ns != 4 && ns != 20) return (double)NAN;
    if (!model->isReversible() || model->getNMixtures() != 1 || model->isSiteSpecificModel()) return (double)NAN;
    if (site_rate->getPInvar() > 0.0) return (double)NAN;   // +I omits ptn_invar in this sweep -> CPU
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

    // build-time invariants: moved roots carry the MOVED length; each sub-root has exactly 2 recorded
    // children; the central edge is excluded from both sub-roots.
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

    // echild[v] = U*exp(eval*rate*parentLen[v]) for every node except the two sub-roots (no parent edge)
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
// gpuScreenNNIFoldCleanRoom: host driver for the resident-postorder + re-pairing-fold screener. Builds the SAME
// physical two-sub-root descriptors as gpuComputeEdgeDervCleanRoom (central edge node1<->node2 excluded; node1's
// subtree contains {S1,Bn}, node2's {S2,Dn} as resident lower partials), then resolves the four surrounding
// subtrees' (echild-node, slot|leaf) and calls gpu_screen_nni_fold_crosscheck, which re-pairs them
// (node1<-{S2,Bn}, node2<-{S1,Dn}) via two k1_node folds + k2_derv at the unchanged central length. The swap is
// purely in the fold grouping: each subtree keeps its own physical echild (its length is unchanged; a moved
// Neighbor keeps its length). *out_lnL = swapped-topology lnL, == gpuScreenNNICleanRoom (the reference) to 1e-9.
// One resident postorder, no extra kernel, no swap-aware DFS. Same eligibility gate; NaN -> CPU.
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

    Node *node = node1, *dadN = node2;   // physical two-sub-root central-edge split (no swap applied)

    // ---- physical two-sub-root DFS (same as gpuComputeEdgeDervCleanRoom) ----
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
// gpuAllBranchUpperCheckCleanRoom: host driver for the persistent-upper preorder validator. Builds a fixed-root
// whole-tree sweep (like gpuComputeTreeLnLCleanRoom: root at R = internal node adjacent to the root leaf), adds
// the per-node expfac (parent-branch factor for kj_pre) and flat child/leaf/slot/parentLen arrays, and calls
// gpu_allbranch_upper_check. It builds one resident postorder + one preorder with a persistent per-node upper
// buffer, computes every edge's lnL = k2_derv(lower_v, pre_v, b_v) and the whole-tree lnL. The invariant (every
// edge lnL == tree lnL for a reversible model) validates the persistent-upper machinery that the batched
// re-pairing screener reuses, without re-pairing. Returns false (ineligible / CUDA error) on NaN. Out: max
// rel-err over edges, #edges, #pass@1e-9, #bit-exact, tree lnL.
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

    // echild[v] = U*exp(eval*rate*b_v); expfac[v] = exp(eval*rate*b_v) (no U; kj_pre applies g_U). Root: zeroed.
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
// gpuScreenNNIBatchCleanRoom: host driver for the batched re-pairing NNI screener. Builds the fixed-root sweep
// (resident lowers + persistent uppers built once), enumerates every inner branch's 2 NNI moves, scores them all
// in one gpu_screen_nni_batch_crosscheck call (cheap folds off the resident state), and cross-checks each move
// vs gpuScreenNNIFoldCleanRoom (the reference, which re-roots a full postorder per move). Times the batched path
// vs the M reference calls. Out: max rel-err vs the reference, #moves, #pass@1e-9, tree lnL, and the two wall
// times. Returns false if ineligible / no moves / CUDA error.
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
    vector<int> orc_n1, orc_n2, orc_w, orc_vswap;   // reference operands (nids)
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

    // ---- batched launcher (timed): 1 postorder + 1 preorder + M cheap folds ----
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

    // ---- per-move reference cross-check (timed): M full-postorder gpuScreenNNIFoldCleanRoom calls ----
    long long nmove=0, npass=0; double maxrel=0.0;
    double t1 = getRealTime();
    for (int m = 0; m < M; m++) {
        PhyloNode *n1 = (PhyloNode*) nodes[orc_n1[m]];
        PhyloNode *n2 = (PhyloNode*) nodes[orc_n2[m]];
        PhyloNeighbor *nei1 = (PhyloNeighbor*) n1->findNeighbor(nodes[orc_w[m]]);
        PhyloNeighbor *nei2 = (PhyloNeighbor*) n2->findNeighbor(nodes[orc_vswap[m]]);
        double oddf=0.0, olnL=(double)NAN;
        if (nei1 && nei2) gpuScreenNNIFoldCleanRoom(n1, n2, nei1, nei2, &oddf, &olnL);
        if (olnL != olnL) continue;          // reference ineligible for this move -> no reference, skip
        nmove++;
        if (moveLnL[m] != moveLnL[m]) {      // batched NaN while the reference is finite is a real failure (don't mask it)
            maxrel = HUGE_VAL;
            continue;                        // counted in nmove, not in npass
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
// gpuScreenNNITileCleanRoom: host driver for the pattern-tiled batched NNI screener. Same build and move
// enumeration as gpuScreenNNIBatchCleanRoom, but the launcher (gpu_screen_nni_tile_crosscheck) tiles nptn so the
// persistent per-node upper fits large alignments. Primary gate: every tiled move == the untiled reference
// (gpuScreenNNIFoldCleanRoom) to 1e-9 at all scales (the reference uses one full postorder). Secondary, at small
// scale when nTile=1 fits: the auto-tiled per-move lnLs are bit-identical to forced nTile in {3,7} and to the
// batch launcher, proving tiling-invariance and that this nTile=1 path matches the batch path. Reports the
// auto-picked nTile and the bit-identity count.
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

    // ---- auto tiled launcher (timed, the reported result): nTile auto-picked from free VRAM ----
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

    // ---- bit-identity check (small scale only): tiling-invariance (forced 3,7) + tile-vs-batch ----
    // nTile=1 footprint (upper + lowers + scratch); if it fits a comfortable budget run the bit-exact checks.
    long long bitexact = -1;   // -1 = skipped (nTile=1 would OOM at this scale)
    size_t nt1Bytes = ((size_t)nNodes + (size_t)nInternal + 5) * (size_t)ncat*ns * (size_t)nptn * sizeof(double);
    if (nt1Bytes < (size_t)90 * 1000000000ULL) {
        bitexact = M;
        auto bitcmp = [&](const vector<double>&a, const vector<double>&b){
            for (int m=0;m<M;m++) if (memcmp(&a[m],&b[m],sizeof(double))!=0) return false; return true; };
        // forced nTile in {3,7} (likely non-divisors of nptn -> exercise the ragged tail)
        for (int fk : {3, 7}) {
            if (fk > nptn) continue;
            vector<double> mv(M, (double)NAN); int nt=0; double r = tileCall(fk, mv, &nt);
            if (r != r || !bitcmp(mv, moveLnL)) bitexact = 0;
        }
        // tiled nTile=1 path vs the batch launcher (regression check)
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

    // ---- per-move reference cross-check (timed, the primary gate): M full-postorder gpuScreenNNIFoldCleanRoom ----
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
    if (out_nmoves_total) *out_nmoves_total = M;   // total enumerated moves (bit-identity over all M; nmove = reference-eligible subset)
    return true;
}

#endif // IQTREE_GPU
