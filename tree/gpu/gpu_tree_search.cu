// gpu_tree_search.cu — GPU NNI screener for tree search.
//
// Scores candidate NNI-swapped topologies at their current branch lengths from a
// single resident postorder over the physical tree plus a re-pairing fold, so no
// swap-aware traversal or per-move recomputation is needed. Three screener entry
// points (fold / batch / tile — the last splits patterns into VRAM-sized chunks)
// plus a persistent-upper preorder pass (gpu_allbranch_upper_check). Its own
// node-space kernels (k1_node_prod, kj_pre_node, make_pmat) live here; the shared
// likelihood kernels (k1_node, k2_derv, k_leaf_eig, kj_pre) and the mixture
// derivative kernels (k2_derv_mix, k2_derv_mix_inv) are declared in gpu_kernels.h.
// The single-model pools are reused via the extern decls in gpu_common.cuh.
// Built only when IQTREE_GPU=ON.
#include <cuda_runtime.h>
#include "gpu_iqtree.h"
#include "gpu_common.cuh"
#include "gpu_kernels.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <functional>   // recursive DFS lambdas over the topology

// =============================== tree-search kernels ===============================
// Persistent-upper node-space partials (precision fix). Storing the preorder
// upper in eigen space round-trips through U*diag(expfac)*Uinv, whose Uinv*prod
// coefficient extraction is ill-conditioned (sign-indefinite Uinv) on a
// near-equilibrium partial (long ancestor branch), producing up to ~1.9%
// intermediate-tree error. The cure (matching the lower/fold path): keep the
// upper as a node-space positive-product vector and apply P(b) as a single
// well-conditioned node-space matvec — no eigen round-trip, no cancellation,
// exact at any branch length. Two kernels:
//   k1_node_prod : root-child seed = prod_{root's other children}(P.L) in node space (k1_node WITHOUT the final Uinv).
//   kj_pre_node  : interior recurrence up_v[x] = (P(b_u).up_u)[x] * fsib[x] (push through b_u first, then .* the
//                  node-space sibling product). eigOut=1 applies a single Uinv (node->eigen) for a move endpoint fed
//                  to k2_derv; eigOut=0 stores node-space for the persistent buffer. P(b_u) is the node-space
//                  transition Pmat_u = U*diag(exp(eval*rate*b_u))*Uinv = echild_u*Uinv, supplied by the host.
__global__ void k1_node_prod(int ns, int nptn, int ncat, double* __restrict__ out, int nchild,
        const double* ec0, const double* p0, const unsigned char* t0,
        const double* ec1, const double* p1, const unsigned char* t1,
        const double* ec2, const double* p2, const unsigned char* t2) {
    int ptn = blockIdx.x*blockDim.x + threadIdx.x; if (ptn>=nptn) return;
    for (int c=0;c<ncat;c++){
        double prod[NS_MAX];
        for (int x=0;x<ns;x++) prod[x]=1.0;
        accum_child(prod,ns,c,ptn,nptn,ec0,p0,t0);
        if (nchild>1) accum_child(prod,ns,c,ptn,nptn,ec1,p1,t1);
        if (nchild>2) accum_child(prod,ns,c,ptn,nptn,ec2,p2,t2);
        double* o = out + (size_t)(c*ns)*nptn + ptn;
        for (int x=0;x<ns;x++) o[(size_t)x*nptn]=prod[x];   // NODE-space write (no final Uinv)
    }
}

__global__ void kj_pre_node(int ns, int nptn, int ncat, int eigOut, double* __restrict__ out,
        const double* __restrict__ up_u,        // node-space upper at u: [c*ns + t]*nptn
        const double* __restrict__ Pmat_u,      // node-space P(b_u): [c*ns*ns + x*ns + t]
        int nsib,
        const double* ec0, const double* sp0, const unsigned char* st0,
        const double* ec1, const double* sp1, const unsigned char* st1){
    int ptn = blockIdx.x*blockDim.x + threadIdx.x; if (ptn>=nptn) return;
    for (int c=0;c<ncat;c++){
        double fsib[NS_MAX]; for (int x=0;x<ns;x++) fsib[x]=1.0;     // node-space sibling product
        accum_child(fsib,ns,c,ptn,nptn,ec0,sp0,st0);
        if (nsib>1) accum_child(fsib,ns,c,ptn,nptn,ec1,sp1,st1);
        const double* uuc = up_u + (size_t)(c*ns)*nptn + ptn;        // u's node-space upper, this cat
        const double* Pc  = Pmat_u + (size_t)c*ns*ns;
        double r[NS_MAX];
        for (int x=0;x<ns;x++){ double v=0.0;                        // r[x] = (P(b_u)·up_u)[x] · fsib[x]
            for (int t=0;t<ns;t++) v += Pc[x*ns+t]*uuc[(size_t)t*nptn];
            r[x] = v*fsib[x]; }
        double* o = out + (size_t)(c*ns)*nptn + ptn;
        if (eigOut){                                                 // move endpoint: SINGLE Uinv -> eigen for k2_derv
            for (int j=0;j<ns;j++){ double v=0.0;
                for (int x=0;x<ns;x++) v += g_Uinv[j*ns+x]*r[x];
                o[(size_t)j*nptn]=v; }
        } else {                                                     // persistent buffer: store node-space
            for (int x=0;x<ns;x++) o[(size_t)x*nptn]=r[x];
        }
    }
}

// make_pmat — derive the node-space transition Pmat[v][c][x][t] = sum_i echild[v][c][x][i]*Uinv[i][t] = P(b_v)[x][t]
// from the already-uploaded echild (= U*diag(exp(eval*rate*b_v))) and g_Uinv. Pattern-independent (no ptn axis),
// run once per launcher; same per-node layout/stride as echild. One thread per (v,c,x) row, writing ns entries (t).
__global__ void make_pmat(int ns, int ncat, int nnodes, const double* __restrict__ echild, double* __restrict__ Pmat){
    int row = blockIdx.x*blockDim.x + threadIdx.x;   // row = (v*ncat+c)*ns + x  over nnodes*ncat*ns rows
    if (row >= nnodes*ncat*ns) return;
    const double* ec = echild + (size_t)row*ns;      // echild[v][c][x][:]  (over i)
    double* pm = Pmat + (size_t)row*ns;              // Pmat[v][c][x][:]    (over t)
    for (int t=0;t<ns;t++){ double v=0.0;
        for (int i=0;i<ns;i++) v += ec[i]*g_Uinv[i*ns+t];   // Σ_i echild[x][i]·Uinv[i][t] = P(b)[x][t]
        pm[t]=v; }
}

static DevBuf gb_n1eig, gb_n2eig;   // re-pairing-fold scratch (node1/node2 swapped directed eigen partials)
static DevBuf gb_valall;            // per-move central-edge {v0,v1,v2} coeff tables (nMoves*3*ncat*ns), uploaded once
static DevBuf gb_baseinvar;         // screener per-pattern invariant base (+I), chunk-sized, uploaded per chunk
static DevBuf gb_uexpfac, gb_upper;  // per-node expfac (parent branch) + persistent per-node upper-partial buffer
static DevBuf gb_pmat;               // per-node node-space transition P(b)=echild*Uinv (derived on-device, screeners)

// gpu_screen_nni_fold_crosscheck — score an NNI-swapped topology at the OLD
// lengths from one resident postorder over the physical (unswapped) tree plus a
// re-pairing fold; no swap-aware DFS, no extra kernel. Reuses gpu_derv_crosscheck's
// prologue + postorder (the descriptor arrays describe the physical two-sub-root
// tree, so d_partial holds every subtree's resident lower partial). Then, instead
// of reading nodeSlot/dadSlot (the unswapped endpoint partials), it re-pairs the
// four surrounding subtrees via two k1_node(isRoot=0,nchild=2) folds: node1's
// swapped directed partial = fold(child n1a, n1b); node2's = fold(n2a, n2b). The
// swap is purely in the fold grouping — each child carries its own physical echild
// matrix, whose length is unchanged by the swap. k2_derv then combines the two
// re-paired endpoint partials across the central edge at the unchanged length t,
// giving the swapped lnL.
//   Re-pairing descriptor per child: (ec = echild node index, slot>=0 internal | leaf>=0 tip; exactly one).
//   For an NNI on (node1,node2) swapping S1<->S2: n1a=S2@L2, n1b=Bn@Lb, n2a=S1@L1, n2b=Dn@Ld.
extern "C" double gpu_screen_nni_fold_crosscheck(
    int nstates, int nptn, int ncat, int ntax, int nnodes, int nInternal,
    const double* Uinv, const double* UinvRowSum, const double* freq, const double* catProp,
    const double* echild, const unsigned char* tip, const double* ptn_freq,
    const int* desc_isRoot, const int* desc_nchild, const int* desc_outSlot,
    const int* desc_childNode, const int* desc_childIsLeaf, const int* desc_childLeaf, const int* desc_childSlot,
    int n1a_ec, int n1a_slot, int n1a_leaf,   int n1b_ec, int n1b_slot, int n1b_leaf,
    int n2a_ec, int n2a_slot, int n2a_leaf,   int n2b_ec, int n2b_slot, int n2b_leaf,
    const double* eval, const double* catRate, double t,
    double* out_ddf, double* out_lnL)
{
    int ns = nstates;
    if (ns > NS_MAX || ncat > 64) { fprintf(stderr,"[GPU-FOLD] unsupported ns=%d ncat=%d\n",ns,ncat); return (double)NAN; }
    (void)desc_isRoot;

    GCK(cudaMemcpyToSymbol(g_Uinv, Uinv, sizeof(double)*ns*ns));
    GCK(cudaMemcpyToSymbol(g_UinvRowSum, UinvRowSum, sizeof(double)*ns));
    GCK(cudaMemcpyToSymbol(g_freq, freq, sizeof(double)*ns));
    GCK(cudaMemcpyToSymbol(g_catw, catProp, sizeof(double)*ncat));

    // central-edge coeffs at the unchanged length t (the swap never touches t) -> __constant__
    std::vector<double> v0((size_t)ncat*ns), v1((size_t)ncat*ns), v2((size_t)ncat*ns);
    for (int c=0;c<ncat;c++){ double rc=catRate[c], pc=catProp[c];
        for (int x=0;x<ns;x++){ double re=rc*eval[x], e=exp(eval[x]*rc*t)*pc;
            v0[c*ns+x]=e; v1[c*ns+x]=re*e; v2[c*ns+x]=re*re*e; } }
    GCK(cudaMemcpyToSymbol(g_val0, v0.data(), sizeof(double)*ncat*ns));
    GCK(cudaMemcpyToSymbol(g_val1, v1.data(), sizeof(double)*ncat*ns));
    GCK(cudaMemcpyToSymbol(g_val2, v2.data(), sizeof(double)*ncat*ns));

    size_t ecStride=(size_t)ncat*ns*ns, slotSz=(size_t)ncat*ns*nptn;
    DEVB(gb_echild, (size_t)nnodes*ecStride*sizeof(double));
    DEVB(gb_tip,    (size_t)ntax*nptn);
    DEVB(gb_partial,(size_t)(nInternal>0?nInternal:1)*slotSz*sizeof(double));
    DEVB(gb_pdf,    (size_t)nptn*sizeof(double));
    DEVB(gb_pddf,   (size_t)nptn*sizeof(double));
    DEVB(gb_patlh,  (size_t)nptn*sizeof(double));
    DEVB(gb_n1eig,  slotSz*sizeof(double));   // re-pairing fold scratch (node1)
    DEVB(gb_n2eig,  slotSz*sizeof(double));
    double *d_echild=(double*)gb_echild.p, *d_partial=(double*)gb_partial.p;
    double *d_pdf=(double*)gb_pdf.p, *d_pddf=(double*)gb_pddf.p, *d_patlh=(double*)gb_patlh.p;
    double *d_n1eig=(double*)gb_n1eig.p, *d_n2eig=(double*)gb_n2eig.p;
    unsigned char *d_tip=(unsigned char*)gb_tip.p;
    GCK(cudaMemcpy(d_echild,echild,(size_t)nnodes*ecStride*sizeof(double),cudaMemcpyHostToDevice));
    GCK(cudaMemcpy(d_tip,tip,(size_t)ntax*nptn,cudaMemcpyHostToDevice));

    int TB=256, GB=(nptn+TB-1)/TB;
    // ---- resident postorder over the physical tree (as in gpu_derv_crosscheck) ----
    for (int idx=0; idx<nInternal; idx++){
        int nchild=desc_nchild[idx];
        double* out=(desc_outSlot[idx]<0)?nullptr:(d_partial+(size_t)desc_outSlot[idx]*slotSz);
        const double* ec[3]={nullptr,nullptr,nullptr};
        const double* p[3]={nullptr,nullptr,nullptr};
        const unsigned char* tp[3]={nullptr,nullptr,nullptr};
        for (int k=0;k<nchild && k<3;k++){ int cn=desc_childNode[idx*3+k];
            if (cn>=0) ec[k]=d_echild+(size_t)cn*ecStride;
            if (desc_childIsLeaf[idx*3+k]) tp[k]=d_tip+(size_t)desc_childLeaf[idx*3+k]*nptn;
            else                          p[k]=d_partial+(size_t)desc_childSlot[idx*3+k]*slotSz; }
        k1_node<<<GB,TB>>>(ns,nptn,ncat,/*isRoot=*/0,out,d_patlh,nchild,
            ec[0],p[0],tp[0], ec[1],p[1],tp[1], ec[2],p[2],tp[2]);
    }
    GCK(cudaDeviceSynchronize()); GCK(cudaGetLastError());

    // ---- re-pairing folds: node1/node2 swapped directed eigen partials (the only new step) ----
    // each child resolves to (ec = echild matrix, p = resident slot | t = tip); exactly one of p/t per child.
    auto ecP = [&](int ecn){ return (ecn>=0)? (const double*)(d_echild+(size_t)ecn*ecStride) : (const double*)nullptr; };
    auto plP = [&](int slot){ return (slot>=0)? (const double*)(d_partial+(size_t)slot*slotSz) : (const double*)nullptr; };
    auto tpP = [&](int leaf){ return (leaf>=0)? (const unsigned char*)(d_tip+(size_t)leaf*nptn) : (const unsigned char*)nullptr; };
    k1_node<<<GB,TB>>>(ns,nptn,ncat,/*isRoot=*/0,d_n1eig,d_patlh,/*nchild=*/2,
        ecP(n1a_ec),plP(n1a_slot),tpP(n1a_leaf), ecP(n1b_ec),plP(n1b_slot),tpP(n1b_leaf), nullptr,nullptr,nullptr);
    k1_node<<<GB,TB>>>(ns,nptn,ncat,/*isRoot=*/0,d_n2eig,d_patlh,/*nchild=*/2,
        ecP(n2a_ec),plP(n2a_slot),tpP(n2a_leaf), ecP(n2b_ec),plP(n2b_slot),tpP(n2b_leaf), nullptr,nullptr,nullptr);
    GCK(cudaDeviceSynchronize()); GCK(cudaGetLastError());

    k2_derv<<<GB,TB>>>(ns,nptn,ncat,d_n1eig,d_n2eig,d_pdf,d_pddf,d_patlh);
    GCK(cudaDeviceSynchronize()); GCK(cudaGetLastError());

    std::vector<double> pdf(nptn),pddf(nptn),patlh(nptn);
    GCK(cudaMemcpy(pdf.data(),d_pdf,(size_t)nptn*sizeof(double),cudaMemcpyDeviceToHost));
    GCK(cudaMemcpy(pddf.data(),d_pddf,(size_t)nptn*sizeof(double),cudaMemcpyDeviceToHost));
    GCK(cudaMemcpy(patlh.data(),d_patlh,(size_t)nptn*sizeof(double),cudaMemcpyDeviceToHost));

    double df=0,kdf=0, ddf=0,kddf=0, lnL=0,kl=0;
    for (int p2=0;p2<nptn;p2++){ double f=ptn_freq[p2];
        { double term=f*pdf[p2],  y=term-kdf,  s=df +y; kdf =(s-df )-y; df =s; }
        { double term=f*pddf[p2], y=term-kddf, s=ddf+y; kddf=(s-ddf)-y; ddf=s; }
        { double term=f*patlh[p2],y=term-kl,   s=lnL+y; kl  =(s-lnL)-y; lnL=s; } }

    if (out_ddf) *out_ddf=ddf;
    if (out_lnL) *out_lnL=lnL;
    return df;
}

// gpu_allbranch_upper_check — one fixed-root postorder (resident lower partials)
// plus one preorder with a persistent per-node upper buffer (d_upper[v], slot =
// node id, not the O(depth) acq/rls pool), then for every internal edge (u=parent,
// v=child) run k2_derv(lower_v, pre_v, t=b_v) and return its lnL. The whole-tree
// lnL is also returned (independent root isRoot=1 reduction). For a reversible
// model every edge's k2_derv lnL equals the tree lnL (the lnL is the contraction
// of lower_v (x) pre_v at the true length b_v, edge-invariant), which validates
// the persistent-upper machinery that the batched screener reuses. Single-model
// only; nTile=1. Reuses k1_node / kj_pre / k2_derv; no extra kernel. The persistent
// (not pooled) upper is the only addition.
//   pre_v: root-child v -> k1_node(isRoot=0) over root's other children (no parent branch); interior v ->
//          kj_pre(pre_u=d_upper[u], expfac_u, siblings = child[u]\{v}). Stored persistently at d_upper[v].
extern "C" double gpu_allbranch_upper_check(
    int nstates, int nptn, int ncat, int ntax, int nnodes, int nInternal, int root,
    const double* Uinv, const double* U, const double* UinvRowSum, const double* freq, const double* catProp,
    const double* eval, const double* catRate,
    const double* echild, const double* expfac, const unsigned char* tip, const double* ptn_freq,
    const int* node_nchild, const int* node_child, const int* node_leaf, const int* node_slot,
    const double* node_parentLen, const int* post_internal,
    double* out_edge_lnL,   // [nnodes]: per-edge lnL via k2_derv(lower_v,pre_v,b_v); non-edge/root entries left as set by the caller
    double* out_tree_lnL)   // whole-tree lnL (independent root isRoot=1 reduction)
{
    int ns = nstates;
    if (ns > NS_MAX || ncat > 64) { fprintf(stderr,"[GPU-UPPER] unsupported ns=%d ncat=%d\n",ns,ncat); return (double)NAN; }
    int TB=256, GB=(nptn+TB-1)/TB, Pn=nptn;

    GCK(cudaMemcpyToSymbol(g_Uinv, Uinv, sizeof(double)*ns*ns));
    GCK(cudaMemcpyToSymbol(g_U,    U,    sizeof(double)*ns*ns));   // kj_pre needs the eigenvectors (up-map)
    GCK(cudaMemcpyToSymbol(g_UinvRowSum, UinvRowSum, sizeof(double)*ns));
    GCK(cudaMemcpyToSymbol(g_freq, freq, sizeof(double)*ns));
    GCK(cudaMemcpyToSymbol(g_catw, catProp, sizeof(double)*ncat));

    size_t ecStride=(size_t)ncat*ns*ns, exStride=(size_t)ncat*ns, slotSz=(size_t)ncat*ns*nptn;
    DEVB(gb_echild, (size_t)nnodes*ecStride*sizeof(double));
    DEVB(gb_uexpfac,(size_t)nnodes*exStride*sizeof(double));
    DEVB(gb_tip,    (size_t)ntax*nptn);
    DEVB(gb_partial,(size_t)(nInternal>0?nInternal:1)*slotSz*sizeof(double));
    DEVB(gb_upper,  (size_t)nnodes*slotSz*sizeof(double));   // PERSISTENT: one upper slot per node
    DEVB(gb_pdf,    (size_t)nptn*sizeof(double));
    DEVB(gb_pddf,   (size_t)nptn*sizeof(double));
    DEVB(gb_patlh,  (size_t)nptn*sizeof(double));
    DEVB(gb_nodeleaf, slotSz*sizeof(double));   // leaf endpoint lower-eigen scratch (k_leaf_eig)
    double *d_echild=(double*)gb_echild.p, *d_expfac=(double*)gb_uexpfac.p, *d_partial=(double*)gb_partial.p;
    double *d_upper=(double*)gb_upper.p, *d_pdf=(double*)gb_pdf.p, *d_pddf=(double*)gb_pddf.p, *d_patlh=(double*)gb_patlh.p;
    double *d_tipeig=(double*)gb_nodeleaf.p;
    unsigned char *d_tip=(unsigned char*)gb_tip.p;
    GCK(cudaMemcpy(d_echild,echild,(size_t)nnodes*ecStride*sizeof(double),cudaMemcpyHostToDevice));
    GCK(cudaMemcpy(d_expfac,expfac,(size_t)nnodes*exStride*sizeof(double),cudaMemcpyHostToDevice));
    GCK(cudaMemcpy(d_tip,tip,(size_t)ntax*nptn,cudaMemcpyHostToDevice));

    // child-args helper (exclude `excl`, -1 for none): echild/partial/tip pointers for k1_node
    auto fillChild=[&](int u,int excl,int& nch,const double** ec,const double** p,const unsigned char** t){
        nch=0; for(int k=0;k<3;k++){ec[k]=nullptr;p[k]=nullptr;t[k]=nullptr;}
        for(int kk=0; kk<node_nchild[u]; kk++){ int c=node_child[u*3+kk]; if(c==excl||nch>=3) continue;
            ec[nch]=d_echild+(size_t)c*ecStride;
            if(node_leaf[c]>=0) t[nch]=d_tip+(size_t)node_leaf[c]*Pn; else p[nch]=d_partial+(size_t)node_slot[c]*slotSz; nch++; } };
    auto edgeNodePtr=[&](int v)->const double*{
        if(node_leaf[v]<0) return d_partial+(size_t)node_slot[v]*slotSz;
        k_leaf_eig<<<GB,TB>>>(ns,Pn,ncat,d_tip+(size_t)node_leaf[v]*Pn,d_tipeig); return d_tipeig; };
    auto setVal=[&](double t){ std::vector<double> v0(ncat*ns),v1(ncat*ns),v2(ncat*ns);
        for(int c=0;c<ncat;c++){ double rc=catRate[c], pc=catProp[c];
            for(int x=0;x<ns;x++){ double re=rc*eval[x], e=exp(eval[x]*rc*t)*pc; v0[c*ns+x]=e; v1[c*ns+x]=re*e; v2[c*ns+x]=re*re*e; } }
        cudaMemcpyToSymbol(g_val0,v0.data(),sizeof(double)*ncat*ns); cudaMemcpyToSymbol(g_val1,v1.data(),sizeof(double)*ncat*ns); cudaMemcpyToSymbol(g_val2,v2.data(),sizeof(double)*ncat*ns); };
    // ptn_freq-weighted Kahan reduce of the patlh channel (after a k2_derv) -> edge lnL
    auto reduceLnL=[&]()->double{ std::vector<double> pl(nptn); cudaMemcpy(pl.data(),d_patlh,(size_t)nptn*sizeof(double),cudaMemcpyDeviceToHost);
        double L=0,k=0; for(int p=0;p<nptn;p++){ double term=ptn_freq[p]*pl[p], y=term-k, s=L+y; k=(s-L)-y; L=s; } return L; };

    // ---- POSTORDER: resident lower partials (skip root) ----
    for (int idx=0; idx<nInternal; idx++){ int u=post_internal[idx]; if(u==root) continue;
        int nch; const double* ec[3]; const double* p[3]; const unsigned char* t[3]; fillChild(u,-1,nch,ec,p,t);
        k1_node<<<GB,TB>>>(ns,Pn,ncat,/*isRoot=*/0,d_partial+(size_t)node_slot[u]*slotSz,d_patlh,nch,
            ec[0],p[0],t[0], ec[1],p[1],t[1], ec[2],p[2],t[2]); }
    GCK(cudaDeviceSynchronize()); GCK(cudaGetLastError());

    // ---- whole-tree lnL: root isRoot=1 fold over root's children (independent cross-check) ----
    { int nch; const double* ec[3]; const double* p[3]; const unsigned char* t[3]; fillChild(root,-1,nch,ec,p,t);
      k1_node<<<GB,TB>>>(ns,Pn,ncat,/*isRoot=*/1,/*out=*/nullptr,d_patlh,nch,
          ec[0],p[0],t[0], ec[1],p[1],t[1], ec[2],p[2],t[2]);
      GCK(cudaDeviceSynchronize()); GCK(cudaGetLastError()); }
    double tree_lnL = reduceLnL();
    if (out_tree_lnL) *out_tree_lnL = tree_lnL;

    // ---- PREORDER: persistent per-node upper d_upper[v] + per-edge k2_derv lnL ----
    // proc returns double (not void) so GCK's injected `return (double)NAN` typechecks; std::function<void(int)>
    // discards it. Every path must return (the deduced type is double; falling off the end would be UB).
    std::function<void(int)> proc=[&](int u)->double{
        for(int kk=0; kk<node_nchild[u]; kk++){
            int v=node_child[u*3+kk];
            double* pre=d_upper+(size_t)v*slotSz;
            if(u==root){   // root child: upper = lower partial of root excluding v (k1_node isRoot=0; no parent branch)
                int nch; const double* ec[3]; const double* p[3]; const unsigned char* t[3]; fillChild(root,v,nch,ec,p,t);
                k1_node<<<GB,TB>>>(ns,Pn,ncat,/*isRoot=*/0,pre,d_patlh,nch, ec[0],p[0],t[0], ec[1],p[1],t[1], ec[2],p[2],t[2]);
            } else {       // interior parent: propagate pre_u through u with siblings-of-v and the parent branch b_u
                const double* ec[2]={0,0}; const double* sp[2]={0,0}; const unsigned char* st[2]={0,0}; int nsb=0;
                for(int jj=0; jj<node_nchild[u]; jj++){ int w=node_child[u*3+jj]; if(w==v||nsb>=2) continue;
                    ec[nsb]=d_echild+(size_t)w*ecStride;
                    if(node_leaf[w]>=0) st[nsb]=d_tip+(size_t)node_leaf[w]*Pn; else sp[nsb]=d_partial+(size_t)node_slot[w]*slotSz; nsb++; }
                kj_pre<<<GB,TB>>>(ns,Pn,ncat,pre,d_upper+(size_t)u*slotSz,d_expfac+(size_t)u*exStride,nsb,
                    ec[0],sp[0],st[0], ec[1],sp[1],st[1]);
            }
            GCK(cudaDeviceSynchronize());
            const double* plv=edgeNodePtr(v); GCK(cudaDeviceSynchronize());
            setVal(node_parentLen[v]);
            k2_derv<<<GB,TB>>>(ns,Pn,ncat,plv,pre,d_pdf,d_pddf,d_patlh); GCK(cudaDeviceSynchronize());
            if(out_edge_lnL) out_edge_lnL[v] = reduceLnL();
            if(node_leaf[v]<0) proc(v);   // recurse (d_upper[v] persists for v's children)
        }
        return (double)0;   // normal exit (see note above)
    };
    proc(root);
    GCK(cudaDeviceSynchronize()); GCK(cudaGetLastError());
    return tree_lnL;
}

// gpu_screen_nni_batch_crosscheck — batched re-pairing NNI screener. Reuses
// gpu_allbranch_upper_check's prologue + one fixed-root postorder (resident lower
// partials d_partial) + one persistent-upper preorder (d_upper[v], slot = node id),
// built once, then scores a host-provided list of NNI moves, each a cheap fold
// reading the resident lowers + persistent uppers (no re-sweep). Per move (parent
// u, child v, swapping u's other child w with a v-child):
//   node1Eig (u side, toward v) = kj_pre(pre_u=d_upper[u], expfac_u, sibling = swapped-in v-child)   [u != root]
//                               = k1_node({swapped-in v-child, u's staying child})                   [u == root: no upper/expfac]
//   node2Eig (v side, toward u) = k1_node({w, v's staying child})
//   lnL = k2_derv(node1Eig, node2Eig, b_v)   (central length unchanged)
// The only new work vs gpu_allbranch_upper_check is the move loop; no extra kernel.
// out_move_lnL[m] = swapped-topology lnL, matching gpu_screen_nni_fold_crosscheck.
// One postorder is shared across all moves, vs M postorders for the per-move path;
// nTile=1.
extern "C" double gpu_screen_nni_batch_crosscheck(
    int nstates, int nptn, int ncat, int ntax, int nnodes, int nInternal, int root,
    const double* Uinv, const double* U, const double* UinvRowSum, const double* freq, const double* catProp,
    const double* eval, const double* catRate,
    const double* echild, const double* expfac, const unsigned char* tip, const double* ptn_freq,
    const int* node_nchild, const int* node_child, const int* node_leaf, const int* node_slot,
    const double* node_parentLen, const int* post_internal,
    int nMoves,
    const int* mv_u, const int* mv_uIsRoot, const double* mv_bv,
    const int* n1a_ec, const int* n1a_slot, const int* n1a_leaf,   // node1 fold child A = swapped-in v-child (kj_pre sibling / k1_node child)
    const int* n1b_ec, const int* n1b_slot, const int* n1b_leaf,   // node1 fold child B = u's staying child (u==root k1_node only)
    const int* n2a_ec, const int* n2a_slot, const int* n2a_leaf,   // node2 fold child A = w (u's moved-out child)
    const int* n2b_ec, const int* n2b_slot, const int* n2b_leaf,   // node2 fold child B = v's staying child
    double* out_move_lnL, double* out_tree_lnL)
{
    int ns = nstates;
    if (ns > NS_MAX || ncat > 64) { fprintf(stderr,"[GPU-BATCH] unsupported ns=%d ncat=%d\n",ns,ncat); return (double)NAN; }
    int TB=256, GB=(nptn+TB-1)/TB, Pn=nptn;

    GCK(cudaMemcpyToSymbol(g_Uinv, Uinv, sizeof(double)*ns*ns));
    GCK(cudaMemcpyToSymbol(g_U,    U,    sizeof(double)*ns*ns));
    GCK(cudaMemcpyToSymbol(g_UinvRowSum, UinvRowSum, sizeof(double)*ns));
    GCK(cudaMemcpyToSymbol(g_freq, freq, sizeof(double)*ns));
    GCK(cudaMemcpyToSymbol(g_catw, catProp, sizeof(double)*ncat));

    size_t ecStride=(size_t)ncat*ns*ns, exStride=(size_t)ncat*ns, slotSz=(size_t)ncat*ns*nptn;
    DEVB(gb_echild, (size_t)nnodes*ecStride*sizeof(double));
    DEVB(gb_uexpfac,(size_t)nnodes*exStride*sizeof(double));
    DEVB(gb_tip,    (size_t)ntax*nptn);
    DEVB(gb_partial,(size_t)(nInternal>0?nInternal:1)*slotSz*sizeof(double));
    DEVB(gb_upper,  (size_t)nnodes*slotSz*sizeof(double));
    DEVB(gb_pmat,   (size_t)nnodes*ecStride*sizeof(double));   // node-space P(b)=echild*Uinv (node-space upper)
    DEVB(gb_pdf,    (size_t)nptn*sizeof(double));
    DEVB(gb_pddf,   (size_t)nptn*sizeof(double));
    DEVB(gb_patlh,  (size_t)nptn*sizeof(double));
    DEVB(gb_nodeleaf, slotSz*sizeof(double));
    DEVB(gb_n1eig,  slotSz*sizeof(double));
    DEVB(gb_n2eig,  slotSz*sizeof(double));
    double *d_echild=(double*)gb_echild.p, *d_expfac=(double*)gb_uexpfac.p, *d_partial=(double*)gb_partial.p;
    double *d_upper=(double*)gb_upper.p, *d_pdf=(double*)gb_pdf.p, *d_pddf=(double*)gb_pddf.p, *d_patlh=(double*)gb_patlh.p;
    double *d_tipeig=(double*)gb_nodeleaf.p, *d_n1eig=(double*)gb_n1eig.p, *d_n2eig=(double*)gb_n2eig.p;
    double *d_pmat=(double*)gb_pmat.p;
    unsigned char *d_tip=(unsigned char*)gb_tip.p;
    GCK(cudaMemcpy(d_echild,echild,(size_t)nnodes*ecStride*sizeof(double),cudaMemcpyHostToDevice));
    GCK(cudaMemcpy(d_expfac,expfac,(size_t)nnodes*exStride*sizeof(double),cudaMemcpyHostToDevice));
    GCK(cudaMemcpy(d_tip,tip,(size_t)ntax*nptn,cudaMemcpyHostToDevice));
    { int rows=nnodes*ncat*ns, gbp=(rows+TB-1)/TB; make_pmat<<<gbp,TB>>>(ns,ncat,nnodes,d_echild,d_pmat);   // P(b) per node,cat
      GCK(cudaDeviceSynchronize()); GCK(cudaGetLastError()); }

    auto fillChild=[&](int u,int excl,int& nch,const double** ec,const double** p,const unsigned char** t){
        nch=0; for(int k=0;k<3;k++){ec[k]=nullptr;p[k]=nullptr;t[k]=nullptr;}
        for(int kk=0; kk<node_nchild[u]; kk++){ int c=node_child[u*3+kk]; if(c==excl||nch>=3) continue;
            ec[nch]=d_echild+(size_t)c*ecStride;
            if(node_leaf[c]>=0) t[nch]=d_tip+(size_t)node_leaf[c]*Pn; else p[nch]=d_partial+(size_t)node_slot[c]*slotSz; nch++; } };
    auto setVal=[&](double t){ std::vector<double> v0(ncat*ns),v1(ncat*ns),v2(ncat*ns);
        for(int c=0;c<ncat;c++){ double rc=catRate[c], pc=catProp[c];
            for(int x=0;x<ns;x++){ double re=rc*eval[x], e=exp(eval[x]*rc*t)*pc; v0[c*ns+x]=e; v1[c*ns+x]=re*e; v2[c*ns+x]=re*re*e; } }
        cudaMemcpyToSymbol(g_val0,v0.data(),sizeof(double)*ncat*ns); cudaMemcpyToSymbol(g_val1,v1.data(),sizeof(double)*ncat*ns); cudaMemcpyToSymbol(g_val2,v2.data(),sizeof(double)*ncat*ns); };
    auto reduceLnL=[&]()->double{ std::vector<double> pl(nptn); cudaMemcpy(pl.data(),d_patlh,(size_t)nptn*sizeof(double),cudaMemcpyDeviceToHost);
        double L=0,k=0; for(int p=0;p<nptn;p++){ double term=ptn_freq[p]*pl[p], y=term-k, s=L+y; k=(s-L)-y; L=s; } return L; };
    // resolve a re-pairing child's (echild, partial|tip) pointers
    auto ecP=[&](int ecn){ return (ecn>=0)? (const double*)(d_echild+(size_t)ecn*ecStride) : (const double*)nullptr; };
    auto plP=[&](int slot){ return (slot>=0)? (const double*)(d_partial+(size_t)slot*slotSz) : (const double*)nullptr; };
    auto tpP=[&](int leaf){ return (leaf>=0)? (const unsigned char*)(d_tip+(size_t)leaf*Pn) : (const unsigned char*)nullptr; };

    // ---- POSTORDER (resident lowers) ----
    for (int idx=0; idx<nInternal; idx++){ int u=post_internal[idx]; if(u==root) continue;
        int nch; const double* ec[3]; const double* p[3]; const unsigned char* t[3]; fillChild(u,-1,nch,ec,p,t);
        k1_node<<<GB,TB>>>(ns,Pn,ncat,/*isRoot=*/0,d_partial+(size_t)node_slot[u]*slotSz,d_patlh,nch,
            ec[0],p[0],t[0], ec[1],p[1],t[1], ec[2],p[2],t[2]); }
    GCK(cudaDeviceSynchronize()); GCK(cudaGetLastError());

    // ---- whole-tree lnL (independent cross-check) ----
    { int nch; const double* ec[3]; const double* p[3]; const unsigned char* t[3]; fillChild(root,-1,nch,ec,p,t);
      k1_node<<<GB,TB>>>(ns,Pn,ncat,/*isRoot=*/1,nullptr,d_patlh,nch, ec[0],p[0],t[0], ec[1],p[1],t[1], ec[2],p[2],t[2]);
      GCK(cudaDeviceSynchronize()); GCK(cudaGetLastError()); }
    if (out_tree_lnL) *out_tree_lnL = reduceLnL();

    // ---- PREORDER: persistent per-node upper d_upper[v] stored node-space: seed root-children with k1_node_prod
    // (no final Uinv); interior up_v = (P(b_u).up_u) .* fsib via kj_pre_node(eigOut=0). No eigen round-trip. ----
    std::function<void(int)> proc=[&](int u)->double{
        for(int kk=0; kk<node_nchild[u]; kk++){
            int v=node_child[u*3+kk];
            double* pre=d_upper+(size_t)v*slotSz;
            if(u==root){
                int nch; const double* ec[3]; const double* p[3]; const unsigned char* t[3]; fillChild(root,v,nch,ec,p,t);
                k1_node_prod<<<GB,TB>>>(ns,Pn,ncat,pre,nch, ec[0],p[0],t[0], ec[1],p[1],t[1], ec[2],p[2],t[2]);
            } else {
                const double* ec[2]={0,0}; const double* sp[2]={0,0}; const unsigned char* st[2]={0,0}; int nsb=0;
                for(int jj=0; jj<node_nchild[u]; jj++){ int w=node_child[u*3+jj]; if(w==v||nsb>=2) continue;
                    ec[nsb]=d_echild+(size_t)w*ecStride;
                    if(node_leaf[w]>=0) st[nsb]=d_tip+(size_t)node_leaf[w]*Pn; else sp[nsb]=d_partial+(size_t)node_slot[w]*slotSz; nsb++; }
                kj_pre_node<<<GB,TB>>>(ns,Pn,ncat,/*eigOut=*/0,pre,d_upper+(size_t)u*slotSz,d_pmat+(size_t)u*ecStride,nsb,
                    ec[0],sp[0],st[0], ec[1],sp[1],st[1]);
            }
            GCK(cudaDeviceSynchronize());
            if(node_leaf[v]<0) proc(v);
        }
        return (double)0;
    };
    proc(root);
    GCK(cudaDeviceSynchronize()); GCK(cudaGetLastError());

    // ---- MOVE LOOP: each move = 2 cheap folds + k2_derv off the resident lowers + persistent uppers ----
    for (int m=0; m<nMoves; m++){
        int u=mv_u[m];
        if (mv_uIsRoot[m]) {   // u==root: no upper/expfac -> k1_node({swapped-in v-child, u's staying child})
            k1_node<<<GB,TB>>>(ns,Pn,ncat,/*isRoot=*/0,d_n1eig,d_patlh,/*nchild=*/2,
                ecP(n1a_ec[m]),plP(n1a_slot[m]),tpP(n1a_leaf[m]), ecP(n1b_ec[m]),plP(n1b_slot[m]),tpP(n1b_leaf[m]), nullptr,nullptr,nullptr);
        } else {               // interior u: node1Eig = Uinv*((P(b_u).up_u) .* (P(b_swapchild).L_swapchild)), eigOut=1
            kj_pre_node<<<GB,TB>>>(ns,Pn,ncat,/*eigOut=*/1,d_n1eig,d_upper+(size_t)u*slotSz,d_pmat+(size_t)u*ecStride,/*nsib=*/1,
                ecP(n1a_ec[m]),plP(n1a_slot[m]),tpP(n1a_leaf[m]), nullptr,nullptr,nullptr);
        }
        k1_node<<<GB,TB>>>(ns,Pn,ncat,/*isRoot=*/0,d_n2eig,d_patlh,/*nchild=*/2,
            ecP(n2a_ec[m]),plP(n2a_slot[m]),tpP(n2a_leaf[m]), ecP(n2b_ec[m]),plP(n2b_slot[m]),tpP(n2b_leaf[m]), nullptr,nullptr,nullptr);
        GCK(cudaDeviceSynchronize());
        setVal(mv_bv[m]);
        k2_derv<<<GB,TB>>>(ns,Pn,ncat,d_n1eig,d_n2eig,d_pdf,d_pddf,d_patlh); GCK(cudaDeviceSynchronize());
        if (out_move_lnL) out_move_lnL[m] = reduceLnL();
    }
    GCK(cudaDeviceSynchronize()); GCK(cudaGetLastError());
    return out_tree_lnL ? *out_tree_lnL : 0.0;
}

// gpu_screen_nni_tile_crosscheck — pattern-tiled batched re-pairing NNI screener.
// Identical math to gpu_screen_nni_batch_crosscheck but nptn is split into nTile
// contiguous chunks so the persistent per-node upper (gb_upper =
// nnodes*ncat*ns*nptn, the OOM surface at large alignments) is sized to chunk0
// instead of nptn. Per chunk: full postorder (resident lower partials) + tree-lnL
// + persistent-upper preorder + the move loop, each scoring all M moves over this
// chunk's patterns. Each move's lnL is a continuous per-move Kahan sum carried
// across chunks (add order 0..nptn-1), so the result is bit-identical to nTile=1
// for any nTile. nTile = forced_ntile>0 ? forced_ntile : mix_pick_ntile(...);
// JOLT_NTILE env overrides the auto path; out_ntile reports the chosen nTile.
// echild/expfac/eigen carry no pattern axis (uploaded once); the move descriptors
// are pattern-independent (enumerated once, reused every chunk). No extra kernel.

extern "C" double gpu_screen_nni_tile_crosscheck(
    int nstates, int nptn, int ncat, int ntax, int nnodes, int nInternal, int root,
    const double* Uinv, const double* U, const double* UinvRowSum, const double* freq, const double* catProp,
    const double* eval, const double* catRate,
    const double* echild, const double* expfac, const unsigned char* tip, const double* ptn_freq,
    const int* node_nchild, const int* node_child, const int* node_leaf, const int* node_slot,
    const double* node_parentLen, const int* post_internal,
    int nMoves,
    const int* mv_u, const int* mv_uIsRoot, const double* mv_bv,
    const int* n1a_ec, const int* n1a_slot, const int* n1a_leaf,
    const int* n1b_ec, const int* n1b_slot, const int* n1b_leaf,
    const int* n2a_ec, const int* n2a_slot, const int* n2a_leaf,
    const int* n2b_ec, const int* n2b_slot, const int* n2b_leaf,
    int forced_ntile,
    double* out_move_lnL, double* out_tree_lnL, int* out_ntile,
    const double* baseinvar, double pinv)   // +I: per-pattern invariant base + pinv; pinv<=0 disables the +I term
{
    int ns = nstates;
    if (ns > NS_MAX || ncat > 64) { fprintf(stderr,"[GPU-TILE] unsupported ns=%d ncat=%d\n",ns,ncat); return (double)NAN; }
    int TB=256;

    GCK(cudaMemcpyToSymbol(g_Uinv, Uinv, sizeof(double)*ns*ns));
    GCK(cudaMemcpyToSymbol(g_U,    U,    sizeof(double)*ns*ns));
    GCK(cudaMemcpyToSymbol(g_UinvRowSum, UinvRowSum, sizeof(double)*ns));
    GCK(cudaMemcpyToSymbol(g_freq, freq, sizeof(double)*ns));
    GCK(cudaMemcpyToSymbol(g_catw, catProp, sizeof(double)*ncat));

    // ---- the screener move loop runs on the default stream with a single scratch slot ----
    const int S = 1;
    // ---- pick nTile: persistent upper (nnodes) + lowers (nInternal) + 5 R*ns scratch + 3 per-pattern scalars ----
    size_t perPtnDoubles = ((size_t)nnodes + (size_t)(nInternal>0?nInternal:1) + 5)*(size_t)ncat*ns + 3;
    int nTile  = (forced_ntile>0) ? forced_ntile : mix_pick_ntile(nptn, perPtnDoubles);
    if (nTile<1) nTile=1; if (nTile>nptn) nTile=nptn;
    int chunk0 = (nptn + nTile - 1) / nTile;
    size_t ecStride=(size_t)ncat*ns*ns, exStride=(size_t)ncat*ns, slotSzMax=(size_t)ncat*ns*chunk0;
    if (getenv("JOLT_DEBUG")) fprintf(stderr,"[TILE] nptn=%d nTile=%d chunk0=%d perPtnDoubles=%zu upperGB(nt=1)=%.2f\n",
        nptn,nTile,chunk0,perPtnDoubles,(double)nnodes*ncat*ns*nptn*8/1.073741824e9);

    // ---- alloc ONCE at chunk0 max width (echild/expfac have NO pattern axis -> full nnodes-size, uploaded once) ----
    DEVB(gb_echild, (size_t)nnodes*ecStride*sizeof(double));
    DEVB(gb_uexpfac,(size_t)nnodes*exStride*sizeof(double));
    DEVB(gb_pmat,   (size_t)nnodes*ecStride*sizeof(double));   // node-space P(b)=echild*Uinv (same per-node stride as echild)
    DEVB(gb_tip,    (size_t)ntax*chunk0);
    DEVB(gb_partial,(size_t)(nInternal>0?nInternal:1)*slotSzMax*sizeof(double));
    DEVB(gb_upper,  (size_t)nnodes*slotSzMax*sizeof(double));
    // The 5 screener move-loop scratch buffers get S private slots (S=1 => one slot). Per-slot strides are chunk0
    // for pdf/pddf/patlh and slotSzMax=ncat*ns*chunk0 for n1eig/n2eig; slot s lives at offset s*<stride>. Move m
    // uses slot (m % S), so within a slot reuse is serialized (k2 -> D2H -> next move's folds), avoiding aliasing.
    DEVB(gb_pdf,    (size_t)S*chunk0*sizeof(double));
    DEVB(gb_pddf,   (size_t)S*chunk0*sizeof(double));
    DEVB(gb_patlh,  (size_t)S*chunk0*sizeof(double));
    DEVB(gb_nodeleaf, slotSzMax*sizeof(double));
    DEVB(gb_n1eig,  (size_t)S*slotSzMax*sizeof(double));
    DEVB(gb_n2eig,  (size_t)S*slotSzMax*sizeof(double));
    DEVB(gb_valall, (size_t)(nMoves>0?nMoves:1)*3*ncat*ns*sizeof(double));   // per-move {v0,v1,v2} tables (pattern-independent), uploaded once
    DEVB(gb_baseinvar, (size_t)chunk0*sizeof(double));   // per-pattern invariant base (+I), uploaded per chunk
    double *d_baseinvar=(double*)gb_baseinvar.p; bool useInv = (pinv > 0.0 && baseinvar != nullptr);
    double *d_echild=(double*)gb_echild.p, *d_expfac=(double*)gb_uexpfac.p, *d_partial=(double*)gb_partial.p;
    double *d_pmat=(double*)gb_pmat.p;
    double *d_upper=(double*)gb_upper.p, *d_pdf=(double*)gb_pdf.p, *d_pddf=(double*)gb_pddf.p, *d_patlh=(double*)gb_patlh.p;
    double *d_n1eig=(double*)gb_n1eig.p, *d_n2eig=(double*)gb_n2eig.p, *d_valall=(double*)gb_valall.p;
    unsigned char *d_tip=(unsigned char*)gb_tip.p;
    GCK(cudaMemcpy(d_echild,echild,(size_t)nnodes*ecStride*sizeof(double),cudaMemcpyHostToDevice));
    GCK(cudaMemcpy(d_expfac,expfac,(size_t)nnodes*exStride*sizeof(double),cudaMemcpyHostToDevice));
    // derive node-space P(b)=echild·Uinv ONCE (pattern-independent) for the stable node-space upper recurrence
    { int rows=nnodes*ncat*ns, gbp=(rows+TB-1)/TB; make_pmat<<<gbp,TB>>>(ns,ncat,nnodes,d_echild,d_pmat);
      GCK(cudaDeviceSynchronize()); GCK(cudaGetLastError()); }

    // ---- chunk-scoped state (the lambdas capture these by ref; updated at the top of each chunk) ----
    int Pn=0, pOff=0, GB=0; size_t slotSz=0;
    std::vector<unsigned char> tipChunk((size_t)ntax*chunk0);

    auto fillChild=[&](int u,int excl,int& nch,const double** ec,const double** p,const unsigned char** t){
        nch=0; for(int k=0;k<3;k++){ec[k]=nullptr;p[k]=nullptr;t[k]=nullptr;}
        for(int kk=0; kk<node_nchild[u]; kk++){ int c=node_child[u*3+kk]; if(c==excl||nch>=3) continue;
            ec[nch]=d_echild+(size_t)c*ecStride;
            if(node_leaf[c]>=0) t[nch]=d_tip+(size_t)node_leaf[c]*Pn; else p[nch]=d_partial+(size_t)node_slot[c]*slotSz; nch++; } };
    // Precompute every move's central-edge coeff table {v0,v1,v2} once. These are pattern-independent (depend only
    // on mv_bv[m], eval, catRate, catProp), so they upload in a single H2D copy instead of a per-move
    // cudaMemcpyToSymbol(g_val*) inside the move loop. The move loop reads each move's slice via k2_derv_mix(R=ncat)
    // device args; the kernel arithmetic and the host-Kahan reduceInto order are unchanged.
    {   const int Smv=3*ncat*ns; std::vector<double> valAll((size_t)(nMoves>0?nMoves:1)*Smv);
        for(int m=0;m<nMoves;m++){ double t=mv_bv[m]; double* vm=&valAll[(size_t)m*Smv];
            for(int c=0;c<ncat;c++){ double rc=catRate[c], pc=catProp[c];
                for(int x=0;x<ns;x++){ double re=rc*eval[x], e=exp(eval[x]*rc*t)*pc;
                    vm[c*ns+x]=e; vm[ncat*ns + c*ns+x]=re*e; vm[2*ncat*ns + c*ns+x]=re*re*e; } } }
        if(nMoves>0) GCK(cudaMemcpy(d_valall, valAll.data(), (size_t)nMoves*Smv*sizeof(double), cudaMemcpyHostToDevice)); }
    auto ecP=[&](int ecn){ return (ecn>=0)? (const double*)(d_echild+(size_t)ecn*ecStride) : (const double*)nullptr; };
    auto plP=[&](int slot){ return (slot>=0)? (const double*)(d_partial+(size_t)slot*slotSz) : (const double*)nullptr; };
    auto tpP=[&](int leaf){ return (leaf>=0)? (const unsigned char*)(d_tip+(size_t)leaf*Pn) : (const unsigned char*)nullptr; };
    // CONTINUOUS per-target Kahan: add THIS chunk's Pn patterns into (acc,accK) weighted by ptn_freq[pOff+p].
    // pOff strictly increases & p runs 0..Pn-1 => global add order 0..nptn-1 => bit-identical to nTile=1.
    std::vector<double> plchunk(chunk0);
    auto reduceInto=[&](double& acc,double& accK){
        cudaMemcpy(plchunk.data(),d_patlh,(size_t)Pn*sizeof(double),cudaMemcpyDeviceToHost);
        double L=acc,k=accK; for(int p=0;p<Pn;p++){ double term=ptn_freq[pOff+p]*plchunk[p], y=term-k, s=L+y; k=(s-L)-y; L=s; }
        acc=L; accK=k; };

    // Persistent upper stored node-space (positive products): seed root-children with k1_node_prod; interior
    // up_v = (P(b_u).up_u) .* fsib via kj_pre_node(eigOut=0). No eigen round-trip, hence no cancellation.
    std::function<void(int)> proc=[&](int u)->double{
        for(int kk=0; kk<node_nchild[u]; kk++){
            int v=node_child[u*3+kk];
            double* pre=d_upper+(size_t)v*slotSz;
            if(u==root){
                int nch; const double* ec[3]; const double* p[3]; const unsigned char* t[3]; fillChild(root,v,nch,ec,p,t);
                k1_node_prod<<<GB,TB>>>(ns,Pn,ncat,pre,nch, ec[0],p[0],t[0], ec[1],p[1],t[1], ec[2],p[2],t[2]);
            } else {
                const double* ec[2]={0,0}; const double* sp[2]={0,0}; const unsigned char* st[2]={0,0}; int nsb=0;
                for(int jj=0; jj<node_nchild[u]; jj++){ int w=node_child[u*3+jj]; if(w==v||nsb>=2) continue;
                    ec[nsb]=d_echild+(size_t)w*ecStride;
                    if(node_leaf[w]>=0) st[nsb]=d_tip+(size_t)node_leaf[w]*Pn; else sp[nsb]=d_partial+(size_t)node_slot[w]*slotSz; nsb++; }
                kj_pre_node<<<GB,TB>>>(ns,Pn,ncat,/*eigOut=*/0,pre,d_upper+(size_t)u*slotSz,d_pmat+(size_t)u*ecStride,nsb,
                    ec[0],sp[0],st[0], ec[1],sp[1],st[1]);
            }
            GCK(cudaDeviceSynchronize());
            if(node_leaf[v]<0) proc(v);
        }
        return (double)0;
    };

    // ---- per-move + tree CONTINUOUS Kahan accumulators carried across chunks (init ONCE) ----
    std::vector<double> accMove(nMoves>0?nMoves:1,0.0), accMoveK(nMoves>0?nMoves:1,0.0);
    double accTree=0.0, accTreeK=0.0;

    // ---- TILE LOOP: full postorder + tree-lnL + persistent-upper preorder + move loop, per chunk ----
    for (int tchunk=0; tchunk<nTile; tchunk++){
        pOff=tchunk*chunk0; int p1=pOff+chunk0; if(p1>nptn)p1=nptn; Pn=p1-pOff; if(Pn<=0) break;
        slotSz=(size_t)ncat*ns*Pn; GB=(Pn+TB-1)/TB;
        for(int a=0;a<ntax;a++) memcpy(&tipChunk[(size_t)a*Pn], tip+(size_t)a*nptn+pOff, (size_t)Pn);   // gather chunk columns
        GCK(cudaMemcpy(d_tip,tipChunk.data(),(size_t)ntax*Pn,cudaMemcpyHostToDevice));
        if (useInv) GCK(cudaMemcpy(d_baseinvar, baseinvar+pOff, (size_t)Pn*sizeof(double), cudaMemcpyHostToDevice));   // this chunk's invariant base (+I)

        // POSTORDER (resident lowers for THIS chunk)
        for (int idx=0; idx<nInternal; idx++){ int u=post_internal[idx]; if(u==root) continue;
            int nch; const double* ec[3]; const double* p[3]; const unsigned char* t[3]; fillChild(u,-1,nch,ec,p,t);
            k1_node<<<GB,TB>>>(ns,Pn,ncat,/*isRoot=*/0,d_partial+(size_t)node_slot[u]*slotSz,d_patlh,nch,
                ec[0],p[0],t[0], ec[1],p[1],t[1], ec[2],p[2],t[2]); }
        GCK(cudaDeviceSynchronize()); GCK(cudaGetLastError());

        // TREE-LNL (independent root isRoot=1 reduction) -> carried accTree. Must precede the preorder (proc clobbers d_patlh).
        { int nch; const double* ec[3]; const double* p[3]; const unsigned char* t[3]; fillChild(root,-1,nch,ec,p,t);
          k1_node<<<GB,TB>>>(ns,Pn,ncat,/*isRoot=*/1,nullptr,d_patlh,nch, ec[0],p[0],t[0], ec[1],p[1],t[1], ec[2],p[2],t[2]);
          GCK(cudaDeviceSynchronize()); GCK(cudaGetLastError()); }
        reduceInto(accTree,accTreeK);

        // PREORDER (persistent per-node upper d_upper[v] for THIS chunk)
        proc(root);
        GCK(cudaDeviceSynchronize()); GCK(cudaGetLastError());

        // MOVE LOOP (each move = 2 cheap folds + k2_derv off the resident lowers + persistent uppers) -> carried accMove[m]
        // Serial path: default stream, single scratch, per-move sync+reduce.
        for (int m=0; m<nMoves; m++){
            int u=mv_u[m];
            if (mv_uIsRoot[m]) {   // u==root: no upper -> k1_node({swapped-in child, root-leaf}) (eigen)
                k1_node<<<GB,TB>>>(ns,Pn,ncat,/*isRoot=*/0,d_n1eig,d_patlh,/*nchild=*/2,
                    ecP(n1a_ec[m]),plP(n1a_slot[m]),tpP(n1a_leaf[m]), ecP(n1b_ec[m]),plP(n1b_slot[m]),tpP(n1b_leaf[m]), nullptr,nullptr,nullptr);
            } else {               // interior u: node1Eig = Uinv*((P(b_u).up_u) .* (P(b_swapchild).L_swapchild)), eigOut=1
                kj_pre_node<<<GB,TB>>>(ns,Pn,ncat,/*eigOut=*/1,d_n1eig,d_upper+(size_t)u*slotSz,d_pmat+(size_t)u*ecStride,/*nsib=*/1,
                    ecP(n1a_ec[m]),plP(n1a_slot[m]),tpP(n1a_leaf[m]), nullptr,nullptr,nullptr);
            }
            k1_node<<<GB,TB>>>(ns,Pn,ncat,/*isRoot=*/0,d_n2eig,d_patlh,/*nchild=*/2,   // node2Eig (lower fold), already eigen
                ecP(n2a_ec[m]),plP(n2a_slot[m]),tpP(n2a_leaf[m]), ecP(n2b_ec[m]),plP(n2b_slot[m]),tpP(n2b_leaf[m]), nullptr,nullptr,nullptr);
            GCK(cudaDeviceSynchronize());
            const double* dv0=d_valall+(size_t)m*3*ncat*ns;   // this move's precomputed table slice (no per-move g_val upload)
            if (useInv)   // +I: add the invariant term pinv*baseinvar[ptn]; the non-+I path uses the same-shaped kernel below
                k2_derv_mix_inv<<<GB,TB>>>(ns,Pn,ncat,d_n1eig,d_n2eig,dv0,dv0+ncat*ns,dv0+2*ncat*ns,pinv,d_baseinvar,d_pdf,d_pddf,d_patlh);
            else
                k2_derv_mix<<<GB,TB>>>(ns,Pn,ncat,d_n1eig,d_n2eig,dv0,dv0+ncat*ns,dv0+2*ncat*ns,d_pdf,d_pddf,d_patlh);
            GCK(cudaDeviceSynchronize());
            reduceInto(accMove[m],accMoveK[m]);
        }
    }
    if (out_move_lnL) for(int m=0;m<nMoves;m++) out_move_lnL[m]=accMove[m];
    if (out_tree_lnL) *out_tree_lnL=accTree;
    if (out_ntile) *out_ntile=nTile;
    GCK(cudaDeviceSynchronize()); GCK(cudaGetLastError());
    return accTree;
}

