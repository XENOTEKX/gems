// gpu_kernels_mixture.cu — profile-mixture likelihood on the GPU.
//
// Kernels and launchers for N-class profile mixtures (e.g. C10/C20/C60, LG+C60):
// each regime r = m*ncat + c is an independent Felsenstein sweep with class m's
// eigensystem and Gamma category c, folded at the root as L_p = sum_r w_r*(pi_m .
// prod_r). Per-class eigensystems live in global memory (gb_m* pools) because
// N*ns*ns exceeds the 64KB __constant__ budget; the O(nptn) partial arenas reuse
// the single-model pools (declared extern in gpu_common.cuh). Three entry points
// mirror the single-model launchers: lnL, single-edge derivative, and all-branch
// derivative. Built only when IQTREE_GPU=ON.
#include <cuda_runtime.h>
#include "gpu_iqtree.h"
#include "gpu_common.cuh"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <functional>   // recursive DFS lambdas over the topology

// =============================== mixture kernels ===============================
// N-class profile-mixture lnL. Each regime r = m*ncat + c is an independent
// Felsenstein sweep with class m's eigen (echild_m,c, Uinv_m) and gamma cat c;
// regimes never mix until the root fold  L_p = sum_r w_r*(pi_m . prod_r),
// w_r = weight_m * catProp_c. Per-class Uinv/freq/rowsum live in global memory
// (N*ns*ns overflows the 64KB __constant__ budget at high regime counts).
// Separate kernel from k1_node so the single-model path is unchanged. One
// thread per pattern.
__device__ __forceinline__ void accum_child_mix(double* prod, int ns, int r, int ptn, int nptn,
        const double* __restrict__ ec, const double* __restrict__ p, const unsigned char* __restrict__ t,
        const double* __restrict__ Uinv_m, const double* __restrict__ UinvRowSum_m) {
    const double* ecc = ec + (size_t)r*ns*ns;
    if (p) {                                   // internal child: its eigen-space partial for regime r
        const double* pc = p + (size_t)r*ns*nptn + ptn;
        for (int x=0;x<ns;x++){ double v=0.0;
            for (int i=0;i<ns;i++) v += ecc[x*ns+i]*pc[(size_t)i*nptn];
            prod[x]*=v; }
    } else {                                   // leaf child: L[i] = column s of class-m U^-1 (row-sum if ambiguous)
        int s = t[ptn];
        for (int x=0;x<ns;x++){ double v=0.0;
            for (int i=0;i<ns;i++){ double Li = (s<ns)? Uinv_m[i*ns+s] : UinvRowSum_m[i]; v += ecc[x*ns+i]*Li; }
            prod[x]*=v; }
    }
}

__global__ void k1_node_mix(int ns, int nptn, int ncat, int nmix, int isRoot,
        double* __restrict__ out, double* __restrict__ patlh,
        const double* __restrict__ d_Uinv,        // [nmix][ns*ns]
        const double* __restrict__ d_UinvRowSum,  // [nmix][ns]
        const double* __restrict__ d_freq,        // [nmix][ns]
        const double* __restrict__ d_wreg,        // [nmix*ncat]  w_m * catProp_c
        double* __restrict__ out_lhcat,           // optional per-class L_{p,m}=w_m*sum_c catProp_c*L_{p,m,c} [nmix][nptn]
        double pinv, const double* __restrict__ clsinv,   // +I: per-class invariant clsinv[m][ptn]=w_m*pinv*base_invar_m (chunk-local [nmix][nptn]); added only at the root path when pinv>0 (pinv<=0 leaves clsinv unread)
        int nchild,
        const double* ec0, const double* p0, const unsigned char* t0,
        const double* ec1, const double* p1, const unsigned char* t1,
        const double* ec2, const double* p2, const unsigned char* t2) {
    int ptn = blockIdx.x*blockDim.x + threadIdx.x; if (ptn>=nptn) return;
    int R = nmix*ncat;
    // 2D-grid regime parallelisation: a 2D launch (gridDim.y==R) assigns each
    // blockIdx.y one regime r, so the GPU saturates on small nptn. A 1D launch
    // (gridDim.y==1 — the root path, which sums lh over all regimes into patlh)
    // keeps the full in-thread r-loop. Non-root writes out[r*ns*nptn+..] are
    // regime-disjoint, so the 2D split is race-free.
    int r0 = (gridDim.y>1) ? blockIdx.y : 0;
    int rN = (gridDim.y>1) ? (blockIdx.y+1) : R;
    double lh = 0.0, clh = 0.0;   // clh = running per-class accumulator (root path only, always 1D)
    for (int r=r0;r<rN;r++){
        int m = r/ncat;
        const double* Uinv_m = d_Uinv + (size_t)m*ns*ns;
        const double* Urs_m  = d_UinvRowSum + (size_t)m*ns;
        double prod[NS_MAX];
        for (int x=0;x<ns;x++) prod[x]=1.0;
        accum_child_mix(prod,ns,r,ptn,nptn,ec0,p0,t0,Uinv_m,Urs_m);
        if (nchild>1) accum_child_mix(prod,ns,r,ptn,nptn,ec1,p1,t1,Uinv_m,Urs_m);
        if (nchild>2) accum_child_mix(prod,ns,r,ptn,nptn,ec2,p2,t2,Uinv_m,Urs_m);
        if (isRoot){
            const double* freq_m = d_freq + (size_t)m*ns;
            double s=0.0;
            for (int x=0;x<ns;x++) s += freq_m[x]*prod[x];
            double contrib = d_wreg[r]*s;   // w_m*catProp_c*(pi_m.prod) = this regime's likelihood contribution
            // +I: add class m's invariant term once per class (at its first gamma cat, c==0). Folding it into contrib
            // makes both the root sum (lh) and the per-class accumulator (clh) carry it: lh gets the full root
            // invariant pinv*sum_m w_m*base_invar_m, out_lhcat[m] gets the per-class invariant.
            if (pinv > 0.0 && (r - m*ncat == 0)) contrib += clsinv[(size_t)m*nptn + ptn];
            lh += contrib;
            if (out_lhcat){                 // accumulate per class m over its ncat gamma cats, flush at c==ncat-1
                clh += contrib;
                if (r - m*ncat == ncat-1){ out_lhcat[(size_t)m*nptn + ptn] = clh; clh = 0.0; }
            }
        } else {
            double* o = out + (size_t)r*ns*nptn + ptn;
            for (int rr=0; rr<ns; rr++){ double v=0.0;
                for (int x=0;x<ns;x++) v += Uinv_m[rr*ns+x]*prod[x];
                o[(size_t)rr*nptn]=v; }
        }
    }
    if (isRoot) patlh[ptn] = log(fabs(lh));
}

// Single-edge branch derivative for profile mixtures: k2_derv generalised to the
// regime axis r = m*ncat + c (R = nmix*ncat). The central-edge coefficients
// dval0/1/2 live in global memory (per-class eigenvalues -> R*ns entries exceed
// the __constant__ 64-cat budget). node_eig/dad_eig are the per-regime eigen-space
// endpoint partials from k1_node_mix (isRoot=0). Per pattern: lh=sum dval0*th,
// d1=sum dval1*th, d2=sum dval2*th (th=node_eig*dad_eig); the per-class weight
// w_m*catProp_c is folded into dval0 (= wreg[r]); pi_m is already in the partials.
__global__ void k2_derv_mix(int ns, int nptn, int R,
        const double* __restrict__ node_eig, const double* __restrict__ dad_eig,
        const double* __restrict__ dval0, const double* __restrict__ dval1, const double* __restrict__ dval2,
        double* __restrict__ pdf, double* __restrict__ pddf, double* __restrict__ patlh) {
    int ptn = blockIdx.x*blockDim.x + threadIdx.x; if (ptn>=nptn) return;
    double lh=0.0, d1=0.0, d2=0.0;
    for (int r=0;r<R;r++) for (int x=0;x<ns;x++){
        size_t o=(size_t)(r*ns+x)*nptn+ptn;
        double th=node_eig[o]*dad_eig[o]; int k=r*ns+x;
        lh+=dval0[k]*th; d1+=dval1[k]*th; d2+=dval2[k]*th;
    }
    double inv=1.0/lh, rr=d1*inv;
    pdf[ptn]=rr; pddf[ptn]=d2*inv-rr*rr; patlh[ptn]=log(fabs(lh));
}

// +I variant of k2_derv_mix: adds the branch/topology-independent invariant term
// pinv*baseinvar[ptn] to the per-pattern likelihood before log. The screener's
// catRate/catProp already carry the 1/(1-pinv) rescale (getRate/getProp for
// RateGammaInvar), so this additive term is the only remaining +I piece. Used by
// the screener move loop only when pinv>0; pinv==0 stays on k2_derv_mix. baseinvar
// is the chunk slice (indexed 0..nptn-1 of the chunk).
__global__ void k2_derv_mix_inv(int ns, int nptn, int R,
        const double* __restrict__ node_eig, const double* __restrict__ dad_eig,
        const double* __restrict__ dval0, const double* __restrict__ dval1, const double* __restrict__ dval2,
        double pinv, const double* __restrict__ baseinvar,
        double* __restrict__ pdf, double* __restrict__ pddf, double* __restrict__ patlh) {
    int ptn = blockIdx.x*blockDim.x + threadIdx.x; if (ptn>=nptn) return;
    double lh=0.0, d1=0.0, d2=0.0;
    for (int r=0;r<R;r++) for (int x=0;x<ns;x++){
        size_t o=(size_t)(r*ns+x)*nptn+ptn;
        double th=node_eig[o]*dad_eig[o]; int k=r*ns+x;
        lh+=dval0[k]*th; d1+=dval1[k]*th; d2+=dval2[k]*th;
    }
    double Lp=fabs(lh)+pinv*baseinvar[ptn]; double inv=1.0/Lp, rr=d1*inv;
    pdf[ptn]=rr; pddf[ptn]=d2*inv-rr*rr; patlh[ptn]=log(Lp);
}

// Leaf endpoint eigen partial for mixtures (per-class). Regime r=m*ncat+c:
// L[r][i] = Uinv_m[i][s] (column s of class m's Uinv; UinvRowSum_m[i] if the state
// is ambiguous), replicated over the ncat cats within class m. Reads the global
// per-class d_Uinv/d_UinvRowSum, not the single-model __constant__ g_Uinv.
__global__ void k_leaf_eig_mix(int ns, int nptn, int ncat, int nmix,
        const unsigned char* __restrict__ tipt,
        const double* __restrict__ d_Uinv, const double* __restrict__ d_UinvRowSum,
        double* __restrict__ out){
    int ptn = blockIdx.x*blockDim.x + threadIdx.x; if (ptn>=nptn) return;
    int s = tipt[ptn];
    int R = nmix*ncat;
    for (int r=0;r<R;r++){ int m=r/ncat;
        const double* Uinv_m = d_Uinv + (size_t)m*ns*ns;
        const double* Urs_m  = d_UinvRowSum + (size_t)m*ns;
        for (int i=0;i<ns;i++){
            double Li = (s<ns) ? Uinv_m[i*ns+s] : Urs_m[i];
            out[(size_t)(r*ns+i)*nptn+ptn] = Li;
        }
    }
}

// Top-down preorder eigen-space partial pre_v for profile mixtures (kj_pre
// generalised to the regime axis r = m*ncat + c). pre_v = "rest of tree above
// edge (parent u)->v", without v's own branch b_v (the per-edge derivative
// reapplies b_v once, via k2_derv_mix's dval). The parent branch b_u is applied
// here through expfac_u[r][i] = exp(eval_m[i]*catRate_c*b_u). Within one regime r
// the class m is constant root-to-tip, so every per-class array (U_m up-map /
// Uinv_m down-map / Urs_m for ambiguous leaf siblings) is indexed by m=r/ncat and
// the per-category rate is c=r%ncat. pi_m is absorbed in the eigen-space partials
// (theta trick), so this kernel must NOT re-apply d_freq[m]. Output layout matches
// k1_node_mix's non-root write so k2_derv_mix reads pl_v (lower) and pre_v (upper)
// endpoints uniformly.
//   pus[t]      = sum_i U_m[t][i]*expfac_u[r][i]*pre_u[r][i]      (map up with eigenvectors U_m)
//   fsib[r][t]  = prod_{siblings of v} ( sum_i echild_sib[r][t][i]*pl_sib[r][i] )   (accum_child_mix, per class m)
//   pre_v[r][j] = sum_t Uinv_m[j][t]*pus[t]*fsib[t]               (map back down with Uinv_m)
__global__ void k7_pre_mix(int ns, int nptn, int ncat, int nmix,
        double* __restrict__ out_pre,
        const double* __restrict__ pre_u,          // [R*ns*nptn]  parent's upper partial
        const double* __restrict__ expfac_u,       // [R*ns]       exp(eval_m[i]*catRate_c*b_u)
        const double* __restrict__ d_U,            // [nmix][ns*ns] eigenvectors (up-map)
        const double* __restrict__ d_Uinv,         // [nmix][ns*ns] inverse eigenvectors (down-map)
        const double* __restrict__ d_UinvRowSum,   // [nmix][ns]    leaf-sibling ambiguous
        int nsib,
        const double* ec0, const double* sp0, const unsigned char* st0,
        const double* ec1, const double* sp1, const unsigned char* st1) {
    int ptn = blockIdx.x*blockDim.x + threadIdx.x; if (ptn>=nptn) return;
    int R = nmix*ncat;
    // 2D-grid regime parallelisation (gridDim.y==R => one regime per blockIdx.y; saturates small nptn).
    // pre_v[r] is computed independently per regime (no cross-regime accumulation), so the 2D split is race-free.
    int r0 = (gridDim.y>1) ? blockIdx.y : 0;
    int rN = (gridDim.y>1) ? (blockIdx.y+1) : R;
    for (int r=r0;r<rN;r++){ int m=r/ncat;
        const double* U_m    = d_U          + (size_t)m*ns*ns;
        const double* Uinv_m = d_Uinv       + (size_t)m*ns*ns;
        const double* Urs_m  = d_UinvRowSum + (size_t)m*ns;
        double fsib[NS_MAX]; for (int t=0;t<ns;t++) fsib[t]=1.0;
        accum_child_mix(fsib,ns,r,ptn,nptn,ec0,sp0,st0,Uinv_m,Urs_m);
        if (nsib>1) accum_child_mix(fsib,ns,r,ptn,nptn,ec1,sp1,st1,Uinv_m,Urs_m);
        const double* puc = pre_u + (size_t)r*ns*nptn + ptn;
        const double* ef  = expfac_u + (size_t)r*ns;
        double pus[NS_MAX];
        for (int t=0;t<ns;t++){ double v=0.0; for (int i=0;i<ns;i++) v += U_m[t*ns+i]*ef[i]*puc[(size_t)i*nptn]; pus[t]=v; }
        double* o = out_pre + (size_t)r*ns*nptn + ptn;
        for (int j=0;j<ns;j++){ double v=0.0; for (int t=0;t<ns;t++) v += Uinv_m[j*ns+t]*pus[t]*fsib[t]; o[(size_t)j*nptn]=v; }
    }
}

// Profile-mixture lnL launcher. Mirrors gpu_lnl_crosscheck but with R = nmix*ncat
// regimes; per-class Uinv/UinvRowSum/freq + per-regime weight live in global memory
// (mix_* pools). echild/tip/partial/patlh reuse the single-model pools (resized via
// DEVB to the larger mixture sizes). NaN on OOM/CUDA error -> CPU fallback.
static DevBuf gb_mUinv, gb_mUrs, gb_mFreq, gb_mWreg, gb_mLhcat;
static DevBuf gb_mClsinv;   // per-chunk per-class invariant clsinv[nmix][Pn] (+I, lnL mix launcher)
static DevBuf gb_mDval0, gb_mDval1, gb_mDval2;   // central-edge derivative coeffs (R*ns, global)

// Pick the pattern-tiling factor for a mixture launcher. perPatternDoubles = the
// O(nptn) device footprint per pattern, in doubles (e.g. nInternal*R*ns for the
// lnL partial arena; +nPool*R*ns for the derivative preorder pool). nTile =
// ceil(one-shot O(nptn) bytes / 80% free VRAM); IQTREE_GPU_NTILE overrides. The result
// is chunk-count-independent (per-pattern values are chunk-independent; the Kahan
// reductions add patterns in order 0..nptn-1), so nTile only trades VRAM for
// kernel launches, never the answer.
int mix_pick_ntile(int nptn, size_t perPatternDoubles) {
    if (const char* e = getenv("IQTREE_GPU_NTILE")) { int t=atoi(e); if(t<1)t=1;
        if(getenv("IQTREE_GPU_DEBUG")) fprintf(stderr,"[MIX-TILE] IQTREE_GPU_NTILE=%d (nptn=%d)\n",t,nptn); return t; }
    size_t foot = perPatternDoubles * (size_t)nptn * sizeof(double);
    size_t freeB=0, totB=0; int nTile=1;
    if (cudaMemGetInfo(&freeB,&totB)==cudaSuccess && freeB>0) {
        double budget = 0.80*(double)freeB;
        int T = (int)ceil((double)foot/budget); if(T<1)T=1; nTile=T;
    }
    if (getenv("IQTREE_GPU_DEBUG")) fprintf(stderr,"[MIX-TILE] nptn=%d perPtnDoubles=%zu O(nptn)foot=%.2fGB freeVRAM=%.1fGB -> nTile=%d (chunk~%d)\n",
        nptn,perPatternDoubles,(double)foot/1.073741824e9,(double)freeB/1.073741824e9,nTile,(nptn+nTile-1)/nTile);
    return nTile;
}
extern "C" double gpu_lnl_crosscheck_mix(
    int nstates, int nptn, int ncat, int nmix, int ntax, int nnodes, int nInternal,
    const double* Uinv, const double* UinvRowSum, const double* freq, const double* wreg,
    const double* echild, const unsigned char* tip, const double* ptn_freq,
    const int* desc_isRoot, const int* desc_nchild, const int* desc_outSlot,
    const int* desc_childNode, const int* desc_childIsLeaf, const int* desc_childLeaf, const int* desc_childSlot,
    double* out_patlh, double* out_lhcat,   // out_lhcat (optional) = per-class L_{p,m}, [nmix][nptn]
    double pinv, const double* clsinv)       // +I: pinv + per-class invariant clsinv[m][ptn]=w_m*pinv*base_invar_m [nmix][nptn]; pinv<=0 leaves clsinv unread
{
    int ns = nstates;
    if (ns > NS_MAX || nmix < 1 || ncat < 1) { fprintf(stderr,"[GPU-XCHECK-MIX] unsupported ns=%d nmix=%d ncat=%d\n",ns,nmix,ncat); return (double)NAN; }
    int R = nmix*ncat;

    // per-class eigen + per-regime weight -> GLOBAL (overflow __constant__ at 320 regimes)
    DEVB(gb_mUinv, (size_t)nmix*ns*ns*sizeof(double));
    DEVB(gb_mUrs,  (size_t)nmix*ns*sizeof(double));
    DEVB(gb_mFreq, (size_t)nmix*ns*sizeof(double));
    DEVB(gb_mWreg, (size_t)R*sizeof(double));
    GCK(cudaMemcpy(gb_mUinv.p, Uinv,       (size_t)nmix*ns*ns*sizeof(double), cudaMemcpyHostToDevice));
    GCK(cudaMemcpy(gb_mUrs.p,  UinvRowSum, (size_t)nmix*ns*sizeof(double),    cudaMemcpyHostToDevice));
    GCK(cudaMemcpy(gb_mFreq.p, freq,       (size_t)nmix*ns*sizeof(double),    cudaMemcpyHostToDevice));
    GCK(cudaMemcpy(gb_mWreg.p, wreg,       (size_t)R*sizeof(double),          cudaMemcpyHostToDevice));
    double *d_Uinv=(double*)gb_mUinv.p, *d_Urs=(double*)gb_mUrs.p, *d_Freq=(double*)gb_mFreq.p, *d_Wreg=(double*)gb_mWreg.p;

    size_t ecStride = (size_t)R*ns*ns;        // echild per node, regime-strided — pattern-independent (built once)

    // Pattern tiling. The partial arena nInternal*R*ns*nptn dominates VRAM;
    // chunking nptn into nTile contiguous pieces and running a full postorder per
    // chunk shrinks every O(nptn) buffer by ~nTile. echild/eigen carry no pattern
    // axis (built once). The per-pattern patlh[p] is chunk-independent and the
    // Kahan lnL accumulator adds patterns in order 0..nptn-1 across chunks, so the
    // result is bit-identical to one-shot for any nTile. Auto-pick from free VRAM
    // (80% target) or IQTREE_GPU_NTILE override.
    int nTile = mix_pick_ntile(nptn, (size_t)(nInternal>0?nInternal:1)*(size_t)R*ns + (size_t)(out_lhcat?nmix:0) + 2);
    int chunk0 = (nptn + nTile - 1) / nTile;
    size_t slotSzMax = (size_t)R*ns*chunk0;
    DEVB(gb_echild, (size_t)nnodes*ecStride*sizeof(double));
    DEVB(gb_tip,    (size_t)ntax*chunk0);
    DEVB(gb_partial,(size_t)(nInternal>0?nInternal:1)*slotSzMax*sizeof(double));
    DEVB(gb_patlh,  (size_t)chunk0*sizeof(double));
    double *d_echild=(double*)gb_echild.p, *d_partial=(double*)gb_partial.p, *d_patlh=(double*)gb_patlh.p;
    unsigned char *d_tip=(unsigned char*)gb_tip.p;
    double *d_lhcat = nullptr;
    if (out_lhcat) { DEVB(gb_mLhcat, (size_t)nmix*chunk0*sizeof(double)); d_lhcat=(double*)gb_mLhcat.p; }
    double *d_clsinv = nullptr;   // per-chunk per-class invariant [nmix][Pn] (+I, only when pinv>0)
    if (pinv > 0.0 && clsinv) { DEVB(gb_mClsinv, (size_t)nmix*chunk0*sizeof(double)); d_clsinv=(double*)gb_mClsinv.p; }
    GCK(cudaMemcpy(d_echild, echild, (size_t)nnodes*ecStride*sizeof(double), cudaMemcpyHostToDevice));      // pattern-independent

    std::vector<unsigned char> tipChunk((size_t)ntax*chunk0);
    std::vector<double> patlh(chunk0), lhcatChunk((size_t)(out_lhcat?nmix:0)*chunk0);
    std::vector<double> clsinvChunk((size_t)(d_clsinv?nmix:0)*chunk0);   // host staging for the +I chunk slice
    double lnL=0.0, kc=0.0;
    int TB=256;
    for (int tchunk=0; tchunk<nTile; tchunk++){
        int pOff=tchunk*chunk0, p1=pOff+chunk0; if(p1>nptn)p1=nptn; int Pn=p1-pOff; if(Pn<=0) break;
        size_t slotSz=(size_t)R*ns*Pn; int GB=(Pn+TB-1)/TB;
        for(int a=0;a<ntax;a++) memcpy(&tipChunk[(size_t)a*Pn], tip+(size_t)a*nptn+pOff, (size_t)Pn);
        GCK(cudaMemcpy(d_tip, tipChunk.data(), (size_t)ntax*Pn, cudaMemcpyHostToDevice));
        if (d_clsinv) {   // upload this chunk's per-class invariant slice [nmix][Pn] (stride Pn matches the kernel's nptn=Pn)
            for(int m=0;m<nmix;m++) memcpy(&clsinvChunk[(size_t)m*Pn], clsinv+(size_t)m*nptn+pOff, (size_t)Pn*sizeof(double));
            GCK(cudaMemcpy(d_clsinv, clsinvChunk.data(), (size_t)nmix*Pn*sizeof(double), cudaMemcpyHostToDevice)); }
        for (int idx=0; idx<nInternal; idx++){
            int isRoot = desc_isRoot[idx], nchild = desc_nchild[idx];
            double* out = (desc_outSlot[idx] < 0) ? nullptr : (d_partial + (size_t)desc_outSlot[idx]*slotSz);
            const double* ec[3]={nullptr,nullptr,nullptr};
            const double* p[3]={nullptr,nullptr,nullptr};
            const unsigned char* t[3]={nullptr,nullptr,nullptr};
            for (int k=0;k<nchild && k<3;k++){
                int cn = desc_childNode[idx*3+k];
                if (cn>=0) ec[k] = d_echild + (size_t)cn*ecStride;
                if (desc_childIsLeaf[idx*3+k]) t[k] = d_tip + (size_t)desc_childLeaf[idx*3+k]*Pn;   // tip stride = chunk width
                else                          p[k] = d_partial + (size_t)desc_childSlot[idx*3+k]*slotSz;
            }
            dim3 gMix = isRoot ? dim3(GB,1) : dim3(GB,R);   // root accumulates over regimes (1D); non-root parallelises them
            k1_node_mix<<<gMix,TB>>>(ns,Pn,ncat,nmix,isRoot,out,d_patlh,d_Uinv,d_Urs,d_Freq,d_Wreg,d_lhcat,pinv,d_clsinv,nchild,
                ec[0],p[0],t[0], ec[1],p[1],t[1], ec[2],p[2],t[2]);
        }
        GCK(cudaDeviceSynchronize());
        GCK(cudaGetLastError());
        GCK(cudaMemcpy(patlh.data(), d_patlh, (size_t)Pn*sizeof(double), cudaMemcpyDeviceToHost));
        if (out_patlh) for (int p2=0; p2<Pn; p2++) out_patlh[pOff+p2] = patlh[p2];
        if (out_lhcat) { GCK(cudaMemcpy(lhcatChunk.data(), d_lhcat, (size_t)nmix*Pn*sizeof(double), cudaMemcpyDeviceToHost));
            for (int m=0;m<nmix;m++) for (int p2=0;p2<Pn;p2++) out_lhcat[(size_t)m*nptn+pOff+p2] = lhcatChunk[(size_t)m*Pn+p2]; }
        for (int p2=0; p2<Pn; p2++){ double term = ptn_freq[pOff+p2]*patlh[p2];   // continuous Kahan, order 0..nptn-1 => bit-identical
            double y=term-kc, t2=lnL+y; kc=(t2-lnL)-y; lnL=t2; }
    }
    return lnL;
}

// Single-edge derivative launcher for profile mixtures. Mirrors gpu_derv_crosscheck
// but: (1) the descriptor sweep runs k1_node_mix (isRoot=0) so each endpoint writes
// its R=nmix*ncat regime eigen partials; (2) the central-edge coefficients
// dval0/1/2[R*ns] are built per-class (eval_{m,x}) with weight w_m*catProp_c=wreg[r]
// and uploaded to global memory (R*ns exceeds the __constant__ 64-cat budget);
// (3) leaf endpoints synthesize per-class tip eigen via k_leaf_eig_mix. Returns
// df=sum_ptn freq*(d1/lh), with *out_ddf=sum freq*(d2/lh-(d1/lh)^2) and
// *out_lnL=tree lnL at t. NaN on CUDA error.
extern "C" double gpu_derv_crosscheck_mix(
    int nstates, int nptn, int ncat, int nmix, int ntax, int nnodes, int nInternal,
    const double* Uinv, const double* UinvRowSum, const double* freq, const double* wreg,
    const double* echild, const unsigned char* tip, const double* ptn_freq,
    const int* desc_isRoot, const int* desc_nchild, const int* desc_outSlot,
    const int* desc_childNode, const int* desc_childIsLeaf, const int* desc_childLeaf, const int* desc_childSlot,
    int nodeSlot, int nodeLeafTax, int dadSlot, int dadLeafTax,
    const double* evalC, const double* catRate, double t,
    double* out_ddf, double* out_lnL)
{
    int ns = nstates;
    if (ns > NS_MAX || nmix < 1 || ncat < 1) { fprintf(stderr,"[GPU-DERV-MIX] unsupported ns=%d nmix=%d ncat=%d\n",ns,nmix,ncat); return (double)NAN; }
    int R = nmix*ncat;
    (void)desc_isRoot;   // all entries isRoot=0 for the derivative sweep

    // per-class eigen + per-regime weight -> GLOBAL (same pools as the lnL mix path)
    DEVB(gb_mUinv, (size_t)nmix*ns*ns*sizeof(double));
    DEVB(gb_mUrs,  (size_t)nmix*ns*sizeof(double));
    DEVB(gb_mFreq, (size_t)nmix*ns*sizeof(double));
    DEVB(gb_mWreg, (size_t)R*sizeof(double));
    GCK(cudaMemcpy(gb_mUinv.p, Uinv,       (size_t)nmix*ns*ns*sizeof(double), cudaMemcpyHostToDevice));
    GCK(cudaMemcpy(gb_mUrs.p,  UinvRowSum, (size_t)nmix*ns*sizeof(double),    cudaMemcpyHostToDevice));
    GCK(cudaMemcpy(gb_mFreq.p, freq,       (size_t)nmix*ns*sizeof(double),    cudaMemcpyHostToDevice));
    GCK(cudaMemcpy(gb_mWreg.p, wreg,       (size_t)R*sizeof(double),          cudaMemcpyHostToDevice));
    double *d_Uinv=(double*)gb_mUinv.p, *d_Urs=(double*)gb_mUrs.p, *d_Freq=(double*)gb_mFreq.p, *d_Wreg=(double*)gb_mWreg.p;

    // central-edge derivative coeffs dval{0,1,2}[r*ns+x], r=m*ncat+c: cof=eval_{m,x}*rate_c; v0=exp(cof*t)*wreg[r];
    // v1=cof*v0; v2=cof*cof*v0. Per-class eigenvalues; pi_m is not here (it is already in the eigen-space partials).
    std::vector<double> v0((size_t)R*ns), v1((size_t)R*ns), v2((size_t)R*ns);
    for (int m=0;m<nmix;m++) for (int c=0;c<ncat;c++){ int r=m*ncat+c; double rc=catRate[c], wr=wreg[r];
        for (int x=0;x<ns;x++){ double cof=evalC[(size_t)m*ns+x]*rc, e=exp(cof*t)*wr;
            v0[(size_t)r*ns+x]=e; v1[(size_t)r*ns+x]=cof*e; v2[(size_t)r*ns+x]=cof*cof*e; } }
    DEVB(gb_mDval0,(size_t)R*ns*sizeof(double)); DEVB(gb_mDval1,(size_t)R*ns*sizeof(double)); DEVB(gb_mDval2,(size_t)R*ns*sizeof(double));
    GCK(cudaMemcpy(gb_mDval0.p, v0.data(), (size_t)R*ns*sizeof(double), cudaMemcpyHostToDevice));
    GCK(cudaMemcpy(gb_mDval1.p, v1.data(), (size_t)R*ns*sizeof(double), cudaMemcpyHostToDevice));
    GCK(cudaMemcpy(gb_mDval2.p, v2.data(), (size_t)R*ns*sizeof(double), cudaMemcpyHostToDevice));
    double *d_v0=(double*)gb_mDval0.p, *d_v1=(double*)gb_mDval1.p, *d_v2=(double*)gb_mDval2.p;

    size_t ecStride=(size_t)R*ns*ns, slotSz=(size_t)R*ns*nptn;
    DEVB(gb_echild, (size_t)nnodes*ecStride*sizeof(double));
    DEVB(gb_tip,    (size_t)ntax*nptn);
    DEVB(gb_partial,(size_t)(nInternal>0?nInternal:1)*slotSz*sizeof(double));
    DEVB(gb_pdf,    (size_t)nptn*sizeof(double));
    DEVB(gb_pddf,   (size_t)nptn*sizeof(double));
    DEVB(gb_patlh,  (size_t)nptn*sizeof(double));
    double *d_echild=(double*)gb_echild.p, *d_partial=(double*)gb_partial.p;
    double *d_pdf=(double*)gb_pdf.p, *d_pddf=(double*)gb_pddf.p, *d_patlh=(double*)gb_patlh.p;
    unsigned char *d_tip=(unsigned char*)gb_tip.p;
    GCK(cudaMemcpy(d_echild,echild,(size_t)nnodes*ecStride*sizeof(double),cudaMemcpyHostToDevice));
    GCK(cudaMemcpy(d_tip,tip,(size_t)ntax*nptn,cudaMemcpyHostToDevice));

    int TB=256, GB=(nptn+TB-1)/TB;
    for (int idx=0; idx<nInternal; idx++){
        int nchild=desc_nchild[idx];
        double* out=(desc_outSlot[idx]<0)?nullptr:(d_partial+(size_t)desc_outSlot[idx]*slotSz);
        const double* ec[3]={nullptr,nullptr,nullptr}; const double* p[3]={nullptr,nullptr,nullptr}; const unsigned char* tp[3]={nullptr,nullptr,nullptr};
        for (int k=0;k<nchild && k<3;k++){ int cn=desc_childNode[idx*3+k];
            if (cn>=0) ec[k]=d_echild+(size_t)cn*ecStride;
            if (desc_childIsLeaf[idx*3+k]) tp[k]=d_tip+(size_t)desc_childLeaf[idx*3+k]*nptn;
            else                          p[k]=d_partial+(size_t)desc_childSlot[idx*3+k]*slotSz; }
        k1_node_mix<<<dim3(GB,R),TB>>>(ns,nptn,ncat,nmix,/*isRoot=*/0,out,d_patlh,d_Uinv,d_Urs,d_Freq,d_Wreg,/*out_lhcat=*/nullptr,/*pinv=*/0.0,/*clsinv=*/nullptr,nchild,
            ec[0],p[0],tp[0], ec[1],p[1],tp[1], ec[2],p[2],tp[2]);
    }
    GCK(cudaDeviceSynchronize()); GCK(cudaGetLastError());

    // resolve node_eig/dad_eig: internal slot, or synthesize a leaf endpoint's per-class tip eigen via k_leaf_eig_mix
    const double *node_eig, *dad_eig;
    if (nodeSlot >= 0) node_eig = d_partial + (size_t)nodeSlot*slotSz;
    else { DEVB(gb_nodeleaf, slotSz*sizeof(double));
           k_leaf_eig_mix<<<GB,TB>>>(ns,nptn,ncat,nmix,d_tip+(size_t)nodeLeafTax*nptn,d_Uinv,d_Urs,(double*)gb_nodeleaf.p); node_eig=(double*)gb_nodeleaf.p; }
    if (dadSlot >= 0) dad_eig = d_partial + (size_t)dadSlot*slotSz;
    else { DEVB(gb_dadleaf, slotSz*sizeof(double));
           k_leaf_eig_mix<<<GB,TB>>>(ns,nptn,ncat,nmix,d_tip+(size_t)dadLeafTax*nptn,d_Uinv,d_Urs,(double*)gb_dadleaf.p); dad_eig=(double*)gb_dadleaf.p; }

    k2_derv_mix<<<GB,TB>>>(ns,nptn,R,node_eig,dad_eig,d_v0,d_v1,d_v2,d_pdf,d_pddf,d_patlh);
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

// All-branch derivative launcher for profile mixtures (linear-time gradient: one
// postorder + one preorder sweep yields df/ddf for every edge, vs the single-edge
// gpu_derv_crosscheck_mix's two-sub-root split per edge). Rooted at an internal
// node `root`; out_df[v]/out_ddf[v] hold d(lnL)/db_v and the 2nd derivative for
// edge v->parent (the root entry stays 0). Reuses k1_node_mix (postorder lower
// partials + root-child upper seeds), k7_pre_mix (preorder upper partials), and
// k2_derv_mix (per-edge reduction). Beyond the lnL/single-edge inputs it uploads
// the eigenvectors d_U (k7_pre_mix's up-map) and the per-node per-regime
// expfac = exp(eval_m[i]*catRate_c*b_u); the O(tree-height) preorder pool recycles
// slots. Returns 0.0 on success (results in out_df/out_ddf), NaN on CUDA error.
static DevBuf gb_mU, gb_expfac, gb_prepool;   // eigenvectors (up-map), per-node expfac, O(depth) preorder pool
static DevBuf gb_baseinv;   // per-chunk combined invariant base_invar_comb[Pn] (+I, all-branch derivative mix launcher)
extern "C" double gpu_allbranch_derv_crosscheck_mix(
    int nstates, int nptn, int ncat, int nmix, int ntax, int nnodes, int root,
    const double* Uinv, const double* U, const double* UinvRowSum, const double* freq, const double* wreg,
    const double* evalC, const double* catRate,
    const double* echild, const double* expfac, const unsigned char* tip, const double* ptn_freq,
    const int* node_nchild, const int* node_child, const int* node_leaf, const double* node_parentLen,
    double* out_df, double* out_ddf,
    double pinv, const double* base_invar_comb)   // +I: pinv + combined invariant sum_m w_m*base_invar_m [nptn]; the +I term is branch-independent, so it enters only the 1/Lp denominator (k2_derv_mix_inv). pinv<=0 routes to k2_derv_mix
{
    int ns = nstates;
    if (ns > NS_MAX || nmix < 1 || ncat < 1) { fprintf(stderr,"[GPU-ALLDERV-MIX] unsupported ns=%d nmix=%d ncat=%d\n",ns,nmix,ncat); return (double)NAN; }
    int R = nmix*ncat;

    // ---- rebuild topology from flat arrays (node ids = caller's DFS index, rooted at `root`) ----
    std::vector<std::vector<int>> child(nnodes);
    std::vector<int> leaf(nnodes);
    std::vector<double> brlen(node_parentLen, node_parentLen+nnodes);
    for (int u=0;u<nnodes;u++){ leaf[u]=node_leaf[u];
        for (int k=0;k<node_nchild[u] && k<3;k++){ int c=node_child[u*3+k]; if (c>=0) child[u].push_back(c); } }
    std::vector<int> postorder; std::vector<int> slot(nnodes,-1);
    std::function<void(int)> dfs=[&](int u){ for(int c:child[u]) dfs(c); if(leaf[u]<0){ slot[u]=(int)postorder.size(); postorder.push_back(u);} };
    dfs(root); int nInternal=(int)postorder.size();
    int treeH=0; std::function<void(int,int)> ddfs=[&](int u,int d){ if(d>treeH)treeH=d; for(int c:child[u]) ddfs(c,d+1); }; ddfs(root,0);
    int nPool=treeH+2;   // O(depth): preorder holds one upper-partial slot per ancestor on the current path (peak == treeH)
    if(getenv("IQTREE_GPU_DEBUG")) fprintf(stderr,"[ALLDERV-DBG] entry ns=%d nptn=%d ncat=%d nmix=%d nnodes=%d root=%d nInternal=%d treeH=%d nPool=%d R=%d\n",ns,nptn,ncat,nmix,nnodes,root,nInternal,treeH,nPool,R);

    size_t ecStride=(size_t)R*ns*ns, exStride=(size_t)R*ns;   // slotSz is chunk-scoped (pattern tiling)

    // ---- per-class eigen (Uinv down-map, U up-map) + per-regime weight -> GLOBAL ----
    DEVB(gb_mUinv, (size_t)nmix*ns*ns*sizeof(double));
    DEVB(gb_mU,    (size_t)nmix*ns*ns*sizeof(double));
    DEVB(gb_mUrs,  (size_t)nmix*ns*sizeof(double));
    DEVB(gb_mFreq, (size_t)nmix*ns*sizeof(double));     // k1_node_mix isRoot=0 ignores freq, but the signature needs it
    DEVB(gb_mWreg, (size_t)R*sizeof(double));
    GCK(cudaMemcpy(gb_mUinv.p, Uinv,       (size_t)nmix*ns*ns*sizeof(double), cudaMemcpyHostToDevice));
    GCK(cudaMemcpy(gb_mU.p,    U,          (size_t)nmix*ns*ns*sizeof(double), cudaMemcpyHostToDevice));
    GCK(cudaMemcpy(gb_mUrs.p,  UinvRowSum, (size_t)nmix*ns*sizeof(double),    cudaMemcpyHostToDevice));
    GCK(cudaMemcpy(gb_mFreq.p, freq,       (size_t)nmix*ns*sizeof(double),    cudaMemcpyHostToDevice));
    GCK(cudaMemcpy(gb_mWreg.p, wreg,       (size_t)R*sizeof(double),          cudaMemcpyHostToDevice));
    double *d_Uinv=(double*)gb_mUinv.p, *d_U=(double*)gb_mU.p, *d_Urs=(double*)gb_mUrs.p, *d_Freq=(double*)gb_mFreq.p, *d_Wreg=(double*)gb_mWreg.p;

    // Pattern tiling for the all-branch gradient. The two O(nptn) partial arenas
    // dominate VRAM: gb_partial = nInternal*R*ns*nptn (all postorder lower partials,
    // resident) + gb_prepool = nPool*R*ns*nptn (O(depth) preorder pool). Chunk nptn
    // into nTile contiguous pieces; per chunk run a full postorder + preorder proc
    // sweep and accumulate each edge's df/ddf into a continuous per-edge Kahan
    // accumulator carried across chunks. echild/expfac/eigen carry no pattern axis
    // (uploaded once). Per-pattern pdf[p]/pddf[p] are chunk-independent and each
    // edge's Kahan reduction adds patterns in order 0..nptn-1, so the result is
    // bit-identical to one-shot for any nTile (mirrors the lnL launcher).
    int nTile  = mix_pick_ntile(nptn, (size_t)(nInternal+nPool+1)*R*ns + 3);
    int chunk0 = (nptn + nTile - 1) / nTile;
    size_t slotSzMax = (size_t)R*ns*chunk0;

    // ---- echild + expfac uploaded ONCE (pattern-independent); partial arenas sized to chunk0 ----
    DEVB(gb_echild, (size_t)nnodes*ecStride*sizeof(double));
    DEVB(gb_expfac, (size_t)nnodes*exStride*sizeof(double));
    DEVB(gb_tip,    (size_t)ntax*chunk0);
    DEVB(gb_partial,(size_t)(nInternal>0?nInternal:1)*slotSzMax*sizeof(double));
    DEVB(gb_prepool,(size_t)nPool*slotSzMax*sizeof(double));
    DEVB(gb_pdf,    (size_t)chunk0*sizeof(double));
    DEVB(gb_pddf,   (size_t)chunk0*sizeof(double));
    DEVB(gb_patlh,  (size_t)chunk0*sizeof(double));
    DEVB(gb_mDval0, (size_t)R*ns*sizeof(double)); DEVB(gb_mDval1,(size_t)R*ns*sizeof(double)); DEVB(gb_mDval2,(size_t)R*ns*sizeof(double));
    DEVB(gb_nodeleaf, slotSzMax*sizeof(double));   // scratch for a leaf endpoint's lower eigen partial (k_leaf_eig_mix)
    double *d_baseinvar=nullptr;   // per-chunk combined invariant slice [Pn] (+I, only when pinv>0)
    if (pinv > 0.0 && base_invar_comb) { DEVB(gb_baseinv, (size_t)chunk0*sizeof(double)); d_baseinvar=(double*)gb_baseinv.p; }
    double *d_echild=(double*)gb_echild.p, *d_expfac=(double*)gb_expfac.p, *d_partial=(double*)gb_partial.p, *d_prepool=(double*)gb_prepool.p;
    double *d_pdf=(double*)gb_pdf.p, *d_pddf=(double*)gb_pddf.p, *d_patlh=(double*)gb_patlh.p;
    double *d_v0=(double*)gb_mDval0.p, *d_v1=(double*)gb_mDval1.p, *d_v2=(double*)gb_mDval2.p, *d_tipeig=(double*)gb_nodeleaf.p;
    unsigned char *d_tip=(unsigned char*)gb_tip.p;
    GCK(cudaMemcpy(d_echild,echild,(size_t)nnodes*ecStride*sizeof(double),cudaMemcpyHostToDevice));   // pattern-independent
    GCK(cudaMemcpy(d_expfac,expfac,(size_t)nnodes*exStride*sizeof(double),cudaMemcpyHostToDevice));   // pattern-independent

    int TB=256;
    // chunk-scoped pattern window (referenced by the lambdas below): updated each tile iteration
    int Pn=0, pOff=0, GB=0; size_t slotSz=0;
    std::vector<unsigned char> tipChunk((size_t)ntax*chunk0);
    // per-edge CONTINUOUS Kahan accumulators (carried across chunks => addition order 0..nptn-1 per edge)
    std::vector<double> accDf(nnodes,0.0), accDfK(nnodes,0.0), accDdf(nnodes,0.0), accDdfK(nnodes,0.0);
    for(int v=0;v<nnodes;v++){ out_df[v]=0.0; out_ddf[v]=0.0; }

    // child-args helper (exclude `excl`, -1 for none): fills echild/partial/tip pointers for k1_node_mix
    auto fillChild=[&](int u,int excl,int& nch,const double** ec,const double** p,const unsigned char** t){
        nch=0; for(int k=0;k<3;k++){ec[k]=nullptr;p[k]=nullptr;t[k]=nullptr;}
        for(int c:child[u]){ if(c==excl||nch>=3) continue; ec[nch]=d_echild+(size_t)c*ecStride;
            if(leaf[c]>=0) t[nch]=d_tip+(size_t)leaf[c]*Pn; else p[nch]=d_partial+(size_t)slot[c]*slotSz; nch++; } };

    // per-edge df/ddf reduction: ADD this chunk's patterns continuously into accDf[v]/accDdf[v] (ptn_freq-weighted
    // Kahan). Carrying (acc,accK) across chunks makes the per-edge sum one continuous Kahan sweep over 0..nptn-1.
    auto reduceInto=[&](int v){
        std::vector<double> pdf(Pn),pddf(Pn);
        cudaMemcpy(pdf.data(),d_pdf,(size_t)Pn*sizeof(double),cudaMemcpyDeviceToHost);
        cudaMemcpy(pddf.data(),d_pddf,(size_t)Pn*sizeof(double),cudaMemcpyDeviceToHost);
        double D=accDf[v],kd=accDfK[v],DD=accDdf[v],kdd=accDdfK[v];
        for(int p=0;p<Pn;p++){ double f=ptn_freq[pOff+p];
            { double term=f*pdf[p],  y=term-kd,  s=D +y; kd =(s-D )-y; D =s; }
            { double term=f*pddf[p], y=term-kdd, s=DD+y; kdd=(s-DD)-y; DD=s; } }
        accDf[v]=D; accDfK[v]=kd; accDdf[v]=DD; accDdfK[v]=kdd; };
    // resolve v's lower partial (internal slot, or synthesise a leaf tip eigen into the scratch buffer)
    auto edgeNodePtr=[&](int v)->const double*{
        if(leaf[v]<0) return d_partial+(size_t)slot[v]*slotSz;
        k_leaf_eig_mix<<<GB,TB>>>(ns,Pn,ncat,nmix,d_tip+(size_t)leaf[v]*Pn,d_Uinv,d_Urs,d_tipeig); return d_tipeig; };
    // central-edge coeffs for t=b_v: dval{0,1,2}[r*ns+x]=exp(eval_m[x]·rate_c·t)·wreg[r] (π_m absorbed in the partials)
    auto setDval=[&](double t){ std::vector<double> v0((size_t)R*ns),v1((size_t)R*ns),v2((size_t)R*ns);
        for(int m=0;m<nmix;m++) for(int c=0;c<ncat;c++){ int r=m*ncat+c; double rc=catRate[c], wr=wreg[r];
            for(int x=0;x<ns;x++){ double cof=evalC[(size_t)m*ns+x]*rc, e=exp(cof*t)*wr; v0[(size_t)r*ns+x]=e; v1[(size_t)r*ns+x]=cof*e; v2[(size_t)r*ns+x]=cof*cof*e; } }
        cudaMemcpy(d_v0,v0.data(),(size_t)R*ns*sizeof(double),cudaMemcpyHostToDevice);
        cudaMemcpy(d_v1,v1.data(),(size_t)R*ns*sizeof(double),cudaMemcpyHostToDevice);
        cudaMemcpy(d_v2,v2.data(),(size_t)R*ns*sizeof(double),cudaMemcpyHostToDevice); };

    // ---- PREORDER pool + proc recursion (the O(depth) slot pool is reset per chunk; proc rebuilt each tile) ----
    std::vector<int> freeSlots; bool poolUnderflow=false; int held=0, maxHeld=0;
    auto acq=[&]()->int{ if(freeSlots.empty()){ poolUnderflow=true; fprintf(stderr,"[ALLDERV-DBG] PREPOOL UNDERFLOW (nPool=%d treeH=%d)\n",nPool,treeH); return 0; } held++; if(held>maxHeld)maxHeld=held; int s=freeSlots.back();freeSlots.pop_back();return s;};
    auto rls=[&](int s){ held--; freeSlots.push_back(s);};
    std::function<void(int,int)> proc=[&](int u,int su){
        if(poolUnderflow) return (double)0;   // (double) to match GCK's injected return type in this lambda; value discarded by std::function<void>
        for(int v:child[u]){
            int sv=acq(); double* pre=d_prepool+(size_t)sv*slotSz;
            if(u==root){   // root child: upper partial = lower partial of root excluding v (k1_node_mix isRoot=0; no pi/wreg)
                int nch; const double* ec[3]; const double* p[3]; const unsigned char* t[3]; fillChild(root,v,nch,ec,p,t);
                k1_node_mix<<<dim3(GB,R),TB>>>(ns,Pn,ncat,nmix,/*isRoot=*/0,pre,d_patlh,d_Uinv,d_Urs,d_Freq,d_Wreg,nullptr,/*pinv=*/0.0,/*clsinv=*/nullptr,nch,
                    ec[0],p[0],t[0], ec[1],p[1],t[1], ec[2],p[2],t[2]);
            } else {       // internal parent: propagate pre_u through u with siblings-of-v and the parent branch b_u
                const double* ec[2]={0,0}; const double* sp[2]={0,0}; const unsigned char* st[2]={0,0}; int nsb=0;
                for(int w:child[u]){ if(w==v||nsb>=2) continue; ec[nsb]=d_echild+(size_t)w*ecStride;
                    if(leaf[w]>=0) st[nsb]=d_tip+(size_t)leaf[w]*Pn; else sp[nsb]=d_partial+(size_t)slot[w]*slotSz; nsb++; }
                k7_pre_mix<<<dim3(GB,R),TB>>>(ns,Pn,ncat,nmix,pre,d_prepool+(size_t)su*slotSz,d_expfac+(size_t)u*exStride,d_U,d_Uinv,d_Urs,nsb,
                    ec[0],sp[0],st[0], ec[1],sp[1],st[1]);
            }
            GCK(cudaDeviceSynchronize());
            const double* plv=edgeNodePtr(v); GCK(cudaDeviceSynchronize());
            setDval(brlen[v]);
            // +I: the invariant is branch-independent, so it enters only the 1/Lp denominator. Route to k2_derv_mix_inv
            // (Lp = variable + pinv*base_invar_comb) when pinv>0; else the plain k2_derv_mix.
            if (d_baseinvar) k2_derv_mix_inv<<<GB,TB>>>(ns,Pn,R,plv,pre,d_v0,d_v1,d_v2,pinv,d_baseinvar,d_pdf,d_pddf,d_patlh);
            else             k2_derv_mix<<<GB,TB>>>(ns,Pn,R,plv,pre,d_v0,d_v1,d_v2,d_pdf,d_pddf,d_patlh);
            GCK(cudaDeviceSynchronize());
            reduceInto(v);
            if(leaf[v]<0) proc(v,sv); rls(sv);   // recurse THEN release (v's pre must stay live for its children)
        }
        return (double)0;   // every control path must return: GCK injects `return (double)NAN`, so this lambda's
                            // deduced return type is double, and falling off the end (normal loop exit) would be UB.
    };

    // ---- TILE LOOP: full postorder + proc per chunk; per-edge Kahan accumulators carry across chunks ----
    for (int tchunk=0; tchunk<nTile; tchunk++){
        pOff=tchunk*chunk0; int p1=pOff+chunk0; if(p1>nptn)p1=nptn; Pn=p1-pOff; if(Pn<=0) break;
        slotSz=(size_t)R*ns*Pn; GB=(Pn+TB-1)/TB;
        for(int a=0;a<ntax;a++) memcpy(&tipChunk[(size_t)a*Pn], tip+(size_t)a*nptn+pOff, (size_t)Pn);   // tip stride = chunk width
        GCK(cudaMemcpy(d_tip,tipChunk.data(),(size_t)ntax*Pn,cudaMemcpyHostToDevice));
        if (d_baseinvar) GCK(cudaMemcpy(d_baseinvar, base_invar_comb+pOff, (size_t)Pn*sizeof(double), cudaMemcpyHostToDevice));   // +I chunk slice (contiguous [pOff..pOff+Pn), kernel reads baseinvar[0..Pn-1])

        // POSTORDER: lower partial pl_u for every internal node (skip root: its lower partial is never used)
        for (int idx=0; idx<nInternal; idx++){ int u=postorder[idx]; if(u==root) continue;
            int nch; const double* ec[3]; const double* p[3]; const unsigned char* t[3]; fillChild(u,-1,nch,ec,p,t);
            k1_node_mix<<<dim3(GB,R),TB>>>(ns,Pn,ncat,nmix,/*isRoot=*/0,d_partial+(size_t)slot[u]*slotSz,d_patlh,d_Uinv,d_Urs,d_Freq,d_Wreg,/*out_lhcat=*/nullptr,/*pinv=*/0.0,/*clsinv=*/nullptr,nch,
                ec[0],p[0],t[0], ec[1],p[1],t[1], ec[2],p[2],t[2]); }
        GCK(cudaDeviceSynchronize()); GCK(cudaGetLastError());

        // PREORDER: reset the O(depth) pool, then proc(root) does upper partials + per-edge reduceInto for this chunk
        freeSlots.clear(); for(int s=nPool-1;s>=0;s--) freeSlots.push_back(s); held=0;
        proc(root,-1);
        if(poolUnderflow) return (double)NAN;
        GCK(cudaDeviceSynchronize()); GCK(cudaGetLastError());
    }
    if(getenv("IQTREE_GPU_DEBUG")) fprintf(stderr,"[ALLDERV-DBG] tiled proc done (nTile=%d underflow=%d maxHeld=%d treeH=%d nInternal=%d)\n",nTile,(int)poolUnderflow,maxHeld,treeH,nInternal);
    for(int v=0;v<nnodes;v++){ out_df[v]=accDf[v]; out_ddf[v]=accDdf[v]; }
    return 0.0;
}

// Single-edge derivative launcher. The descriptor list covers both subtrees split
// by the central edge (two sub-roots: the edge endpoints), every entry isRoot=0 so
// each internal node (incl. the two endpoints) writes its eigen-space partial to
// its slot. node_eig=d_partial[nodeSlot], dad_eig=d_partial[dadSlot] are the two
// endpoint partials excluding the central edge's transition (applied via
// val0=exp(eval*r*t)). Returns df = d(lnL)/dt = sum_ptn ptn_freq*(d1/lh);
// *out_ddf = sum_ptn ptn_freq*(d2/lh-(d1/lh)^2); *out_lnL = the tree lnL at t (a
// free cross-check). NaN on CUDA error.
