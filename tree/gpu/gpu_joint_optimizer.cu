// gpu_joint_optimizer.cu — JOLT joint-gradient model optimiser (GPU launcher).
//
// One entry point (gpu_jolt_optimize) optimises every continuous parameter of a
// model on one fixed topology — branch lengths, Gamma alpha, p-inv, +R
// rates/weights, and free-Q exchangeabilities — in a joint Levenberg-Marquardt
// loop, evaluating the likelihood and its full gradient on the GPU each step.
// The optimum matches the CPU optimiser to rel<=1e-12. The optimiser's private
// derivative/reduction kernels (kj_derv_fused, kj_reduce3, kj_invl,
// kj_reduce_gradnum) live here; the shared postorder/preorder kernels it also
// launches (k1_node, k_leaf_eig, kj_pre) are declared in gpu_kernels.h.
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
#include <mutex>        // serialize GPU access — ModelFinder is across-model OpenMP-parallel
#include <chrono>       // host-side timer for the per-eval echild rebuild (--jolt-diag)

// =============================== JOLT private kernels ===============================
// kj_derv_fused: compute theta = node*dad in registers (never materialised to a
// d_theta arena), then emit patlh/pdf/pddf AND (if rnum!=null) accumulate the
// per-category rate-gradient numerator (rnum[c] += g_rscale[c]*sum_x g_val1[c,x]*theta)
// in one pass — node+dad read once each, the theta VRAM round-trip eliminated.
// On this bandwidth-bound kernel that is the win. Bit-identical to the unfused
// path: the per-(c,x) products are summed in the same order (FP64 store/load is
// lossless), so the optimiser's accept/reject trajectory is reproducible.
// rnum==null => derv-only (the evalLnL path, no rate gradient).
// d1 = sum_c rc reuses the per-cat sum.
__global__ void kj_derv_fused(int ns, int nptn, int ncat,
        const double* __restrict__ node, const double* __restrict__ dad,
        double pinv, const double* __restrict__ baseinvar,
        double* __restrict__ patlh, double* __restrict__ pdf, double* __restrict__ pddf,
        double* __restrict__ rnum, double* __restrict__ wnum){   // wnum[c]=Lc(p) per-category likelihood (weight-grad numerator)
    int ptn = blockIdx.x*blockDim.x + threadIdx.x; if (ptn>=nptn) return;
    double lh=0.0,d1=0.0,d2=0.0;
    for (int c=0;c<ncat;c++){
        double rc=0.0, lcc=0.0;
        for (int x=0;x<ns;x++){ int k=c*ns+x; size_t o=(size_t)k*nptn+ptn;
            double th=node[o]*dad[o];
            lcc+=g_val0[k]*th; rc+=g_val1[k]*th; d2+=g_val2[k]*th; }
        lh+=lcc; d1+=rc;
        if (rnum) rnum[(size_t)c*nptn+ptn]+=g_rscale[c]*rc;
        if (wnum) wnum[(size_t)c*nptn+ptn]=lcc;   // Lc(p)=sum_x g_val0[c,x]*theta (category weight w_c already folded into g_val0)
    }
    double Lp=fabs(lh)+pinv*baseinvar[ptn]; double inv=1.0/Lp, r=d1*inv;
    patlh[ptn]=log(Lp); pdf[ptn]=r; pddf[ptn]=d2*inv-r*r;
}

// kj_reduce3 — deterministic block-level FP64 reduction of the 3 per-pattern
// derivative channels {patlh,pdf,pddf}, each weighted by ptn_freq. Each block
// tree-reduces its blockDim patterns into 3 partial sums out3[ch*nblk + blockIdx];
// the host then Kahan-combines the (small) nblk per-block partials in channel
// order. The within-block reduce is a fixed-blockDim shared-memory tree (not
// atomicAdd), so it is bit-reproducible across launches and the LM accept/reject
// trajectory is stable. Requires blockDim a power of two (TB=256).
__global__ void kj_reduce3(int nptn, const double* __restrict__ patlh, const double* __restrict__ pdf,
        const double* __restrict__ pddf, const double* __restrict__ ptnfreq, int nblk, double* __restrict__ out3){
    extern __shared__ double sm[];                  // 3*blockDim doubles
    double* sL=sm; double* sD=sm+blockDim.x; double* sDD=sm+2*blockDim.x;
    int tid=threadIdx.x; int p=blockIdx.x*blockDim.x+tid;
    double f = (p<nptn) ? ptnfreq[p] : 0.0;
    sL[tid]  = (p<nptn) ? f*patlh[p] : 0.0;
    sD[tid]  = (p<nptn) ? f*pdf[p]   : 0.0;
    sDD[tid] = (p<nptn) ? f*pddf[p]  : 0.0;
    __syncthreads();
    for (int s=blockDim.x>>1; s>0; s>>=1){
        if (tid<s){ sL[tid]+=sL[tid+s]; sD[tid]+=sD[tid+s]; sDD[tid]+=sDD[tid+s]; }
        __syncthreads();
    }
    if (tid==0){ out3[blockIdx.x]=sL[0]; out3[(size_t)nblk+blockIdx.x]=sD[0]; out3[(size_t)2*nblk+blockIdx.x]=sDD[0]; }
}
// kj_invl — 1/L_p = exp(-patlh[p]) on-device (the base-edge inverse likelihood).
__global__ void kj_invl(int nptn, const double* __restrict__ patlh, double* __restrict__ invl){
    int p=blockIdx.x*blockDim.x+threadIdx.x; if(p>=nptn) return; invl[p]=exp(-patlh[p]);
}
// kj_reduce_gradnum — per-category deterministic block reduction of
// ptn_freq[p]*rnum[c][p]*invl[p] (the +R/alpha rate-gradient numerator).
// out[c*nblk + blockIdx] = per-block partial; the host sums the (small) nblk
// partials per category, then scales by catProp_v[c]. Deterministic shared-mem
// tree reduce (no atomicAdd). The common factor ptn_freq*invl is loaded once per
// thread; the ncat passes reuse one shared array.
__global__ void kj_reduce_gradnum(int nptn, int ncat, const double* __restrict__ rnum,
        const double* __restrict__ invl, const double* __restrict__ ptnfreq, int nblk, double* __restrict__ out){
    extern __shared__ double sm[];                  // blockDim doubles
    int tid=threadIdx.x; int p=blockIdx.x*blockDim.x+tid;
    double fw = (p<nptn) ? ptnfreq[p]*invl[p] : 0.0;
    for (int c=0;c<ncat;c++){
        sm[tid] = (p<nptn) ? fw*rnum[(size_t)c*nptn+p] : 0.0;
        __syncthreads();
        for (int s=blockDim.x>>1; s>0; s>>=1){ if (tid<s) sm[tid]+=sm[tid+s]; __syncthreads(); }
        if (tid==0) out[(size_t)c*nblk+blockIdx.x]=sm[0];
        __syncthreads();
    }
}

// =============================== JOLT joint-gradient optimiser launcher ===============================
// Mean-rate discrete-Gamma (Yang 1994; IQ-TREE's "mean of the portion"):
// r_c = K*[P(alpha+1, alpha*b_c) - P(alpha+1, alpha*b_{c-1})].
static double jolt_gammp_reg(double a, double x){   // regularized lower incomplete gamma P(a,x) (gser/gcf)
    if (x<=0.0) return 0.0; double gln=lgamma(a);
    if (x<a+1.0){ double ap=a,sum=1.0/a,del=sum; for(int n=1;n<=300;n++){ ap+=1.0; del*=x/ap; sum+=del; if(fabs(del)<fabs(sum)*1e-16) break; }
        return sum*exp(-x+a*log(x)-gln); }
    double b=x+1.0-a,c=1e300,d=1.0/b,h=d;
    for(int i=1;i<=300;i++){ double an=-(double)i*((double)i-a); b+=2.0; d=an*d+b; if(fabs(d)<1e-300)d=1e-300; c=b+an/c; if(fabs(c)<1e-300)c=1e-300; d=1.0/d; double del=d*c; h*=del; if(fabs(del-1.0)<1e-16) break; }
    return 1.0-exp(-x+a*log(x)-gln)*h;
}
static double jolt_gammp_inv(double a, double p){    // inverse: x s.t. P(a,x)=p, by bracketed bisection
    if (p<=0.0) return 0.0; if (p>=1.0) return 1e300;
    double lo=0.0,hi=a+10.0*sqrt(a+1.0)+20.0; int guard=0; while(jolt_gammp_reg(a,hi)<p && guard++<200) hi*=2.0;
    for(int it=0;it<200;it++){ double mid=0.5*(lo+hi); if(jolt_gammp_reg(a,mid)<p) lo=mid; else hi=mid; if(hi-lo<1e-13*(mid+1e-13)) break; }
    return 0.5*(lo+hi);
}
static void jolt_discreteGammaMean(double alpha, int K, double* rates){
    if (K==1){ rates[0]=1.0; return; }
    double prev=0.0;
    for(int c=0;c<K;c++){ double hi;
        if(c==K-1) hi=1.0;
        else { double bc=jolt_gammp_inv(alpha,(double)(c+1)/(double)K)/alpha; hi=jolt_gammp_reg(alpha+1.0, alpha*bc); }
        rates[c]=(double)K*(hi-prev); prev=hi; }
}
// Host shim so the mixture joint-optimiser's alpha override can recompute mean-1
// discrete-gamma rates at an iterate alpha, matching the live RateGamma mean-rate
// (GAMMA_CUT_MEAN) path.
extern "C" void gpu_discrete_gamma_mean(double alpha, int K, double* rates){ jolt_discreteGammaMean(alpha, K, rates); }

// JOLT persistent device buffers (separate from the lnL/derv pools; same alloc-once / reuse policy).
static DevBuf gbj_echild, gbj_partial, gbj_patlh, gbj_pdf, gbj_pddf,
              gbj_pretmp, gbj_tipeig, gbj_prepool, gbj_expfac, gbj_rnum, gbj_tip, gbj_baseinvar,
              gbj_ptnfreq, gbj_redpart,   // on-device ptn_freq + per-block reduction partials
              gbj_invlbase, gbj_redR,     // base-edge 1/L_p + per-category gradR partials
              gbj_wnum, gbj_redW;         // +R per-category Lc(p) (weight-grad numerator) + its block-reduction partials

// --jolt-diag: per-eval host echild-rebuild cost (host loop + 2 blocking H2D in rebuildEchild). Gated by env
// JOLT_DIAG (set by --jolt-diag; the CUDA TU cannot see Params). Accumulated across the LM loop; reported per call.
static bool   g_jdiag_init = false;
static bool   g_jdiag = false;
static double g_jd_echild_sec = 0.0;
static long   g_jd_echild_n = 0;

extern "C" double gpu_jolt_optimize(
    int nstates, int nptn, int ncat, int ntax, int nnodes, int root,
    const double* Uinv, const double* UinvRowSum, const double* U, const double* eval,
    const double* catProp, const unsigned char* tip, const double* ptn_freq,
    const int* node_nchild, const int* node_child, const int* node_leaf, const double* node_parentLen,
    double alpha0, int optAlpha, int maxiter,
    const double* base_invar, double pinv0, int optPinv, double pinvMin, double pinvMax,
    const double* catRate0, int freeRate,   // +R FreeRate — catRate0=rates[c] (else nullptr); freeRate=1 seeds rates directly (no alpha)
    int nFreeQ, const double* q0, jolt_qdecompose_fn qdecompose, void* qctx, double* out_q,   // DNA free-Q (eigensystem moves)
    double* out_brlen, double* out_alpha, double* out_pinv, int* out_iters,
    double* out_rates, double* out_props)   // +R optimised rates/weights (nullptr unless freeRate==1)
{
    int ns = nstates;
    if (ns > NS_MAX || ncat > 64) { fprintf(stderr,"[JOLT] unsupported ns=%d ncat=%d\n",ns,ncat); return (double)NAN; }

    // ModelFinder evaluates candidates across-model in parallel, so this launcher
    // can be entered by many threads at once. The single GPU's __constant__ symbols
    // (g_Uinv/g_U/g_val*/g_rscale) and the static DevBuf pool (gbj_*) are
    // process-global device state, so concurrent use would clobber. Serialize the
    // whole GPU computation: JOLT models run one at a time on the GPU while other
    // threads keep optimising CPU-fallback candidates.
    static std::mutex jolt_gpu_mtx;
    std::lock_guard<std::mutex> jolt_lock(jolt_gpu_mtx);
    // --jolt-diag: init the gate + snapshot the per-call echild baseline inside the lock, so the across-model
    // OpenMP concurrency can't tear the baseline read or race g_jdiag_init.
    if (!g_jdiag_init) { g_jdiag = (std::getenv("JOLT_DIAG") != nullptr); g_jdiag_init = true; }
    double _jd_ech0 = g_jd_echild_sec; long _jd_echn0 = g_jd_echild_n;
    // Reopt proof-counters (JOLT_DEBUG): per-call tally of reopt coefficient uploads. Function-local (reset per
    // call); guarded by the GPU mutex above.
    long ts_reopt_mcs = 0;   // cudaMemcpyToSymbol count (g_val0/1/2 + g_rscale per edge)
    long ts_reopt_vp  = 0;   // cudaMemcpyAsync-to-valpool count

    // alpha-independent eigen constants — upload once (the base-Q eigensystem). For free-Q (nFreeQ>0) qApply()
    // re-uploads these whenever an exchangeability changes; for fixed-Q this is the only upload.
    GCK(cudaMemcpyToSymbol(g_Uinv, Uinv, sizeof(double)*ns*ns));
    GCK(cudaMemcpyToSymbol(g_U,    U,    sizeof(double)*ns*ns));
    GCK(cudaMemcpyToSymbol(g_UinvRowSum, UinvRowSum, sizeof(double)*ns));

    // Free-Q: mutable working copies of the eigensystem (refreshed per Q change via qApply). For fixed-Q
    // (nFreeQ==0) evalP/UP alias the passed-in const arrays (the lambdas below use evalP/UP). For free-Q they
    // point at the buffers that qApply overwrites in place.
    std::vector<double> evalB, UB, UinvB;
    const double *evalP = eval, *UP = U;
    if (nFreeQ > 0) { evalB.assign(eval, eval+ns); UB.assign(U, U+ns*ns); UinvB.assign(Uinv, Uinv+ns*ns);
                      evalP = evalB.data(); UP = UB.data(); }
    auto qApply = [&](const double* q) -> void {   // re-decompose for a trial Q and re-upload eval/U/Uinv; rebuildEchild()/setVal() then use the new evalP/UP
        // Plain cudaMemcpyToSymbol (not GCK): GCK's `return (double)NAN` would return from this void lambda,
        // swallowing the error. Any failure is caught by the final cudaGetLastError() backstop (the sticky
        // last-error persists to the end of gpu_jolt_optimize -> NaN -> CPU).
        qdecompose(qctx, q, evalB.data(), UB.data(), UinvB.data());
        cudaMemcpyToSymbol(g_Uinv, UinvB.data(), sizeof(double)*ns*ns);
        cudaMemcpyToSymbol(g_U,    UB.data(),    sizeof(double)*ns*ns);
        double rs[NS_MAX]; for(int i=0;i<ns;i++){ double s=0; for(int j=0;j<ns;j++) s+=UinvB[i*ns+j]; rs[i]=s; }
        cudaMemcpyToSymbol(g_UinvRowSum, rs, sizeof(double)*ns); };

    // ---- rebuild topology from flat arrays (node ids = caller's DFS index) ----
    std::vector<std::vector<int>> child(nnodes);
    std::vector<int> leaf(nnodes);
    for (int u=0; u<nnodes; u++){ leaf[u]=node_leaf[u];
        for (int k=0; k<node_nchild[u] && k<3; k++){ int c=node_child[u*3+k]; if (c>=0) child[u].push_back(c); } }
    std::vector<double> brlen(node_parentLen, node_parentLen+nnodes);

    std::vector<int> postorder; std::vector<int> slot(nnodes,-1);
    std::function<void(int)> dfs=[&](int u){ for(int c:child[u]) dfs(c); if(leaf[u]<0){ slot[u]=(int)postorder.size(); postorder.push_back(u);} };
    dfs(root); int nInternal=(int)postorder.size();
    int c0=-1; for(int c:child[root]) if(leaf[c]<0){ c0=c; break; } if(c0<0){ fprintf(stderr,"[JOLT] no internal root child\n"); return (double)NAN; }
    std::vector<int> edgeV; for(int u=0;u<nnodes;u++) for(int v:child[u]) edgeV.push_back(v); int nedge=(int)edgeV.size();
    int treeH=0; std::function<void(int,int)> ddfs=[&](int u,int d){ if(d>treeH)treeH=d; for(int c:child[u]) ddfs(c,d+1); }; ddfs(root,0); int nPool=treeH+2;

    size_t ecStride=(size_t)ncat*ns*ns;
    int TB=256;

    // ===== PATTERN TILING — fit the O(nptn) partial arenas on smaller GPUs =====
    // Every JOLT quantity (lnL, df_e, ddf_e, gradR_c) is a sum over patterns, so partitioning the nptn patterns into
    // nTile contiguous chunks, running a full postorder+preorder sweep per chunk, and Kahan-accumulating each chunk's
    // contribution reproduces the one-shot result to rel<=1e-12 (the same additivity that underlies the ptn_freq-
    // weighted reductions). Every O(nptn) device arena (the dominant postorder gbj_partial, the preorder pool,
    // scratch, tip/patlh/...) shrinks by ~nTile. The chunk-independent echild/expfac/eigen constants are built once
    // per (brlen,alpha,pinv,Q) point (rebuildEchild), not per chunk.
    int nTile = 1;
    if (const char* e = getenv("JOLT_NTILE")) { nTile = atoi(e); if (nTile < 1) nTile = 1; }
    else {
        // auto-pick from free VRAM: estimate the one-shot footprint, target 80% of free, round up.
        size_t slot1 = (size_t)ncat*ns*nptn*sizeof(double);
        size_t foot  = (size_t)(nInternal + nPool + 3) * slot1                 // partial + prepool + (pretmp/tipeig+slack)
                     + (size_t)ncat*nptn*sizeof(double)                        // rnum
                     + (size_t)6*nptn*sizeof(double)                           // patlh/pdf/pddf/baseinvar/ptnfreq/invlbase
                     + (size_t)ntax*nptn                                       // tip
                     + (size_t)nnodes*ecStride*sizeof(double);                 // echild (chunk-independent; not tiled)
        size_t freeB=0, totB=0;
        if (cudaMemGetInfo(&freeB,&totB)==cudaSuccess && freeB>0) {
            double budget = 0.80 * (double)freeB;
            int T = (int)ceil((double)foot / budget); if (T<1) T=1;
            nTile = T;
        }
    }
    if (freeRate) nTile = 1;   // +R (freeRate 1 and 2) runs on full-nptn buffers; no pattern tiling
    if (getenv("JOLT_DEBUG")) {
        size_t fB=0,tB=0; cudaMemGetInfo(&fB,&tB);
        fprintf(stderr,"[JOLT-TILE] nptn=%d ns=%d ncat=%d nInternal=%d nPool=%d -> nTile=%d (chunk~%d ptn); freeVRAM=%.1f GB\n",
                nptn,ns,ncat,nInternal,nPool,nTile,(nptn+nTile-1)/nTile,(double)fB/1073741824.0); fflush(stderr);
    }

    int    chunk0    = (nptn + nTile - 1) / nTile;   // max chunk width; all per-pattern buffers are sized to this
    size_t slotSzMax = (size_t)ncat*ns*chunk0;
    int    GBmax     = (chunk0 + TB - 1) / TB;
    // current-chunk state (mutable; set by setChunk; the sweep closures capture these by reference). At nTile==1
    // chunk0==nptn / Pn==nptn / pOff==0 / slotSz==slotSzMax / GB==GBmax (the single-chunk case).
    int    Pn     = nptn;                            // current chunk's pattern count
    int    pOff   = 0;                               // current chunk's first pattern index (into the host inputs)
    size_t slotSz = (size_t)ncat*ns*nptn;            // current chunk's [cat][state][ptn] slot stride
    int    GB     = (nptn + TB - 1) / TB;            // current chunk's grid
    (void)pOff;

    DEVB(gbj_echild, (size_t)nnodes*ecStride*sizeof(double));
    DEVB(gbj_partial,(size_t)(nInternal>0?nInternal:1)*slotSzMax*sizeof(double));
    // No d_theta arena: kj_derv_fused computes theta in registers.
    DEVB(gbj_patlh,  (size_t)chunk0*sizeof(double)); DEVB(gbj_pdf,(size_t)chunk0*sizeof(double)); DEVB(gbj_pddf,(size_t)chunk0*sizeof(double));
    DEVB(gbj_pretmp, slotSzMax*sizeof(double)); DEVB(gbj_tipeig, slotSzMax*sizeof(double));
    DEVB(gbj_prepool,(size_t)nPool*slotSzMax*sizeof(double));
    DEVB(gbj_expfac, (size_t)nnodes*ncat*ns*sizeof(double));
    DEVB(gbj_rnum,   (size_t)ncat*chunk0*sizeof(double));
    DEVB(gbj_tip,    (size_t)ntax*chunk0);
    DEVB(gbj_baseinvar, (size_t)chunk0*sizeof(double));   // +I: pinv-independent invariant base per pattern
    DEVB(gbj_ptnfreq,   (size_t)chunk0*sizeof(double));   // pattern weights, constant across the optimise call
    double *d_echild=(double*)gbj_echild.p,*d_partial=(double*)gbj_partial.p;
    double *d_patlh=(double*)gbj_patlh.p,*d_pdf=(double*)gbj_pdf.p,*d_pddf=(double*)gbj_pddf.p;
    double *d_pretmp=(double*)gbj_pretmp.p,*d_tipeig=(double*)gbj_tipeig.p,*d_prepool=(double*)gbj_prepool.p;
    double *d_expfac=(double*)gbj_expfac.p,*d_rnum=(double*)gbj_rnum.p,*d_baseinvar=(double*)gbj_baseinvar.p;
    double *d_ptnfreq=(double*)gbj_ptnfreq.p;
    unsigned char* d_tip=(unsigned char*)gbj_tip.p;
    // tip/ptn_freq/base_invar are constant across the optimise call; setChunk uploads the current chunk's slice. (At
    // nTile==1 this uploads the whole arrays once per sweep.)
    std::vector<unsigned char> tipChunk((size_t)ntax*chunk0);
    std::vector<double> biFull(nptn, 0.0); if (base_invar) for (int p=0;p<nptn;p++) biFull[p]=base_invar[p];
    auto setChunk=[&](int t){
        int p0=t*chunk0, p1=p0+chunk0; if(p1>nptn)p1=nptn; int cw=p1-p0;
        Pn=cw; pOff=p0; slotSz=(size_t)ncat*ns*cw; GB=(cw+TB-1)/TB;
        for(int a=0;a<ntax;a++) memcpy(&tipChunk[(size_t)a*cw], tip+(size_t)a*nptn+p0, (size_t)cw);
        cudaMemcpy(d_tip, tipChunk.data(), (size_t)ntax*cw, cudaMemcpyHostToDevice);
        cudaMemcpy(d_ptnfreq, ptn_freq+p0, (size_t)cw*sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(d_baseinvar, biFull.data()+p0, (size_t)cw*sizeof(double), cudaMemcpyHostToDevice);
    };

    DEVB(gbj_redpart, (size_t)3*GBmax*sizeof(double));   // 3 channels x GBmax per-block partial sums
    double* d_redpart=(double*)gbj_redpart.p;
    std::vector<double> h_redpart((size_t)3*GBmax);
    DEVB(gbj_invlbase, (size_t)chunk0*sizeof(double)); double* d_invLbase=(double*)gbj_invlbase.p;  // base-edge 1/L_p
    DEVB(gbj_redR,     (size_t)ncat*GBmax*sizeof(double)); double* d_redR=(double*)gbj_redR.p;
    std::vector<double> h_redR((size_t)ncat*GBmax);
    // +R: weight-grad numerator Lc(p) buffer + its per-block reduction (only referenced when freeRate==1 engages the
    // +R joint LM below). Independent cudaMalloc, so no effect on any other buffer's layout.
    double* d_wnum=nullptr; double* d_redW=nullptr; std::vector<double> h_redW;
    if(freeRate==1){ DEVB(gbj_wnum,(size_t)ncat*chunk0*sizeof(double)); d_wnum=(double*)gbj_wnum.p;
                     DEVB(gbj_redW,(size_t)ncat*GBmax*sizeof(double)); d_redW=(double*)gbj_redW.p; h_redW.assign((size_t)ncat*GBmax,0.0); }
    std::vector<double> h_echild((size_t)nnodes*ecStride), h_expfac((size_t)nnodes*ncat*ns);
    std::vector<double> patlh(nptn),pdf(nptn),pddf(nptn);
    std::vector<double> catRate(ncat,1.0), catProp_v(catProp, catProp+ncat);
    std::vector<double> meanR(ncat,1.0);   // mean-1 discrete-gamma rates (alpha-dependent only)
    double curAlpha=alpha0;
    auto applyAlpha=[&](double a){ jolt_discreteGammaMean(a,ncat,meanR.data()); };  // -> meanR (mean 1)
    if (ncat>1 && !freeRate) applyAlpha(curAlpha);   // +R seeds rates directly (below), not from alpha
    // +I: IQ-TREE's RateGammaInvar uses getProp(c)=(1-pinv)/K and rescales the gamma rates to meanR[c]/(1-pinv), so
    // the overall mean rate including invariant sites at rate 0 stays 1. Both the rate (up by 1/(1-pinv)) and the
    // prop (down by (1-pinv)) move with pinv. catProp[c] arrives as (1-pinv0)/K, so the pinv-free base prop is
    // bprop=catProp/(1-pinv0). For non-+I (optPinv=0): f=1 => catRate=meanR, catProp_v=catProp.
    std::vector<double> bprop(ncat);
    for(int c=0;c<ncat;c++) bprop[c] = optPinv ? catProp[c]/(1.0-pinv0) : catProp[c];
    // +R FreeRate: seed the pinv-free basis meanR=rho_c (mean-1 rates, sum w.rho=1), bprop=w_c (weights), so
    // applyPinv(pinv0) reproduces catRate=getRate, catProp_v=getProp. RateFreeInvar: getRate=rho/(1-p),
    // getProp=(1-p)w => meanR=catRate0*f0, bprop=catProp/f0 with f0=(1-pinv0). For pure +R (optPinv=0) f0=1.
    if (freeRate) { double f0 = optPinv ? (1.0-pinv0) : 1.0;
                    for(int c=0;c<ncat;c++){ meanR[c]=catRate0[c]*f0; bprop[c]=catProp[c]/f0; } }
    double curPinv = optPinv ? pinv0 : 0.0;
    auto applyPinv=[&](double p){ double f = optPinv ? (1.0-p) : 1.0;
        for(int c=0;c<ncat;c++){ catRate[c]=meanR[c]/f; catProp_v[c]=f*bprop[c]; } };
    applyPinv(curPinv);

    auto childArgs=[&](int u,int excl,int& nch,const double** ec,const double** p,const unsigned char** t){
        nch=0; for(int k=0;k<3;k++){ec[k]=p[k]=nullptr;t[k]=nullptr;}
        for(int c:child[u]){ if(c==excl) continue; ec[nch]=d_echild+(size_t)c*ecStride;
            if(leaf[c]>=0) t[nch]=d_tip+(size_t)leaf[c]*Pn; else p[nch]=d_partial+(size_t)slot[c]*slotSz; nch++; } };   // tip stride = current chunk width Pn
    auto sibArg=[&](int w,const double*& ec,const double*& sp,const unsigned char*& st){
        ec=d_echild+(size_t)w*ecStride; sp=nullptr; st=nullptr;
        if(leaf[w]>=0) st=d_tip+(size_t)leaf[w]*Pn; else sp=d_partial+(size_t)slot[w]*slotSz; };
    auto rebuildEchild=[&](){
        std::chrono::steady_clock::time_point _jd_e0; if(g_jdiag) _jd_e0 = std::chrono::steady_clock::now();   // --jolt-diag timer
        for(int c=0;c<nnodes;c++){ if(c==root){ for(size_t z=0;z<ecStride;z++) h_echild[(size_t)c*ecStride+z]=0.0; continue; }
            for(int cat=0;cat<ncat;cat++){ double len=brlen[c]*catRate[cat]; double ex[NS_MAX]; for(int i=0;i<ns;i++) ex[i]=exp(evalP[i]*len);
                double* e=&h_echild[(size_t)c*ecStride+(size_t)cat*ns*ns]; for(int x=0;x<ns;x++) for(int i=0;i<ns;i++) e[x*ns+i]=UP[x*ns+i]*ex[i];
                for(int i=0;i<ns;i++) h_expfac[(size_t)c*ncat*ns+cat*ns+i]=ex[i]; } }
        cudaMemcpy(d_echild,h_echild.data(),(size_t)nnodes*ecStride*sizeof(double),cudaMemcpyHostToDevice);
        cudaMemcpy(d_expfac,h_expfac.data(),(size_t)nnodes*ncat*ns*sizeof(double),cudaMemcpyHostToDevice);
        if(g_jdiag){ g_jd_echild_sec += std::chrono::duration<double>(std::chrono::steady_clock::now()-_jd_e0).count(); g_jd_echild_n++; } };   // --jolt-diag: echild cost
    auto postorderFill=[&](){
        for(int idx=0; idx<nInternal; idx++){ int u=postorder[idx]; if(u==root) continue;
            int nch; const double* ec[3]; const double* p[3]; const unsigned char* t[3]; childArgs(u,-1,nch,ec,p,t);
            k1_node<<<GB,TB>>>(ns,Pn,ncat,0,d_partial+(size_t)slot[u]*slotSz,d_patlh,nch,ec[0],p[0],t[0],ec[1],p[1],t[1],ec[2],p[2],t[2]); }
        cudaDeviceSynchronize(); };
    // Central-edge coefficient tables, split into a pure host-build helper (setValBuild) and the upload (setVal).
    // setValBuild fills v0/v1/v2 (exp(evalP[x]*rc*t)*pcw factor order); setVal does the 3 cudaMemcpyToSymbol of
    // those bytes. v0/v1/v2 must be sized ncat*ns by the caller.
    auto setValBuild=[&](double t,std::vector<double>& v0,std::vector<double>& v1,std::vector<double>& v2){
        for(int c=0;c<ncat;c++){ double rc=catRate[c],pcw=catProp_v[c]; for(int x=0;x<ns;x++){ double re=rc*evalP[x],e=exp(evalP[x]*rc*t)*pcw;
            v0[c*ns+x]=e; v1[c*ns+x]=re*e; v2[c*ns+x]=re*re*e; } } };
    auto setVal=[&](double t){ std::vector<double> v0(ncat*ns),v1(ncat*ns),v2(ncat*ns);
        setValBuild(t,v0,v1,v2);
        cudaMemcpyToSymbol(g_val0,v0.data(),sizeof(double)*ncat*ns); cudaMemcpyToSymbol(g_val1,v1.data(),sizeof(double)*ncat*ns); cudaMemcpyToSymbol(g_val2,v2.data(),sizeof(double)*ncat*ns);
        ts_reopt_mcs += 3; };   // proof-counter (3 constant-memory uploads per setVal)
    auto reduceDerv=[&](double& lnL,double& df,double& ddf){
        // On-device ptn_freq-weighted block reduction -> 3*GB partials D2H, replacing a 3x nptn D2H + single-thread
        // host Kahan. The cross-block combine stays on host in channel order (Kahan) for bit-reproducibility, which
        // keeps the LM accept/reject trajectory stable.
        kj_reduce3<<<GB,TB,(size_t)3*TB*sizeof(double)>>>(Pn,d_patlh,d_pdf,d_pddf,d_ptnfreq,GB,d_redpart);
        cudaMemcpy(h_redpart.data(),d_redpart,(size_t)3*GB*sizeof(double),cudaMemcpyDeviceToHost);
        double L=0,kc=0,D=0,kd=0,DD=0,kdd=0;
        for(int b=0;b<GB;b++){
            { double term=h_redpart[b],            y=term-kc, s=L +y; kc =(s-L )-y; L =s; }
            { double term=h_redpart[(size_t)GB+b], y=term-kd, s=D +y; kd =(s-D )-y; D =s; }
            { double term=h_redpart[(size_t)2*GB+b],y=term-kdd,s=DD+y; kdd=(s-DD)-y; DD=s; } }
        lnL=L; df=D; ddf=DD; };
    // Resolve v's eigen node-partial pointer (synthesising a leaf tip vector into d_tipeig if needed) without
    // materialising theta — kj_derv_fused reads node+dad directly.
    auto edgeNodePtr=[&](int v)->const double*{
        if(leaf[v]<0) return d_partial+(size_t)slot[v]*slotSz;
        k_leaf_eig<<<GB,TB>>>(ns,Pn,ncat,d_tip+(size_t)leaf[v]*Pn,d_tipeig); return d_tipeig; };

    // ============ +R weight-gradient finite-difference self-check (gated by JOLT_RGRADCHECK) ============
    // Validates the softmax weight gradient gz_c = WN_c - w_c*N against central FD on the real GPU path. Weights
    // enter only through g_val0 (setVal), not the partials/echild, and Lc(p) is edge-invariant, so WN_c comes from
    // the base edge and each FD perturbation re-runs only setVal + one base-edge kj_derv_fused.
    if (freeRate) {
        if (getenv("JOLT_RGRADCHECK")) {
            setChunk(0);   // nTile==1 for +R — upload the (whole) tip/ptn_freq/base_invar slice
            rebuildEchild(); postorderFill();
            int nch; const double* ec[3]; const double* p[3]; const unsigned char* tp[3]; childArgs(root,c0,nch,ec,p,tp);
            k1_node<<<GB,TB>>>(ns,nptn,ncat,0,d_pretmp,d_patlh,nch,ec[0],p[0],tp[0],ec[1],p[1],tp[1],ec[2],p[2],tp[2]); cudaDeviceSynchronize();
            const double* pl0=edgeNodePtr(c0); cudaDeviceSynchronize();
            cudaMemset(d_rnum,0,(size_t)ncat*nptn*sizeof(double));   // reuse the rnum buffer as wnum (rnum unused here)
            setVal(brlen[c0]);
            kj_derv_fused<<<GB,TB>>>(ns,nptn,ncat,pl0,d_pretmp,0.0,d_baseinvar,d_patlh,d_pdf,d_pddf,nullptr,d_rnum); cudaDeviceSynchronize();
            double lnL0,dtmp,ddtmp; reduceDerv(lnL0,dtmp,ddtmp);
            kj_invl<<<GB,TB>>>(nptn,d_patlh,d_invLbase); cudaDeviceSynchronize();
            kj_reduce_gradnum<<<GB,TB,(size_t)TB*sizeof(double)>>>(nptn,ncat,d_rnum,d_invLbase,d_ptnfreq,GB,d_redR); cudaDeviceSynchronize();
            cudaMemcpy(h_redR.data(),d_redR,(size_t)ncat*GB*sizeof(double),cudaMemcpyDeviceToHost);
            std::vector<double> WN(ncat,0.0); double sumWN=0;
            for(int c=0;c<ncat;c++){ long double a=0; for(int b=0;b<GB;b++) a+=(long double)h_redR[(size_t)c*GB+b]; WN[c]=(double)a; sumWN+=WN[c]; }
            double Ntot=0; for(int pp=0;pp<nptn;pp++) Ntot+=ptn_freq[pp];
            std::vector<double> w(catProp_v), gz(ncat); double sumgz=0;
            for(int c=0;c<ncat;c++){ gz[c]=WN[c]-w[c]*Ntot; sumgz+=gz[c]; }
            auto lnlW=[&](const std::vector<double>& wv)->double{   // lnL re-evaluated at perturbed weights (partials unchanged)
                std::vector<double> save=catProp_v; catProp_v=wv; setVal(brlen[c0]);
                kj_derv_fused<<<GB,TB>>>(ns,nptn,ncat,pl0,d_pretmp,0.0,d_baseinvar,d_patlh,d_pdf,d_pddf,nullptr,nullptr); cudaDeviceSynchronize();
                double l,a2,b2; reduceDerv(l,a2,b2); catProp_v=save; return l; };
            auto softmax=[&](const std::vector<double>& z){ double mx=z[0]; for(int c=1;c<ncat;c++) if(z[c]>mx)mx=z[c];
                std::vector<double> o(ncat); double s=0; for(int c=0;c<ncat;c++){o[c]=exp(z[c]-mx); s+=o[c];} for(int c=0;c<ncat;c++)o[c]/=s; return o; };
            double eps=1e-4, maxrel=0;
            for(int d=0; d<ncat; d++){
                std::vector<double> zp(ncat),zm(ncat); for(int c=0;c<ncat;c++){ zp[c]=log(w[c]); zm[c]=zp[c]; }
                zp[d]+=eps; zm[d]-=eps;
                double lp=lnlW(softmax(zp)), lm=lnlW(softmax(zm));
                double fd=(lp-lm)/(2.0*eps); double rel=fabs(gz[d]-fd)/(fabs(fd)+1e-30);
                if(rel>maxrel) maxrel=rel;
                fprintf(stderr,"[RGRADCHECK] c=%d WN=%.6e w=%.6f gz=%.6e FD=%.6e rel=%.3e\n",d,WN[d],w[d],gz[d],fd,rel);
            }
            double relWN=fabs(sumWN-Ntot)/Ntot;
            fprintf(stderr,"[RGRADCHECK] ncat=%d lnL0=%.6f sumWN=%.6f N=%.0f relWN=%.3e sumGz=%.3e maxrel=%.3e -> %s\n",
                ncat,lnL0,sumWN,Ntot,relWN,sumgz,maxrel,(maxrel<1e-4 && relWN<1e-9)?"RGRADCHECK PASS":"RGRADCHECK FAIL");
            fflush(stderr);
        }
        // freeRate==1 engages the +R joint LM (falls through to the gradient infra + LM loop below). freeRate==2 is
        // the gradient-check-only diagnostic (an ineligible regime the gate let through only for the FD check), so it
        // declines to CPU after the check.
        if (freeRate != 1) return (double)NAN;
    }

    long nGradSweeps=0,nLnLEval=0;
    // Base-sweep skip: record the (brlen,alpha,pinv) the device echild/partial were last built for (by evalLnL) so
    // computeGradient can skip the redundant rebuild+postorder when its base already matches. The recorded values are
    // copies of the same candidate vectors (no recompute), so exact == is reliable; any mismatch falls back to rebuild.
    std::vector<double> devB; double devA=1e300, devP=1e300; bool devValid=false;
    auto evalLnL=[&](const std::vector<double>& cand_b,double cand_a,double cand_pinv,const double* cand_q)->double{
        if(nFreeQ>0 && cand_q) qApply(cand_q);   // re-decompose+reupload the trial Q -> rebuildEchild() below uses the new evalP/UP
        if(ncat>1 && !freeRate) applyAlpha(cand_a); applyPinv(cand_pinv); brlen=cand_b;   // +R seeds rates directly -> never re-derive from alpha (would clobber meanR)
        rebuildEchild();   // chunk-independent (echild/expfac carry no nptn) — build once per eval, reused across all chunks
        double Lacc=0,Lk=0;   // Kahan accumulator of lnL over the pattern chunks (rel<=1e-12 vs one-shot)
        for(int t=0;t<nTile;t++){
            setChunk(t); postorderFill();
            int nch; const double* ec[3]; const double* p[3]; const unsigned char* tp[3]; childArgs(root,c0,nch,ec,p,tp);
            k1_node<<<GB,TB>>>(ns,Pn,ncat,0,d_pretmp,d_patlh,nch,ec[0],p[0],tp[0],ec[1],p[1],tp[1],ec[2],p[2],tp[2]); cudaDeviceSynchronize();
            const double* pl0=edgeNodePtr(c0); cudaDeviceSynchronize();   // fused — no theta materialisation
            // evalLnL's base-edge derv is derv-only (rnum=wnum=nullptr).
            double l,d,dd;
            setVal(brlen[c0]); kj_derv_fused<<<GB,TB>>>(ns,Pn,ncat,pl0,d_pretmp,cand_pinv,d_baseinvar,d_patlh,d_pdf,d_pddf,nullptr,nullptr); cudaDeviceSynchronize();
            reduceDerv(l,d,dd);
            double y=l-Lk, s=Lacc+y; Lk=(s-Lacc)-y; Lacc=s;   // Kahan add this chunk's lnL contribution
        }
        devB=cand_b; devA=cand_a; devP=cand_pinv; devValid=true;   // echild matches this base (full postorder present on device only when nTile==1)
        return Lacc; };

    std::vector<double> g_df(nedge,0.0),g_ddf(nedge,0.0),gradR(ncat,0.0);
    // +R: weight-grad outputs filled by computeGradient when freeRate==1. WNc[c]=sum_p Lc(p)/L_p*freq (edge-invariant,
    // taken at the base edge), gzR[c]=WNc[c]-w_c*N (softmax weight gradient). N computed once.
    std::vector<double> WNc(freeRate==1?ncat:0,0.0), gzR(freeRate==1?ncat:0,0.0);
    double rN=0.0; if(freeRate==1){ for(int p=0;p<nptn;p++) rN+=ptn_freq[p]; }
    auto computeGradient=[&](double& lnLout,double& galphaOut){
        applyPinv(curPinv);   // align catRate=meanR/(1-curPinv) and catProp_v to the base pinv before the sweep
        // Base-sweep skip (tiling-aware): echild/expfac are chunk-independent, so skip the rebuild when the device
        // echild already matches this base point (built by the immediately-preceding accepted evalLnL). The postorder
        // partials are present on device for all patterns only when nTile==1; with nTile>1 they hold just the last
        // chunk, so postorderFill must rerun per chunk below. The skip tracks (brlen,alpha,pinv) only — a Q-FD step
        // moves the eigensystem without moving those, so it is disabled when nFreeQ>0.
        bool devMatch = (nFreeQ==0) && devValid && devA==curAlpha && devP==curPinv && (int)devB.size()==nnodes;
        if(devMatch) for(int z=0;z<nnodes;z++) if(devB[z]!=brlen[z]){ devMatch=false; break; }
        if(!devMatch) rebuildEchild();
        bool postValid = devMatch && (nTile==1);
        nGradSweeps++;
        // Cross-chunk Kahan accumulators for the per-pattern sums (df_e, ddf_e, lnL, and the raw rate-grad numerator
        // per category). Deterministic chunk order 0..nTile-1 => reproducible; rel<=1e-12 vs the one-shot sum.
        std::vector<double> accDf(nedge,0.0),accDfK(nedge,0.0),accDdf(nedge,0.0),accDdfK(nedge,0.0);
        std::vector<double> accR(ncat,0.0),accRk(ncat,0.0);
        std::vector<double> accW(freeRate==1?ncat:0,0.0),accWk(freeRate==1?ncat:0,0.0);   // +R weight-grad numerator (Kahan across chunks)
        double Lacc=0,Lk=0;
        for(int t=0;t<nTile;t++){
            setChunk(t);
            if(!postValid) postorderFill();
            cudaMemset(d_rnum,0,(size_t)ncat*Pn*sizeof(double));
            std::vector<int> freeSlots; for(int s=nPool-1;s>=0;s--) freeSlots.push_back(s);
            auto acq=[&](){int s=freeSlots.back();freeSlots.pop_back();return s;}; auto rls=[&](int s){freeSlots.push_back(s);};
            std::vector<double> dfC(nnodes,0.0),ddfC(nnodes,0.0); bool gotL=false; double lnLfirst=0;
            std::function<void(int,int)> proc=[&](int u,int su){
                for(int v:child[u]){
                    int sv=acq(); double* pre=d_prepool+(size_t)sv*slotSz;
                    if(u==root){ int nch; const double* ec[3]; const double* p[3]; const unsigned char* tp[3]; childArgs(root,v,nch,ec,p,tp);
                        k1_node<<<GB,TB>>>(ns,Pn,ncat,0,pre,d_patlh,nch,ec[0],p[0],tp[0],ec[1],p[1],tp[1],ec[2],p[2],tp[2]); }
                    else { const double* ec[2]={0,0}; const double* sp[2]={0,0}; const unsigned char* st[2]={0,0}; int nsb=0;
                        for(int w:child[u]){ if(w==v||nsb>=2) continue; sibArg(w,ec[nsb],sp[nsb],st[nsb]); nsb++; }
                        kj_pre<<<GB,TB>>>(ns,Pn,ncat,pre,d_prepool+(size_t)su*slotSz,d_expfac+(size_t)u*ncat*ns,nsb,ec[0],sp[0],st[0],ec[1],sp[1],st[1]); }
                    cudaDeviceSynchronize();
                    const double* plv=edgeNodePtr(v);   // fused theta+derv+ratenum, no d_theta round-trip
                    double bv=brlen[v]; std::vector<double> rs(ncat); for(int c=0;c<ncat;c++) rs[c]=bv/(catRate[c]*catProp_v[c]);
                    double l,d,dd;
                    cudaMemcpyToSymbol(g_rscale,rs.data(),sizeof(double)*ncat); setVal(bv); cudaDeviceSynchronize();
                    kj_derv_fused<<<GB,TB>>>(ns,Pn,ncat,plv,pre,curPinv,d_baseinvar,d_patlh,d_pdf,d_pddf,d_rnum,nullptr); cudaDeviceSynchronize();
                    ts_reopt_mcs += 1;   // proof-counter for the per-edge g_rscale upload (setVal counts the 3 g_val* uploads)
                    reduceDerv(l,d,dd);
                    dfC[v]=d; ddfC[v]=dd;
                    if(!gotL){ lnLfirst=l;
                        // 1/L_p on-device from the base-edge patlh.
                        kj_invl<<<GB,TB>>>(Pn,d_patlh,d_invLbase); gotL=true;
                        // +R: per-category likelihood Lc(p) (weight-grad numerator) is edge-invariant, so it is captured
                        // once at this first edge of the chunk (re-derv with the wnum output; setVal/plv/pre/g_rscale
                        // still set for this edge). kj_invl ran first so d_invLbase holds the correct base 1/L_p before
                        // this derv overwrites d_patlh/d_pdf/d_pddf (already reduced for edge v above).
                        if(freeRate==1){
                            cudaMemset(d_wnum,0,(size_t)ncat*Pn*sizeof(double));
                            // +R: per-category likelihood Lc(p) re-derv with the wnum output (g_val0 still set for this edge).
                            kj_derv_fused<<<GB,TB>>>(ns,Pn,ncat,plv,pre,curPinv,d_baseinvar,d_patlh,d_pdf,d_pddf,nullptr,d_wnum); cudaDeviceSynchronize();
                            kj_reduce_gradnum<<<GB,TB,(size_t)TB*sizeof(double)>>>(Pn,ncat,d_wnum,d_invLbase,d_ptnfreq,GB,d_redW);
                            cudaMemcpy(h_redW.data(),d_redW,(size_t)ncat*GB*sizeof(double),cudaMemcpyDeviceToHost);
                            for(int c=0;c<ncat;c++){ long double a=0; for(int b=0;b<GB;b++) a+=(long double)h_redW[(size_t)c*GB+b];
                                double term=(double)a; double y=term-accWk[c], s=accW[c]+y; accWk[c]=(s-accW[c])-y; accW[c]=s; } } }
                    if(leaf[v]<0) proc(v,sv); rls(sv);
                } };
            proc(root,-1); cudaDeviceSynchronize();
            // accumulate this chunk's per-edge df/ddf and base-edge lnL (Kahan):
            for(int e=0;e<nedge;e++){
                double td=dfC[edgeV[e]];  { double y=td-accDfK[e],  s=accDf[e] +y; accDfK[e] =(s-accDf[e]) -y; accDf[e] =s; }
                double t2=ddfC[edgeV[e]]; { double y=t2-accDdfK[e], s=accDdf[e]+y; accDdfK[e]=(s-accDdf[e])-y; accDdf[e]=s; } }
            { double y=lnLfirst-Lk, s=Lacc+y; Lk=(s-Lacc)-y; Lacc=s; }
            // Per-category block reduction of ptn_freq*rnum[c]*invL (on-device); accumulate the raw numerator across
            // chunks (the catProp_v[c] factor is applied once after the chunk loop).
            kj_reduce_gradnum<<<GB,TB,(size_t)TB*sizeof(double)>>>(Pn,ncat,d_rnum,d_invLbase,d_ptnfreq,GB,d_redR);
            cudaMemcpy(h_redR.data(),d_redR,(size_t)ncat*GB*sizeof(double),cudaMemcpyDeviceToHost);
            for(int c=0;c<ncat;c++){ long double a=0; for(int b=0;b<GB;b++) a+=(long double)h_redR[(size_t)c*GB+b];
                double term=(double)a; double y=term-accRk[c], s=accR[c]+y; accRk[c]=(s-accR[c])-y; accR[c]=s; }
        }
        for(int e=0;e<nedge;e++){ g_df[e]=accDf[e]; g_ddf[e]=accDdf[e]; }
        for(int c=0;c<ncat;c++) gradR[c]=catProp_v[c]*accR[c];
        // Softmax weight gradient gz_c = WN_c - w_c*(sum_p freq*S_p/L_p), w_c=bprop[c]. For pure +R the normalizer
        // sum freq*S/L = N = rN. For +I, L_p=(1-p)S_p+p*I_p, and sum_k WN_k = sum freq*(1-p)S/L equals that normalizer
        // ((1-p) cancels into WN), so use sum_k WN_k under +I (no extra reduction). At pinv=0 bprop==catProp_v and
        // sum WN==rN, but rN is kept to avoid a ~1e-12 sum-order drift.
        if(freeRate==1){ double sumWN=0; for(int c=0;c<ncat;c++) sumWN+=accW[c];
            double wnorm = optPinv ? sumWN : rN;
            for(int c=0;c<ncat;c++){ WNc[c]=accW[c]; gzR[c]=WNc[c]-bprop[c]*wnorm; } }
        double ga=0;
        // alpha gradient: ga = sum_c (d catRate[c]/dalpha)*gradR[c]; catRate[c]=meanR[c]/f, so the perturbed mean-1
        // rate rp[c] must be scaled by 1/f too (else mixing scaled/unscaled rates gives the wrong alpha grad on +I).
        if(ncat>1 && !freeRate){ double f = optPinv ? (1.0-curPinv) : 1.0; double rp[64]; jolt_discreteGammaMean(curAlpha+1e-5,ncat,rp);
            for(int c=0;c<ncat;c++) ga+=((rp[c]/f-catRate[c])/1e-5)*gradR[c]; }   // no alpha for +R (rates are free params, not gamma-derived)
        lnLout=Lacc; galphaOut=ga; };

    // ---- single joint LM diagonal-Newton optimise from the provided (warm) start ----
    std::vector<double> cand(nnodes,0.0), base;
    std::vector<double> startB(node_parentLen, node_parentLen+nnodes);   // distinct from brlen (evalLnL overwrites brlen)
    curAlpha=alpha0;
    // Free-Q: qcur = running free-Q vector (seeded from q0); set the device to base Q before the first eval so
    // computeGradient's rebuildEchild (skip disabled for free-Q) reads the correct eigensystem.
    const double MINQ=1e-4, MAXQ=100.0;   // == MIN_RATE / MAX_RATE (modelmarkov.h)
    std::vector<double> qcur(nFreeQ>0?nFreeQ:0), qPrev(nFreeQ>0?nFreeQ:0,0.0), gqPrev(nFreeQ>0?nFreeQ:0,0.0);
    if(nFreeQ>0){ for(int k=0;k<nFreeQ;k++) qcur[k]=q0[k]; qApply(qcur.data()); }
    // ===== +R FreeRate joint-LM state (empty/untouched unless freeRate==1) =====
    // y_c=log(r_c) (log-rate space, so the LM moves rates multiplicatively); z_c = softmax weight-logit. gaugeFix pins
    // sum w.r=1 by rescaling rates and folding the scale into branch lengths (lnL-invariant), fixing the FreeRate
    // rate<->branch-scale degeneracy.
    std::vector<double> zR(freeRate==1?ncat:0,0.0), ryPrev(freeRate==1?ncat:0,0.0), rgyPrev(freeRate==1?ncat:0,0.0),
                        rzPrev(freeRate==1?ncat:0,0.0), rgzPrev(freeRate==1?ncat:0,0.0);
    std::vector<double> baseR_save(freeRate==1?ncat:0,0.0), baseW_save(freeRate==1?ncat:0,0.0);
    auto gaugeFix=[&](){ double m=0; for(int c=0;c<ncat;c++) m+=catProp_v[c]*catRate[c];   // m = overall mean rate (sum catProp_v.catRate; the +I invariant class adds 0) — pin to 1
        if(m>0){ for(int c=0;c<ncat;c++) catRate[c]/=m; for(int v=0;v<nnodes;v++){ brlen[v]*=m; if(brlen[v]>20.0) brlen[v]=20.0; } }
        double f = optPinv ? (1.0-curPinv) : 1.0;   // meanR is the pinv-free rate rho=catRate*(1-pinv); applyPinv(curPinv) then reproduces catRate. f==1 for pure +R.
        for(int c=0;c<ncat;c++) meanR[c]=catRate[c]*f; };
    auto softmaxApply=[&](const std::vector<double>& z,std::vector<double>& w){
        double mx=z[0]; for(int c=1;c<ncat;c++) if(z[c]>mx) mx=z[c];
        double s=0; for(int c=0;c<ncat;c++){ w[c]=exp(z[c]-mx); s+=w[c]; } for(int c=0;c<ncat;c++) w[c]/=s;
        double tot=0; for(int c=0;c<ncat;c++){ if(w[c]<1e-4) w[c]=1e-4; tot+=w[c]; } for(int c=0;c<ncat;c++) w[c]/=tot; };
    // Warm-seed the +R rates from the CPU's separated estimates. Starting from flat rates ~1 is a symmetric
    // stationary point: for ncat>=2 with equal rates the rate-separation gradient can vanish, trapping +R at a
    // collapsed (sub-optimal) optimum. Instead start from the separated rate estimates meanR=rho_c (seeded above
    // from getRate), gauged to the mean-1 base sum bprop.meanR=1 with the scale folded into startB. Operates on the
    // pinv-free basis meanR/bprop, unifying the pure-+R and +I+R paths.
    if(freeRate==1){ for(int c=0;c<ncat;c++) zR[c]=log(bprop[c]);   // weight-logits from the pinv-free weights w_c (== log(catProp_v) at pinv=0; softmax shift-invariant)
        double m=0; for(int c=0;c<ncat;c++) m+=bprop[c]*meanR[c];   // mean-1 base constraint sum w.rho
        if(m>0){ for(int c=0;c<ncat;c++) meanR[c]/=m; for(int v=0;v<nnodes;v++){ startB[v]*=m; if(startB[v]>20.0) startB[v]=20.0; } } }
    double lnL=evalLnL(startB,curAlpha,curPinv,nullptr); nLnLEval++;
    double mu=1.0, tol=1e-7; int it=0,nRej=0; bool conv=false;
    double aPrev=0,gaPrev=0; bool haveSec=false;
    double pPrev=0,gpPrev=0;   // pinv secant curvature (mirrors the alpha secant)
    for(it=1; it<=maxiter; it++){
        base=brlen; double baseA=curAlpha, baseP=curPinv; if(ncat>1 && !freeRate) applyAlpha(baseA);
        double lg,ga; computeGradient(lg,ga);
        // pinv gradient by forward finite difference (robust to the rate<->prop<->pinv coupling that the 1/(1-pinv)
        // rate rescaling introduces). One extra postorder lnL eval. lg = lnL at the base point (from above).
        double gradPinv=0.0;
        if(optPinv){ double ep=1e-4, pp=baseP+ep, dep;
            if(pp>pinvMax){ pp=baseP-ep; if(pp<pinvMin)pp=pinvMin; }   // backward FD at the upper boundary (avoid a stuck zero gradient)
            dep=pp-baseP;
            if(fabs(dep)>1e-9){ double lpe=evalLnL(base,baseA,pp,nullptr); nLnLEval++; gradPinv=(lpe-lg)/dep; } }
        // Free-Q gradient by forward finite difference (the CPU optimises Q by FD too). Each free exchangeability:
        // perturb in rate-class space (qApply re-decomposes Q + re-uploads), one extra lnL eval. lg = base lnL. The
        // device is left at base Q afterwards (qApply(qcur)).
        std::vector<double> gradQ(nFreeQ>0?nFreeQ:0,0.0), ddQ(nFreeQ>0?nFreeQ:0,-1e6);
        if(nFreeQ>0){
            std::vector<double> qp(qcur);
            for(int k=0;k<nFreeQ;k++){
                double save=qp[k], hq=1e-4*fabs(save); if(hq==0.0)hq=1e-4; double qpk=save+hq;
                if(qpk>MAXQ){ qpk=save-hq; if(qpk<MINQ)qpk=MINQ; }   // backward FD at the upper bound
                double dq=qpk-save;
                if(fabs(dq)>1e-12){ qp[k]=qpk; double lq=evalLnL(base,baseA,baseP,qp.data()); nLnLEval++; gradQ[k]=(lq-lg)/dq; qp[k]=save; }
            }
            qApply(qcur.data());   // restore the device eigensystem to base Q (the last FD eval left it at a perturbation)
        }
        // +R: log-rate / softmax-logit gradients + per-component secant curvature (mirrors the alpha ddA path).
        // g_y=r*gradR (chain rule for y=log r); g_z=gzR (softmax weight grad). haveSec is the pre-update flag here
        // (set true at the bottom of the iteration), so iteration 1 uses the -1e6 floor like alpha. baseR/W saved for
        // the reject restore (the trial staging below overwrites meanR/bprop in place).
        std::vector<double> baseY(freeRate==1?ncat:0),baseZ(freeRate==1?ncat:0),g_y(freeRate==1?ncat:0),g_z(freeRate==1?ncat:0);
        std::vector<double> ddY(freeRate==1?ncat:0,-1e6),ddZ(freeRate==1?ncat:0,-1e6);
        if(freeRate==1){
            baseR_save=catRate; baseW_save=catProp_v;
            // The log-rate arm lives in the pinv-free basis y=log(meanR=rho) (the trial staging writes meanR=exp(y)).
            // g_y = d lnL/dy = meanR*dL/dmeanR = meanR*(gradR/(1-p)) = catRate*gradR (the (1-p) cancels). For pure +R meanR==catRate.
            for(int c=0;c<ncat;c++){ baseY[c]=log(meanR[c]); baseZ[c]=zR[c]; g_y[c]=catRate[c]*gradR[c]; g_z[c]=gzR[c]; }
            if(haveSec) for(int c=0;c<ncat;c++){
                if(fabs(baseY[c]-ryPrev[c])>1e-9) ddY[c]=(g_y[c]-rgyPrev[c])/(baseY[c]-ryPrev[c]);
                if(fabs(baseZ[c]-rzPrev[c])>1e-9) ddZ[c]=(g_z[c]-rgzPrev[c])/(baseZ[c]-rzPrev[c]); }
            for(int c=0;c<ncat;c++){ ryPrev[c]=baseY[c]; rgyPrev[c]=g_y[c]; rzPrev[c]=baseZ[c]; rgzPrev[c]=g_z[c]; }
        }
        double ddA=(haveSec && fabs(baseA-aPrev)>1e-9)?(ga-gaPrev)/(baseA-aPrev):-1e6;
        double ddP=(haveSec && fabs(baseP-pPrev)>1e-12)?(gradPinv-gpPrev)/(baseP-pPrev):-1e6;
        for(int k=0;k<nFreeQ;k++) ddQ[k]=(haveSec && fabs(qcur[k]-qPrev[k])>1e-12)?(gradQ[k]-gqPrev[k])/(qcur[k]-qPrev[k]):-1e6;
        aPrev=baseA; gaPrev=ga; pPrev=baseP; gpPrev=gradPinv;
        for(int k=0;k<nFreeQ;k++){ qPrev[k]=qcur[k]; gqPrev[k]=gradQ[k]; }
        haveSec=true;
        bool acc=false;
        {
            for(int bt=0; bt<14; bt++){
                cand=base; for(int e=0;e<nedge;e++){ int v=edgeV[e]; double dn=fabs(g_ddf[e])+mu; double nb=base[v]+g_df[e]/dn; if(nb<1e-6)nb=1e-6; if(nb>20.0)nb=20.0; cand[v]=nb; }
                double ca=baseA; if(optAlpha && ncat>1){ double da=ga/(fabs(ddA)+mu); ca=baseA+da; if(ca<0.02)ca=0.02; if(ca>50.0)ca=50.0; }
                double cp=baseP; if(optPinv){ double dp=gradPinv/(fabs(ddP)+mu); cp=baseP+dp; if(cp<pinvMin)cp=pinvMin; if(cp>pinvMax)cp=pinvMax; }
                std::vector<double> cq(nFreeQ>0?nFreeQ:0);
                for(int k=0;k<nFreeQ;k++){ double dn=fabs(ddQ[k])+mu; double nq=qcur[k]+gradQ[k]/dn; if(nq<MINQ)nq=MINQ; if(nq>MAXQ)nq=MAXQ; cq[k]=nq; }
                // +R: log-rate / softmax-weight arms at the same mu (mirror the alpha/pinv diagonal arms). Stage the
                // trial (cr,cw) into the pinv-free basis meanR/bprop so evalLnL's applyPinv(cp) evaluates
                // catRate=cr/(1-cp), catProp_v=(1-cp)cw — the +I-correct rates/props. cr lives in meanR=rho space (baseY=log meanR).
                std::vector<double> cr,cw,cz;
                if(freeRate==1){ cr.resize(ncat); cw.resize(ncat); cz.resize(ncat);
                    for(int c=0;c<ncat;c++){ double ny=baseY[c]+g_y[c]/(fabs(ddY[c])+mu); double r=exp(ny);
                        if(r<1e-4)r=1e-4; if(r>1000.0)r=1000.0; cr[c]=r; cz[c]=baseZ[c]+g_z[c]/(fabs(ddZ[c])+mu); }
                    softmaxApply(cz,cw);
                    for(int c=0;c<ncat;c++){ meanR[c]=cr[c]; bprop[c]=cw[c]; } }
                double ln=evalLnL(cand,ca,cp, nFreeQ>0?cq.data():nullptr); nLnLEval++;
                if(ln>lnL+1e-9){ double dl=ln-lnL; brlen=cand; curAlpha=ca; curPinv=cp; if(nFreeQ>0) qcur=cq;
                    // Accept the staged pinv-free meanR/bprop, then derive catRate/catProp_v via applyPinv(curPinv)
                    // (== cr/cw at pinv=0), then gauge.
                    if(freeRate==1){ for(int c=0;c<ncat;c++){ meanR[c]=cr[c]; bprop[c]=cw[c]; } zR=cz; applyPinv(curPinv); gaugeFix(); }
                    lnL=ln; mu=fmax(mu*0.5,1e-9); acc=true; if(dl<tol)conv=true; break; }
                else { mu*=4.0; nRej++; } }
        }
        if(!acc){ brlen=base; curAlpha=baseA; curPinv=baseP;
            // Restore catRate/catProp_v and the pinv-free basis. baseR_save/baseW_save are catRate/catProp_v at base;
            // meanR=rho=catRate*(1-baseP), bprop=w=catProp_v/(1-baseP). For pure +R meanR=baseR_save.
            if(freeRate==1){ catRate=baseR_save; catProp_v=baseW_save; double f=optPinv?(1.0-baseP):1.0;
                for(int c=0;c<ncat;c++){ meanR[c]=baseR_save[c]*f; bprop[c]=baseW_save[c]/f; } }
            if(nFreeQ>0) qApply(qcur.data()); break; }
        if(conv) break; }

    if (cudaGetLastError()!=cudaSuccess) return (double)NAN;   // any launch/sync error -> caller falls back to CPU
    for(int v=0;v<nnodes;v++) out_brlen[v]=brlen[v];
    if(out_alpha) *out_alpha=curAlpha;
    if(out_pinv)  *out_pinv = optPinv ? curPinv : pinv0;
    if(nFreeQ>0 && out_q) for(int k=0;k<nFreeQ;k++) out_q[k]=qcur[k];
    if(freeRate==1 && out_rates && out_props) for(int c=0;c<ncat;c++){ out_rates[c]=catRate[c]; out_props[c]=catProp_v[c]; }   // optimised +R rates/weights
    if(out_iters) *out_iters=it;
    // --jolt-diag: nRej = rejected backtracks (each a discarded full postorder), nLnLEval = total evalLnL postorders.
    if(g_jdiag) printf("JOLT-DIAG-CU echild=%.6f n=%ld iters=%d nRej=%d nLnLEval=%ld nptn=%d\n", g_jd_echild_sec-_jd_ech0, g_jd_echild_n-_jd_echn0, it, nRej, nLnLEval, nptn);
    (void)ts_reopt_mcs; (void)ts_reopt_vp;
    return lnL;
}
