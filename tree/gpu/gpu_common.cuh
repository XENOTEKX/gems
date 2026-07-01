// gpu_common.cuh — shared declarations for the in-tree GPU module.
//
// Every gpu_*.cu translation unit includes this header. The device model
// constants are declared extern here and defined once in gpu_kernels_lnl.cu;
// cross-translation-unit access to them relies on CUDA_SEPARABLE_COMPILATION
// (see CMakeLists.txt). Built only when IQTREE_GPU=ON.
#pragma once

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>

#define NS_MAX 20

// ---- device model constants (set per launcher call; one tree, one model) ----
// Defined once in gpu_kernels_lnl.cu; declared extern here for the other TUs.
extern __constant__ double g_Uinv[NS_MAX*NS_MAX];
extern __constant__ double g_U[NS_MAX*NS_MAX];   // eigenvectors, needed by the preorder kernel kj_pre
extern __constant__ double g_UinvRowSum[NS_MAX];
extern __constant__ double g_freq[NS_MAX];
extern __constant__ double g_catw[64];
// single-edge derivative coefficients (per cat,state at the central branch length t):
extern __constant__ double g_val0[64*NS_MAX];   // exp(eval[x]*rate_c*t) * prop_c
extern __constant__ double g_val1[64*NS_MAX];   // (rate_c*eval[x]) * val0
extern __constant__ double g_val2[64*NS_MAX];   // (rate_c*eval[x]) * val1
extern __constant__ double g_rscale[64];        // per-cat edge scale b_e/(r_k*w_k) for the +R/alpha rate-grad numerator

// per-child probability-space contribution: prod[x] *= sum_i echild[c][x][i] * L_child[c][i]
static __device__ __forceinline__ void accum_child(double* prod, int ns, int c, int ptn, int nptn,
        const double* __restrict__ ec, const double* __restrict__ p, const unsigned char* __restrict__ t) {
    const double* ecc = ec + (size_t)c*ns*ns;
    if (p) {                              // internal child: read its eigen-space partial (coalesced over ptn)
        const double* pc = p + (size_t)(c*ns)*nptn + ptn;
        for (int x=0;x<ns;x++){ double v=0.0;
            for (int i=0;i<ns;i++) v += ecc[x*ns+i]*pc[(size_t)i*nptn];
            prod[x]*=v; }
    } else {                              // leaf child: L[i] = column s of U^-1 (row-sum if ambiguous)
        int s = t[ptn];
        for (int x=0;x<ns;x++){ double v=0.0;
            for (int i=0;i<ns;i++){ double Li = (s<ns)? g_Uinv[i*ns+s] : g_UinvRowSum[i]; v += ecc[x*ns+i]*Li; }
            prod[x]*=v; }
    }
}

// Persistent device-buffer pool entry. Allocated once and grown on demand
// (never freed; released at process exit). Contents are overwritten every call.
struct DevBuf { void* p = nullptr; size_t cap = 0; };

// Ensure a buffer holds at least `need` bytes; returns false on cudaMalloc failure.
// Defined in gpu_kernels_lnl.cu.
bool devbuf_ensure(DevBuf& b, size_t need);

// Pattern-tile count for a per-pattern working-set of `perPatternDoubles` doubles,
// chosen from free VRAM. Defined in gpu_kernels_mixture.cu.
int mix_pick_ntile(int nptn, size_t perPatternDoubles);

// CUDA error check for the double-returning launchers (return NaN on failure).
#define GCK(x) do{ cudaError_t _e=(x); if(_e!=cudaSuccess){ \
    fprintf(stderr,"[GPU-XCHECK] %s failed at %s:%d: %s\n",#x,__FILE__,__LINE__,cudaGetErrorString(_e)); \
    return (double)NAN; } }while(0)

// Ensure a device buffer, returning NaN from the enclosing launcher on failure.
#define DEVB(b, bytes) do{ if(!devbuf_ensure((b),(size_t)(bytes))){ \
    fprintf(stderr,"[GPU] devbuf_ensure failed (%zu bytes) at %s:%d\n",(size_t)(bytes),__FILE__,__LINE__); \
    return (double)NAN; } }while(0)
