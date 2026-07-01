// gpu_parsimony.cu — GPU parsimony (bit-packed Fitch) launchers.
//
// Scores candidate insertion branches for one taxon in a single launch,
// bit-identical to the CPU parsimony score. A stateless per-call launcher
// (gpu_parsimony_score_branches) and a device-resident path
// (set_leaves / set_leaf_row / build_and_score, recompute from resident leaves).
// Built only when IQTREE_GPU=ON.
#include <cuda_runtime.h>
#include "gpu_iqtree.h"
#include "gpu_common.cuh"
#include <mutex>

// GPU parsimony — batched per-taxon-insertion branch scoring. The stepwise-addition
// parsimony builder is O(nseq^2 * patterns) and CPU-only, scaling linearly with the
// pattern count. This kernel scores ALL candidate insertion branches for one taxon
// in a single launch (one block per branch), reproducing the CPU Fitch
// (computeParsimonyBranchFast + the internal combine in computePartialParsimonyFast)
// exactly. partial_pars is bit-packed: 32 sites / UINT, nstates UINTs per site-block.
// Per branch b (endpoints L,R):
//   z[i] = L[i]&R[i];  w1 = ~OR_i(z[i]);  local += popc(w1);  z[i] |= w1 & (L[i]|R[i]);     (internal combine)
//   w2 = ~OR_i(tip[i]&z[i]);  branch += popc(w2);                                            (branch score)
//   out[b] = tipScore + scoreL[b] + scoreR[b] + sum_blk(local+branch)
// Pure integer, so bit-identical to the CPU (integer add is associative; padding
// bits are pre-set in the leaf partials as the CPU expects). The host argmin uses
// first-index tie-break to match `score < best_pars`.
static DevBuf gb_parsTip, gb_parsEndL, gb_parsEndR, gb_parsScoreL, gb_parsScoreR, gb_parsOut;
#define PARS_MAXST 32
#define PARS_TPB 256
__global__ void k_pars_score_branches(
        const unsigned int* __restrict__ tip,     // nstates*nsblk (broadcast across branches)
        const unsigned int* __restrict__ endL,    // B * nstates*nsblk
        const unsigned int* __restrict__ endR,    // B * nstates*nsblk
        const unsigned int* __restrict__ scoreL,  // B  (accumulated subtree score, endpoint L)
        const unsigned int* __restrict__ scoreR,  // B
        unsigned int tipScore,
        int nstates, int nsblk, int B,
        unsigned int* __restrict__ out)           // B
{
    int b = blockIdx.x;
    if (b >= B) return;
    const unsigned int* eL = endL + (size_t)b * nstates * nsblk;
    const unsigned int* eR = endR + (size_t)b * nstates * nsblk;
    unsigned int acc = 0u;
    for (int s = threadIdx.x; s < nsblk; s += blockDim.x) {
        const unsigned int* x = eL  + (size_t)s * nstates;
        const unsigned int* y = eR  + (size_t)s * nstates;
        const unsigned int* t = tip + (size_t)s * nstates;
        unsigned int z[PARS_MAXST];
        unsigned int w1 = 0u;
        for (int i = 0; i < nstates; i++) { z[i] = x[i] & y[i]; w1 |= z[i]; }
        w1 = ~w1;
        acc += __popc(w1);
        unsigned int w2 = 0u;
        for (int i = 0; i < nstates; i++) { z[i] |= w1 & (x[i] | y[i]); w2 |= t[i] & z[i]; }
        w2 = ~w2;
        acc += __popc(w2);
    }
    __shared__ unsigned int sh[PARS_TPB];
    sh[threadIdx.x] = acc;
    __syncthreads();
    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) sh[threadIdx.x] += sh[threadIdx.x + stride];
        __syncthreads();
    }
    if (threadIdx.x == 0)
        out[b] = sh[0] + tipScore + scoreL[b] + scoreR[b];
}

// Returns 0 on success (h_out[0..B-1] filled), -1 on any failure (caller falls back to CPU).
extern "C" int gpu_parsimony_score_branches(
        const unsigned int* h_tip, unsigned int tipScore,
        const unsigned int* h_endL, const unsigned int* h_endR,
        const unsigned int* h_scoreL, const unsigned int* h_scoreR,
        int nstates, int nsblk, int B, unsigned int* h_out)
{
    if (nstates < 1 || nstates > PARS_MAXST || nsblk < 1 || B < 1) return -1;
    static std::mutex pars_mtx; std::lock_guard<std::mutex> lk(pars_mtx);
    size_t setU   = (size_t)nstates * nsblk;
    size_t setB   = setU * sizeof(unsigned int);
    size_t packB  = (size_t)B * setB;
    size_t scB    = (size_t)B * sizeof(unsigned int);
    if (!devbuf_ensure(gb_parsTip,    setB))  return -1;
    if (!devbuf_ensure(gb_parsEndL,   packB)) return -1;
    if (!devbuf_ensure(gb_parsEndR,   packB)) return -1;
    if (!devbuf_ensure(gb_parsScoreL, scB))   return -1;
    if (!devbuf_ensure(gb_parsScoreR, scB))   return -1;
    if (!devbuf_ensure(gb_parsOut,    scB))   return -1;
    if (cudaMemcpy(gb_parsTip.p,    h_tip,    setB,  cudaMemcpyHostToDevice) != cudaSuccess) return -1;
    if (cudaMemcpy(gb_parsEndL.p,   h_endL,   packB, cudaMemcpyHostToDevice) != cudaSuccess) return -1;
    if (cudaMemcpy(gb_parsEndR.p,   h_endR,   packB, cudaMemcpyHostToDevice) != cudaSuccess) return -1;
    if (cudaMemcpy(gb_parsScoreL.p, h_scoreL, scB,   cudaMemcpyHostToDevice) != cudaSuccess) return -1;
    if (cudaMemcpy(gb_parsScoreR.p, h_scoreR, scB,   cudaMemcpyHostToDevice) != cudaSuccess) return -1;
    k_pars_score_branches<<<B, PARS_TPB>>>(
        (const unsigned int*)gb_parsTip.p, (const unsigned int*)gb_parsEndL.p, (const unsigned int*)gb_parsEndR.p,
        (const unsigned int*)gb_parsScoreL.p, (const unsigned int*)gb_parsScoreR.p,
        tipScore, nstates, nsblk, B, (unsigned int*)gb_parsOut.p);
    if (cudaGetLastError() != cudaSuccess) return -1;
    if (cudaMemcpy(h_out, gb_parsOut.p, scB, cudaMemcpyDeviceToHost) != cudaSuccess) return -1;
    if (cudaDeviceSynchronize() != cudaSuccess) return -1;
    return 0;
}

// 2D-grid parsimony variants (blockIdx.y = pattern-chunk) — saturate the GPU on a
// single launch. blockIdx.y indexes a chunk of CH sblk, so grid = cnt*nChunk blocks.
// The popcount is split across chunks and accumulated with integer atomicAdd
// (associative, so bit-identical to a sequential reduction). The child sub-scores
// (sA+sB) are added exactly once, by the blockIdx.y==0 block. arenaScore/out MUST
// be pre-zeroed by the host.
__global__ void k_pars_combine_to_arena_2d(
        const unsigned int* __restrict__ leaves, unsigned int* __restrict__ arena,
        unsigned int* __restrict__ arenaScore, const int* __restrict__ taskOut,
        const int* __restrict__ taskA, const int* __restrict__ taskB,
        int nstates, int nsblk, int T, int CH)
{
    int tk = blockIdx.x; if (tk >= T) return;
    int s0 = blockIdx.y * CH; if (s0 >= nsblk) return; int s1 = min(s0 + CH, nsblk);
    size_t setU = (size_t)nstates * nsblk;
    int a = taskA[tk], b = taskB[tk];
    const unsigned int* X = (a >= 0) ? arena + (size_t)a*setU : leaves + (size_t)(-(a+1))*setU;
    const unsigned int* Y = (b >= 0) ? arena + (size_t)b*setU : leaves + (size_t)(-(b+1))*setU;
    unsigned int* O = arena + (size_t)taskOut[tk] * setU;
    unsigned int acc = 0u;
    for (int s = s0 + threadIdx.x; s < s1; s += blockDim.x) {
        const unsigned int* x = X + (size_t)s*nstates; const unsigned int* y = Y + (size_t)s*nstates;
        unsigned int* o = O + (size_t)s*nstates;
        unsigned int z[PARS_MAXST]; unsigned int w1 = 0u;
        for (int i = 0; i < nstates; i++) { z[i] = x[i] & y[i]; w1 |= z[i]; } w1 = ~w1; acc += __popc(w1);
        for (int i = 0; i < nstates; i++) { z[i] |= w1 & (x[i] | y[i]); o[i] = z[i]; }
    }
    __shared__ unsigned int sh[PARS_TPB]; sh[threadIdx.x] = acc; __syncthreads();
    for (int st = blockDim.x >> 1; st > 0; st >>= 1) { if (threadIdx.x < st) sh[threadIdx.x] += sh[threadIdx.x+st]; __syncthreads(); }
    if (threadIdx.x == 0) {
        unsigned int add = sh[0];
        if (blockIdx.y == 0) {                                   // add the child sub-scores exactly once
            unsigned int sA = (a >= 0) ? arenaScore[a] : 0u;
            unsigned int sB = (b >= 0) ? arenaScore[b] : 0u;
            add += sA + sB;
        }
        atomicAdd(&arenaScore[taskOut[tk]], add);
    }
}
__global__ void k_pars_score_indexed_2d(
        const unsigned int* __restrict__ leaves, const unsigned int* __restrict__ arena,
        const unsigned int* __restrict__ arenaScore, const int* __restrict__ candL,
        const int* __restrict__ candR, const unsigned int* __restrict__ tipPart,
        int nstates, int nsblk, int nCand, int CH, unsigned int* __restrict__ out)
{
    int k = blockIdx.x; if (k >= nCand) return;
    int s0 = blockIdx.y * CH; if (s0 >= nsblk) return; int s1 = min(s0 + CH, nsblk);
    size_t setU = (size_t)nstates * nsblk;
    int a = candL[k], b = candR[k];
    const unsigned int* X = (a >= 0) ? arena + (size_t)a*setU : leaves + (size_t)(-(a+1))*setU;
    const unsigned int* Y = (b >= 0) ? arena + (size_t)b*setU : leaves + (size_t)(-(b+1))*setU;
    unsigned int acc = 0u;
    for (int s = s0 + threadIdx.x; s < s1; s += blockDim.x) {
        const unsigned int* x = X + (size_t)s*nstates; const unsigned int* y = Y + (size_t)s*nstates;
        const unsigned int* t = tipPart + (size_t)s*nstates;
        unsigned int z[PARS_MAXST]; unsigned int w1 = 0u;
        for (int i = 0; i < nstates; i++) { z[i] = x[i] & y[i]; w1 |= z[i]; } w1 = ~w1; acc += __popc(w1);
        unsigned int w2 = 0u; for (int i = 0; i < nstates; i++) { z[i] |= w1 & (x[i] | y[i]); w2 |= t[i] & z[i]; } w2 = ~w2; acc += __popc(w2);
    }
    __shared__ unsigned int sh[PARS_TPB]; sh[threadIdx.x] = acc; __syncthreads();
    for (int st = blockDim.x >> 1; st > 0; st >>= 1) { if (threadIdx.x < st) sh[threadIdx.x] += sh[threadIdx.x+st]; __syncthreads(); }
    if (threadIdx.x == 0) {
        unsigned int add = sh[0];
        if (blockIdx.y == 0) {                                   // tipScore = 0 (leaf); add child sub-scores once
            unsigned int sA = (a >= 0) ? arenaScore[a] : 0u;
            unsigned int sB = (b >= 0) ? arenaScore[b] : 0u;
            add += sA + sB;
        }
        atomicAdd(&out[k], add);
    }
}

// ---- device-resident parsimony state ----
static DevBuf gb_pbLeaves, gb_pbArena, gb_pbArenaScore;
static DevBuf gb_pbTaskOut, gb_pbTaskA, gb_pbTaskB, gb_pbCandL, gb_pbCandR, gb_pbScores;
static int g_pb_nLeaf = 0, g_pb_setU = 0, g_pb_nstates = 0, g_pb_nsblk = 0;
static const void* g_pb_alnId = nullptr;   // identity of the alignment whose leaves are resident
static std::mutex g_pb_mtx;   // shared across set_leaves / set_leaf_row / build_and_score (OpenMP tree generation)

// Allocate the resident leaf buffer: nLeaf rows of setU=nstates*nsblk UINTs (rows filled via set_leaf_row).
// Idempotent — safe to call from every OpenMP thread's computeParsimonyTree (dims must match the alignment).
// alnId = an opaque per-alignment token (the Alignment*): build_and_score declines if a different alignment has
// since clobbered the shared resident leaves (mismatch => CPU fallback).
extern "C" int gpu_parsimony_set_leaves(int nLeaf, int nstates, int nsblk, const void* alnId) {
    if (nLeaf < 1 || nstates < 1 || nstates > PARS_MAXST || nsblk < 1) return -1;
    std::lock_guard<std::mutex> lk(g_pb_mtx);
    size_t setU = (size_t)nstates * nsblk;
    if (!devbuf_ensure(gb_pbLeaves, (size_t)nLeaf * setU * sizeof(unsigned int))) return -1;
    g_pb_nLeaf = nLeaf; g_pb_setU = (int)setU; g_pb_nstates = nstates; g_pb_nsblk = nsblk; g_pb_alnId = alnId;
    return 0;
}

// Upload ONE bit-packed leaf row (setU UINTs) into resident slot lid (= taxon node->id). Idempotent across
// threads (same lid => byte-identical data, since leaf packs are tree-independent).
extern "C" int gpu_parsimony_set_leaf_row(const unsigned int* h_row, int lid) {
    std::lock_guard<std::mutex> lk(g_pb_mtx);
    if (g_pb_nLeaf < 1 || lid < 0 || lid >= g_pb_nLeaf) return -1;
    unsigned int* base = (unsigned int*)gb_pbLeaves.p;
    if (cudaMemcpy(base + (size_t)lid * g_pb_setU, h_row, (size_t)g_pb_setU * sizeof(unsigned int),
                   cudaMemcpyHostToDevice) != cudaSuccess) return -1;
    return 0;
}

// Recompute all directed partials from resident leaves via the level schedule, then score candidate branches.
extern "C" int gpu_parsimony_build_and_score(
        int maxSlots,
        const int* h_taskOut, const int* h_taskA, const int* h_taskB,
        const int* h_levelStart, int nLevel, int nTask,
        const int* h_candL, const int* h_candR, int tipLeaf,
        int nstates, int nsblk, int nCand, unsigned int* h_scores, const void* alnId) {
    std::lock_guard<std::mutex> lk(g_pb_mtx);   // take the lock before reading the shared g_pb_* globals
    if (g_pb_nLeaf < 1) return -1;                                   // leaves not uploaded
    if (alnId != g_pb_alnId) return -1;          // a different alignment clobbered residency => CPU fallback
    if (nstates != g_pb_nstates || nsblk != g_pb_nsblk) return -1;   // must match resident leaves
    if (maxSlots < 1 || nCand < 1 || nTask < 0 || nLevel < 0) return -1;
    if (tipLeaf < 0 || tipLeaf >= g_pb_nLeaf) return -1;
    size_t setU = (size_t)nstates * nsblk;
    if (!devbuf_ensure(gb_pbArena,      (size_t)maxSlots * setU * sizeof(unsigned int))) return -1;
    if (!devbuf_ensure(gb_pbArenaScore, (size_t)maxSlots * sizeof(unsigned int)))        return -1;
    if (nTask > 0) {
        if (!devbuf_ensure(gb_pbTaskOut, (size_t)nTask * sizeof(int))) return -1;
        if (!devbuf_ensure(gb_pbTaskA,   (size_t)nTask * sizeof(int))) return -1;
        if (!devbuf_ensure(gb_pbTaskB,   (size_t)nTask * sizeof(int))) return -1;
        if (cudaMemcpy(gb_pbTaskOut.p, h_taskOut, (size_t)nTask*sizeof(int), cudaMemcpyHostToDevice)!=cudaSuccess) return -1;
        if (cudaMemcpy(gb_pbTaskA.p,   h_taskA,   (size_t)nTask*sizeof(int), cudaMemcpyHostToDevice)!=cudaSuccess) return -1;
        if (cudaMemcpy(gb_pbTaskB.p,   h_taskB,   (size_t)nTask*sizeof(int), cudaMemcpyHostToDevice)!=cudaSuccess) return -1;
    }
    if (!devbuf_ensure(gb_pbCandL,  (size_t)nCand*sizeof(int)))          return -1;
    if (!devbuf_ensure(gb_pbCandR,  (size_t)nCand*sizeof(int)))          return -1;
    if (!devbuf_ensure(gb_pbScores, (size_t)nCand*sizeof(unsigned int))) return -1;
    if (cudaMemcpy(gb_pbCandL.p, h_candL, (size_t)nCand*sizeof(int), cudaMemcpyHostToDevice)!=cudaSuccess) return -1;
    if (cudaMemcpy(gb_pbCandR.p, h_candR, (size_t)nCand*sizeof(int), cudaMemcpyHostToDevice)!=cudaSuccess) return -1;
    const unsigned int* dLeaves = (const unsigned int*)gb_pbLeaves.p;
    unsigned int* dArena  = (unsigned int*)gb_pbArena.p;
    unsigned int* dScore  = (unsigned int*)gb_pbArenaScore.p;
    const int* dTaskOut = (const int*)gb_pbTaskOut.p;
    const int* dTaskA   = (const int*)gb_pbTaskA.p;
    const int* dTaskB   = (const int*)gb_pbTaskB.p;
    const unsigned int* dTip = dLeaves + (size_t)tipLeaf * setU;
    // 2D-grid (saturating, atomicAdd) — the production path.
    const int CH = 128;                                      // tuned; nChunk=ceil(nsblk/CH) saturates SMs
    const int nChunk = (nsblk + CH - 1) / CH;
    // atomicAdd accumulators MUST start at 0 (combine writes touched slots; score writes nCand)
    if (cudaMemset(dScore, 0, (size_t)maxSlots * sizeof(unsigned int)) != cudaSuccess) return -1;
    if (cudaMemset(gb_pbScores.p, 0, (size_t)nCand * sizeof(unsigned int)) != cudaSuccess) return -1;
    for (int l = 0; l < nLevel; l++) {                       // sequential launches => level L+1 reads level L's scores
        int base = h_levelStart[l], cnt = h_levelStart[l+1] - base;
        if (cnt <= 0) continue;
        k_pars_combine_to_arena_2d<<<dim3(cnt, nChunk), PARS_TPB>>>(dLeaves, dArena, dScore,
            dTaskOut + base, dTaskA + base, dTaskB + base, nstates, nsblk, cnt, CH);
        if (cudaGetLastError() != cudaSuccess) return -1;
    }
    k_pars_score_indexed_2d<<<dim3(nCand, nChunk), PARS_TPB>>>(dLeaves, dArena, dScore,
        (const int*)gb_pbCandL.p, (const int*)gb_pbCandR.p, dTip, nstates, nsblk, nCand, CH,
        (unsigned int*)gb_pbScores.p);
    if (cudaGetLastError() != cudaSuccess) return -1;
    if (cudaMemcpy(h_scores, gb_pbScores.p, (size_t)nCand*sizeof(unsigned int), cudaMemcpyDeviceToHost)!=cudaSuccess) return -1;
    if (cudaDeviceSynchronize() != cudaSuccess) return -1;
    return 0;
}
