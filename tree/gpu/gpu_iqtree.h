#ifndef IQTREE_GPU_IQTREE_H
#define IQTREE_GPU_IQTREE_H

// C-linkage entry points for the in-tree CUDA GPU module (Phase G.1+).
//
// These declarations are compiled into the `iqtree_gpu` static library, which
// is built ONLY when the CMake option IQTREE_GPU is ON (see CMakeLists.txt:
// `if(IQTREE_GPU) enable_language(CUDA) ... add_library(iqtree_gpu ...)`).
//
// We use C linkage so the nvcc-compiled .cu translation units and the host C++
// objects (main/main.cpp etc.) agree on symbol names without C++ mangling.
//
// Host callers must guard the include and the call with `#ifdef IQTREE_GPU`
// (the macro comes from the generated <iqtree_config.h>), so a CPU-only build
// neither references these symbols nor links the GPU library.

#ifdef __cplusplus
extern "C" {
#endif

// Phase G.6 — host callback that re-decomposes the rate matrix for a trial free-Q parameter vector. The launcher
// calls it inside the FD/LM loop whenever an exchangeability changes: q[nFreeQ] (the model's free params) -> the
// fresh eigensystem eval[ns], U[ns*ns] (eigenvectors), Uinv[ns*ns] (inverse eigenvectors), all row-major. ctx is the
// opaque host context (the live model). Plain C ABI so the nvcc TU and the host C++ agree without mangling.
typedef void (*jolt_qdecompose_fn)(void* ctx, const double* q, double* eval, double* U, double* Uinv);

// Phase G.1.0 build-scaffold diagnostic. Enumerates the CUDA device(s), prints
// device name / compute capability / VRAM, launches a trivial kernel, verifies
// it executed (marker readback) and that cudaGetLastError() == cudaSuccess,
// then returns. Pure side effect (stdout/stderr); never throws. A no-op-ish
// failure path just prints a message if no device is present.
void iqtree_gpu_diag();

// Lightweight device-info query for the startup banner (no kernel launch, no allocation).
// Fills `name` (caller buffer of `name_len` bytes) with the active device name, *vram_gb with its
// total VRAM in GiB, and *rt_major/*rt_minor with the CUDA runtime version. Returns 0 on success,
// -1 if no CUDA device is present (caller then prints nothing). Never throws.
int iqtree_gpu_info(char *name, int name_len, double *vram_gb, int *rt_major, int *rt_minor);

// Phase G.2.0a — clean-room GPU log-likelihood cross-check launcher.
//
// Runs the validated K1 eigen-space postorder partial-likelihood sweep on the GPU from host-prepared,
// already-extracted arrays (eigen factors, per-child echild, compact tip states, per-internal-node child
// descriptors, pattern frequencies) and returns the total tree log-likelihood
//   tree_lh = sum_ptn ptn_freq[ptn] * log|lh_ptn|.
// It is the in-tree analog of the standalone gpu_k1_lnl.cu harness; the host caller
// (PhyloTree::gpuLnLCrossCheckOnce) builds the inputs from the LIVE model/tree/alignment and compares the
// result against IQ-TREE's own curScore. NORM_LH / unscaled (AA-100K, leafNum<2000); FP64; native-20 path
// works for nstates in {4,20} (no 32-pad). Pure compute + a few cudaMalloc/Memcpy; returns NaN on any
// internal CUDA error (caller treats NaN as "cross-check unavailable", never aborts the CPU run).
//
// Descriptor arrays are length nInternal (postorder) except the *_child* arrays which are nInternal*3
// (up to 3 children per node; unused slots = -1). echild is indexed [child_node_id][cat][x][i] with stride
// ncat*nstates*nstates; the root node's echild slot is unused. outSlot/childSlot index the partials arena
// in units of (ncat*nstates*nptn). childIsLeaf!=0 => use childLeaf (taxon id) + the tip array; else childSlot.
//
// G.2.0b: if out_patlh != NULL it is filled with the per-pattern log-likelihoods log|lh_ptn| (nptn entries,
// pattern order) — exactly what IQ-TREE's _pattern_lh[] holds under NORM_LH (no scaling). This lets the
// Branch-pointer override (PhyloTree::computeLikelihoodBranchGPU) feed the host _pattern_lh[] so the
// downstream (unconditional) computeLogLVariance()/computePatternLikelihood() produce the correct s.e.
double gpu_lnl_crosscheck(
    int nstates, int nptn, int ncat, int ntax, int nnodes, int nInternal,
    const double* Uinv,          // nstates*nstates (row-major; IQ-TREE inverse eigenvectors)
    const double* UinvRowSum,    // nstates (row sums of Uinv, for fully-ambiguous tip states)
    const double* freq,          // nstates (state frequencies, for the root reduction)
    const double* catProp,       // ncat (category proportions / weights)
    const double* echild,        // nnodes * ncat*nstates*nstates
    const unsigned char* tip,    // ntax * nptn (compact states; >=nstates means ambiguous)
    const double* ptn_freq,      // nptn (pattern multiplicities)
    const int* desc_isRoot,      // nInternal
    const int* desc_nchild,      // nInternal
    const int* desc_outSlot,     // nInternal (partials-arena slot; root = -1)
    const int* desc_childNode,   // nInternal*3 (child node id -> echild base; -1 = unused)
    const int* desc_childIsLeaf, // nInternal*3
    const int* desc_childLeaf,   // nInternal*3 (taxon id if leaf child)
    const int* desc_childSlot,   // nInternal*3 (partials slot if internal child)
    double* out_patlh);          // nptn (optional; per-pattern log|lh_ptn| if non-NULL) — G.2.0b

// Phase G.8.0 — profile-mixture clean-room lnL (C20/C60/MEOW80). Same descriptor scheme as gpu_lnl_crosscheck but
// with R = nmix*ncat regimes (r = m*ncat + c): per-class Uinv/UinvRowSum/freq are [nmix][...] arrays (GLOBAL on
// device), wreg is [nmix*ncat] (= weight_m * catProp_c), echild is [nnodes][R][ns*ns], partial slots index in units
// of (R*nstates*nptn). Each regime is an independent Felsenstein sweep; combined only at the root fold
// L_p = Σ_r wreg_r·(freq_m · prod_r). Returns the whole-tree lnL; NaN on OOM/CUDA error.
double gpu_lnl_crosscheck_mix(
    int nstates, int nptn, int ncat, int nmix, int ntax, int nnodes, int nInternal,
    const double* Uinv,          // nmix * nstates*nstates (per-class inverse eigenvectors)
    const double* UinvRowSum,    // nmix * nstates
    const double* freq,          // nmix * nstates (per-class state frequencies)
    const double* wreg,          // nmix*ncat (weight_m * catProp_c)
    const double* echild,        // nnodes * (nmix*ncat)*nstates*nstates
    const unsigned char* tip,    // ntax * nptn
    const double* ptn_freq,      // nptn
    const int* desc_isRoot, const int* desc_nchild, const int* desc_outSlot,
    const int* desc_childNode, const int* desc_childIsLeaf, const int* desc_childLeaf, const int* desc_childSlot,
    double* out_patlh, double* out_lhcat,   // out_lhcat (optional, G.8.1): per-class L_{p,m} = w_m*Σ_c catProp_c*L_{p,m,c}, [nmix][nptn]
    double pinv = 0.0, const double* clsinv = nullptr);   // A1 (+I): pinv + per-class invariant clsinv[m][ptn]=w_m*pinv*base_invar_m [nmix][nptn]; defaults keep non-+I callers unchanged

// Phase G.2.1a — clean-room single-edge branch-length derivative launcher (K2 in-tree). The descriptor list
// covers BOTH subtrees split by the central edge (two sub-roots = the edge endpoints), all entries isRoot=0 so
// every internal node (incl. the two endpoints) writes its eigen-space partial to its slot. nodeSlot/dadSlot
// index the two endpoint partials. Returns df = d(lnL)/dt (= Σ ptn_freq·d1/lh); *out_ddf = the second
// derivative (Σ ptn_freq·(d2/lh−(d1/lh)²)); *out_lnL = the tree lnL at the central length t (free cross-check).
// val0/val1/val2 are built on-device from eval/catRate/catProp/t. Returns NaN on CUDA error.
double gpu_derv_crosscheck(
    int nstates, int nptn, int ncat, int ntax, int nnodes, int nInternal,
    const double* Uinv, const double* UinvRowSum, const double* freq, const double* catProp,
    const double* echild, const unsigned char* tip, const double* ptn_freq,
    const int* desc_isRoot, const int* desc_nchild, const int* desc_outSlot,
    const int* desc_childNode, const int* desc_childIsLeaf, const int* desc_childLeaf, const int* desc_childSlot,
    int nodeSlot, int nodeLeafTax,   // node endpoint: slot>=0 if internal, else leaf taxon id (G.2.1b)
    int dadSlot,  int dadLeafTax,    // dad  endpoint: slot>=0 if internal, else leaf taxon id (G.2.1b)
    const double* eval,          // nstates (eigenvalues)
    const double* catRate,       // ncat (per-category rates)
    double t,                    // central branch length
    double* out_ddf,             // out: second derivative
    double* out_lnL);            // out: tree lnL at t

// TS.2 Increment 3a — score an NNI-swapped topology @ OLD lengths from ONE resident postorder over the PHYSICAL
// tree + a re-pairing fold (no swap-aware DFS, no new kernel). The desc_* arrays are the physical two-sub-root
// postorder (as for gpu_derv_crosscheck). The 4 re-pairing children (node1's swapped {n1a,n1b}, node2's {n2a,n2b})
// each carry (ec = echild node index for that child's UNCHANGED length, slot>=0 internal | leaf>=0 tip taxon).
// Two k1_node(isRoot=0,nchild=2) folds build the swapped endpoint partials; k2_derv combines them at central
// length t. *out_lnL = swapped-topology lnL (== gpuScreenNNICleanRoom). Returns df; NaN on CUDA error.
double gpu_screen_nni_fold_crosscheck(
    int nstates, int nptn, int ncat, int ntax, int nnodes, int nInternal,
    const double* Uinv, const double* UinvRowSum, const double* freq, const double* catProp,
    const double* echild, const unsigned char* tip, const double* ptn_freq,
    const int* desc_isRoot, const int* desc_nchild, const int* desc_outSlot,
    const int* desc_childNode, const int* desc_childIsLeaf, const int* desc_childLeaf, const int* desc_childSlot,
    int n1a_ec, int n1a_slot, int n1a_leaf,   int n1b_ec, int n1b_slot, int n1b_leaf,
    int n2a_ec, int n2a_slot, int n2a_leaf,   int n2b_ec, int n2b_slot, int n2b_leaf,
    const double* eval, const double* catRate, double t,
    double* out_ddf, double* out_lnL);

// TS.2 Increment 3b-i — ONE fixed-root postorder (resident lowers) + ONE preorder with a PERSISTENT per-node upper
// buffer (slot = node id, not the O(depth) pool); for EVERY internal edge runs k2_derv(lower_v, pre_v, b_v) -> its
// lnL into out_edge_lnL[v]; returns the whole-tree lnL (independent root reduction) in *out_tree_lnL. The invariant
// (every edge lnL == tree lnL) validates the persistent-upper machinery. Single-model, nTile=1. Reuses k1_node /
// kj_pre / k2_derv; NO new kernel. node_* arrays index the fixed-root DFS; post_internal = postorder internal ids.
double gpu_allbranch_upper_check(
    int nstates, int nptn, int ncat, int ntax, int nnodes, int nInternal, int root,
    const double* Uinv, const double* U, const double* UinvRowSum, const double* freq, const double* catProp,
    const double* eval, const double* catRate,
    const double* echild, const double* expfac, const unsigned char* tip, const double* ptn_freq,
    const int* node_nchild, const int* node_child, const int* node_leaf, const int* node_slot,
    const double* node_parentLen, const int* post_internal,
    double* out_edge_lnL, double* out_tree_lnL);

// TS.2 Increment 3b-ii — BATCHED re-pairing NNI screener. Same fixed-root postorder + persistent-upper preorder
// as gpu_allbranch_upper_check (built ONCE), then scores nMoves NNI moves as cheap folds off the resident lowers +
// persistent uppers (kj_pre with swapped sibling for the parent endpoint / k1_node for the child / k2_derv at b_v).
// out_move_lnL[m] = swapped-topology lnL (== gpuScreenNNIFoldCleanRoom, the 3a oracle, to 1e-9). NO new kernel.
double gpu_screen_nni_batch_crosscheck(
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
    double* out_move_lnL, double* out_tree_lnL);

// TS.2 Increment 3c — PATTERN-TILED batched NNI screener. Identical math to gpu_screen_nni_batch_crosscheck but
// nptn is split into nTile chunks so the persistent per-node upper (the AA-1M OOM surface) is sized to chunk0, not
// nptn. Per-move lnL is a continuous carried Kahan sum over chunks => bit-identical to nTile=1 for any nTile.
// forced_ntile>0 forces nTile (0=auto via free VRAM; JOLT_NTILE env overrides auto). out_ntile = chosen nTile.
double gpu_screen_nni_tile_crosscheck(
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
    const double* baseinvar = nullptr, double pinv = 0.0);   // A3 (+I): default keeps non-+I callers unchanged

// G.8.1b — profile-mixture single-edge derivative (df/ddf class-summed). Mirrors gpu_derv_crosscheck but the
// sweep is per-regime (k1_node_mix), the central-edge coeffs are per-CLASS in global memory, and weights are
// per-regime (wreg[r]=w_m*catProp_c). evalC = per-class eigenvalues [nmix*nstates].
double gpu_derv_crosscheck_mix(
    int nstates, int nptn, int ncat, int nmix, int ntax, int nnodes, int nInternal,
    const double* Uinv, const double* UinvRowSum, const double* freq, const double* wreg,
    const double* echild, const unsigned char* tip, const double* ptn_freq,
    const int* desc_isRoot, const int* desc_nchild, const int* desc_outSlot,
    const int* desc_childNode, const int* desc_childIsLeaf, const int* desc_childLeaf, const int* desc_childSlot,
    int nodeSlot, int nodeLeafTax,
    int dadSlot,  int dadLeafTax,
    const double* evalC,         // nmix*nstates (per-class eigenvalues)
    const double* catRate,       // ncat
    double t,
    double* out_ddf,
    double* out_lnL);

// Phase G.8.2.1a — clean-room ALL-BRANCH derivative launcher for profile mixtures (Ji-2020 linear-time gradient:
// one postorder + one preorder sweep -> df/ddf for every edge). out_df[v]/out_ddf[v] = d(lnL)/db_v and 2nd deriv for
// edge v->parent (root entry stays 0). Returns 0.0 on success, NaN on CUDA error. d_U (eigenvectors, k7_pre_mix's
// up-map) and per-node expfac=exp(eval_m·rate_c·b_u) are new vs the lnL/single-edge mix launchers.
double gpu_allbranch_derv_crosscheck_mix(
    int nstates, int nptn, int ncat, int nmix, int ntax, int nnodes, int root,
    const double* Uinv, const double* U, const double* UinvRowSum, const double* freq, const double* wreg,
    const double* evalC, const double* catRate,
    const double* echild, const double* expfac, const unsigned char* tip, const double* ptn_freq,
    const int* node_nchild, const int* node_child, const int* node_leaf, const double* node_parentLen,
    double* out_df, double* out_ddf,
    double pinv = 0.0, const double* base_invar_comb = nullptr);   // A1 (+I): pinv + COMBINED invariant Σ_m w_m·base_invar_m [nptn]; defaults keep non-+I callers unchanged

// Phase G.8.2.1b — host shim exposing the JOLT mean-1 discrete-gamma discretiser so the mixture joint-optimiser
// kill-switch can recompute catRate[] at an iterate alpha (bit-identical to the live GAMMA_CUT_MEAN rates).
void gpu_discrete_gamma_mean(double alpha, int K, double* rates);

// Phase G.4.2 — in-tree JOLT joint-gradient optimiser launcher. Runs the validated G.4.1b standalone driver
// (gpu_k8b_jolt_alpha.cu) clean-room from host-prepared arrays built from the LIVE model/tree/alignment:
// a SINGLE joint LM diagonal-Newton loop steps ALL branches AND (if optAlpha) the gamma shape alpha at once,
// replacing IQ-TREE's per-edge Gauss-Seidel optimizeAllBranches + alpha-Brent. The whole host control loop +
// the device kernels (k1_node postorder, k7_pre preorder all-branch gradient, k2_theta/k2_derv edge reduction,
// k_ratenum +R/alpha rate gradient) run inside this one call; only the optimised (brlen, alpha, lnL) come back.
//
// Reductions are ptn_freq-WEIGHTED (compressed patterns) — the in-tree analog of the standalone's weight-1 sites.
// The eigen factors (U=evec, Uinv=inv_evec, eval) come from the LIVE model, so this works for any FIXED-Q
// reversible model (empirical AA matrix etc.); the caller gates eligibility (ns in {4,20}, no +I, gamma-only or
// no rate het, model->getNDim()==0). Topology is passed as flat per-node arrays (the launcher rebuilds its own
// post/preorder DFS); node ids = the caller's DFS index, so out_brlen[v] is the optimised length of edge
// (v -> its parent) for the caller to write back. NORM_LH / unscaled; FP64. Returns NaN on any CUDA error.
double gpu_jolt_optimize(
    int nstates, int nptn, int ncat, int ntax, int nnodes, int root,
    const double* Uinv,          // nstates*nstates (inverse eigenvectors)
    const double* UinvRowSum,    // nstates (row sums, for ambiguous tips)
    const double* U,             // nstates*nstates (eigenvectors; needed by k7_pre step 1)
    const double* eval,          // nstates (eigenvalues)
    const double* catProp,       // ncat (category weights, e.g. 1/K for +G)
    const unsigned char* tip,    // ntax * nptn (compact states; >=nstates means ambiguous)
    const double* ptn_freq,      // nptn (pattern multiplicities)
    const int* node_nchild,      // nnodes
    const int* node_child,       // nnodes*3 (child node ids; -1 = unused)
    const int* node_leaf,        // nnodes (taxon id if leaf, else -1)
    const double* node_parentLen,// nnodes (initial edge length to parent; root entry = 0)
    double alpha0, int optAlpha, int maxiter,
    // G.4.3b — +I (proportion of invariant sites) joint support (ncat>1 / +I+G only):
    const double* base_invar,    // nptn (pinv-independent invariant base = ptn_invar/pinv; 0 if not +I)
    double pinv0, int optPinv,   // initial pinv; optimise it jointly if optPinv (else held at pinv0; 0 => no +I)
    double pinvMin, double pinvMax, // clamp bounds (MIN_PINVAR, aln->frac_const_sites)
    const double* catRate0,      // G.5.1: ncat FreeRate rates[c] (nullptr unless freeRate); seeds rates directly
    int freeRate,                // G.5.1b: 0=off; 1=ENGAGE the +R joint LM (catRate/catProp free, gaugeFix Σwr=1, no alpha);
                                 //         2=RGRADCHECK-only diagnostic then decline to CPU (high-ncat FD validation path)
    // G.6 — DNA free-Q (the exchangeabilities are optimised; the eigensystem MOVES). nFreeQ free params q0[0..nFreeQ-1]
    // (the model's getVariables()[1..nFreeQ], raw rates), perturbed by FD inside the LM loop; each change re-decomposes
    // the rate matrix via the host callback (which applies param_spec + the gauge) and re-uploads eval/U/Uinv. nFreeQ==0
    // => fixed-Q (AA / DNA JC,F81), the launcher is byte-unchanged. G.5.1c (ladder 2a): freeRate and nFreeQ may now BOTH
    // be active (GTR+R) — independent diagonal-LM axes (free-Q FD grad ⟂ the log-rate/softmax-weight arms).
    int nFreeQ,                  // number of free exchangeabilities (0 = fixed-Q; 1..5 for DNA HKY..GTR)
    const double* q0,            // nFreeQ initial free params (nullptr if nFreeQ==0)
    jolt_qdecompose_fn qdecompose, void* qctx,   // ctx-bound host callback: q[nFreeQ] -> eval[ns],U[ns*ns],Uinv[ns*ns]
    double* out_q,               // nFreeQ (out: optimised free params; untouched if nFreeQ==0)
    double* out_brlen,           // nnodes (out: optimised parentLen per node; root entry untouched)
    double* out_alpha,           // out: optimised alpha (unchanged if !optAlpha)
    double* out_pinv,            // out: optimised pinv (unchanged if !optPinv)
    int* out_iters,              // out: joint-iteration count (the headline)
    double* out_rates = nullptr, // G.5.1b: ncat optimised +R rates (gauged Σ w·r=1; nullptr unless freeRate==1)
    double* out_props = nullptr);// G.5.1b: ncat optimised +R weights (nullptr unless freeRate==1)

// TS.8 GPU PARSIMONY (Phase A) — score ALL candidate insertion branches for one taxon in a single launch.
// Bit-packed Fitch (32 sites/UINT, nstates UINTs per site-block). endL/endR are B packed endpoint state-set
// arrays (each nstates*nsblk UINTs); scoreL/scoreR are their accumulated subtree scores; tip is the broadcast
// new-taxon leaf partial. h_out[b] = exact CPU parsimony score of inserting at branch b. Returns 0 / -1 (fallback).
int gpu_parsimony_score_branches(
    const unsigned int* h_tip, unsigned int tipScore,
    const unsigned int* h_endL, const unsigned int* h_endR,
    const unsigned int* h_scoreL, const unsigned int* h_scoreR,
    int nstates, int nsblk, int B, unsigned int* h_out);

// TS.8 GPU PARSIMONY (Phase B v1 — device-resident, recompute-from-resident-leaves). Three calls:
//  (1) set_leaves: ALLOCATE the resident leaf buffer (nLeaf rows of setU=nstates*nsblk UINTs); idempotent.
//  (2) set_leaf_row: upload ONE bit-packed leaf row into slot lid (= taxon node->id), as each taxon enters the
//      tree (leaf packs are tree-independent so idempotent across the 98 OpenMP trees). Returns 0 / -1.
//  (3) build_and_score: per insertion, recompute ALL directed internal partials from the resident leaves via a
//      level schedule (taskOut/A/B; refs >=0 = arena slot, <0 = leaf -(id+1); levelStart groups independent tasks),
//      then score the candidate branches (candL/candR refs + the new-taxon tipLeaf). h_scores[k] = exact CPU
//      parsimony score of inserting at candidate branch k. Returns 0 (filled) / -1 (caller -> CPU loop, byte-identical).
int gpu_parsimony_set_leaves(int nLeaf, int nstates, int nsblk, const void* alnId);
int gpu_parsimony_set_leaf_row(const unsigned int* h_row, int lid);
int gpu_parsimony_build_and_score(
    int maxSlots,
    const int* h_taskOut, const int* h_taskA, const int* h_taskB,
    const int* h_levelStart, int nLevel, int nTask,
    const int* h_candL, const int* h_candR, int tipLeaf,
    int nstates, int nsblk, int nCand, unsigned int* h_scores, const void* alnId);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // IQTREE_GPU_IQTREE_H
