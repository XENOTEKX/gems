// gpu_kernels.h — forward declarations for GPU kernels launched across
// translation units.
//
// A kernel defined in one gpu_*.cu but launched from another is declared here;
// CUDA_SEPARABLE_COMPILATION (see CMakeLists.txt) links the cross-TU launch.
// The shared postorder/preorder likelihood kernels are defined in the core
// kernel TU and reused by the joint optimiser (gpu_joint_optimizer.cu).
// Built only when IQTREE_GPU=ON.
#pragma once

#include <cuda_runtime.h>

// Postorder eigen-space partial for one internal node (up to 3 children).
__global__ void k1_node(int ns, int nptn, int ncat, int isRoot, double* __restrict__ out, double* __restrict__ patlh,
        int nchild,
        const double* ec0, const double* p0, const unsigned char* t0,
        const double* ec1, const double* p1, const unsigned char* t1,
        const double* ec2, const double* p2, const unsigned char* t2);

// Leaf eigen-space partial: column of U^-1 for the observed state (row-sum if ambiguous).
__global__ void k_leaf_eig(int ns, int nptn, int ncat, const unsigned char* __restrict__ tipt, double* __restrict__ out);

// Preorder "rest of tree" eigen-space partial above edge u->v (parent branch applied).
__global__ void kj_pre(int ns, int nptn, int ncat, double* __restrict__ out_pre,
        const double* __restrict__ pre_u, const double* __restrict__ expfac_u,
        int nsib, const double* ec0, const double* sp0, const unsigned char* st0,
                 const double* ec1, const double* sp1, const unsigned char* st1);
