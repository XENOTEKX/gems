// tree/gpu/gpu_diag.cu — Phase G.1.0 build-scaffold diagnostic for the IQ-TREE 3
// GPU ModelFinder port.
//
// Built into the `iqtree_gpu` static library, only when the CMake option
// IQTREE_GPU is ON. Invoked from main/main.cpp's `--gpu` diagnostic hook.
//
// Purpose: prove the full toolchain end-to-end with ZERO numerics yet — nvcc
// compiles a .cu, it links into the iqtree3 executable, and a kernel actually
// launches and runs on the device. This is the foundation that the later phases
// build on (G.1.1 postorder lnL kernel K1, G.1.2 single-edge derivative K2,
// G.1.3 CUDA-graph capture). See gpu-modelfinder-design.md PART II.

#include "tree/gpu/gpu_iqtree.h"

#include <cuda_runtime.h>
#include <cstdio>

// Trivial kernel: one thread writes a recognizable marker so the host can
// confirm device-side execution actually happened (not just a clean launch).
__global__ void iqtree_gpu_hello_kernel(int *flag) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        *flag = 0xC0DE;
    }
}

extern "C" int iqtree_gpu_info(char *name, int name_len, double *vram_gb,
                               int *rt_major, int *rt_minor) {
    int device_count = 0;
    if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0)
        return -1;
    int dev = 0;
    cudaGetDevice(&dev);
    cudaDeviceProp prop;
    if (cudaGetDeviceProperties(&prop, dev) != cudaSuccess)
        return -1;
    if (name && name_len > 0) {
        std::snprintf(name, name_len, "%s", prop.name);
        name[name_len - 1] = '\0';
    }
    if (vram_gb)
        *vram_gb = prop.totalGlobalMem / 1073741824.0;
    int rt = 0;
    cudaRuntimeGetVersion(&rt);
    if (rt_major) *rt_major = rt / 1000;
    if (rt_minor) *rt_minor = (rt % 1000) / 10;
    return 0;
}

extern "C" void iqtree_gpu_diag() {
    std::printf("GPU: ===== IQ-TREE GPU diagnostic (Phase G.1.0 scaffold) =====\n");

    int device_count = 0;
    cudaError_t err = cudaGetDeviceCount(&device_count);
    if (err != cudaSuccess) {
        std::fprintf(stderr, "GPU: cudaGetDeviceCount failed (%s)\n",
                     cudaGetErrorString(err));
        return;
    }
    if (device_count == 0) {
        std::fprintf(stderr, "GPU: no CUDA device found (device_count=0)\n");
        return;
    }

    int dev = 0;
    cudaGetDevice(&dev);
    cudaDeviceProp prop;
    if (cudaGetDeviceProperties(&prop, dev) == cudaSuccess) {
        size_t free_b = 0, total_b = 0;
        cudaMemGetInfo(&free_b, &total_b);
        std::printf("GPU: device %d/%d = \"%s\", compute capability %d.%d\n",
                    dev, device_count, prop.name, prop.major, prop.minor);
        std::printf("GPU: VRAM %.1f GB total / %.1f GB free, %d SMs, %.0f GB/s peak BW\n",
                    total_b / 1073741824.0, free_b / 1073741824.0,
                    prop.multiProcessorCount,
                    2.0 * prop.memoryClockRate * (prop.memoryBusWidth / 8) / 1.0e6);
    }

    // Launch the kernel and read the marker back to prove it ran on-device.
    int *d_flag = nullptr, h_flag = 0;
    err = cudaMalloc(&d_flag, sizeof(int));
    if (err != cudaSuccess) {
        std::fprintf(stderr, "GPU: cudaMalloc failed: %s\n", cudaGetErrorString(err));
        return;
    }
    cudaMemset(d_flag, 0, sizeof(int));
    iqtree_gpu_hello_kernel<<<1, 32>>>(d_flag);
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        std::fprintf(stderr, "GPU: cudaDeviceSynchronize failed: %s\n",
                     cudaGetErrorString(err));
        cudaFree(d_flag);
        return;
    }
    cudaMemcpy(&h_flag, d_flag, sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(d_flag);

    err = cudaGetLastError();
    if (err == cudaSuccess && h_flag == 0xC0DE) {
        std::printf("GPU: hello-world kernel OK "
                    "(marker=0x%X, cudaGetLastError=cudaSuccess)\n", h_flag);
        std::printf("GPU: ===== diagnostic PASSED =====\n");
    } else {
        std::fprintf(stderr, "GPU: diagnostic FAILED (marker=0x%X, err=%s)\n",
                     h_flag, cudaGetErrorString(err));
    }
}
