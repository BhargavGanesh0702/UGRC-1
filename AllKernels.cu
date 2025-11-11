#include <chrono>
#include <fstream>
#include <iostream>
#include <stdio.h>
#include <cuda.h>
#include "AllKernels.h"
#include <cuda_runtime.h>

__global__ void tanh_kernel(float* activations, float* outputs, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        outputs[idx] = tanhf(activations[idx]);
    }
}

void TanhActivation(float* activations, float* outputs, int n) {
    int blockSize = 1024;
    int numBlocks = (n + blockSize - 1) / blockSize;
    tanh_kernel<<<numBlocks, blockSize>>>(activations, outputs, n);
    cudaDeviceSynchronize();
}

__global__ void tanh_derivative_kernel(float* activations, float* derivatives, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float t = tanhf(activations[idx]);
        derivatives[idx] = 1.0f - t * t;
    }
}

float* TanhDerivative(float* activations, float* derivatives, int n) {
    int blockSize = 1024;
    int numBlocks = (n + blockSize - 1) / blockSize;
    tanh_derivative_kernel<<<numBlocks, blockSize>>>(activations, derivatives, n);
    cudaDeviceSynchronize();
    return derivatives;
}