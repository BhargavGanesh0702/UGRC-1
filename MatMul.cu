
#include <iostream>
#include <mma.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <chrono>
#include <cuda_fp16.h>
#include "MatMul.h"
using namespace nvcuda;

__global__ void copyUnpadKernel(const float* C_pad, float* C, int M, int N, int N_pad) {
    int r = blockIdx.y * blockDim.y + threadIdx.y;
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (r < M && c < N) {
        C[r * N + c] = C_pad[r * N_pad + c];
    }
}


__global__ void padConvert(const float *A, __half *A_pad,
                            int M, int K, int M_pad, int K_pad) {
    int r = blockIdx.y * blockDim.y + threadIdx.y;
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (r < M_pad && c < K_pad) {
        if (r < M && c < K) {
            A_pad[r * K_pad + c] = __float2half(A[r * K + c]);
        }
    }
}


__global__ void MatMulWMMAKernel(__half* A, __half* B, float* C, int M, int N, int K) {
    
    int row = blockIdx.x * 16;
    int col = blockIdx.y * 16;

    wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;

    wmma::fill_fragment(c_frag, 0.0f);

    for (int k = 0; k < K; k += 16) {
        if (row < M && k < K && col < N) {
            wmma::load_matrix_sync(a_frag, A + row * K + k, K);
            wmma::load_matrix_sync(b_frag, B + k * N + col, N);
            wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
        }
    }

    if (row < M && col < N) {
        wmma::store_matrix_sync(C + row * N + col, c_frag, N, wmma::mem_row_major);
    }
}

void MatMul_WMMA(float* A, float* B, float* C, int M, int N, int K) {
 
    // auto start = std::chrono::high_resolution_clock::now();

    int M_pad = ((M + 15) / 16) * 16;
    int N_pad = ((N + 15) / 16) * 16;
    int K_pad = ((K + 15) / 16) * 16;

    __half *A_pad, *B_pad;
    float  *C_pad;
    cudaMalloc(&A_pad, M_pad * K_pad * sizeof(__half));
    cudaMalloc(&B_pad, K_pad * N_pad * sizeof(__half));
    cudaMalloc(&C_pad, M_pad * N_pad * sizeof(float));
    cudaMemset(A_pad, 0.0, M_pad * K_pad * sizeof(__half));
    cudaMemset(B_pad, 0.0, K_pad * N_pad * sizeof(__half));
    cudaMemset(C_pad, 0.0, M_pad * N_pad * sizeof(float));
    //padding
    dim3 block(32, 32);
    dim3 gridA((K_pad + 31)/32, (M_pad + 31)/32);
    padConvert<<<gridA, block>>>(A, A_pad, M, K, M_pad, K_pad);
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cout << "Kernel launch failed: " << cudaGetErrorString(err) << std::endl;
    }
    dim3 gridB((N_pad + 31)/32, (K_pad + 31)/32);
    padConvert<<<gridB, block>>>(B, B_pad, K, N, K_pad, N_pad);
    cudaDeviceSynchronize();
     err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cout << "Kernel launch failed: " << cudaGetErrorString(err) << std::endl;
    }
    
    // auto end = std::chrono::high_resolution_clock::now();
    // std::chrono::duration<double, std::milli> duration = end - start;
    // printf("%f \n", duration.count());
    

    // start = std::chrono::high_resolution_clock::now();  
    
    
    //MatMul
    dim3 blockDim(32, 1);
    dim3 gridDim(M_pad / 16, N_pad / 16);
    MatMulWMMAKernel<<<gridDim, blockDim>>>(A_pad, B_pad, C_pad, M_pad, N_pad, K_pad);
    cudaDeviceSynchronize();
     err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cout << "Kernel launch failed: " << cudaGetErrorString(err) << std::endl;
    }
    
    // end = std::chrono::high_resolution_clock::now();
    // duration = end - start;
    // printf("%f \n", duration.count());

    // start = std::chrono::high_resolution_clock::now();   
    // for (int i = 0; i < M; ++i)
    //     cudaMemcpy(C + i * N, C_pad + i * N_pad, N * sizeof(float), cudaMemcpyDeviceToDevice);
    //unpadding
    dim3 blockC(32, 32);
    dim3 gridC((N + 31)/32, (M + 31)/32);
    copyUnpadKernel<<<gridC, blockC>>>(C_pad, C, M, N, N_pad);
    cudaDeviceSynchronize();
     err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cout << "Copy kernel launch failed: " << cudaGetErrorString(err) << std::endl;
    }


    // end = std::chrono::high_resolution_clock::now();
    // duration = end - start;
    // printf("%f \n", duration.count());
    cudaFree(A_pad); cudaFree(B_pad); cudaFree(C_pad);

    return;
}

// int main(){
//     float* A;
//     float* B;
//     float* C;
//     int M, N, K;
//     scanf("%d %d %d", &M, &N, &K);
//     A = new float[M*K];
//     B = new float[K*N];
//     C = new float[M*N];
//     for(int i=0;i<M*K;i++) scanf("%f", &A[i]);
//     for(int i=0;i<K*N;i++) scanf("%f", &B[i]);

//      float *d_A, *d_B, *d_C;
//     cudaMalloc(&d_A, M * K * sizeof(float));
//     cudaMalloc(&d_B, K * N * sizeof(float));
//     cudaMalloc(&d_C, M * N * sizeof(float));
//     cudaMemcpy(d_A, A, M * K * sizeof(float), cudaMemcpyHostToDevice);
//     cudaMemcpy(d_B, B, K * N * sizeof(float), cudaMemcpyHostToDevice);

//     auto start = std::chrono::high_resolution_clock::now();
//     MatMul_WMMA(d_A, d_B, d_C, M, N, K);
//     auto end = std::chrono::high_resolution_clock::now();
//     std::chrono::duration<double, std::milli> duration = end - start;
//     cudaMemcpy(C, d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost);
    
//     FILE* fp;
//     fp = fopen("Parallel_Out.txt","w");
//     if(fp==NULL){
//         printf("File not able to open\n");
//         return 1;
//     }
//     for(int i=0;i<M;i++){
//         for(int j=0;j<N;j++){
//             fprintf(fp,"%f ", C[i*N+j]);
//         }
//         fprintf(fp,"\n");
//     }
//     printf("%f \n", duration.count());
// }