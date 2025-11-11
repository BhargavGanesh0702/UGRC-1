#pragma once

#ifdef __cplusplus
extern "C" {
#endif

// A = M*K, B= K*N, C=M*N
void MatMul_WMMA(float* A, float* B, float* C, int M, int N, int K);// C = AB

#ifdef __cplusplus
}
#endif
