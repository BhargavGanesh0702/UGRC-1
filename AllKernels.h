#pragma once

#ifdef __cplusplus
extern "C" {
#endif

void TanhActivation(float* activations, float* outputs, int n);
float* TanhDerivative(float* activations, float* derivatives, int n);

#ifdef __cplusplus
}
#endif
