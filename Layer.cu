
#include <iostream>
#include "Layer.h"
#include "AllKernels.h"
#include "MatMul.h"
#include <cuda_runtime.h>
#include <iomanip>
#include <cmath>
#include <vector>
#include <random>

#include <thrust/transform.h>
#include <thrust/transform_reduce.h>
#include <thrust/device_ptr.h>
#include <thrust/functional.h>
#include <thrust/execution_policy.h>
#include <thrust/device_vector.h>
using namespace std;


struct multiply_by_constant {
    const float c;
    multiply_by_constant(float _c) : c(_c) {}
    __host__ __device__ float operator()(const float& x) const {
        return x * c;
    }
};

void Layer::init_Layer(int n, int w){
    n_neurons = n;
    n_wt_params = w;
    cudaMalloc(&weights, n*w*sizeof(float));
    // cudaError_t err = cudaGetLastError();
    // if (err != cudaSuccess) {
    //     std::cout << "Kernel launch failed: " << cudaGetErrorString(err) << std::endl;
    // }
    cudaMalloc(&biases, n*sizeof(float));
    cudaMalloc(&activations, n*sizeof(float));
    cudaMalloc(&outputs, n*sizeof(float));
    cudaMalloc(&deltas, n*sizeof(float));
    // cudaMemset(weights,1,n*w*sizeof(float));
    // cudaMemset(biases,1,n*sizeof(float));
    vector<float> h_weights(n*w);
    vector<float> h_biases(n);
    std::random_device rd;
    std::mt19937 gen(rd());
    float limit = std::sqrt(6.0f/(n+w));
    std::uniform_real_distribution<float>dist(-limit,limit);
    for(auto &wv:h_weights){
        wv = dist(gen);
    }
    for(auto &bv:h_biases){
        bv = dist(gen);
    }
    cudaMemcpy(weights, h_weights.data(), n*w*sizeof(float), cudaMemcpyHostToDevice);// stored as n*w but printed as w*n (stored as W_transpose printed as W)
    cudaMemcpy(biases, h_biases.data(), n*sizeof(float), cudaMemcpyHostToDevice);
    return;
    
}

void Layer::Matrix_operation(float* X){

    // weights is n_neurons x n_wt_params
    // X is n_wt_params x 1
    // activations is n_neurons x 1
    // MatMul_WMMA(A, B, C, M, N, K) C = AB, A = M*K, B=K*N, C=M*N
    MatMul_WMMA(weights, X, activations, n_neurons, 1, n_wt_params);

    // vector<float> h_activations(n_neurons);
    // cudaMemcpy(h_activations.data(), activations, n_neurons * sizeof(float), cudaMemcpyDeviceToHost);
    // cout<<"Before adding biases, activations are: ";
    // for (auto val : h_activations) {
    //     cout << val << " ";  
    // }cout<<endl;

    thrust::device_ptr<float> d_activations = thrust::device_pointer_cast(activations);
    thrust::device_ptr<float> d_biases = thrust::device_pointer_cast(biases);

    thrust::transform(thrust::device,d_activations,d_activations+n_neurons,d_biases,d_activations,thrust::plus<float>());

    // cout<<"After adding biases, activations are: ";
    // cudaMemcpy(h_activations.data(), activations, n_neurons * sizeof(float), cudaMemcpyDeviceToHost);
    // for (auto val : h_activations) {
    //     cout << val << " ";
    // }cout<<endl;
    return;
}

void Layer::Activation_function(bool is_out_layer){
    if(!is_out_layer){
        TanhActivation(activations, outputs, n_neurons);
    }else{
        cudaMemcpy(outputs,activations,n_neurons*sizeof(float),cudaMemcpyDeviceToDevice);
    }
    // cout<<"After activation, outputs are: ";
    // vector<float> h_outputs(n_neurons);
    // cudaMemcpy(h_outputs.data(), outputs, n_neurons * sizeof(float), cudaMemcpyDeviceToHost);
    // for (auto val : h_outputs) {
    //     cout << val << " ";
    // }cout<<endl;
    return;
}

float* Layer::Activation_function_derivatives(){

    float* d_derivatives;
    cudaMalloc(&d_derivatives, n_neurons * sizeof(float));
    return TanhDerivative(activations, d_derivatives, n_neurons);
}

void Layer::Delta_Calculation(float* Wt_Lplus1, float* D_Lplus1, int n_neurons_Lplus1, int n_wt_p_Lplus1){
    MatMul_WMMA(D_Lplus1, Wt_Lplus1, deltas, 1, n_wt_p_Lplus1, n_neurons_Lplus1);
    float* derivatives = Activation_function_derivatives();
    thrust::device_ptr<float> d_deltas = thrust::device_pointer_cast(deltas);


    thrust::device_ptr<float> d_ptr_B = thrust::device_pointer_cast(derivatives);
    thrust::transform(d_deltas, d_deltas + n_neurons, d_ptr_B, d_deltas, thrust::multiplies<float>());

    // float h_derivatives[n_neurons];

    // cudaMemcpy(h_derivatives, derivatives, n_neurons * sizeof(float), cudaMemcpyDeviceToHost);
    // for(auto it:h_derivatives){
    //     cout<<it<<" ";  
    // }cout<<endl;
    cudaFree(derivatives);
    return;
}

void Layer:: weights_update(float* prev_layer_outputs, float learning_rate){
    float* delta_Wt;
    cudaMalloc(&delta_Wt, n_neurons * n_wt_params * sizeof(float));
    MatMul_WMMA(deltas, prev_layer_outputs, delta_Wt, n_neurons, n_wt_params, 1);

    thrust::device_ptr<float> d_ptr = thrust::device_pointer_cast(delta_Wt);
    thrust::transform(d_ptr, d_ptr + n_neurons * n_wt_params, d_ptr, multiply_by_constant(learning_rate));

    thrust::device_ptr<float> d_weights = thrust::device_pointer_cast(weights);
    thrust::device_ptr<float> d_delta_wts = thrust::device_pointer_cast(delta_Wt);

    thrust::transform(thrust::device,d_weights,d_weights+(n_neurons*n_wt_params),d_delta_wts,d_weights,thrust::plus<float>());

    thrust::device_ptr<float> d_deltas = thrust::device_pointer_cast(deltas);
    thrust::device_vector<float> d_delta_biases(d_deltas,d_deltas+n_neurons);
    thrust::transform(d_delta_biases.begin(), d_delta_biases.end(), d_delta_biases.begin(), multiply_by_constant(learning_rate));
    thrust::device_ptr<float> d_biases = thrust::device_pointer_cast(biases);
    thrust::transform(thrust::device,d_biases,d_biases+n_neurons,d_delta_biases.begin(),d_biases,thrust::plus<float>());
    // float* H_delta_Wt = new float[n_neurons * n_wt_params];
    // cudaMemcpy(H_delta_Wt, delta_Wt, n_neurons * n_wt_params * sizeof(float), cudaMemcpyDeviceToHost);
    
    // for (int i = 0; i < n_wt_params; ++i) {
    //     for (int j = 0; j < n_neurons; ++j) {
    //         cout << H_delta_Wt[j * n_wt_params + i] << " ";
    //     }
    //     cout << endl;
    // }
    cudaFree(delta_Wt);
    
}

void Layer::print_Layer() {
    vector<float> h_weights(n_neurons * n_wt_params);
    vector<float> h_biases(n_neurons);
    cudaMemcpy(h_weights.data(), weights, n_neurons * n_wt_params * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_biases.data(), biases, n_neurons * sizeof(float), cudaMemcpyDeviceToHost);
    cout << "Weights: "<<n_wt_params<< "*"<<n_neurons << endl;
    std::cout << std::fixed << std::setprecision(5);
    for (int i = 0; i < n_wt_params; ++i) {
        for (int j = 0; j < n_neurons; ++j) {
            cout << h_weights[j * n_wt_params + i] << " ";
        }
        cout << endl;
    }
    cout << "Biases:" << endl;
    for (int i = 0; i < n_neurons; ++i) {
        cout << h_biases[i] << " ";
    }
    cout << endl;
}