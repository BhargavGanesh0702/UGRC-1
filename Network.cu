#include <iostream>
#include <cuda_runtime.h>
#include "Network.h"
#include "AllKernels.h"
#include <vector>
#include <iomanip> 
#include <chrono>

#include <thrust/transform.h>
#include <thrust/transform_reduce.h>
#include <thrust/device_ptr.h>
#include <thrust/functional.h>
#include <thrust/execution_policy.h>
#include <thrust/device_vector.h>


struct square {
    __host__ __device__ float operator()(const float& a) const {
        return a * a;
    }
};

struct multiply_by_constant {
    const float c;
    multiply_by_constant(float _c) : c(_c) {}
    __host__ __device__ float operator()(const float& x) const {
        return x * c;
    }
};

float loss(float* d_output, float* d_expected, int n) {
    thrust::device_ptr<const float> output_ptr(d_output);
    thrust::device_ptr<const float> expected_ptr(d_expected);

    thrust::device_vector<float> diff(n);
    thrust::transform(thrust::device,output_ptr, output_ptr + n,expected_ptr,diff.begin(),thrust::minus<float>()); //computes output[i] -expected[i]
    
    float sum_sq_error = thrust::transform_reduce(thrust::device,diff.begin(),diff.end(),square(),0.0f,thrust::plus<float>());

    return sum_sq_error/2; 
}

void Network::init_Network(int inp_layer_size,vector<int>& hid_layer_sizes,int out_layer_size){
    int n_hidden = hid_layer_sizes.size();
    n_Layers = n_hidden+1;
    Layers.clear();

    add_Layer(hid_layer_sizes[0],inp_layer_size);
    for(int i=1;i<n_hidden;i++){
        add_Layer(hid_layer_sizes[i],hid_layer_sizes[i-1]);
    }
    add_Layer(out_layer_size,hid_layer_sizes[n_hidden-1]);
    return;
}

void Network::add_Layer(int n_neu,int n_wts){
    Layer L;
    L.init_Layer(n_neu,n_wts);
    Layers.push_back(L);
    return;
}

void Network::Forward_pass(float* X){
    // Pass X to the first layer
    Curr_X = X;
    Layers[0].Matrix_operation(X);
    Layers[0].Activation_function();
    // For next layers, use prev outputs
    for(int i=1;i<n_Layers;i++){
        Layers[i].Matrix_operation(Layers[i-1].outputs);
        bool is_out = i==n_Layers-1;
        Layers[i].Activation_function(is_out);
    }
    return ;
}

void Network::Backward_pass(float* Expected_output){


    thrust::device_ptr<float> d_expected = thrust::device_pointer_cast(Expected_output);
    thrust::device_ptr<float> d_output = thrust::device_pointer_cast(Layers[n_Layers-1].outputs);
    thrust::device_ptr<float> d_deltas = thrust::device_pointer_cast(Layers[n_Layers-1].deltas);

    thrust::transform(thrust::device,d_expected,d_expected+Layers[n_Layers-1].n_neurons,d_output,d_deltas,thrust::minus<float>());


    for(int i=n_Layers-2;i>=0;i--){
        // cout<<"Derivatives of Hidden layer "<<i<<" are: ";
        Layers[i].Delta_Calculation(Layers[i+1].weights, Layers[i+1].deltas,Layers[i+1].n_neurons,Layers[i+1].n_wt_params);
    }
}

void Network::Weights_update(float learning_rate){
    float* prev = Curr_X;
    for(int i=0;i<n_Layers;i++){
        // cout<<"Calculated Delta Weights for Layer "<<i<<":"<<endl;
        Layers[i].weights_update(prev, learning_rate);
        prev = Layers[i].outputs;
    }
}

vector<float> Network::Train(vector<vector<float>>Data,vector<vector<float>>Expected_output,int n_epochs,float learning_rate){
    int N = Data.size();
    int input_dim = Data[0].size();
    int output_dim = Expected_output[0].size();
    vector<float>h_Data(N*input_dim);
    vector<float>h_Exp_out(N*output_dim);
    for(int i=0;i<N;i++){
        copy(Data[i].begin(),Data[i].end(),h_Data.begin()+i*input_dim);
        copy(Expected_output[i].begin(),Expected_output[i].end(),h_Exp_out.begin()+i*output_dim);
    }
    float* d_Data,*d_Exp_out;
    cudaMalloc(&d_Data, h_Data.size()*sizeof(float));
    cudaMalloc(&d_Exp_out, h_Exp_out.size()*sizeof(float));

    cudaMemcpy(d_Data, h_Data.data(), h_Data.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_Exp_out, h_Exp_out.data(), h_Exp_out.size() * sizeof(float), cudaMemcpyHostToDevice);

    vector<float>epoch_errors_list;
    auto start = std::chrono::high_resolution_clock::now();
    for (int epoch= 0;epoch<n_epochs;epoch++) {
        float error = 0;
        for (int j =0; j<N;j++) {
            float* d_sample_X = d_Data + j*input_dim;
            float* d_sample_Y = d_Exp_out + j*output_dim;

            Forward_pass(d_sample_X);
            error += loss(Layers[n_Layers-1].outputs,d_sample_Y,output_dim);
            Backward_pass(d_sample_Y);
            Weights_update(learning_rate);
        }
        error = error / N;
        epoch_errors_list.push_back(error);
    }
    auto stop = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = stop - start;

    std::cout << elapsed.count() << std::endl;
    cudaFree(d_Data);
    cudaFree(d_Exp_out);
    return epoch_errors_list;
}

float* Network::predict(float*X){//needs GPU pointer
    Forward_pass(X);
    return Layers[n_Layers-1].outputs;
}

void Network::print_Network(){
    cout<<"Hidden Layer "<<1<<endl<<endl;
    Layers[0].print_Layer();
    cout<<endl;
    for(int i=1;i<n_Layers-1;i++){
        cout<<"Hidden Layer "<<i+1<<endl<<endl;
        Layers[i].print_Layer();
        cout<<endl;
    }
    cout<<"Output Layer: "<<endl<<endl;
    Layers[n_Layers-1].print_Layer();
    cout<<endl;
}
