#include <iostream>
#include "Network.h"
#include <cuda_runtime.h>
int main(){

    cout<<"Entered main in Main.cpp"<<endl;
    float X[] = {23.0, -85.0, 14.0, 92.0, -61.0,  -10.0, 48.0, -55.0, 30.0, 6.0};
    Network NN;
    NN.init_Network(10,8,5,4);
    cout<<"Network initialised"<<endl;
    NN.print_Network();

    float *d_X;
    cudaMalloc(&d_X, 10 * sizeof(float));
    cudaMemcpy(d_X, X, 10 * sizeof(float), cudaMemcpyHostToDevice);
    NN.Forward_pass(d_X);
    cout<<"Forward pass done"<<endl;
     
    vector<float>outs(5);
    cudaError_t err = cudaMemcpy(outs.data(),NN.Layers[NN.n_Layers-1].outputs,5*sizeof(float),cudaMemcpyDeviceToHost);
    
    if (err != cudaSuccess) {
        std::cout << "cudaMemcpy failed: " << cudaGetErrorString(err) << std::endl;
    }
    std::cout << "About to print outputs..." << std::endl;

    for(auto it:outs){
        cout<<it<<" ";
    }
    cout<<endl<<endl;
    NN.print_Network();

    float Expected_output[] = {0.0, 1.0, 0.0, 0.0, 0.0};
    float *d_Expected;
    cudaMalloc(&d_Expected, 5 * sizeof(float));
    cudaMemcpy(d_Expected, Expected_output, 5 * sizeof(float), cudaMemcpyHostToDevice);
    NN.Backward_pass(d_Expected);
    cout<<"Backward pass done"<<endl;
    NN.print_Network();
    NN.Weights_update(0.01);
    cout<<"Weights updated"<<endl;
    NN.print_Network();
    cudaFree(d_X);
    cudaFree(d_Expected);
    return 0;
}