#pragma once
#include "Layer.h"
#include <vector>

using namespace std;

class Network
{
public:
    Network(){
        ;
    }
    ~Network(){
        ;
    }

    int n_Layers;
    vector <Layer> Layers;
    float* Curr_X;
    void init_Network(int inp_layer_size,vector<int>& hidden_layer_sizes,int out_layer_size);
    void add_Layer(int n_neu,int n_wt);
    void Forward_pass(float* X);
    void Backward_pass(float* Expected_output);
    void Weights_update(float learning_rate);
    vector<float> Train(vector<vector<float>>Data,vector<vector<float>>Expected_output,int n_epochs,float learning_rate);
    float* predict(float* X);
    void print_Network();
};
