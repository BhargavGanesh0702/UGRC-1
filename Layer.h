#pragma once

class Layer
{
public:
    Layer(){
        ;
    }
    ~Layer(){
        ;
    }

    int n_neurons;
    int n_wt_params;
    float* weights;
    float* biases;
    float* activations;
    float* outputs;
    float* deltas;

    void init_Layer(int n, int w);
    void Matrix_operation(float* X);
    void Activation_function(bool is_out_layer = false);
    float* Activation_function_derivatives();
    void Delta_Calculation(float* Wt_Lplus1, float* D_Lplus1, int n_neurons_Lplus1, int n_wt_p_Lplus1);
    void weights_update(float* prev_layer_outputs, float learning_rate);
    void print_Layer();
};

