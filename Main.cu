#include <iostream>
#include <vector>
#include "Network.h"
#include <cmath>
#include <iomanip> 
#include <chrono>
#include <fstream>
#include <sstream>
#include <cuda_runtime.h>
#include <string>
using namespace std;


double mean_squared_error(vector<vector<float>>& Y1,vector<vector<float>>& Y2) {
    double mse = 0.0;
    for(int j=0;j<Y1.size();j++){
        vector<float>y1 = Y1[j];
        vector<float>y2 = Y2[j];
        for(size_t i = 0; i < y1.size(); i++) {
            double diff = y2[i] - y1[i];
            mse += diff * diff;
        }
    }
    return mse/Y1.size() ;
}

void readCSV(const string &filename,
             int n_input_features,
             int n_output_features,
             vector<vector<float>> &X,
             vector<vector<float>> &Y)
{
    ifstream file(filename);
    if(!file.is_open()){
        cerr << "Error: Cannot open file " << filename << endl;
        return;
    }

    string line;
    getline(file, line);

    while(getline(file, line)){
        stringstream ss(line);
        string val;
        vector<float> x_row, y_row;

        int col_idx = 0;
        while(getline(ss, val, ',')){
            float fval = stof(val);
            if(col_idx < n_input_features)
                x_row.push_back(fval);
            else
                y_row.push_back(fval);
            col_idx++;
        }

        if(x_row.size() == n_input_features && y_row.size() == n_output_features){
            X.push_back(x_row);
            Y.push_back(y_row);
        } else {
            cout<< x_row.size()<<" "<<y_row.size()<<endl;
            cout << "Wrong format" << line << endl;
        }
    }

    file.close();
}

int main(){
    auto start = std::chrono::high_resolution_clock::now();
    // cout<<"Entered main in Main.cpp"<<endl;

    int n_input_features = 7;
    int n_output_features = 1;

    vector<vector<float>> X, Expected_output;
    readCSV("Testcases/scaled_real_estate_train.csv", n_input_features, n_output_features, X, Expected_output);// Makesure files are scaled using mean variance scaling
    // cout << "Read " << X.size() << " rows from train CSV." << endl;

    Network NN;
    vector<int>hidden_layer_sizes = {20,10};
    NN.init_Network(n_input_features,hidden_layer_sizes,n_output_features);
    // cout<<"Network initialised"<<endl;
    // NN.print_Network();

    int n_epochs = 250;
    float learning_rate = 0.01;
    vector<float>errors_list = NN.Train(X,Expected_output,n_epochs,learning_rate);

    cout<<'['<<errors_list[0];
    for(int i=1;i<n_epochs;i++){
        cout<<", "<<errors_list[i];
    }
    cout<<']'<<endl;
    // vector<float>outs(n_output_features);
    // cudaError_t err = cudaMemcpy(outs.data(),NN.Layers[NN.n_Layers-1].outputs,n_output_features*sizeof(float),cudaMemcpyDeviceToHost);
    
    // if (err != cudaSuccess) {
    //     std::cout << "cudaMemcpy failed: " << cudaGetErrorString(err) << std::endl;
    // }
    // std::cout << "About to print outputs..." << std::endl;

    // for(auto it:outs){
    //     cout<<it<<" ";
    // }
    // cout<<endl<<endl;
    // NN.print_Network();
    auto stop = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = stop - start;


    vector<vector<float>> X_test, Y_test;
    readCSV("Testcases/scaled_real_estate_test.csv", n_input_features, n_output_features, X_test, Y_test);
    // cout << "Read " << X_test.size() << " rows from test CSV." << endl;

    vector<vector<float>>outputs;
    for(auto it:X_test){
        float* d_it;
        cudaMalloc(&d_it, n_input_features * sizeof(float));
        cudaMemcpy(d_it, it.data(), n_input_features * sizeof(float), cudaMemcpyHostToDevice);

        float *d_out = NN.predict(d_it);
        vector<float> out(n_output_features);
        cudaMemcpy(out.data(), d_out, n_output_features * sizeof(float), cudaMemcpyDeviceToHost);
        outputs.push_back(out);
        cudaFree(d_it);
    }

    double total_mse = mean_squared_error(outputs,Y_test);
    cout  << total_mse << endl;
    return 0;
}