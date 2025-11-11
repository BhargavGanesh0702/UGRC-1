#include <iostream>
#include <chrono>
using namespace std;


void NaiveMatMul(float* A, float* B, float* C, int M, int N, int K){
    for(int i=0;i<M;i++){
        for(int j=0;j<N;j++){
            C[i*N+j] = 0;
            for(int k=0;k<K;k++){
                C[i*N+j] += A[i*K+k]*B[k*N+j];
            }
        }
    }
}

int main(){
    float* A;
    float* B;
    float* C;
    int M, N, K;
    cin >> M >> N >> K;
    A = new float[M*K];
    B = new float[K*N];
    C = new float[M*N];
    for(int i=0;i<M*K;i++) cin >> A[i];
    for(int i=0;i<K*N;i++) cin >> B[i];

    auto start2 = chrono::high_resolution_clock::now();
    NaiveMatMul(A, B, C, M, N, K);
    auto end2 = chrono::high_resolution_clock::now();
    chrono::duration<double, std::milli> duration2 = end2 - start2;
    
    FILE* fp;
    fp = fopen("Sequential_Out.txt","w");
    if(fp==NULL){
        printf("File not able to open\n");
        return 1;
    }
    for(int i=0;i<M;i++){
        for(int j=0;j<N;j++){
            fprintf(fp,"%f ", C[i*N+j]);
        }
        fprintf(fp,"\n");
    }
    cout << duration2.count()  << endl;
    delete[] A;
    delete[] B;
    delete[] C;
    return 0;

}