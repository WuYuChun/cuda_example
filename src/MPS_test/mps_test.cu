//
// Created by dji on 2020/6/19.
//

#include <cuda_runtime.h>
#include <iostream>
#include <unistd.h>

#include "timer.hpp"

__global__ void testMaxFlopsKernel(float *pData, int nRepeats, float v1, float v2){
    int tid = blockIdx.x* blockDim.x+ threadIdx.x;

    float s = pData[tid], s2 = 10.0f - s, s3 = 9.0f - s, s4 = 9.0f - s2;
    for(int i = 0; i < nRepeats; i++){
        s=v1-s*v2;
        s2=v1-s*v2;
        s3=v1-s2*v2;
        s4=v1-s3*v2;
    }
    pData[tid] = ((s+s2)+(s3+s4));
}

int test_mps_function(){

    common::PreciseCpuTimer timer{};

    int N = 1<<10;
    int nRepeats = 10000000;
    float *x;
    float *x_device;

    //cpu data
    x = (float *)malloc(N*sizeof(float));

    // initialize x and y arrays on the host
    for (int i = 0; i < N; i++) {
        x[i] = 1.0f;
    }

    cudaMalloc(&x_device,N*sizeof(float));
    cudaMemcpy(x_device,x,N*sizeof(float),cudaMemcpyHostToDevice);
    dim3 blockSize(1,1,1);
    dim3 numBlocks(1,1,1);

    timer.start();
    testMaxFlopsKernel<<<numBlocks,blockSize>>>(x_device,nRepeats,0,0);
    cudaDeviceSynchronize();
    timer.stop();
    std::cout << ">>>>debug: cost time: " << timer.milliseconds() << std::endl;


    free(x);
    cudaFree(x_device);
}

int main(int argc, char *argv[]){

    test_mps_function();

    return 0;
}