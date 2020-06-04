#include <iostream>
#include <math.h>
#include <chrono>

// Kernel function to add the elements of two arrays
__global__ void add(int n, float *x, float *y) {
    for (int i = 0; i < n; i++)
        y[i] = x[i] + y[i];
}

int main(int argc ,char *argv[]){

    std::chrono::high_resolution_clock::time_point  start_beig,stop_end;

    int N = 1<<20;
    float *x, *y;

    // Allocate Unified Memory – accessible from CPU or GPU
    cudaMallocManaged(&x, N*sizeof(float));
    cudaMallocManaged(&y, N*sizeof(float));

    // initialize x and y arrays on the host
    for (int i = 0; i < N; i++) {
        x[i] = 1.0f;
        y[i] = 2.0f;
    }

    start_beig = std::chrono::high_resolution_clock::now();

    // Run kernel on 1M elements on the GPU
    add<<<1, 1>>>(N, x, y);
    // Wait for GPU to finish before accessing on host
    cudaDeviceSynchronize();

    stop_end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop_end - start_beig).count();
    std::cout << "gpu(only one thread) : " << duration / 1000.0f << " ms\n";

    // Check for errors (all values should be 3.0f)
    float maxError = 0.0f;
    for (int i = 0; i < N; i++){
        maxError = fmax(maxError, fabs(y[i]-3.0f));
    }

    std::cout << "Max error: " << maxError << std::endl;

    // Free memory
    cudaFree(x);
    cudaFree(y);

    return 0;
}