#include <iostream>
#include <math.h>
#include <chrono>

#include "timer.hpp"

// function to add the elements of two arrays
void add(int n, float *x, float *y){
    for (int i = 0; i < n; i++) {
        y[i] = x[i] + y[i];
    }
}

int main(int argc ,char *argv[]){
    std::chrono::high_resolution_clock::time_point  start_beig,stop_end;

    int N = 1<<20; // 1M elements
    std::cout << "N: " << N << std::endl;

    float *x = new float[N];
    float *y = new float[N];

    // initialize x and y arrays on the host
    for (int i = 0; i < N; i++) {
        x[i] = 1.0f;
        y[i] = 2.0f;
    }

    start_beig = std::chrono::high_resolution_clock::now();
    // Run kernel on 1M elements on the CPU
    add(N, x, y);
    stop_end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop_end - start_beig).count();
    std::cout << "cpu : " << duration / 1000.0f << " ms\n";

    // Check for errors (all values should be 3.0f)
    float maxError = 0.0f;
    for (int i = 0; i < N; i++) {
        maxError = fmax(maxError, fabs(y[i] - 3.0f));
    }
    std::cout << "Max error: " << maxError << std::endl;

    // Free memory
    delete [] x;
    delete [] y;

    return 0;
}