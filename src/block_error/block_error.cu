#include <cuda_runtime.h>


#define THREADS 128

__shared__ int smem[THREADS];

__global__ void sumKernel(int *data_in, int *sum_out) {
    int tx = threadIdx.x;
    smem[tx] = data_in[tx] + tx;

    if (tx == 0) {
        *sum_out = 0;
        for (int i = 0; i < THREADS; ++i)
                *sum_out += smem[i];
    }
}

int main(int argc, char **argv){
    int *data_in = NULL;
    int *sum_out = NULL;
    cudaMalloc((void**)&data_in, sizeof(int) * THREADS);
    cudaMalloc((void**)&sum_out, sizeof(int));
    cudaMemset(data_in, 0, sizeof(int) * THREADS);
    sumKernel<<<1, THREADS>>>(data_in, sum_out);
    cudaDeviceSynchronize();
    cudaFree(data_in);
    cudaFree(sum_out);
    return 0;
}