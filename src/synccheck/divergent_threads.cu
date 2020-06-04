#include <cuda_runtime.h>

#define THREADS 64
#define DATA_BLOCKS 16
__shared__ int smem[THREADS];

__global__ void myKernel(int *data_in, int *sum_out, const int size) {
    int tx = threadIdx.x;
    smem[tx] = 0;
    __syncthreads();
    for (int b = 0; b < DATA_BLOCKS; ++b) {
        const int offset = THREADS * b + tx;
        if (offset < size) {
            smem[tx] += data_in[offset];
            __syncthreads();
        }
    }
    if (tx == 0) {
        *sum_out = 0;
        for (int i = 0; i < THREADS; ++i)
        *sum_out += smem[i];
    }
}

int main(int argc, char *argv[]){

    const int SIZE = (THREADS * DATA_BLOCKS) - 16;
    int *data_in = NULL;
    int *sum_out = NULL;
    cudaMalloc((void**)&data_in, SIZE * sizeof(int));
    cudaMalloc((void**)&sum_out, sizeof(int));
    myKernel<<<1,THREADS>>>(data_in, sum_out, SIZE);
    cudaDeviceSynchronize();
    cudaFree(data_in);
    cudaFree(sum_out);
    return 0;
}