#include <cuda_runtime.h>

#define THREADS 128
#define BLOCKS 2
__global__ void vectorAdd(int *v) {
    int tx = threadIdx.x + blockDim.x * blockIdx.x;
    v[tx] += tx;
}


int main(int argc, char **argv) {

    int *d_vec = NULL;
    cudaMalloc((void**)&d_vec, sizeof(int) * BLOCKS * THREADS);
    cudaMemset(d_vec, 0, BLOCKS * THREADS);
    vectorAdd<<<BLOCKS, THREADS>>>(d_vec);
    cudaDeviceSynchronize();
    cudaFree(d_vec);
    return 0;
}