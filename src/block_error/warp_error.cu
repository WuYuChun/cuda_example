#include <cuda_runtime.h>

#define WARPS 2
#define WARP_SIZE 32
#define THREADS (WARPS * WARP_SIZE)
__shared__ int smem_first[THREADS];
__shared__ int smem_second[WARPS];
__global__ void sumKernel(int *data_in, int *sum_out) {
    int tx = threadIdx.x;
    smem_first[tx] = data_in[tx] + tx;
    if (tx % WARP_SIZE == 0) {
        int wx = tx / WARP_SIZE;
        smem_second[wx] = 0;
        for (int i = 0; i < WARP_SIZE; ++i){
            smem_second[wx] += smem_first[wx * WARP_SIZE + i];
        }
    }
    __syncthreads();

    if (tx == 0) {
        *sum_out = 0;
        for (int i = 0; i < WARPS; ++i){
            *sum_out += smem_second[i];
        }
    }
}

int main(int argc, char **argv) {
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