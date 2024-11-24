#include <stdio.h>

__global__ void kernel1(void) {
    int bid = blockIdx.x +
                blockIdx.y * gridDim.x + 
                blockIdx.z * gridDim.y * gridDim.x;

    int t_offset = threadIdx.x + 
                threadIdx.y * blockDim.x + 
                threadIdx.z * blockDim.y * blockDim.x;

    int b_offset = bid * blockDim.x * blockDim.y * blockDim.z;


    int id = b_offset + t_offset;

    printf("-------------------------\nId: %d\nBlock: (%d %d %d)\nThread: (%d %d %d)\n-------------------------\n", 
    id,
    blockIdx.x, blockIdx.y, blockIdx.z, 
    threadIdx.x, threadIdx.y, threadIdx.z );

}

int main(int argc, char **argv) {
    const int b_x = 2;
    const int b_y = 3;
    const int b_z = 4;

    const int t_x = 3;
    const int t_y = 3;
    const int t_z = 3;

    int blocks_PerGrid = b_x * b_y * b_z;
    int threads_PerBlock = t_x * t_y * t_z;
    printf("Total number of blocks/grid %d\n",blocks_PerGrid);
    printf("Total number of threads/block %d\n",threads_PerBlock);
    printf("Total number of threads %d\n",blocks_PerGrid * threads_PerBlock);

    dim3 blocksPerGrid(b_x, b_y, b_z);
    dim3 threadsPerBlock(t_x, t_y, t_z);

    kernel1<<<blocksPerGrid, threadsPerBlock>>>();
    cudaDeviceSynchronize();
}