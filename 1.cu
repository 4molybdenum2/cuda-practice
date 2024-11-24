#include <stdio.h>

__global__ void kernel1(void) {
    printf("kernel1\n");
}

__global__ void kernel2(void) {
    printf("kernel2\n");
}

int main(int argc, char **argv) {
    kernel1<<<1, 1>>>();
    printf("CPU here\n");
    kernel2<<<1, 1>>>();
    cudaDeviceSynchronize();
    printf("CPU also here\n");
}