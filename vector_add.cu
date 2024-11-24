#include <stdio.h>
#include <time.h>
#include <cuda_runtime.h>
#include <stdlib.h>

#define N 10000000
#define BLOCK_SIZE 256

// Function to measure execution time
double get_time() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

void init_vec(float *a, int n) {
    for(int i=0; i<n; i++) {
        a[i] = (float)rand()/ 100;
    }
}

void vector_add_cpu(float *a, float *b, float *c, int n) {
    for(int i=0; i<n; i++) {
        c[i] = a[i] + b[i];
    }
}

__global__ void vector_add_gpu(float *a, float *b, float *c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < n) {
        c[i] = a[i] + b[i];
    }
}

int main(int argc, char **argv) {
    srand(time(NULL));
    float *h_a, *h_b, *h_c_cpu, *h_c_gpu;
    float *d_a, *d_b, *d_c;
    size_t size = sizeof(float) * N;

    h_a = (float*)malloc(size);
    h_b = (float*)malloc(size);
    h_c_cpu = (float*)malloc(size);
    h_c_gpu = (float*)malloc(size);

    init_vec(h_a, N);
    init_vec(h_b, N);


    cudaMalloc(&d_a, size);
    cudaMalloc(&d_b, size);
    cudaMalloc(&d_c, size);


    cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);


    dim3 block(BLOCK_SIZE);
    dim3 grid((N + block.x - 1) / block.x);

    printf("grid.x %d block.x %d \n",grid.x, block.x);

    printf("benchmarking cpu impl...\n");
    double cpu_start_time = get_time();
    vector_add_cpu(h_a, h_b, h_c_cpu, N);
    double cpu_end_time = get_time();

    printf("benchmarking gpu impl...\n");
    double gpu_start_time = get_time();
    vector_add_gpu<<<grid, block>>>(d_a, d_b, d_c, N);
    cudaDeviceSynchronize();
    double gpu_end_time = get_time();


    // Print results
    printf("CPU time: %f milliseconds\n", (cpu_end_time - cpu_start_time)*1000);
    printf("GPU time: %f milliseconds\n", (gpu_end_time - gpu_start_time)*1000);


    cudaMemcpy(h_c_gpu, d_c, size, cudaMemcpyDeviceToHost);

    bool correct = true;
    for (int i = 0; i < N; i++) {
        if (fabs(h_c_cpu[i] - h_c_gpu[i]) > 1e-5) {
            correct = false;
            break;
        }
    }
    printf("Results are %s\n", correct ? "correct" : "incorrect");

    free(h_a);
    free(h_b);
    free(h_c_cpu);
    free(h_c_gpu);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    cudaDeviceReset();
    return 0;
    
}   