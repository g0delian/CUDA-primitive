#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <iostream>

#define BLOCK_SIZE 2
#define N 128

__global__ void scanKernel(float* data) {
    int tid = threadIdx.x;                          // thread in block
    int id = threadIdx.x + blockDim.x * blockIdx.x;    // thread among all blocks
    __shared__ float arr[BLOCK_SIZE];                 // shared array

    arr[tid] = data[id];               // give input to the array
    __syncthreads();                   // sync to make sure it is completed

    for (int o = 1; o < N; o *= 2) {
        if (tid >= o) {
            arr[tid] += arr[tid - o];  // perform scan
        }
        __syncthreads();                // ensure all threads written offset
    }
    data[id] = arr[tid];            // write from shared to global
}

int main(void) {
    static int GRID_SIZE = N / BLOCK_SIZE;     // number of blocks
    size_t size = N * sizeof(float);           // size of input array
    float h_data[N];                         // input array
    float reducedData[GRID_SIZE];           // storing reduced value for each block
    float* d_data;

    cudaEvent_t start, stop;         // timer setup
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaMalloc(&d_data, size);
    for (int i = 0; i < N; i++) {         // populate the input array and copy contents to device global memory
        h_data[i] = 1.0f;
    }
    cudaMemcpy(d_data, h_data, size, cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(BLOCK_SIZE);    // declare kernel dimensions
    dim3 blocksPerGrid(GRID_SIZE);

    cudaEventRecord(start);
    scanKernel<<<blocksPerGrid, threadsPerBlock>>>(d_data);    // run kernel
    cudaDeviceSynchronize();
    cudaMemcpy(h_data, d_data, size, cudaMemcpyDeviceToHost);   // copy array from device to host memory
    cudaFree(d_data);                 // free array from device memory

    //for (int i = 0; i < N; i++) {
        //printf("%f ", h_data[i]);
    //}
    //printf("\n");

    int c = 0;
    for (int i = BLOCK_SIZE - 1; i < N; i += BLOCK_SIZE) {
        reducedData[c++] = h_data[i];          // get final reduced value from each block
    }

    for (int i = 1; i < GRID_SIZE; i++) {
        reducedData[i] += reducedData[i - 1];     // perform scan operation on reduced data
    }

    for (int i = BLOCK_SIZE; i < N; i++) {
        int block = i / BLOCK_SIZE;          // map (add) reduced values to the input array to get final scanned array
        h_data[i] = h_data[i] + reducedData[block - 1];
    }
    cudaEventRecord(stop);

    printf("%f\n", h_data[N - 1]);             // print last array index which is the result of the scan

    float time;

    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);          // print results
    printf("Execution time: %f", time);

    return 0;
}