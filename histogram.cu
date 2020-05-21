%%cu
#include <stdio.h> 
#include <numeric> 
#include <stdlib.h> 
 
#include <cuda.h> 
#define BLOCK_SIZE 2
#define BIN_SIZE 8
 
 
__global__ void shared_histogram(int *d_bins, const int* d_in) {
   int id = threadIdx.x + blockDim.x * blockIdx.x;
   int tid = threadIdx.x;

   int item = d_in[id];
   __shared__ float shared_bins[BIN_SIZE];
   int binId = item % BIN_SIZE;
   __syncthreads();

   atomicAdd(&(shared_bins[binId]), 1);
   __syncthreads();
   if (tid == 0) {
    for (int i = 0; i < BIN_SIZE; i++) {
        atomicAdd(&(d_bins[i]), shared_bins[i]);
        __syncthreads();
    }
  }
}  
__global__ void simple_histogram(int *d_bins, const int *d_in, const int BIN_COUNT); 
 
int main(void) {     
    int N = 64;                // Size of the input array     
    int *d_bins;                // Array that will contain the histogram     
    int *d_in;            // Array that will contain the input data     
    int BIN_COUNT = 8;    // Number of bins in the histogram 
 
    cudaEvent_t start, stop;
    cudaError_t err;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // We will use the CUDA unified memory model to ensure data is transferred between host and device     
    cudaMallocManaged(&d_bins, BIN_COUNT*sizeof(int));     
    cudaMallocManaged(&d_in, N*sizeof(int)); 
 
    // Now we need to generate some input data     // You can invent any strategy you like for generating the input data!     
    for (int i=0; i < N; i++)     {         
        d_in[i] = i;     
        } 
 
    // We also need to initialise the bins in the histogram     
    for (int i=0; i < BIN_COUNT; i++)     {         
        d_bins[i] = 0;     
        } 
 
    // Now we need to set up the grid size. Work on the assumption that N is an exact multiple     //  of BLOCK_SIZE     // for the moment     
    int grid_size = N/BLOCK_SIZE; 
    cudaEventRecord(start);
    shared_histogram<<<grid_size, BLOCK_SIZE>>>(d_bins, d_in); 
    cudaDeviceSynchronize();  
    err = cudaDeviceSynchronize();
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Elapsed time was: %f milliseconds\n", milliseconds);
 
    for (int i = 0; i < BIN_COUNT; i++)     {         
        printf("Bin no. %d: Count = %d\n", i, d_bins[i]);     
        } 
 
    return 0; 
    }