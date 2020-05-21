#include <stdio.h> 
#include <numeric> 
#include <stdlib.h> 
#include <cuda.h> 

#define BLOCK_SIZE 512
#define N 512


__global__ void reduceKernel(float *d_out, float *d_in); 
 int main(void) {     
     size_t size = N*sizeof(float);     
     size_t size_o = size/BLOCK_SIZE; 
     float h_in[N];     
     float h_out[N/BLOCK_SIZE]; 
     float *d_in, *d_out; 
     cudaEvent_t start, stop;     
     cudaEventCreate(&start);     
     cudaEventCreate(&stop); 
     cudaError_t err; 
 
 
    for (int i = 0; i < N; i++)     {         
        h_in[i] = 1.0f;
         } 
     cudaMalloc((void**)&d_in, size);
     cudaMemcpy(d_in, h_in, size, cudaMemcpyHostToDevice); 
     cudaMalloc((void**)&d_out, size_o); 
     int grid_size = N/BLOCK_SIZE;     
     printf("Grid Size is: %d\n", grid_size);
     printf("Block Size is: %d\n", BLOCK_SIZE); 
     dim3 threadsPerBlock(BLOCK_SIZE);     
     dim3 blocks(grid_size); 
     cudaEventRecord(start);     
     reduceKernel<<<blocks, threadsPerBlock>>>(d_out, d_in);     // Wait for GPU to finish before accessing on host     
     err = cudaDeviceSynchronize(); 
     // err = cudaThreadSynchronize();     
     printf("Run kernel: %s\n", cudaGetErrorString(err)); 

    printf("Input Array: \n");         
    for (int n=0; n < N; n++)         
    {             
        printf("%f ", h_in[n]);         
     }         
     printf("\n"); 
     err = cudaMemcpy(h_out, d_out, size_o, cudaMemcpyDeviceToHost);     
     printf("Copy h_out off device: %s\n",cudaGetErrorString(err)); 
     for (int n=0; n < grid_size; n++)     {         
         printf("%f ", h_out[n]);     
         }     
         printf("\n"); 
     float final_reduction = 0.0f; 
     for (int i = 0; i < grid_size; i++) {         
         final_reduction += h_out[i];     
         }
      cudaEventRecord(stop);
      cudaEventSynchronize(stop);
      float elapsedTime;
      cudaEventElapsedTime(&elapsedTime, start, stop);     
      printf("Elapsed time: %f\n", elapsedTime); 
      printf("And the final reduction is: %f\n", final_reduction); 
 
 
     cudaFree(d_in);
     cudaFree(d_out); 

    } 
 __global__ void reduceKernel(float* d_out, float* d_in) {     
     int myId = threadIdx.x + blockDim.x * blockIdx.x;    // ID relative to whole array     
     int tid = threadIdx.x;                                // Local ID within the current block 
     __shared__ float temp[BLOCK_SIZE];     
     temp[tid] = d_in[myId];     
     __syncthreads(); 
     // do reduction in shared memory     
     for (unsigned int s = blockDim.x/2; s >= 1; s >>= 1)     
     {         
         if (tid < s)         
         {             
             temp[tid] += temp[tid + s];         
          }         
      __syncthreads();     
      } 
      if (tid == 0)     {         
          d_out[blockIdx.x] = temp[tid];     
          } 
      }