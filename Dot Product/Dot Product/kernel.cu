#ifndef __CUDACC__  
	#define __CUDACC__
#endif

#include <device_launch_parameters.h>
#include <cuda_runtime.h>

#include "options.h"

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

void dotProductCPU()
{
	int* a = (int*)malloc(sizeof(int) * VECTOR_SIZE);
	int* b = (int*)malloc(sizeof(int) * VECTOR_SIZE);

	for (int i = 0; i < VECTOR_SIZE; i++)
	{
		a[i] = -i;
		b[i] = i * i;
	}

	int dotProduct = 0;
	clock_t start = clock();
	for (int iter = 0; iter < ITER; iter++)
	{
		dotProduct = 0;
		for (int i = 0; i < VECTOR_SIZE; i++)
			dotProduct += a[i] * b[i];
	}
	clock_t end = clock();

	printf("Dot product: %d\n", dotProduct);
	printf("CPU Time: %.3fms\n", (double)(end - start)/ITER);

	free(a);
	free(b);
}

__global__ void initVectors(int* a, int* b)
{
	int threadId = blockIdx.x * blockDim.x + threadIdx.x;

	while (threadId < VECTOR_SIZE)
	{
		a[threadId] = -threadId;
		b[threadId] = threadId * threadId;

		threadId += gridDim.x * blockDim.x;
	}
}

__global__ void dotProductKernel(int* a, int* b, int* partial_sum)
{
	__shared__ int cachedProduct[THREADS_PER_BLOCK];
	cachedProduct[threadIdx.x] = 0;
	__syncthreads();

	int threadId = blockIdx.x * blockDim.x + threadIdx.x;
	int threadIdInBlock = threadIdx.x;

	int localSum = 0;
	while (threadId < VECTOR_SIZE)
	{
		localSum += a[threadId] * b[threadId];

		threadId += gridDim.x * blockDim.x;
	}

	cachedProduct[threadIdInBlock] = localSum;

	__syncthreads();

	int i = blockDim.x / 2;
	while (i > 0)
	{
		if(threadIdInBlock < i)//Only working with the first half of the threads
			cachedProduct[threadIdInBlock] += cachedProduct[threadIdInBlock + i];
		__syncthreads();

		i /= 2;
	}

	if (threadIdInBlock == 0)//We only want one thread to do that
		partial_sum[blockIdx.x] = cachedProduct[0];
}

void dotProductGPU()
{
	int* dev_a, *dev_b;
	int* dev_sum;
	int* host_sum = (int*)malloc(sizeof(int) * BLOCKS_PER_GRID);
	memset(host_sum, 0, sizeof(int) * BLOCKS_PER_GRID);

	if (cudaMalloc(&dev_a, sizeof(int) * VECTOR_SIZE) != 0)
	{
		printf("Malloc error\n");
		exit(-1);
	}
	if (cudaMalloc(&dev_b, sizeof(int) * VECTOR_SIZE) != 0)
	{
		printf("Malloc error\n");
		exit(-1);
	}
	cudaMalloc(&dev_sum, sizeof(int) * BLOCKS_PER_GRID);

	initVectors << <128, 128 >> > (dev_a, dev_b);

	cudaEvent_t start, stop;

	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	cudaEventRecord(start, 0);
	for(int iter = 0; iter < ITER; iter++)
		dotProductKernel<<<BLOCKS_PER_GRID, THREADS_PER_BLOCK>>>(dev_a, dev_b, dev_sum);
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);

	float duration;
	cudaEventElapsedTime(&duration, start, stop);

	cudaMemcpy(host_sum, dev_sum, sizeof(int) * BLOCKS_PER_GRID, cudaMemcpyDeviceToHost);

	int finalSum = 0;
	for (int i = 0; i < BLOCKS_PER_GRID; i++)
		finalSum += host_sum[i];

	printf("\n");
	printf("Dot product: %d\n", finalSum);
	printf("GPU Time: %.3fms\n", duration / ITER);

	cudaFree(dev_a);
	cudaFree(dev_b);
	cudaFree(dev_sum);
}

int main()
{
	dotProductCPU();
	dotProductGPU();
}
