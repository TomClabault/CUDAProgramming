#ifndef __CUDACC__
	#define __CUDACC__
#endif

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <stdio.h>
#include <stdlib.h>

#define BLOCKS_PER_GRID 8
#define THREADS_PER_BLOCK 8

#define VECTOR_SIZE 500000000

__global__ void dotProductGPU(int* a, int* b, int* c)
{
	__shared__ int cachedProduct[THREADS_PER_BLOCK];
	cachedProduct[threadIdx.x] = 0;
	__syncthreads();

	int threadId = blockDim.x * blockIdx.x + threadIdx.x;

	while (threadId < VECTOR_SIZE)
	{
		cachedProduct[threadIdx.x] += a[threadId] * b[threadId];

		threadId += gridDim.x * blockDim.x;
	}
	__syncthreads();

	int i = THREADS_PER_BLOCK / 2;
	while (i > 0)
	{
		if (threadIdx.x < i)
			cachedProduct[threadIdx.x] += cachedProduct[threadIdx.x + i];

		i /= 2;

		__syncthreads();
	}

	if (threadIdx.x == 0)
		c[blockIdx.x] = cachedProduct[0];
}

int main()
{
	int* dev_a, *dev_b, *dev_c;
	int* a, * b;

	a = (int*)malloc(sizeof(int) * VECTOR_SIZE);
	b = (int*)malloc(sizeof(int) * VECTOR_SIZE);

	cudaMalloc(&dev_a, sizeof(int) * VECTOR_SIZE);
	cudaMalloc(&dev_b, sizeof(int) * VECTOR_SIZE);
	cudaMalloc(&dev_c, sizeof(int) * BLOCKS_PER_GRID);

	int* host_c = (int*)malloc(sizeof(int) * BLOCKS_PER_GRID);

	for (int i = 0; i < VECTOR_SIZE; i++)
	{
		a[i] = -i;
		b[i] = i * i;
	}

	/*for (int i = 0; i < VECTOR_SIZE; i++)
		printf("%d, ", a[i]);
	printf("\n");

	for (int i = 0; i < VECTOR_SIZE; i++)
		printf("%d, ", b[i]);
	printf("\n");*/

	cudaMemcpy(dev_a, a, sizeof(int) * VECTOR_SIZE, cudaMemcpyHostToDevice);
	cudaMemcpy(dev_b, b, sizeof(int) * VECTOR_SIZE, cudaMemcpyHostToDevice);

	cudaEvent_t start, stop;

	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	cudaEventRecord(start, 0);
	dotProductGPU << <BLOCKS_PER_GRID, THREADS_PER_BLOCK >> >(dev_a, dev_b, dev_c);
	cudaEventRecord(stop, 0);

	cudaEventSynchronize(stop);

	float duration;
	cudaEventElapsedTime(&duration, start, stop);

	printf("GPU Time: %.3fms\n", duration);

	cudaMemcpy(host_c, dev_c, sizeof(int) * BLOCKS_PER_GRID, cudaMemcpyDeviceToHost);

	long int totalSum = 0;
	for (int i = 0; i < BLOCKS_PER_GRID; i++)
	{
		totalSum += host_c[i];
	}

	printf("%d\n", totalSum);
}
