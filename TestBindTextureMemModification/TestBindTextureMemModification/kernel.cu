#ifndef __CUDACC__
#define __CUDACC__
#endif

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <stdio.h>
#include <stdlib.h>

texture<int> textureArray;

__global__ void displayArray(int* arrayInt)
{
	int offset = threadIdx.x;

	printf("[%d]: %d\n", offset, arrayInt[offset]);
}

__global__ void displayTexture()
{
	int offset = threadIdx.x;

	printf("[%d]: %d\n", offset, tex1Dfetch(textureArray, offset));

	if(offset == 0)
		printf("\n\n");
}

__global__ void incrementArray(int* arrayInt)
{
	int offset = threadIdx.x;

	arrayInt[offset]++;
}

int main()
{
	int* arrayIntDevice;
	int* arrayInt = (int*)malloc(sizeof(int) * 5);
	for (int i = 0; i < 5; i++)
	{
		arrayInt[i] = i + 1;
		printf("%d\n", arrayInt[i]);
	}

	if(cudaMalloc(&arrayIntDevice, sizeof(int) * 5) != 0)
		exit(-1);
	if (cudaMemcpy(arrayIntDevice, arrayInt, sizeof(int) * 5, cudaMemcpyHostToDevice) != 0)
		exit(-2);

	incrementArray <<<1, 5>>> (arrayIntDevice);

	cudaChannelFormatDesc desc = cudaCreateChannelDesc<int>();
	if (cudaBindTexture(NULL, &textureArray, arrayIntDevice, &desc, sizeof(int) * 5) != 0)
		exit(-3);

	displayTexture << <1, 5 >> > ();

	incrementArray << <1, 5 >> > (arrayIntDevice);

	displayTexture << <1, 5 >> > ();
}
