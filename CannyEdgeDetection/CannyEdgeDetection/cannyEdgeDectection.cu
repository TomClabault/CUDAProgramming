#ifndef __CUDACC__
#define __CUDACC__
#endif

#define M_PI       3.14159265358979323846   // pi

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <stdio.h>

#include "cannyEdgeDectection.h"
#include "common.h"

__constant__ float cm_gaussianKernel5x5[5 * 5];
__constant__ int cm_sobelKernelHorizontal[3 * 3] = {-1, 0, 1, -2, 0, 2, -1, 0, 1};
__constant__ int cm_sobelKernelVertical[3 * 3] =  {-1, -2, -1, 0, 0, 0, 1, 2, 1};

__host__ void computeGaussianKernel(float* gaussianKernel, int size, float sigma) 
{
	int halfSize = size / 2;

	float sigmaSquare = sigma * sigma;

	float normal = 1 / (2 * M_PI * sigmaSquare);

	for (int y = 1; y <= size; y++)
	{
		for (int x = 1; x <= size; x++)
		{
			int leftNumerator = (x - (halfSize + 1)) * (x - (halfSize + 1)); 
			int rightNumerator = (y - (halfSize + 1)) * (y - (halfSize + 1));
			gaussianKernel[x - 1 + (y - 1) * size] = normal * exp(-(leftNumerator + rightNumerator)/(2 * sigmaSquare));
		}
	}
}

__host__ cudaError_t copyGaussianKernelToConstMem(float* gaussianKernel)
{
	return cudaMemcpyToSymbol(cm_gaussianKernel5x5, gaussianKernel, sizeof(float) * 5 * 5);
}

__device__ void applyGaussianKernel(unsigned char* grayscaleInput, int width, int height, unsigned char* grayscaleOutput)
{
	int threadId = threadIdx.x + blockDim.x * blockIdx.x;

	while (threadId < width * height)
	{
		float outPixelValue = 0.0;
		float kernelSum = 0.0;

		int centerPixelX = threadId % width;
		int centerPixelY = threadId / width;

		for (int yKernel = -2; yKernel <= 2; yKernel++)
		{
			int pixelY = centerPixelY + yKernel;
			pixelY = max(0, pixelY);
			pixelY = min(pixelY, height - 1);

			for (int xKernel = -2; xKernel <= 2; xKernel++)
			{
				int pixelX = centerPixelX + xKernel;
				pixelX = max(0, pixelX);
				pixelX = min(pixelX, width - 1);

				int kernelLinearIndex = (yKernel + 2) * 5 + xKernel + 2;
				int pixelLinearIndex = pixelX + pixelY * width;

				outPixelValue += grayscaleInput[pixelLinearIndex] * cm_gaussianKernel5x5[kernelLinearIndex];
				kernelSum += cm_gaussianKernel5x5[kernelLinearIndex];
			}
		}

		int centerPixelLinearIndex = centerPixelX + centerPixelY * width;
		grayscaleOutput[centerPixelLinearIndex] = (unsigned char)(outPixelValue/kernelSum);

		threadId += gridDim.x * blockDim.x;
	}
}

__device__ void computeGradientIntensityEdgeDir(unsigned char* input, int width, int height, unsigned char* gradientIntensityOutput, float* edgeDirectionOutput)
{
	int threadId = threadIdx.x + blockIdx.x * blockDim.x;

	while (threadId < width * height)
	{
		int centerPixelX = threadId % width;
		int centerPixelY = threadId / width;

		int outValueVertical = 0;
		int outValueHorizontal = 0;
		for (int yKernel = -1; yKernel <= 1; yKernel++)
		{
			int pixelY = centerPixelY + yKernel;
			pixelY = max(0, pixelY);
			pixelY = min(pixelY, height - 1);

			for (int xKernel = -1; xKernel <= 1; xKernel++)
			{
				int pixelX = centerPixelX + xKernel;
				pixelX = max(0, pixelX);
				pixelX = min(pixelX, width - 1);

				int kernelLinearIndex = (yKernel + 1) * 3 + xKernel + 1;
				int pixelLinearIndex = pixelX + pixelY * width;

				outValueHorizontal += input[pixelLinearIndex] * cm_sobelKernelHorizontal[kernelLinearIndex];
				outValueVertical += input[pixelLinearIndex] * cm_sobelKernelVertical[kernelLinearIndex];
			}
		}

		int centerPixelLinearIndex = centerPixelX + centerPixelY * width;
		
		gradientIntensityOutput[centerPixelLinearIndex] = abs(outValueHorizontal) + abs(outValueVertical);
		edgeDirectionOutput[centerPixelLinearIndex] = atan2((float)outValueVertical, (float)outValueHorizontal);

		threadId += gridDim.x * blockDim.x;
	}
}


__global__ void cuda_cannyEdgeDectection(unsigned char* grayscaleInput, int width, int height, unsigned char* grayscaleOutput, unsigned char* gradientIntensityOutput, float* edgeDirectionOutput)
{
	applyGaussianKernel(grayscaleInput, width, height, grayscaleOutput);
	__syncthreads();
	computeGradientIntensityEdgeDir(grayscaleOutput, width, height, gradientIntensityOutput, edgeDirectionOutput);
}

#define BENCHMARK_BLOCK_SIZE_THREAD_COUNT(iteration, blockSizeLimit, threadCountLimit, ...) do {\
	float minAverageTime = 10000000;\
	int bestBlockSize = 0;\
	int bestThreadsCount = 0;\
	\
	for (int blockSize = 8; blockSize < blockSizeLimit; blockSize += 16)\
	{\
		for (int threadCount = 8; threadCount < threadCountLimit; threadCount += 8)\
		{\
			float totalIterationsDuration = 0.0;\
			\
			for (int i = 0; i < iteration; i++)\
			{\
				cudaEvent_t start, stop;\
				cudaEventCreate(&start);\
				cudaEventCreate(&stop);\
				\
				cudaEventRecord(start);\
				__VA_ARGS__;\
				cudaEventRecord(stop);\
				cudaEventSynchronize(stop);\
				\
				float iterationDuration = 0;\
				cudaEventElapsedTime(&iterationDuration, start, stop);\
				cudaEventDestroy(start);\
				cudaEventDestroy(stop);\
				\
				totalIterationsDuration += iterationDuration;\
			}\
			\
			printf("[%03d, %03d] took %.3fms\n", blockSize, threadCount, totalIterationsDuration / iteration);\
			\
			if (totalIterationsDuration / iteration < minAverageTime)\
			{\
				bestBlockSize = blockSize;\
				bestThreadsCount = threadCount;\
				\
				minAverageTime = totalIterationsDuration / iteration;\
			}\
		}\
	}\
	printf("Best settings are: [%d, %d] with %.3fms\n", bestBlockSize, bestThreadsCount, minAverageTime);\
	} while(0)

//TODO cannyEdgeDetection alloc function qui fait l'allocation nécessaire de l'image pour pas avoir à la refaire à chaque fois qu'on cal cannyEdge et mettre les buffers en variables globales
__host__ void cannyEdgeDetection(unsigned char* grayscaleInput, int width, int height, unsigned char* grayscaleOutput)
{
	unsigned char* gradientIntensityBuffer;
	float* edgeDirectionBuffer;

	cudaMalloc(&gradientIntensityBuffer, sizeof(unsigned char) * width * height);
	cudaMalloc(&edgeDirectionBuffer, sizeof(float) * width * height);

	BENCHMARK_BLOCK_SIZE_THREAD_COUNT(25, 2048, 512, cuda_cannyEdgeDectection << <blockSize, threadCount >> > (grayscaleInput, width, height, grayscaleOutput, gradientIntensityBuffer, edgeDirectionBuffer));

	//CUDA_TICKTOCK_DECLARE();
	//CUDA_TICK();
	
	//CUDA_TOCK("Canny edge detection");

	cudaMemcpy(grayscaleOutput, gradientIntensityBuffer, sizeof(unsigned char) * width * height, cudaMemcpyDeviceToHost);
}
