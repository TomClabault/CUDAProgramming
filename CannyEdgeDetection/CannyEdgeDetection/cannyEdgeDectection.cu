#ifndef __CUDACC__
#define __CUDACC__
#endif


#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#define _USE_MATH_DEFINES
#include <math.h>
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

__host__ void copyGaussianKernelToConstMem(float* gaussianKernel)
{
	CUDA_HANDLE_ERROR(cudaMemcpyToSymbol(cm_gaussianKernel5x5, gaussianKernel, sizeof(float) * 5 * 5), "Cuda memcpy to symbol gaussianKernel");
}

__device__ void applyGaussianKernel(unsigned char* grayscaleInput, int width, int height, unsigned char* grayscaleOutput)
{
	int threadId = threadIdx.x + blockDim.x * blockIdx.x;

	while (threadId < width * height)
	{
		float outPixelValue = 0.0;

		int centerPixelX = threadId % width;
		int centerPixelY = threadId / width;

		for (int yKernel = -2; yKernel <= 2; yKernel++)
		{
			int pixelY = centerPixelY + yKernel;
			if (pixelY < 0)
				pixelY = 0;
			else if (pixelY >= height)
				pixelY = height - 1;

			for (int xKernel = -2; xKernel <= 2; xKernel++)
			{
				int pixelX = centerPixelX + xKernel;
				if (pixelX < 0)
					pixelX = 0;
				else if (pixelX >= width)
					pixelX = width - 1;

				int kernelLinearIndex = (yKernel + 2) * 5 + xKernel + 2;
				int pixelLinearIndex = pixelX + pixelY * width;

				outPixelValue += grayscaleInput[pixelLinearIndex] * cm_gaussianKernel5x5[kernelLinearIndex];
			}
		}

		int centerPixelLinearIndex = centerPixelX + centerPixelY * width;
		grayscaleOutput[centerPixelLinearIndex] = (unsigned char)outPixelValue;

		threadId += gridDim.x * blockDim.x;
	}
}

__device__ void computeGradientIntensity(unsigned char* input, int width, int height, unsigned char* output)
{
	int threadId = threadIdx.x + blockIdx.x * blockDim.x;

	while (threadId < width * height)
	{


		threadId += gridDim.x * blockDim.x;
	}
}


__global__ void cuda_cannyEdgeDectection(unsigned char* grayscaleInput, int width, int height, unsigned char* grayscaleOutput, unsigned char* gradientIntensityBuffer)
{
	applyGaussianKernel(grayscaleInput, width, height, grayscaleOutput);
	__syncthreads();
	computeGradientIntensity(grayscaleOutput, width, height, gradientIntensityBuffer);
}

__host__ void cannyEdgeDetection(unsigned char* grayscaleInput, int width, int height, unsigned char* grayscaleOutput)
{
	unsigned char* gradientIntensityBuffer;
	cudaMalloc(&gradientIntensityBuffer, sizeof(unsigned char) * width * height);

	CUDA_TICKTOCK_DECLARE();
	CUDA_TICK();
	cuda_cannyEdgeDectection<<<128, 128>>>(grayscaleInput, width, height, grayscaleOutput, gradientIntensityBuffer);
	CUDA_TOCK("Canny edge detection");
}
