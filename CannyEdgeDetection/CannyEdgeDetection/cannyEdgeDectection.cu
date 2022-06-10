#ifndef __CUDACC__
#define __CUDACC__
#endif

#define M_PI       (float)(3.14159265358979323846)// pi

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <stdio.h>

#include "cannyEdgeDectection.h"
#include "common.h"

__constant__ float cm_gaussianKernel[GAUSSIAN_KERNEL_SIZE * GAUSSIAN_KERNEL_SIZE];
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
	return cudaMemcpyToSymbol(cm_gaussianKernel, gaussianKernel, sizeof(float) * GAUSSIAN_KERNEL_SIZE * GAUSSIAN_KERNEL_SIZE);
}

__global__ void applyGaussianKernel(unsigned char* grayscaleInput, int width, int height, unsigned char* grayscaleOutput)
{
	int threadId = threadIdx.x + blockDim.x * blockIdx.x;

	while (threadId < width * height)
	{
		float outPixelValue = 0.0;
		float kernelSum = 0.0;

		int centerPixelX = threadId % width;
		int centerPixelY = threadId / width;

		for (int yKernel = -(GAUSSIAN_KERNEL_SIZE / 2); yKernel <= GAUSSIAN_KERNEL_SIZE / 2; yKernel++)
		{
			int pixelY = centerPixelY + yKernel;
			pixelY = max(0, pixelY);
			pixelY = min(pixelY, height - 1);

			for (int xKernel = -(GAUSSIAN_KERNEL_SIZE / 2); xKernel <= GAUSSIAN_KERNEL_SIZE / 2; xKernel++)
			{
				int pixelX = centerPixelX + xKernel;
				pixelX = max(0, pixelX);
				pixelX = min(pixelX, width - 1);

				int kernelLinearIndex = (yKernel + GAUSSIAN_KERNEL_SIZE / 2) * GAUSSIAN_KERNEL_SIZE + xKernel + GAUSSIAN_KERNEL_SIZE / 2;
				int pixelLinearIndex = pixelX + pixelY * width;

				outPixelValue += grayscaleInput[pixelLinearIndex] * cm_gaussianKernel[kernelLinearIndex];
				kernelSum += cm_gaussianKernel[kernelLinearIndex];
			}
		}

		int centerPixelLinearIndex = centerPixelX + centerPixelY * width;
		grayscaleOutput[centerPixelLinearIndex] = (unsigned char)(outPixelValue/kernelSum);

		threadId += gridDim.x * blockDim.x;
	}
}

__global__ void computeGradientIntensityEdgeDir(unsigned char* input, int width, int height, unsigned char* gradientIntensityOutput, float* edgeDirectionOutput)
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
		
		gradientIntensityOutput[threadId] = abs(outValueHorizontal) + abs(outValueVertical);
		edgeDirectionOutput[threadId] = atan2((float)outValueVertical, (float)outValueHorizontal);

		threadId += gridDim.x * blockDim.x;
	}
}

__global__ void nonMaximumSuppression(unsigned char* gradientIntensityBuffer, float* edgeDirectionBuffer, const int width, const int height, unsigned char* outputBuffer)
{
	int threadId = threadIdx.x + blockDim.x * blockIdx.x;

	unsigned char xOffsetsLeft[8] = { -1, -1, -1, 0, 0, -1, -1, -1 };
	unsigned char xOffsetsRight[8] = { +1, +1, +1, 0, 0, +1, +1, +1 };

	unsigned char yOffsetsRight[8] = { 0, +1, +1, +1, +1, -1, -1, 0 };
	unsigned char yOffsetsLeft[8] = { 0, -1, -1, -1, -1, +1, +1, 0 };

	while (threadId < width * height)
	{
		float angle = edgeDirectionBuffer[threadId] * 180 / M_PI;
		angle += 180 * (angle < 0);

		unsigned char pixelCenterValue = gradientIntensityBuffer[threadId];
		unsigned char pixelLeftValue, pixelRightValue;

		int indexLeft, indexRight;

		//int angleIndex = (int)(angle / 22.5);
		//angleIndex = min(angleIndex, 7);
		////printf("%.3f -> %d\n", angle, angleIndex);
		//indexLeft = threadId + xOffsetsLeft[angleIndex] + yOffsetsLeft[angleIndex] * width;
		//indexRight = threadId + xOffsetsRight[angleIndex] + yOffsetsRight[angleIndex] * width;

		if (angle <= 22.5 || angle >= 157.5)
		{
			indexLeft = threadId - 1;
			indexRight = threadId + 1;
		}
		else if (angle <= 67.5)
		{
			indexLeft = threadId - 1 + width;
			indexRight = threadId + 1 - width;
		}
		else if (angle <= 112.5)
		{
			indexLeft = threadId + width;
			indexRight = threadId - width;
		}
		else
		{
			indexLeft = threadId - 1 - width;
			indexRight = threadId + 1 + width;
		}

		if (!(indexLeft < 0 || indexLeft >= width * height || indexRight < 0 || indexRight >= width * height))
		{
			pixelLeftValue = gradientIntensityBuffer[indexLeft];
			pixelRightValue = gradientIntensityBuffer[indexRight];

			if (pixelLeftValue > pixelCenterValue || pixelRightValue > pixelCenterValue)
				outputBuffer[threadId] = 0;
			else
				outputBuffer[threadId] = pixelCenterValue;
		}

		threadId += gridDim.x * blockDim.x;
	}
}

__global__ void findMaximumIntensityOfImage(unsigned char* inputBuffer, int width, int height, unsigned char* outputMaxIntensities)
{
	int threadId = threadIdx.x + blockIdx.x * blockDim.x;

	__shared__ unsigned char localMaxIntensities[256];
	localMaxIntensities[threadIdx.x] = 0;
	__syncthreads();

	while (threadId < width * height)
	{
		unsigned char pixelValue = inputBuffer[threadId];
		localMaxIntensities[threadIdx.x] = max(localMaxIntensities[threadIdx.x], pixelValue);

		threadId += gridDim.x * blockDim.x;
	}

	int i = 256 / 2;
	while (i > 0)
	{
		if (threadIdx.x < i)
			localMaxIntensities[threadIdx.x] = max(localMaxIntensities[threadIdx.x], localMaxIntensities[threadIdx.x + i]);

		__syncthreads();

		i /= 2;
	}

	if (threadIdx.x == 0)
		outputMaxIntensities[blockIdx.x] = localMaxIntensities[0];
}

__global__ void findMaximumIntensityOfBuffer(unsigned char* maximumIntensitiesBuffer, unsigned char* outputMaximumIntensity)
{
	int i = 512 / 2;
	while (i > 0)
	{
		if (threadIdx.x < i)
			maximumIntensitiesBuffer[threadIdx.x] = max(maximumIntensitiesBuffer[threadIdx.x], maximumIntensitiesBuffer[threadIdx.x + i]);

		__syncthreads();

		i /= 2;
	}

	if (threadIdx.x == 0)
		*outputMaximumIntensity = maximumIntensitiesBuffer[0];
}

__global__ void highLowThreshold(unsigned char* nonMaximumBuffer, int width, int height, unsigned char* maximumIntensity, float lowThresholdRatio, float highTresholdRatio, unsigned char* outputWeakBuffer, unsigned char* outputBuffer)
{
	int threadId = threadIdx.x + blockIdx.x * blockDim.x;

	unsigned char highTreshold = highTresholdRatio * *maximumIntensity;
	unsigned char lowTreshold = highTreshold * lowThresholdRatio;

	while (threadId < width * height)
	{
		if (nonMaximumBuffer[threadId] <= lowTreshold)
			outputBuffer[threadId] = 0;
		else if (nonMaximumBuffer[threadId] >= highTreshold)
			outputBuffer[threadId] = 255;
		else
		{
			outputBuffer[threadId] = 25;//Arbitrary but these pixels are going to be replaced anyway
			outputWeakBuffer[threadId] = 1;
		}

		threadId += gridDim.x * blockDim.x;
	}
}

__global__ void hysteresis(unsigned char* highLowTresholdBuffer, unsigned char* weakPixelsInput, int width, int height, unsigned char* hysteresisOutput)
{
	int threadId = threadIdx.x + blockIdx.x * blockDim.x;

	while (threadId < width * height)
	{
		if (weakPixelsInput[threadId] == 1)//We found a weak pixel
		{
			bool strongFound = false;
			//We're going to look around for a strong pixel. If we find one, the weak pixel is declared as strong and if we don't , it's declared as non-relevant
			for (int y = -1; y <= 1; y++)
			{
				if (threadId + y * width < 0 || threadId + y * width >= width * height)
					continue;//Out of bounds

				for (int x = -1; x <= 1; x++)
				{
					int aroundPixelIndex = threadId + x + y * width;
					if (aroundPixelIndex < 0 || aroundPixelIndex >= width * height)
						continue;

					if (highLowTresholdBuffer[aroundPixelIndex] == 255)//This a strong pixel
					{
						hysteresisOutput[threadId] = 255;

						strongFound = true;

						break;
					}
				}

				if (strongFound)
					break;
			}

			if (!strongFound)
				hysteresisOutput[threadId] = 0;
		}
		else
			hysteresisOutput[threadId] = highLowTresholdBuffer[threadId];

		threadId += gridDim.x * blockDim.x;
	}
}

//TODO cannyEdgeDetection alloc function qui fait l'allocation nécessaire de l'image pour pas avoir à la refaire à chaque fois qu'on cal cannyEdge et mettre les buffers en variables globales
__host__ void cannyEdgeDetection(unsigned char* grayscaleInput, int width, int height, unsigned char* grayscaleOutput)
{
	unsigned char* gradientIntensityBuffer, *nonMaximumBuffer, *hysteresisBuffer;
	float* edgeDirectionBuffer;
	unsigned char* nonMaximumMaximumIntensities;
	unsigned char* nonMaximumMaximumIntensity;

	cudaMalloc(&gradientIntensityBuffer, sizeof(unsigned char) * width * height);
	cudaMalloc(&edgeDirectionBuffer, sizeof(float) * width * height);
	cudaMalloc(&nonMaximumBuffer, sizeof(unsigned char) * width * height);
	cudaMalloc(&hysteresisBuffer, sizeof(unsigned char) * width * height);
	cudaMalloc(&nonMaximumMaximumIntensities, sizeof(unsigned char) * 512);// * 512 for the number of blocks
	cudaMalloc(&nonMaximumMaximumIntensity, sizeof(unsigned char));

	CUDA_TIME_EXECUTION("Apply gaussian kernel", applyGaussianKernel<<<512, 256>>>(grayscaleInput, width, height, grayscaleOutput));
	CUDA_TIME_EXECUTION("Gradient intensity and edge direction", computeGradientIntensityEdgeDir<<<512, 256>>>(grayscaleOutput, width, height, gradientIntensityBuffer, edgeDirectionBuffer));
	CUDA_TIME_EXECUTION("Non-maximum suppression", nonMaximumSuppression<<<512, 256>>>(gradientIntensityBuffer, edgeDirectionBuffer, width, height, nonMaximumBuffer));
	CUDA_TIME_EXECUTION("Find maximum intensity of non-maximum image", findMaximumIntensityOfImage<<<512, 256>>>(nonMaximumBuffer, width, height, nonMaximumMaximumIntensities));
	CUDA_TIME_EXECUTION("Find maximum intensity of buffer", findMaximumIntensityOfBuffer<<<1, 256>>>(nonMaximumMaximumIntensities, nonMaximumMaximumIntensity));


	//Simple variable renaming for clarity and not to have to allocate new buffers
	unsigned char* highLowTresholdBuffer = gradientIntensityBuffer;
	unsigned char* weakPixelsBuffer = grayscaleOutput;
	CUDA_TIME_EXECUTION("High-low threshold", highLowThreshold<<<512, 256>>>(nonMaximumBuffer, width, height, nonMaximumMaximumIntensity, 0.05f, 0.09f, weakPixelsBuffer, highLowTresholdBuffer));
	CUDA_TIME_EXECUTION("Hysteresis", hysteresis<<<512, 256>>> (highLowTresholdBuffer, weakPixelsBuffer, width, height, hysteresisBuffer));
	
	cudaMemcpy(grayscaleOutput, hysteresisBuffer, sizeof(unsigned char) * width * height, cudaMemcpyDeviceToHost);
}
