#ifndef __CUDACC__
#define __CUDACC__
#endif


#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#define _USE_MATH_DEFINES
#include <math.h>
#include <stdio.h>

void computeGaussianKernel(float* gaussianKernel, int size, float sigma) 
{
	int halfSize = size / 2;

	float sigmaSquare = sigma * sigma;

	float normal = 1 / (2 * M_PI * sigmaSquare);

	for (int y = 0; y < size; y++)
	{
		for (int x = 0; x < size; x++)
		{
			int leftNumerator = (y - (halfSize + 1)) * (y - (halfSize + 1)); 
			int rightNumerator = (x - (halfSize + 1)) * (x - (halfSize + 1));
			gaussianKernel[x + y * size] = normal * exp(-(leftNumerator + rightNumerator)/(2 * sigmaSquare));
		}
	}
}

__constant__ float constGaussianKernel5x5[5*5];

__global__ void cannyEdgeDectection(unsigned char* grayscaleInput, int width, int height, unsigned char* grayscaleOutput)
{
	if (threadIdx.x == 0 && blockIdx.x == 0)
	{
		for (int i = 0; i < 5; i++)
		{
			for (int j = 0; j < 5; j++)
			{
				printf("%.3f ", constGaussianKernel5x5[i * 5 + j]);
			}
			printf("\n");
		}
	}

	return;
}
