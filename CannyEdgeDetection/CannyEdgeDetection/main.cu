#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <stdio.h>
#include <stdlib.h>

#include "bitmap.h"

__global__ void rgbToGrayscale(unsigned char* input, int width, int height, unsigned char* output);

__host__ void computeGaussianKernel(float* gaussianKernel, int size, float sigma);
__global__ void cannyEdgeDectection(unsigned char* grayscaleInput, int width, int height, unsigned char* grayscaleOutput);

extern __constant__ float constGaussianKernel5x5[5*5];

int main()
{
	unsigned char* inputImageBytes;
	unsigned char* dev_inputImageBytes;

	int imageWidth, imageHeight;
	readBitmapImage(&inputImageBytes, &imageWidth, &imageHeight, "lizard.bmp");

	BGRToRGB(inputImageBytes, imageWidth, imageHeight);

	unsigned char* imageGrayScale = (unsigned char*)malloc(sizeof(unsigned char) * imageWidth * imageHeight);
	unsigned char* dev_imageGrayScale;

	cudaMalloc(&dev_inputImageBytes, sizeof(unsigned char) * imageWidth * imageHeight);
	cudaMalloc(&dev_imageGrayScale, sizeof(unsigned char) * imageWidth * imageHeight);
	cudaMemcpy(dev_inputImageBytes, inputImageBytes, sizeof(unsigned char) * imageWidth * imageHeight, cudaMemcpyHostToDevice);

	float* gaussianKernel5x5 = (float*)malloc(sizeof(float*) * 5 * 5);

	computeGaussianKernel(gaussianKernel5x5, 5, 1);
	for (int y = 0; y < 5; y++)
	{
		for (int x = 0; x < 5; x++)
			printf("%.3f ", gaussianKernel5x5[y * 5 + x]);
		printf("\n");
	}

	cudaMemcpyToSymbol(constGaussianKernel5x5, gaussianKernel5x5, sizeof(float) * 5 * 5, 0, cudaMemcpyHostToDevice);

	rgbToGrayscale<<<128, 128>>>(dev_inputImageBytes, imageWidth, imageHeight, dev_imageGrayScale);
	cannyEdgeDectection<<<128, 128>>>(dev_imageGrayScale, imageWidth, imageHeight, dev_imageGrayScale);

	generateBitmapImage(inputImageBytes, imageHeight, imageWidth, "outGray.bmp");
}
