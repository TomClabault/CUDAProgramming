#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <iostream>
#include <stdexcept>
#include <stdio.h>
#include <stdlib.h>
#include <string>

#include "bitmap.h"
#include "cannyEdgeDectection.h"
#include "common.h"
#include "rgbManip.h"

#define IN_IMAGE_NAME "mhwWallpaper.bmp"

void saveImageAndOpen(unsigned char* fullBytesImageBuffer, int width, int height)
{
	char outName[256];
	sprintf(outName, "out%s", IN_IMAGE_NAME);
	generateBitmapImage(fullBytesImageBuffer, height, width, outName);

	char startString[256];
	sprintf(startString, "start %s", outName);
	system(startString);
}

char* execSystemCommand(const char* cmd) 
{
	char buffer[128];
	char* result = (char*)calloc(65536, sizeof(char));
	FILE* pipe = _popen(cmd, "r");
	if (!pipe) throw std::runtime_error("popen() failed!");
	try {
		while (fgets(buffer, sizeof buffer, pipe) != NULL) 
		{
			strcat(result, buffer);
		}
	}
	catch (...) {
		_pclose(pipe);
		throw;
	}
	_pclose(pipe);
	return result;
}

void killAllProcessesByName(const char* processName)
{
	char command[256];
	sprintf(command, "tasklist /FI \"IMAGENAME eq %s\" /FO LIST | findstr \"PID:\"", processName);

	char* PIDList = execSystemCommand(command);
	char PIDString[16] = {0};
	while (*PIDList != 0)
	{
		int index = 0;
		if (*PIDList >= '0' && *PIDList <= '9')
		{
			while (*PIDList >= '0' && *PIDList <= '9')
			{
				PIDString[index++] = *PIDList;

				PIDList++;
			}

			char killCommand[128];
			sprintf(killCommand, "taskkill /pid %d", atoi(PIDString));
			system(killCommand);
		}
		else
			PIDList++;
	}
}

int main()
{
	unsigned char* fullBytesImageBuffer;
	int width, height;

	killAllProcessesByName("ImageGlass.exe");

	readBitmapImage(&fullBytesImageBuffer, &width, &height, IN_IMAGE_NAME);

	unsigned char* dev_fullBytesImageBuffer;
	CUDA_HANDLE_ERROR(cudaMalloc(&dev_fullBytesImageBuffer, sizeof(unsigned char) * width * height * 3), "Cuda malloc dev_fullBytesImageBuffer");
	CUDA_HANDLE_ERROR(cudaMemcpy(dev_fullBytesImageBuffer, fullBytesImageBuffer, sizeof(unsigned char) * width * height * 3, cudaMemcpyHostToDevice), "Cuda memcpy to dev_fullBytesImageBuffer");
	
	float gaussianKernel[GAUSSIAN_KERNEL_SIZE * GAUSSIAN_KERNEL_SIZE];
	computeGaussianKernel(gaussianKernel, GAUSSIAN_KERNEL_SIZE, GAUSSIAN_KERNEL_SIGMA);
	CUDA_HANDLE_ERROR(copyGaussianKernelToConstMem(gaussianKernel), "Cuda memcpy to symbol gaussianKernel");

	unsigned char* dev_grayBytesImageBuffer;
	unsigned char* dev_grayBytesOutCannyDetectionBuffer;
	CUDA_HANDLE_ERROR(cudaMalloc(&dev_grayBytesImageBuffer, sizeof(unsigned char) * width * height), "Cuda malloc gray bytes buffer");
	CUDA_HANDLE_ERROR(cudaMalloc(&dev_grayBytesOutCannyDetectionBuffer, sizeof(unsigned char) * width * height), "Cuda malloc gray bytes canny out buffer");

	CUDA_TIME_EXECUTION("rgbToGrayscale", cuda_rgbToGrayscale << <512, 512 >> > (dev_fullBytesImageBuffer, width, height, dev_grayBytesImageBuffer));

	cannyEdgeDetection(dev_grayBytesImageBuffer, width, height, dev_grayBytesOutCannyDetectionBuffer);//Already tick-tocked

	CUDA_TIME_EXECUTION("grayToGray3Bytes", cuda_grayBytesToGray3Bytes << <128, 128 >> > (dev_grayBytesOutCannyDetectionBuffer, width, height, dev_fullBytesImageBuffer));

	CUDA_HANDLE_ERROR(cudaMemcpy(fullBytesImageBuffer, dev_fullBytesImageBuffer, sizeof(unsigned char) * width * height * 3, cudaMemcpyDeviceToHost), "Memcpy device to host fullImageBytesBuffer");

	saveImageAndOpen(fullBytesImageBuffer, width, height);
}

//int main()
//{
//	unsigned char** imageBytes;
//	unsigned char* dev_inputImageBytes;
//
//	int imageWidth, imageHeight;
//	readBitmapImage2D(&imageBytes, &imageWidth, &imageHeight, IN_IMAGE_NAME);
//	BGRToRGB2D(imageBytes, imageWidth, imageHeight);//TODO, this is unecessary because we can just treat the bytes in the BGR order in the code
//
//	unsigned char* imageGrayScale = (unsigned char*)malloc(sizeof(unsigned char) * imageWidth * imageHeight);
//	unsigned char* dev_cannyEdgeResult;
//
//	cudaArray_t dev_imageGrayScaleArray;
//	cudaChannelFormatDesc desc = cudaCreateChannelDesc<unsigned char>();
//	CUDA_HANDLE_ERROR(cudaMallocArray(&dev_imageGrayScaleArray, &desc, imageWidth, imageHeight, 0), "Cuda malloc array");
//
//	CUDA_HANDLE_ERROR(cudaMalloc(&dev_inputImageBytes, sizeof(unsigned char) * imageWidth * imageHeight * 3), "Cuda malloc inputImageBytes");
//	CUDA_HANDLE_ERROR(cudaMalloc(&dev_cannyEdgeResult, sizeof(unsigned char) * imageWidth * imageHeight), "Cuda malloc dev_cannyEdgeResult");
//
//	CUDA_HANDLE_ERROR(cudaMemcpy(dev_inputImageBytes, imageBytes, sizeof(unsigned char) * imageWidth * imageHeight * 3, cudaMemcpyHostToDevice), "Cuda memcpy inputImageBytes");
//
//	float gaussianKernel5x5[5 * 5];
//
//	computeGaussianKernel(gaussianKernel5x5, 5, 1);
//	copyGaussianKernelToConstMem(gaussianKernel5x5);
//
//	rgbToGrayscale<<<128, 128>>>(dev_inputImageBytes, imageWidth, imageHeight, dev_imageGrayScale);
//	CUDA_HANDLE_ERROR(cudaFree(dev_inputImageBytes), "Cuda free dev_inputImagesBytes");
//	bindImageToTextureMemory(dev_imageGrayScaleArray, imageWidth, imageHeight);
//
//	cudaEvent_t start, stop;
//	cudaEventCreate(&start);
//	cudaEventCreate(&stop);
//
//	cudaEventRecord(start);
//	cannyEdgeDectection<<<128, 128>>>(dev_cannyEdgeResult, imageWidth, imageHeight);
//	cudaEventRecord(stop);
//	cudaEventSynchronize(stop);
//
//	float cannyEdgeDectectionDuration = 0;
//	cudaEventElapsedTime(&cannyEdgeDectectionDuration, start, stop);
//	printf("Canny edge dectection time: %.3fms\n", cannyEdgeDectectionDuration);
//
//	CUDA_HANDLE_ERROR(cudaMemcpy(imageGrayScale, dev_cannyEdgeResult, sizeof(unsigned char) * imageWidth * imageHeight, cudaMemcpyDeviceToHost), "Cuda memcpy dev_imageGrayScale to imageBytes");
//
//	grayscaleToGrayscale3Bytes2D(imageGrayScale, imageWidth, imageHeight, imageBytes);
//	//BGRToRGB(imageBytes, imageWidth, imageHeight);//TODO, this is unecessary because we can just treat the bytes in the BGR order in the code
//
//	char imageOutName[256];
//	sprintf(imageOutName, "out%s", IN_IMAGE_NAME);
//
//	generateBitmapImage(imageBytes, imageHeight, imageWidth, imageOutName);
//
//	printf("\nDone.\n");
//}
