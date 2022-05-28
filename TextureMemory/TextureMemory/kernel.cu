#ifndef __CUDACC__
#define __CUDACC__
#endif

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <stdio.h>
#include <stdlib.h>

#include "bitmap.h"

#define NB_FRAMES 40
#define NB_HEATERS 10
#define HEATER_SIZE 3//Each heater will be HEATER_SIZE * HEATER_SIZE
#define HEATER_BASE_HEAT 5.0

#define SIMULATION_WIDTH 1280
#define SIMULATION_HEIGHT 720

#define max(a, b) (a > b ? a : b)

#define THREADS_PER_BLOCK 32
#define BLOCKS_PER_GRID max(((SIMULATION_WIDTH + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK), ((SIMULATION_HEIGHT + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK))

__global__ void copyConstHeaters(float* inFrameTex, const float* constantHeaters)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int offset = x + y * gridDim.x * blockDim.x;

	while (offset < SIMULATION_WIDTH * SIMULATION_HEIGHT)
	{
		if (constantHeaters[offset] != 0)
			inFrameTex[offset] = constantHeaters[offset];

		offset += gridDim.x * gridDim.y * blockDim.x * blockDim.y;
	}
}

texture<float> constHeatersTex;
texture<float> inFrameTex;
texture<float> outFrameTex;

__global__ void computeFrame(float* out, bool dest)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int offset = x + y * gridDim.x * blockDim.x;

	while (offset < SIMULATION_WIDTH * SIMULATION_HEIGHT)
	{
		int left;
		int right;
		int top;
		int bottom;

		int current;
		
		if (dest)
		{
			left = tex1Dfetch(inFrameTex, offset - 1);
			right = tex1Dfetch(inFrameTex, offset + 1);
			top = tex1Dfetch(inFrameTex, offset - SIMULATION_WIDTH);
			bottom = tex1Dfetch(inFrameTex, offset + SIMULATION_WIDTH);
			current = tex1Dfetch(inFrameTex, offset);
		}
		else
		{
			left = tex1Dfetch(outFrameTex, offset - 1);
			right = tex1Dfetch(outFrameTex, offset + 1);
			top = tex1Dfetch(outFrameTex, offset - SIMULATION_WIDTH);
			bottom = tex1Dfetch(outFrameTex, offset + SIMULATION_WIDTH);
			current = tex1Dfetch(outFrameTex, offset);
		}

		out[offset] = current + 4 * (left + right + top + bottom -4 * current);

		offset += gridDim.x * gridDim.y * blockDim.x * blockDim.y;
	}
}

void floatToRGB(float* inFloat, int width, int height, unsigned char* outRGB)
{
	for(int y = 0; y < height; y++)
	{
		for (int x = 0; x < width; x++)
		{
			int index = x + y * width;
			int floatGrey = (int)(inFloat[x + y * width] * 255);

			outRGB[index * 3 + 0] = floatGrey;
			outRGB[index * 3 + 1] = floatGrey;
			outRGB[index * 3 + 2] = floatGrey;
		}
	}
}

int main()
{
	float* constant_heaters = (float*)calloc(SIMULATION_HEIGHT * SIMULATION_WIDTH, sizeof(float));
	float* hostFrameBuffer = (float*)malloc(sizeof(float) * SIMULATION_WIDTH * SIMULATION_HEIGHT);
	unsigned char* current_frame_rgb = (unsigned char*)malloc(sizeof(unsigned char*) * SIMULATION_WIDTH * SIMULATION_HEIGHT * 3);

	for (int i = 0; i < NB_HEATERS; i++)//Generating random heaters positions
	{
		//This is not a good way to do it because two heaters could be generated at the same place...

		int randX = rand() % SIMULATION_WIDTH;
		int randY = rand() % SIMULATION_HEIGHT;

		for (int heaterSizeY = 0; heaterSizeY < HEATER_SIZE; heaterSizeY++)
		{
			for (int heaterSizeX = 0; heaterSizeX < HEATER_SIZE; heaterSizeX++)
			{
				int finalY = randY - HEATER_SIZE / 2 + heaterSizeY;
				int finalX = randX - HEATER_SIZE / 2 + heaterSizeX;

				constant_heaters[finalY * SIMULATION_WIDTH + finalX] = HEATER_BASE_HEAT;
			}
		}
	}

	float* device_constant_heaters, *device_current_frame, *device_out_frame;
	cudaMalloc(&device_constant_heaters, sizeof(float) * SIMULATION_WIDTH * SIMULATION_HEIGHT);
	cudaMalloc(&device_out_frame, sizeof(float) * SIMULATION_WIDTH * SIMULATION_HEIGHT);
	cudaMalloc(&device_current_frame, sizeof(float) * SIMULATION_WIDTH * SIMULATION_HEIGHT);
	cudaBindTexture(NULL, constHeatersTex, device_constant_heaters, sizeof(float) * SIMULATION_WIDTH * SIMULATION_HEIGHT);
	cudaBindTexture(NULL, inFrameTex, device_current_frame, sizeof(float) * SIMULATION_WIDTH * SIMULATION_HEIGHT);
	cudaBindTexture(NULL, outFrameTex, device_out_frame, sizeof(float) * SIMULATION_WIDTH * SIMULATION_HEIGHT);

	cudaMemcpy(device_constant_heaters, constant_heaters, sizeof(float) * SIMULATION_WIDTH * SIMULATION_HEIGHT, cudaMemcpyHostToDevice);


	//Writing the very first frame
	copyConstHeaters << <dim3(BLOCKS_PER_GRID, BLOCKS_PER_GRID), dim3(THREADS_PER_BLOCK, THREADS_PER_BLOCK) >> > (device_current_frame, device_constant_heaters);
	cudaMemcpy(hostFrameBuffer, device_current_frame, sizeof(float) * SIMULATION_WIDTH * SIMULATION_HEIGHT, cudaMemcpyDeviceToHost);
	char bitmapFileName[128];
	sprintf(bitmapFileName, "Output%d.bmp", 0);
	floatToRGB(hostFrameBuffer, SIMULATION_WIDTH, SIMULATION_HEIGHT, current_frame_rgb);
	generateBitmapImage(current_frame_rgb, SIMULATION_HEIGHT, SIMULATION_WIDTH, bitmapFileName);

	float totalDuration = 0;
	bool destFlag = true;
	for (int frame = 0; frame < NB_FRAMES; frame++)
	{
		cudaEvent_t start, stop;
		cudaEventCreate(&start);
		cudaEventCreate(&stop);

		cudaEventRecord(start, 0);

		float* inFrame, *outFrame;
		if (destFlag)
		{
			inFrame = device_current_frame;
			outFrame = device_out_frame;
		}
		else
		{
			inFrame = device_out_frame;
			outFrame = device_current_frame;
		}

		copyConstHeaters << <dim3(BLOCKS_PER_GRID, BLOCKS_PER_GRID), dim3(THREADS_PER_BLOCK, THREADS_PER_BLOCK) >> > (inFrame, device_constant_heaters);
		computeFrame << <dim3(BLOCKS_PER_GRID, BLOCKS_PER_GRID), dim3(THREADS_PER_BLOCK, THREADS_PER_BLOCK) >> > (outFrame, destFlag);
		cudaMemcpy(hostFrameBuffer, outFrame, sizeof(float) * SIMULATION_WIDTH * SIMULATION_HEIGHT, cudaMemcpyDeviceToHost);
		cudaEventRecord(stop, 0);
		cudaEventSynchronize(stop);

		float localDuration;
		cudaEventElapsedTime(&localDuration, start, stop);

		totalDuration += localDuration;

		char bitmapFileName[128];
		sprintf(bitmapFileName, "Output%d.bmp", frame + 1);
		floatToRGB(hostFrameBuffer, SIMULATION_WIDTH, SIMULATION_HEIGHT, current_frame_rgb);
		generateBitmapImage(current_frame_rgb, SIMULATION_HEIGHT, SIMULATION_WIDTH, bitmapFileName);

		destFlag = !destFlag;
	}

	printf("GPU Time: %.3fms\n", totalDuration);
	printf("Per frame: %.3fms\n", totalDuration / NB_FRAMES);
}