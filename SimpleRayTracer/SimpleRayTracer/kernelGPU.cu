#ifndef __CUDACC__
#define __CUDACC__
#endif // __CUDACC__

#include "bitmap.h"
#include "YUV4MPEG2.cuh"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <stdio.h>
#include <stdlib.h>

#define WIDTH 1280
#define HEIGHT 720

#define NB_FRAMES 5
#define FRAMES_PER_SECOND 1

#define NB_SPHERES 20

#define THREAD_PER_BLOCKS 32

#define INF 2e10
#define rnd( x ) (x * rand() / RAND_MAX)

#define HANDLE_ERROR(code) if(code != 0) printf("Error code: %d\n", code);

typedef struct Sphere
{
	float x, y, z;

	float radius;

	float r, g, b;

	__device__ float hit(float ox, float oy, float* n) 
	{
		float dx = ox - x;
		float dy = oy - y;
		if (dx * dx + dy * dy < radius * radius) {
			float dz = sqrtf(radius * radius - dx * dx - dy * dy);
			*n = dz / sqrtf(radius * radius);
			return dz + z;
		}
		return -INF;
	}

} Sphere;

__constant__ Sphere spheres[NB_SPHERES];

__global__ void generateRTFrameYCbCr(unsigned char* output)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int threadId = x + y * blockDim.x * gridDim.x;

	float originX = x - WIDTH / 2;
	float originY = y - HEIGHT / 2;

	float r = 0;
	float g = 0;
	float b = 0;
	float maxz = -INF;

	while (threadId < WIDTH * HEIGHT && x < WIDTH && y < HEIGHT)
	{
		for (int i = 0; i < NB_SPHERES; i++)
		{
			float n;
			float t = spheres[i].hit(originX, originY, &n);

			if (t > maxz)
			{
				float fscale = n;

				r = spheres[i].r * fscale;
				g = spheres[i].g * fscale;
				b = spheres[i].b * fscale;
			}
		}

		output[threadId * 3 + 0] = (int)(r * 255),
		output[threadId * 3 + 1] = (int)(b * 255),
		output[threadId * 3 + 2] = (int)(r * 255),

		/*output[threadId * 3 + 0] = (int)(0.299 * rInt + 0.587 * gInt + 0.114 * bInt);
		output[threadId * 3 + 1] = (int)(- 0.1687 * rInt - 0.3313 * gInt + 0.5 * bInt + 128);
		output[threadId * 3 + 2] = (int)(0.5 * rInt -0.4187 * gInt - 0.0813 * bInt + 128);*/

		threadId += gridDim.x * gridDim.y * blockDim.x * blockDim.y;
	}
}

int benchmarkGPU()
{
	char header[128];
	generateYUV4MPEG2Header(header, WIDTH, HEIGHT, FRAMES_PER_SECOND, "444");

	FILE* output = fopen("OutputVideo.y4m", "w+");
	fwrite(header, sizeof(unsigned char), strlen(header), output);
	fclose(output);

	Sphere* host_spheres = (Sphere*)malloc(sizeof(Sphere) * NB_SPHERES);

	unsigned char* host_output = (unsigned char*)malloc(sizeof(unsigned char) * WIDTH * HEIGHT * 3);
	unsigned char* dev_output;
	HANDLE_ERROR(cudaMalloc(&dev_output, sizeof(unsigned char) * WIDTH * HEIGHT * 3));
	
	dim3 blocks((WIDTH + 15) / THREAD_PER_BLOCKS, (HEIGHT + 15) / THREAD_PER_BLOCKS);
	dim3 threads(THREAD_PER_BLOCKS, THREAD_PER_BLOCKS);
	for (int frame = 0; frame < NB_FRAMES; frame++)
	{
		for (int i = 0; i < NB_SPHERES; i++)
		{
			host_spheres[i].r = rnd(1.0f);
			host_spheres[i].g = rnd(1.0f);
			host_spheres[i].b = rnd(1.0f);
			host_spheres[i].x = rnd(1000) - 500;
			host_spheres[i].y = rnd(1000) - 500;
			host_spheres[i].z = rnd(1000) - 500;
			host_spheres[i].radius = rnd(100.0f) + 20;
		}

		cudaMemcpyToSymbol(spheres, host_spheres, sizeof(Sphere) * NB_SPHERES);

		generateRTFrameYCbCr <<<blocks, threads>>> (dev_output);

		cudaMemcpy(host_output, dev_output, sizeof(unsigned char) * WIDTH * HEIGHT * 3, cudaMemcpyDeviceToHost);

		addFrameToFile((char*)"OutputVideo.y4m", WIDTH, HEIGHT, host_output);
	}

	system("start OutputVideo.y4m");
	return 0;
}
