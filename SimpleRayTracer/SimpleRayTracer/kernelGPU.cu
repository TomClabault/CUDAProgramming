#ifndef __CUDACC__
#define __CUDACC__
#endif // __CUDACC__

#include "bitmap.h"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <stdio.h>
#include <stdlib.h>

#define NB_SPHERES 20

#define WIDTH 3840*4
#define HEIGHT 2160*4

#define THREAD_PER_BLOCKS 16

#define INF 2e10
#define rnd( x ) (x * rand() / RAND_MAX)

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

__global__ void rayTracerGPU(unsigned char* output)
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

		output[threadId * 3 + 0] = (int)(r*255);
		output[threadId * 3 + 1] = (int)(g*255);
		output[threadId * 3 + 2] = (int)(b*255);

		threadId += gridDim.x * gridDim.y * blockDim.x * blockDim.y;
	}
}

int benchmarkGPU()
{
	Sphere* host_spheres = (Sphere*)malloc(sizeof(Sphere) * NB_SPHERES);
	unsigned char* host_output = (unsigned char*)malloc(sizeof(unsigned char) * WIDTH * HEIGHT * 3);
	unsigned char* dev_output;

	cudaMalloc(&dev_output, sizeof(unsigned char) * WIDTH * HEIGHT * 3);

	for (int i = 0; i < NB_SPHERES; i++)
	{
		host_spheres[i].r = rnd(1.0f);
		host_spheres[i].g = rnd(1.0f);
		host_spheres[i].b = rnd(1.0f);
		host_spheres[i].x = rnd(1000.0f) - 500;
		host_spheres[i].y = rnd(1000.0f) - 500;
		host_spheres[i].z = rnd(1000.0f) - 500;
		host_spheres[i].radius = rnd(100.0f) + 20;
	}

	cudaMemcpyToSymbol(spheres, host_spheres, sizeof(Sphere) * NB_SPHERES);


	dim3 blocks((WIDTH + 15) / THREAD_PER_BLOCKS, (HEIGHT + 15) / THREAD_PER_BLOCKS);
	dim3 threads(THREAD_PER_BLOCKS, THREAD_PER_BLOCKS);

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	cudaEventRecord(start, 0);
	rayTracerGPU << <blocks, threads >> > (dev_output);
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);

	float duration;
	cudaEventElapsedTime(&duration, start, stop);

	printf("GPU Time: %.3fms\n", duration);

	cudaMemcpy(host_output, dev_output, sizeof(unsigned char) * WIDTH * HEIGHT * 3, cudaMemcpyDeviceToHost);

	generateBitmapImage(host_output, HEIGHT, WIDTH, "OutputRTGPU.bmp");

	return 0;
}
