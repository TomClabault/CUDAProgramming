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

#define INF 2e10
#define rnd( x ) (x * rand() / RAND_MAX)

typedef struct Sphere
{
	float x, y, z;

	float radius;

	float r, g, b;

	float hit(float ox, float oy, float* n)
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

void rayTracerCPU(Sphere* spheres, unsigned char* output)
{
	for (int y = 0; y < HEIGHT; y++)
	{
		for (int x = 0; x < WIDTH; x++)
		{
			float originX = x - WIDTH / 2;
			float originY = y - HEIGHT / 2;

			float r = 0;
			float g = 0;
			float b = 0;
			float maxz = -INF;

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

			int pixelIndex = x + y * WIDTH;
			output[pixelIndex * 3 + 0] = (int)(r * 255);
			output[pixelIndex * 3 + 1] = (int)(g * 255);
			output[pixelIndex * 3 + 2] = (int)(b * 255);
		}
	}
}

int benchmarkCPU()
{
	Sphere* host_spheres = (Sphere*)malloc(sizeof(Sphere) * NB_SPHERES);
	unsigned char* host_output = (unsigned char*)malloc(sizeof(unsigned char) * WIDTH * HEIGHT * 3);

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

	clock_t start = clock();
	rayTracerCPU(host_spheres, host_output);
	clock_t end = clock();
	
	float duration = end-start;

	printf("CPU Time: %.3fms\n", duration);

	generateBitmapImage(host_output, HEIGHT, WIDTH, "OutputRTCPU.bmp");

	return 0;
}
