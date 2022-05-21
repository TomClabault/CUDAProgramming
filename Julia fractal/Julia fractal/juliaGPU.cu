/*
 * Copyright 1993-2010 NVIDIA Corporation.  All rights reserved.
 *
 * NVIDIA Corporation and its licensors retain all intellectual property and
 * proprietary rights in and to this software and related documentation.
 * Any use, reproduction, disclosure, or distribution of this software
 * and related documentation without an express license agreement from
 * NVIDIA Corporation is strictly prohibited.
 *
 * Please refer to the applicable NVIDIA end user license agreement (EULA)
 * associated with this source code for terms and conditions that govern
 * your use of this NVIDIA software.
 *
 */

#include "benchOptions.h"

#include "device_launch_parameters.h"
#include "cudart_platform.h"
#include "cuda_runtime.h"

#include <time.h>
#include <stdlib.h>
#include <stdio.h>

struct cuComplex 
{
    float   r;
    float   i;

    __device__ cuComplex(float a, float b) : r(a), i(b) {}

    __device__ float magnitude2(void) 
    { 
        return r * r + i * i; 
    }

    __device__ cuComplex operator*(const cuComplex& a) 
    {
        return cuComplex(r * a.r - i * a.i, i * a.r + r * a.i);
    }

    __device__  cuComplex operator+(const cuComplex& a)
    {
        return cuComplex(r + a.r, i + a.i);
    }
};

__device__  int juliaGPU(int x, int y)
{
    float jx = SCALE * (float)(WIDTH / 2 - x) / (WIDTH / 2);
    float jy = SCALE * (float)(HEIGHT / 2 - y) / (HEIGHT / 2);

    cuComplex c(-0.8, 0.156);
    cuComplex a(jx, jy);

    int i = 0;
    for (i = 0; i < MAX_JULIA_ITERATION; i++) {
        a = a * a + c;
        if (a.magnitude2() > 1000)
            return 0;
    }

    return 1;
}

__global__ void kernelGPU(unsigned char* ptr) 
{
    int threadId = blockIdx.x * blockDim.x + threadIdx.x;

    while (threadId < WIDTH * HEIGHT)
    {
        int x = threadId % WIDTH;
        int y = threadId / WIDTH;

        int juliaValue = juliaGPU(x, y);
        ptr[threadId * 3 + 0] = 255 * juliaValue;
        ptr[threadId * 3 + 1] = 0;
        ptr[threadId * 3 + 1] = 0;

        threadId += gridDim.x * blockDim.x;
    }
}
