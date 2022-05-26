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

#include <math.h>
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
    const int rainbowColors[6][3] =
    {
        {255, 0, 0},   //Red
        {255, 255, 0}, //Yellow
        {0, 255, 0},   //Green
        {0, 255, 255}, //Cyan
        {0, 0, 255},   //Blue
        {255, 0, 255}, //Magenta
    };

    //TODO calculer x et Y d'abord et ensuite thread id, ça évitera le modulo et la div dans le while
    int threadId = gridDim.x * blockDim.x * blockDim.y * blockIdx.y + blockIdx.x * blockDim.x * blockDim.y + threadIdx.x + threadIdx.y * blockDim.x;

    while(threadId < WIDTH * HEIGHT)
    {
        int x = threadId % WIDTH;
        int y = threadId / WIDTH;

        int juliaValue = juliaGPU(x, y);

        float RGBpercent = (float)x / (WIDTH - 1);
        int firstColorIndex = (RGBpercent != 0) * ((int)ceil(RGBpercent * 6) - 1);

        int color[3] = { 0, 0, 0 };
        int secondColorIndex = (firstColorIndex + 1) % 6;

        float lowerBound = ((float)firstColorIndex / 6) * (firstColorIndex != 0);
        float upperBound = 1 * (firstColorIndex + 1 == 6) + ((float)secondColorIndex / 6) * (firstColorIndex + 1 != 6);//TODO remplacer par lower bound + 1/6

        float lerpPercent = 1.0 / ((upperBound - lowerBound) / (RGBpercent - lowerBound));

        color[0] = rainbowColors[firstColorIndex][0] * (1 - lerpPercent) + rainbowColors[secondColorIndex][0] * lerpPercent;
        color[1] = rainbowColors[firstColorIndex][1] * (1 - lerpPercent) + rainbowColors[secondColorIndex][1] * lerpPercent;
        color[2] = rainbowColors[firstColorIndex][2] * (1 - lerpPercent) + rainbowColors[secondColorIndex][2] * lerpPercent;

        ptr[threadId * 3 + 0] = color[0] * juliaValue;
        ptr[threadId * 3 + 1] = color[1] * juliaValue;
        ptr[threadId * 3 + 2] = color[2] * juliaValue;

        threadId += gridDim.x * gridDim.y * blockDim.x * blockDim.y;
    }
}
