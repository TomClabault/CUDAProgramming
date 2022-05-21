#include "device_launch_parameters.h"
#include "cudart_platform.h"
#include "cuda_runtime.h"

#include <time.h>
#include <stdlib.h>
#include <stdio.h>

#include "benchOptions.h"
#include "bitmap.h"
#include "juliaCPU.cuh"
#include "juliaGPU.cuh"

void writeToDisk(unsigned char* data, char* fileName)
{
    /*FILE* output = fopen(fileName, "w+");
    for (int y = 0; y < HEIGHT; y++)
    {
        for (int x = 0; x < WIDTH; x++)
        {
            fprintf(output, "%d ", data[x + y * WIDTH]);
        }
        fprintf(output, "\n");
    }
    fclose(output);*/

    generateBitmapImage(data, HEIGHT, WIDTH, fileName);
}

void juliaGPU(unsigned char* ptr)
{
    unsigned char* dev_ptr;

    if (cudaMalloc(&dev_ptr, sizeof(unsigned char) * WIDTH * HEIGHT * 3) != 0)
    {
        printf("Malloc error");

        exit(-1);
    }

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start, 0);
    for (int i = 0; i < ITER; i++)
        kernelGPU<<<128, 128 >>> (dev_ptr);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    float duration;
    cudaEventElapsedTime(&duration, start, stop);

    cudaMemcpy(ptr, dev_ptr, sizeof(unsigned char) * WIDTH * HEIGHT * 3, cudaMemcpyDeviceToHost);
    cudaFree(dev_ptr);

    printf("GPU: %.3fms\n", duration / ITER);

    writeToDisk(ptr, "OutputGPU.bmp");
}

void juliaCPU(unsigned char* ptr)
{
    clock_t start = clock();
    for (int i = 0; i < ITER; i++)
        kernelCPU(ptr);
    clock_t end = clock();

    printf("CPU: %.3fms\n", (double)(end-start) / ITER);

    writeToDisk(ptr, "OutputCPU.bmp");
}

int main(void)
{
    unsigned char* ptr = (unsigned char*)malloc(sizeof(unsigned char) * WIDTH * HEIGHT * 3);

    juliaGPU(ptr);
    juliaCPU(ptr);

    free(ptr);
}