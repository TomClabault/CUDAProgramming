#include "device_launch_parameters.h"
#include "cudart_platform.h"
#include "cuda_runtime.h"

#include <time.h>
#include <stdlib.h>
#include <stdio.h>

#include "benchOptions.h"

struct cuComplex
{
    float   r;
    float   i;

    cuComplex(float a, float b) : r(a), i(b) {}

    float magnitude2(void)
    {
        return r * r + i * i;
    }

    cuComplex operator*(const cuComplex& a)
    {
        return cuComplex(r * a.r - i * a.i, i * a.r + r * a.i);
    }

    cuComplex operator+(const cuComplex& a)
    {
        return cuComplex(r + a.r, i + a.i);
    }
};

int juliaCPU(int x, int y)
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

void kernelCPU(unsigned char* ptr)
{
    for (int y = 0; y < HEIGHT; y++)
    {
        for (int x = 0; x < WIDTH; x++)
        {
            int offset = x + y * WIDTH;

            int juliaValue = juliaCPU(x, y);
            ptr[offset * 3 + 0] = 255 * juliaValue;
            ptr[offset * 3 + 1] = 0;
            ptr[offset * 3 + 1] = 0;
        }
    }
}
