#include <cuda_runtime.h>

__host__ void cannyEdgeDetection(unsigned char* graysclaeInput, int width, int height, unsigned char* grayscaleOutput);
//__global__ void cannyEdgeDectection2D(unsigned char* grayscaleOutput, int width, int height);

__host__ void computeGaussianKernel(float* gaussianKernel, int size, float sigma);
__host__ void copyGaussianKernelToConstMem(float* gaussianKernel);
