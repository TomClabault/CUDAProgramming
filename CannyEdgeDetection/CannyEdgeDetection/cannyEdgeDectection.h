#include <cuda_runtime.h>

#define GAUSSIAN_KERNEL_SIZE 15
#define GAUSSIAN_KERNEL_SIGMA 5

__host__ void cannyEdgeDetection(unsigned char* graysclaeInput, int width, int height, unsigned char* grayscaleOutput);

__host__ void computeGaussianKernel(float* gaussianKernel, int size, float sigma);
__host__ cudaError_t copyGaussianKernelToConstMem(float* gaussianKernel);
