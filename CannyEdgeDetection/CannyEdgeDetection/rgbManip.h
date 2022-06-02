#include <cuda_runtime.h>

__global__ void cuda_rgbToGrayscale(unsigned char* input, int width, int height, unsigned char* output);
__global__ void cuda_rgbToGrayscale2D(unsigned char** input, int width, int height, unsigned char** output);

__host__ void BGRToRGB(unsigned char* imageBytes, int width, int height);
__host__ void BGRToRGB2D(unsigned char** imageBytes, int width, int height);

__global__ void cuda_grayBytesToGray3Bytes(unsigned char* gray1ByteInput, int width, int height, unsigned char* gray3BytesOutput);

__host__ void grayscaleToGrayscale3Bytes(unsigned char* grayscaleInput, int width, int height, unsigned char* grayscale3Output);
__host__ void grayscaleToGrayscale3Bytes2D(unsigned char** grayscaleInput, int width, int height, unsigned char** grayscale3Output);
