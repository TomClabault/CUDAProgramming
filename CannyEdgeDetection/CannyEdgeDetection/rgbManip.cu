#include <cuda_runtime.h>
#include <device_launch_parameters.h>

__global__ void cuda_rgbToGrayscale(unsigned char* input, int width, int height, unsigned char* output)
{
	int offset = threadIdx.x + blockIdx.x * blockDim.x;

	while (offset < width * height)
	{
		unsigned char red = input[offset * 3 + 0];
		unsigned char green = input[offset * 3 + 1];
		unsigned char blue = input[offset * 3 + 2];

		output[offset] = red * 0.299 + green * 0.587 + blue * 0.114;

		offset += gridDim.x * blockDim.x;
	}
}

__global__ void cuda_rgbToGrayscale2D(unsigned char** input, int width, int height, unsigned char** output)
{
	int offset = threadIdx.x + blockIdx.x * blockDim.x + threadIdx.y + blockIdx.y * blockDim.y * gridDim.x * blockDim.x * blockDim.y;

	while (offset < width * height)
	{
		int x = offset % width;
		int y = offset / width;

		unsigned char red   = input[y][x + 0];
		unsigned char green = input[y][x + 1];
		unsigned char blue  = input[y][x + 2];

		output[y][x] = red * 0.299 + green * 0.587 + blue * 0.114;

		offset += gridDim.x * blockDim.x;
	}
}

__host__ void BGRToRGB(unsigned char* imageBytes, int width, int height)
{
    for (int y = 0; y < height; y++)
    {
        for (int x = 0; x < width; x++)
        {
            int offset = (x + y * width) * 3;

            int currentBlue = imageBytes[offset + 0];
            int currentRed = imageBytes[offset + 2];

            imageBytes[offset + 0] = currentRed;
            imageBytes[offset + 2] = currentBlue;
        }
    }
}

__host__ void BGRToRGB2D(unsigned char** imageBytes, int width, int height)
{
    for (int y = 0; y < height; y++)
    {
        for (int x = 0; x < width; x++)
        {
            int currentBlue = imageBytes[y][x];
            int currentRed = imageBytes[y][x];

            imageBytes[y][x] = currentRed;
            imageBytes[y][x] = currentBlue;
        }
    }
}

__global__ void cuda_grayBytesToGray3Bytes(unsigned char* gray1ByteInput, int width, int height, unsigned char* gray3BytesOutput)
{
    int offset = threadIdx.x + blockIdx.x * blockDim.x;

    while (offset < width * height)
    {
        gray3BytesOutput[offset * 3 + 0] = gray1ByteInput[offset];
        gray3BytesOutput[offset * 3 + 1] = gray1ByteInput[offset];
        gray3BytesOutput[offset * 3 + 2] = gray1ByteInput[offset];

        offset += blockDim.x * gridDim.x;
    }
}

__host__ void grayscaleToGrayscale3Bytes(unsigned char* grayscaleInput, int width, int height, unsigned char* grayscale3Output)
{
    for (int y = 0; y < height; y++)
    {
        for (int x = 0; x < width; x++)
        {
            int index = y * width + x;

            grayscale3Output[index * 3 + 0] = grayscaleInput[index];
            grayscale3Output[index * 3 + 1] = grayscaleInput[index];
            grayscale3Output[index * 3 + 2] = grayscaleInput[index];
        }
    }
}

__host__ void grayscaleToGrayscale3Bytes2D(unsigned char** grayscaleInput, int width, int height, unsigned char** grayscale3Output)
{
    for (int y = 0; y < height; y++)
    {
        for (int x = 0; x < width; x++)
        {
            grayscale3Output[y][x + 0] = grayscaleInput[y][x];
            grayscale3Output[y][x + 1] = grayscaleInput[y][x];
            grayscale3Output[y][x + 2] = grayscaleInput[y][x];
        }
    }
}
