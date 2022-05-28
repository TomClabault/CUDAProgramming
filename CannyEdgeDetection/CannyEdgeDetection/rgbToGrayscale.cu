#include <cuda_runtime.h>
#include <device_launch_parameters.h>

__global__ void rgbToGrayscale(unsigned char* input, int width, int height, unsigned char* output)
{
	int offset = threadIdx.x;

	while (offset < width * height)
	{
		unsigned char red = input[offset * 3 + 0];
		unsigned char green = input[offset * 3 + 1];
		unsigned char blue = input[offset * 3 + 2];

		output[offset] = red * 0.299 + green * 0.587 + blue * 0.114;

		offset += gridDim.x * blockDim.x;
	}
}
