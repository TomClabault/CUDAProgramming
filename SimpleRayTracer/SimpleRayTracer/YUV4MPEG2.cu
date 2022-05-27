#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void generateYUV4MPEG2Header(char* outHeader, int width, int height, int framesPerSecond, char* colorSpace)
{
	sprintf(outHeader, "YUV4MPEG2 W%d H%d F%d:1 C%s\n", width, height, framesPerSecond, colorSpace);
}

void addFrameToFile(char* filePath, int width, int height , unsigned char* imageBytes)
{
	FILE* output = fopen(filePath, "ab");

	char frameStart[] = "FRAME\n";
	fwrite(frameStart, sizeof(unsigned char), strlen(frameStart), output);
	fwrite(imageBytes, sizeof(unsigned char), width * height * 3, output);

	fclose(output);
}
