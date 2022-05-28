#include <stdio.h>
#include <stdlib.h>

const int BYTES_PER_PIXEL = 3; /// red, green, & blue
const int FILE_HEADER_SIZE = 14;
const int INFO_HEADER_SIZE = 40;

unsigned char* createBitmapFileHeader(int height, int stride)
{
    int fileSize = FILE_HEADER_SIZE + INFO_HEADER_SIZE + (stride * height);

    static unsigned char fileHeader[] = {
        0,0,     /// signature
        0,0,0,0, /// image file size in bytes
        0,0,0,0, /// reserved
        0,0,0,0, /// start of pixel array
    };

    fileHeader[0] = (unsigned char)('B');
    fileHeader[1] = (unsigned char)('M');
    fileHeader[2] = (unsigned char)(fileSize);
    fileHeader[3] = (unsigned char)(fileSize >> 8);
    fileHeader[4] = (unsigned char)(fileSize >> 16);
    fileHeader[5] = (unsigned char)(fileSize >> 24);
    fileHeader[10] = (unsigned char)(FILE_HEADER_SIZE + INFO_HEADER_SIZE);

    return fileHeader;
}

unsigned char* createBitmapInfoHeader(int height, int width)
{
    static unsigned char infoHeader[] = {
        0,0,0,0, /// header size
        0,0,0,0, /// image width
        0,0,0,0, /// image height
        0,0,     /// number of color planes
        0,0,     /// bits per pixel
        0,0,0,0, /// compression
        0,0,0,0, /// image size
        0,0,0,0, /// horizontal resolution
        0,0,0,0, /// vertical resolution
        0,0,0,0, /// colors in color table
        0,0,0,0, /// important color count
    };

    infoHeader[0] = (unsigned char)(INFO_HEADER_SIZE);
    infoHeader[4] = (unsigned char)(width);
    infoHeader[5] = (unsigned char)(width >> 8);
    infoHeader[6] = (unsigned char)(width >> 16);
    infoHeader[7] = (unsigned char)(width >> 24);
    infoHeader[8] = (unsigned char)(height);
    infoHeader[9] = (unsigned char)(height >> 8);
    infoHeader[10] = (unsigned char)(height >> 16);
    infoHeader[11] = (unsigned char)(height >> 24);
    infoHeader[12] = (unsigned char)(1);
    infoHeader[14] = (unsigned char)(BYTES_PER_PIXEL * 8);

    return infoHeader;
}

void generateBitmapImage(unsigned char* image, int height, int width, char* imageFileName)
{
    int widthInBytes = width * BYTES_PER_PIXEL;

    unsigned char padding[3] = { 0, 0, 0 };
    int paddingSize = (4 - (widthInBytes) % 4) % 4;

    int stride = (widthInBytes)+paddingSize;

    FILE* imageFile = fopen(imageFileName, "wb");

    unsigned char* fileHeader = createBitmapFileHeader(height, stride);
    fwrite(fileHeader, 1, FILE_HEADER_SIZE, imageFile);

    unsigned char* infoHeader = createBitmapInfoHeader(height, width);
    fwrite(infoHeader, 1, INFO_HEADER_SIZE, imageFile);

    int i;
    for (i = 0; i < height; i++) {
        fwrite(image + (i * widthInBytes), BYTES_PER_PIXEL, width, imageFile);
        fwrite(padding, 1, paddingSize, imageFile);
    }

    fclose(imageFile);
}

void readWidthHeight(FILE* inputFile, int* width, int* height)
{
    int infoHeaderSize;

    fseek(inputFile, FILE_HEADER_SIZE, SEEK_SET);

    fread(&infoHeaderSize, sizeof(int), 1, inputFile);
    fread(width, sizeof(int), 1, inputFile);
    fread(height, sizeof(int), 1, inputFile);
    fseek(inputFile, infoHeaderSize - 12, SEEK_CUR);//-12 because we already read 3 ints
}

/*
* Reads a bmp image from disk, allocates the necessary space in 'output' and fills 'output' with the image bytes.
* Also fills 'width' and 'height' with the width and the height of the read image
*/
void readBitmapImage(unsigned char** output, int* width, int* height, char* imageFileName)
{
    FILE* inputFile = fopen(imageFileName, "rb");
    readWidthHeight(inputFile, width, height);

    *output = (unsigned char*)malloc(sizeof(unsigned char) * *width * *height * 3);
    if (*output == NULL)
        return;

    int padding = (*width * 3) % 4;
    if(padding != 0)
        for (int i = 0; i < *height; i++)
        {
            fread(*output, sizeof(unsigned char), *width * 3, inputFile);
            fseek(inputFile, padding, SEEK_CUR);//Skipping the padding
        }
    else
        fread(*output, sizeof(unsigned char), *width * *height * 3, inputFile);

}

void BGRToRGB(unsigned char* imageBytes, int width, int height)
{
    for (int y = 0; y < height; y++)
    {
        for (int x = 0; x < width; x++)
        {
            int offset = (x + y * width) * 3;

            int currentBlue  = imageBytes[offset + 0];
            int currentRed   = imageBytes[offset + 2];

            imageBytes[offset + 0] = currentRed;
            imageBytes[offset + 2] = currentBlue;
        }
    }
}
