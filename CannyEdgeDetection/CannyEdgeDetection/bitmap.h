void BGRToRGB(unsigned char* imageBytes, int width, int height);

extern void generateBitmapImage(unsigned char* image, int height, int width, char* imageFileName);

void readBitmapImage(unsigned char** output, int* width, int* height, char* imageFileName);
