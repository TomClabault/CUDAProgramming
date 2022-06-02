void BGRToRGB(unsigned char* imageBytes, int width, int height);
void BGRToRGB2D(unsigned char** imageBytes, int width, int height);

void generateBitmapImage(unsigned char* image, int height, int width, char* imageFileName);
void generateBitmapImageFrom2D(unsigned char** image, int height, int width, char* imageFileName);
void grayscaleToGrayscale3Bytes(unsigned char* grayscaleInput, int width, int height, unsigned char* grayscale3Output);

void readBitmapImage(unsigned char** output, int* width, int* height, char* imageFileName);
void readBitmapImage2D(unsigned char*** output, int* width, int* height, char* imageFileName);

