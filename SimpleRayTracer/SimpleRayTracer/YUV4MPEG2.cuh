void generateYUV4MPEG2Header(char* outHeader, int width, int height, int framesPerSecond, char* colorSpace);

void addFrameToFile(char* filePath, int width, int height, unsigned char* imageBytes);