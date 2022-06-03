#include <cuda_runtime.h>

#define CUDA_HANDLE_ERROR(errorCode, message) if (errorCode != cudaSuccess) { printf("[%s] CUDA Error: %s\n", message, cudaGetErrorString(errorCode)); exit(-1); }

#define CUDA_TIME_EXECUTION(label, ...) \
	do {\
		cudaEvent_t start, stop;\
		float duration;\
		\
		cudaEventCreate(&start);\
		cudaEventCreate(&stop);	\
		cudaEventRecord(start);\
		__VA_ARGS__;\
		cudaEventRecord(stop);\
		cudaEventSynchronize(stop);	\
		cudaEventElapsedTime(&duration, start, stop);\
		printf("GPU Time [%s]: %.3fms\n", label, duration);\
	} while(0)

#define BENCHMARK_BLOCK_SIZE_THREAD_COUNT(iteration, blockSizeLimit, threadCountLimit, ...) do {\
	float minAverageTime = 10000000;\
	int bestBlockSize = 0;\
	int bestThreadsCount = 0;\
	\
	for (int blockSize = 8; blockSize < blockSizeLimit; blockSize += 16)\
	{\
		for (int threadCount = 8; threadCount < threadCountLimit; threadCount += 8)\
		{\
			float totalIterationsDuration = 0.0;\
			\
			for (int i = 0; i < iteration; i++)\
			{\
				cudaEvent_t start, stop;\
				cudaEventCreate(&start);\
				cudaEventCreate(&stop);\
				\
				cudaEventRecord(start);\
				__VA_ARGS__;\
				cudaEventRecord(stop);\
				cudaEventSynchronize(stop);\
				\
				float iterationDuration = 0;\
				cudaEventElapsedTime(&iterationDuration, start, stop);\
				cudaEventDestroy(start);\
				cudaEventDestroy(stop);\
				\
				totalIterationsDuration += iterationDuration;\
			}\
			\
			printf("[%03d, %03d] took %.3fms\n", blockSize, threadCount, totalIterationsDuration / iteration);\
			\
			if (totalIterationsDuration / iteration < minAverageTime)\
			{\
				bestBlockSize = blockSize;\
				bestThreadsCount = threadCount;\
				\
				minAverageTime = totalIterationsDuration / iteration;\
			}\
		}\
	}\
	printf("Best settings are: [%d, %d] with %.3fms\n", bestBlockSize, bestThreadsCount, minAverageTime);\
	} while(0)
