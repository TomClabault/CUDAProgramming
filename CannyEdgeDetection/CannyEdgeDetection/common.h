#include <cuda_runtime.h>

#define CUDA_HANDLE_ERROR(errorCode, message) if (errorCode != cudaSuccess) { printf("[%s] CUDA Error: %s\n", message, cudaGetErrorString(errorCode)); exit(-1); }

#define CUDA_TICKTOCK_DECLARE() cudaEvent_t start, stop;\
								float duration;

#define CUDA_TICK()	cudaEventCreate(&start);\
					cudaEventCreate(&stop);	\
					cudaEventRecord(start);
#define CUDA_TOCK(label)	cudaEventRecord(stop);		\
							cudaEventSynchronize(stop);	\
							cudaEventElapsedTime(&duration, start, stop);\
							printf("GPU Time [%s]: %.3fms\n", label, duration);
