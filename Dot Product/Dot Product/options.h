#pragma once

#define min(a, b) (a < b ? a : b)

#define VECTOR_SIZE 650000000
#define ITER 10

#define THREADS_PER_BLOCK 256 //Has to be a power of 2
#define BLOCKS_PER_GRID min(32, ((VECTOR_SIZE + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK))
