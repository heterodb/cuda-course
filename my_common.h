#ifndef _MY_COMMON_H_
#define _MY_COMMON_H_
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <unistd.h>
#include <cuda.h>

#define get_global_id()		(blockDim.x * blockIdx.x + threadIdx.x)
#define get_global_size()	(blockDim.x * gridDim.x)
#define get_local_id()		(threadIdx.x)
#define get_local_size()	(blockDim.x)

/* error handling */
#define __(stmt)														\
	do {																\
		cudaError_t	__rc = stmt;										\
																		\
		if (__rc != cudaSuccess)										\
		{																\
			fprintf(stderr,												\
					"[%s:%d] failed on " #stmt " = %s (%s)\n",			\
					__FILE__, __LINE__,									\
					cudaGetErrorName(__rc),								\
					cudaGetErrorString(__rc));							\
			_exit(1);													\
		}																\
	} while(0)

#endif	/* _MY_COMMON_H_ */
