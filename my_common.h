#ifndef _MY_COMMON_H_
#define _MY_COMMON_H_
#include <unistd.h>
#include <cuda.h>

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
