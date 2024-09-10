#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include "../my_common.h"

#define N_DIM		4000
#define WARP_SIZE	32

#define KERN_VECTOR_MATRIX_PRODUCT(SUFFIX,V_TYPE,M_TYPE,D_TYPE)		\
	__global__ void													\
	kern_vector_matrix_product_##SUFFIX(V_TYPE V,					\
										M_TYPE M,					\
										D_TYPE D)					\
	{																\
		__shared__ float	work[WARP_SIZE];						\
																	\
		for (int k=blockIdx.x; k < N_DIM; k += gridDim.x)			\
		{															\
			const float * __restrict__ M_line = M + k * N_DIM;		\
			double	sum = 0.0;										\
																	\
			if (threadIdx.x < WARP_SIZE)							\
				work[threadIdx.x] = 0.0;							\
			__syncthreads();										\
																	\
			for (int base=0; base < N_DIM; base += blockDim.x)		\
			{														\
				int		index = base + threadIdx.x;					\
				float	fval = 0.0;									\
																	\
				if (index < N_DIM)									\
					fval = V[index] * M_line[index];				\
																	\
				fval += __shfl_xor_sync(0xffffffffU, fval,  1);		\
				fval += __shfl_xor_sync(0xffffffffU, fval,  2);		\
				fval += __shfl_xor_sync(0xffffffffU, fval,  4);		\
				fval += __shfl_xor_sync(0xffffffffU, fval,  8);		\
				fval += __shfl_xor_sync(0xffffffffU, fval, 16);		\
				if ((threadIdx.x & (WARP_SIZE-1)) == 0)				\
					work[threadIdx.x / WARP_SIZE] = fval;			\
				__syncthreads();									\
																	\
				if (threadIdx.x < WARP_SIZE)						\
				{													\
					fval = work[threadIdx.x];						\
					fval += __shfl_xor_sync(0xffffffffU, fval,  1);	\
					fval += __shfl_xor_sync(0xffffffffU, fval,  2);	\
					fval += __shfl_xor_sync(0xffffffffU, fval,  4);	\
					fval += __shfl_xor_sync(0xffffffffU, fval,  8);	\
					fval += __shfl_xor_sync(0xffffffffU, fval, 16);	\
																	\
					sum += fval;									\
				}													\
				__syncthreads();									\
			}														\
			if (threadIdx.x == 0)									\
				D[k] = sum;											\
		}															\
	}

KERN_VECTOR_MATRIX_PRODUCT(full_cached,
						   const float * __restrict__,
						   const float * __restrict__,
						   float *)

KERN_VECTOR_MATRIX_PRODUCT(half_cached,
						   const float * __restrict__,
						   const float *,
						   float *)

KERN_VECTOR_MATRIX_PRODUCT(none_cached,
						   float *,
						   float *,
						   float *)

int main(int argc, char *argv[])
{
	float  *V, *M, *D;
	cudaEvent_t ev1, ev2, ev3, ev4;
	float	tm1, tm2, tm3;
	int		my_device;

	__(cudaEventCreate(&ev1));
	__(cudaEventCreate(&ev2));
	__(cudaEventCreate(&ev3));
	__(cudaEventCreate(&ev4));
	
	__(cudaMallocManaged(&V, sizeof(float) * N_DIM));
	__(cudaMallocManaged(&M, sizeof(float) * N_DIM * N_DIM));
	__(cudaMallocManaged(&D, sizeof(float) * N_DIM));
	srand48(time(NULL));
	for (int i=0; i < N_DIM; i++)
		V[i] = drand48();
	for (int i=0; i < N_DIM * N_DIM; i++)
		M[i] = drand48();
	__(cudaGetDevice(&my_device));
	__(cudaMemPrefetchAsync(V, sizeof(float) * N_DIM, my_device));
	__(cudaMemPrefetchAsync(M, sizeof(float) * N_DIM * N_DIM, my_device));

	cudaEventRecord(ev1);
	kern_vector_matrix_product_full_cached<<<1,320>>>(V, M, D);
	cudaEventRecord(ev2);
	kern_vector_matrix_product_half_cached<<<1,320>>>(V, M, D);
	cudaEventRecord(ev3);
	kern_vector_matrix_product_none_cached<<<1,320>>>(V, M, D);
	cudaEventRecord(ev4);
	__(cudaStreamSynchronize(NULL));

	__(cudaEventElapsedTime(&tm1, ev1, ev2));
	__(cudaEventElapsedTime(&tm2, ev2, ev3));
	__(cudaEventElapsedTime(&tm3, ev3, ev4));

	printf("kernel execution time: full-cached=%.3fms half-cached=%.3fms non-cached=%.3fms\n", tm1, tm2, tm3);

	return 0;
}
