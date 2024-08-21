#include <stdio.h>
#include <stdlib.h>
#include "../my_common.h"
#define NITEMS		1000000

__device__ double	sum_x = 0.0;

__global__ void my_gpu_average(float *x)
{
	int		index;

	for (index = blockDim.x * blockIdx.x + threadIdx.x;
		 index < NITEMS;
		 index += blockDim.x * gridDim.x)
		atomicAdd(&sum_x, x[index]);
}

int main(int argc, char *argv[])
{
	float	   *host_x = (float *)calloc(NITEMS, sizeof(float));
	float	   *dev_x;
	double		host_sum_x = 0.0;
	double		gpu_sum_x;

	/* initialization */
	for (int i=0; i < NITEMS; i++)
	{
		host_x[i] = 100.0 * drand48();
		host_sum_x += host_x[i];
	}
	/* data transfer CPU-->GPU */
	__(cudaMalloc(&dev_x, sizeof(float) * NITEMS));
	__(cudaMemcpy(dev_x, host_x, sizeof(float) * NITEMS,
				  cudaMemcpyHostToDevice));
	/* launch GPU kernel */
	my_gpu_average<<<8,128>>>(dev_x);
	__(cudaDeviceSynchronize());
	/* data transfer GPU-->CPU */
	__(cudaMemcpyFromSymbol(&gpu_sum_x, sum_x, sizeof(double)));

	printf("average by CPU = %f, GPU = %f\n",
		   host_sum_x / (double)NITEMS,
		   gpu_sum_x / (double)NITEMS);
	return 0;
}
