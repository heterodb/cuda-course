#include <stdio.h>
#include <stdlib.h>
#define NITEMS		1000000

__device__ float	dev_x[NITEMS];
__device__ double	sum_x = 0.0;

__global__ void my_gpu_average(float *x)
{
	/*
	 * 1. read all the items in x[] array
	 * 2. increment sum_x by the value read above
	 */
}

int main(int argc, char *argv[])
{
	float		host_x[NITEMS];
	double		host_sum_x = 0.0;
	double		gpu_sum_x;

	/* initialization */
	for (int i=0; i < NITEMS; i++)
	{
		host_x[i] = 100.0 * drand48();
		host_sum_x += host_x[i];
	}
	/* data transfer CPU-->GPU */
	cudaMemcpy(dev_x, host_x, sizeof(float) * NITEMS,
			   cudaMemcpyHostToDevice);
	/* launch GPU kernel */
	my_gpu_average<<<8,128>>>(dev_x);
	cudaDeviceSynchronize();
	/* data transfer GPU-->CPU */
	cudaMemcpyFromSymbol(&gpu_sum_x, sum_x, sizeof(double));

	printf("average by CPU = %f, GPU = %f\n",
		   host_sum_x / (double)NITEMS,
		   gpu_sum_x / (double)NITEMS);
	return 0;
}
