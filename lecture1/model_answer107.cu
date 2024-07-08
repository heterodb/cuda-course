#include <stdio.h>
#include <stdlib.h>
#define NITEMS		10000000

__managed__ double	gpu_sum_x = 0.0;

__global__ void my_gpu_average(float *x)
{
	int		index;

	for (index = blockDim.x * blockIdx.x + threadIdx.x;
		 index < NITEMS;
		 index += blockDim.x * gridDim.x)
		atomicAdd(&gpu_sum_x, x[index]);
}

int main(int argc, char *argv[])
{
	float  *fval;
	double	host_sum_x = 0.0;
	int		my_gpu;

	/* initialization */
	cudaMallocManaged(&fval, NITEMS * sizeof(float));
	for (int i=0; i < NITEMS; i++)
	{
		fval[i] = 100.0 * drand48();
		host_sum_x += fval[i];
	}
	/* preferch managed memory */
	cudaGetDevice(&my_gpu);
	cudaMemPrefetchAsync(fval, NITEMS * sizeof(float), my_gpu);
	/* launch GPU kernel */
	my_gpu_average<<<8,128>>>(fval);
	cudaDeviceSynchronize();

	/* fetch result */
	printf("average by CPU = %f, GPU = %f\n",
		   host_sum_x / (double)NITEMS,
		   gpu_sum_x / (double)NITEMS);
	return 0;
}
