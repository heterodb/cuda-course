#include <stdio.h>
#include <stdlib.h>
#define NITEMS		100

__device__ float	dev_r[NITEMS], dev_x[NITEMS], dev_y[NITEMS];

__global__ void my_gpu_func(float *r, float *x, float *y)
{
	int		index = blockDim.x * blockIdx.x + threadIdx.x;

	if (index < NITEMS)
		r[index] = x[index] + y[index];
}

int main(int argc, char *argv[])
{
	float		host_r[NITEMS], host_x[NITEMS], host_y[NITEMS];

	for (int i=0; i < NITEMS; i++)
	{
		host_x[i] = 100.0 * drand48();
		host_y[i] = 100.0 * drand48();
	}
	cudaMemcpy(dev_x, host_x, sizeof(float) * NITEMS,
			   cudaMemcpyHostToDevice);
	cudaMemcpy(dev_y, host_y, sizeof(float) * NITEMS,
			   cudaMemcpyHostToDevice);
	my_gpu_func<<<8,128>>>(dev_r, dev_x, dev_y);
	cudaDeviceSynchronize();
	cudaMemcpy(host_r, dev_r, sizeof(float) * NITEMS,
			   cudaMemcpyDeviceToHost);
	for (int i=0; i < NITEMS; i++)
		printf("%d: %f * %f = %f\n", i, host_x[i], host_y[i], host_r[i]);
	return 0;
}
