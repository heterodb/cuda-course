#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#define NITEMS		1000000000
#define GB_PER_SEC(nbytes,usec)										\
	(((double)(nbytes) * 1000000.0) / ((double)(usec) * 1073741824.0))

int main(int argc, char *argv[])
{
	float	   *host_x;
	float	   *host_y;
	float	   *dev_z;
	size_t		length = NITEMS * sizeof(float);
	struct timeval tv1, tv2, tv3;
	double		usec1, usec2;

	/* allocation of normal host memory */
	gettimeofday(&tv1, NULL);
	host_x = (float *)calloc(NITEMS, sizeof(float));
	/* allocation of page-locked host memory */
	gettimeofday(&tv2, NULL);
	cudaMallocHost(&host_y, length);
	gettimeofday(&tv3, NULL);
	printf("malloc: %.3fms, cudaMallocHost: %.3fms\n",
		   (double)(tv2.tv_sec  - tv1.tv_sec)  * 1000.0 +
		   (double)(tv2.tv_usec - tv1.tv_usec) / 1000.0,
		   (double)(tv3.tv_sec  - tv2.tv_sec)  * 1000.0 +
		   (double)(tv3.tv_usec - tv2.tv_usec) / 1000.0);
	/* allocation of device memory */
	cudaMalloc(&dev_z, length);
	for (int count=0; count < 3; count++)
	{
		/* Test1: normal host memory <-> device memory */
		gettimeofday(&tv1, NULL);
		cudaMemcpy(dev_z, host_x, length,
				   cudaMemcpyHostToDevice);
		/* data transfer GPU-->CPU */
		cudaMemcpy(host_x, dev_z, length,
				   cudaMemcpyDeviceToHost);
		gettimeofday(&tv2, NULL);
		/* Test2: page-locked host memory <-> device memory */
		cudaMemcpy(dev_z, host_y, length,
				   cudaMemcpyHostToDevice);
		/* data transfer GPU-->CPU */
		cudaMemcpy(host_y, dev_z, length,
				   cudaMemcpyDeviceToHost);
		gettimeofday(&tv3, NULL);
		/* print */
		usec1 = ((tv2.tv_sec  - tv1.tv_sec) * 1000000 +
				 (tv2.tv_usec - tv1.tv_usec));
		usec2 = ((tv3.tv_sec  - tv2.tv_sec) * 1000000 +
				 (tv3.tv_usec - tv2.tv_usec));
		printf("normal: %.2fGB/s [%.3fms], page-locked: %.2fGB/s [%.3fms]\n",
			   GB_PER_SEC(length, usec1), usec1 / 1000.0,
			   GB_PER_SEC(length, usec2), usec2 / 1000.0);
	}
	return 0;
}
