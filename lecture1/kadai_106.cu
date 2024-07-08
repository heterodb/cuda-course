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
	struct timeval tv1, tv2;
	double		usec1, usec2;

	/* allocation of normal host memory */
	gettimeofday(&tv1, NULL);
	host_x = (float *)calloc(NITEMS, sizeof(float));
	gettimeofday(&tv2, NULL);
	usec1 = ((tv2.tv_sec  - tv1.tv_sec) * 1000000 +
			 (tv2.tv_usec - tv1.tv_usec));

	/* allocation of page-locked host memory */
	// cudaMallocHost を用いて page-locked host memory を
	// 割り当て、その前後の時刻を gettimeofday(2) を用いて
	// 計測する。
	usec2 = ((tv2.tv_sec  - tv1.tv_sec) * 1000000 +
			 (tv2.tv_usec - tv1.tv_usec));
	printf("malloc: %.3fms, cudaMallocHost: %.3fms\n",
		   usec1 / 1000.0,
		   usec2 / 1000.0);
	/* allocation of device memory */
	cudaMalloc(&dev_z, length);
	for (int count=0; count < 3; count++)
	{
		/* Test1: normal host memory <-> device memory */
		gettimeofday(&tv1, NULL);
		// 通常のホストメモリ host_x から dev_z にデータを
		// 転送し、その直後、何もせずに dev_z から host_x に
		// データを書き戻す。
		// その前後の時刻を gettimeofday(2) を用いてtv1, tv2に
		// 記録する。
		//
		gettimeofday(&tv2, NULL);
		usec1 = ((tv2.tv_sec  - tv1.tv_sec) * 1000000 +
				 (tv2.tv_usec - tv1.tv_usec));
		/* Test2: page-locked host memory <-> device memory */
		// page-locked メモリ host_y から dev_z にデータを
		// 転送し、何もせずに dev_z から host_x に
        // データを書き戻す。
        // その前後の時刻を gettimeofday(2) を用いてtv1, tv2に
        // 記録する。
		//
		usec2 = ((tv2.tv_sec  - tv1.tv_sec) * 1000000 +
				 (tv2.tv_usec - tv1.tv_usec));
		/* print */
		printf("normal: %.2fGB/s [%.3fms], page-locked: %.2fGB/s [%.3fms]\n",
			   GB_PER_SEC(length, usec1), usec1 / 1000.0,
			   GB_PER_SEC(length, usec2), usec2 / 1000.0);
	}
	return 0;
}
