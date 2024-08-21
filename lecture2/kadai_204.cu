#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include "../my_common.h"

#define UNLOCK		0
#define LOCKED		1
__device__ static int		count_lock = UNLOCK;
__managed__ unsigned int	tid_counter_1 = 0;
__managed__ unsigned int	tid_counter_2 = 0;

__global__ void tid_count_by_lock(void)
{
	unsigned int tindex = blockIdx.x * blockDim.x + threadIdx.x;
	bool	jobs_done = false;

	// ここに Spin-Lock を用いた排他処理を実装する。
	// 各スレッド 1回ずつ tid_counter_1 に tindex を加算する。
	// ロック変数が LOCKED 状態の時は、他のスレッドは何もせず、
	// __syncthreads() でロックを取ったスレッドの処理が終わるのを
	// 同期する。

	do {
		// ここに排他ロックを実装
	} while (__syncthreads_count(!jobs_done) != 0);
}

__global__ void tid_count_by_atomic(void)
{
	unsigned int tindex = blockIdx.x * blockDim.x + threadIdx.x;

	atomicAdd(&tid_counter_2, tindex);
}

int main(int argc, char *argv[])
{
	struct timeval  tv1, tv2, tv3;

	gettimeofday(&tv1, NULL);
	tid_count_by_lock<<<10,320>>>();
	__(cudaDeviceSynchronize());

	gettimeofday(&tv2, NULL);
    tid_count_by_atomic<<<10,320>>>();
    __(cudaDeviceSynchronize());

	gettimeofday(&tv3, NULL);

	printf("tid-count: by lock [%u] %luus, by atomic [%u] %luus\n",
		   tid_counter_1,
		   (tv2.tv_sec  - tv1.tv_sec) * 1000000 +
           (tv2.tv_usec - tv1.tv_usec),
		   tid_counter_2,
		   (tv3.tv_sec  - tv2.tv_sec) * 1000000 +
           (tv3.tv_usec - tv2.tv_usec));
	return 0;
}
