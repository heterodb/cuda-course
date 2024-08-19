#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

#define UNLOCK		0
#define LOCKED		1
__device__ static int		count_lock = UNLOCK;
__managed__ unsigned int	tid_counter_1 = 0;
__managed__ unsigned int	tid_counter_2 = 0;

__global__ void tid_count_by_lock(void)
{
	unsigned int tindex = blockIdx.x * blockDim.x + threadIdx.x;
	bool	jobs_done = false;

	do {
		if (!jobs_done &&
			atomicCAS(&count_lock, UNLOCK, LOCKED) == UNLOCK)
		{
			int		oldval;

			// tid_counter_1 を更新
			tid_counter_1 += tindex;

			// tid_counter_1 の更新が、LOCKED -> UNLOCK への
			// 変更よりも後にくることの無いように memory fence
			// 命令を挟んでおく。
			__threadfence();

			// ロック変数の状態を UNLOCK に戻す。
			// この時、元のロック変数は LOCKED でなければならない。
			oldval = atomicExch(&count_lock, UNLOCK);
			assert(oldval == LOCKED);

			// このスレッドはもう排他処理を必要としない。
			jobs_done = true;
		}
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
	cudaDeviceSynchronize();

	gettimeofday(&tv2, NULL);
    tid_count_by_atomic<<<10,320>>>();
    cudaDeviceSynchronize();

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
