#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include "../my_common.h"

// __pow2_next: returns the smallest 2^N value that is larger than or
//              equal to the supplied 'val'
__host__
static int __pow2_next(int val)
{
	if (val == 0 || val == 1)
		return val;
	return (1U << (32 - __builtin_clz(val-1)));
}

__global__ void
kern_bitonic_sorting_step(int nrooms, float *g_buffer, int scale, int step)
{
	/* buffer length must be 2^N */
	int		nr_threads = (nrooms / 2);
	int		tid;

	assert((nrooms & (nrooms-1)) == 0);

	for (tid = blockDim.x * blockIdx.x + threadIdx.x;
		 tid < nr_threads;
		 tid += blockDim.x * gridDim.x)
	{
		bool	direction = (tid & (1U<<scale));
		int		base = ((tid >> scale) << (scale+1));
		int		m_bits = (tid & ((1U<<scale)-1)) >> step;
		int		l_bits = (tid & ((1U<<step)-1));
		int		index = base + (m_bits << (step+1)) + l_bits;
		int		buddy = index + (1U << step);

		if (!direction
			? g_buffer[index] < g_buffer[buddy]
			: g_buffer[index] > g_buffer[buddy])
		{
			float	temp = g_buffer[index];
			g_buffer[index] = g_buffer[buddy];
			g_buffer[buddy] = temp;
		}
	}
}

static void
launch_kernel(float *h_buffer,
			  float *g_buffer,
			  int nrooms,
			  cudaStream_t stream,
			  cudaEvent_t ev1,
			  cudaEvent_t ev2,
			  int grid_sz,
			  int block_sz)
{
	int			max_scale = (31-__builtin_clz(nrooms-1));
	size_t		bufsz = sizeof(float) * nrooms;
	float		tm1;

	// streamにhost->deviceのデータ転送を投入
	__(cudaMemcpyAsync(g_buffer, h_buffer, bufsz,
					   cudaMemcpyHostToDevice,
					   stream));
	// イベント1
	__(cudaEventRecord(ev1, stream));
	// Bitonic-sortingのパラメータを変化させつつ、
	// GPU kernel functionの実行をstreamに投入
	for (int scale=0; scale <= max_scale; scale++)
	{
		for (int step=scale; step >= 0; step--)
		{
			kern_bitonic_sorting_step<<<grid_sz, block_sz, 0, stream>>>
				(nrooms, g_buffer, scale, step);
		}
	}
	// イベント2
	__(cudaEventRecord(ev2, stream));
	// streamに投入済みのタスクが終了するまで待機
	__(cudaStreamSynchronize(stream));

	// 実行時間を表示
	cudaEventElapsedTime(&tm1, ev1, ev2);
	printf("elapsed times: GPU Kernel(grid_sz: %d, block_sz: %d) %.3fms\n",
		   grid_sz, block_sz, tm1);
}

int main(int argc, char *argv[])
{
	int			nitems = 4000000;
	int			nrooms = __pow2_next(nitems);
	size_t		bufsz = sizeof(float) * nrooms;
	float	   *h_buffer;
	float	   *g_buffer;
	cudaStream_t stream;
	cudaEvent_t	ev1, ev2;
	int			grid_sz;
	int			block_sz;

	// Host/Deviceバッファの作成
	// 非同期メモリ転送を行うため、Managedメモリは使用しない。
	__(cudaMallocHost(&h_buffer, bufsz));
	__(cudaMalloc(&g_buffer, bufsz));
	__(cudaStreamCreate(&stream));

	// Eventオブジェクトの作成（時間計測用）
	__(cudaEventCreate(&ev1));
	__(cudaEventCreate(&ev2));

	// ソートすべきデータの作成
	// アルゴリズムの都合上、バッファ長は 2^N でなければならないが、
	// nitems番目以降は -INFINITY を設定し、必ず nitems 番目よりも
	// 後ろにソートされるようにする。
	srand48(time(NULL));
	for (int i=0; i < nitems; i++)
		h_buffer[i] = 1000 * drand48();
	for (int i=nitems; i < nrooms; i++)
		h_buffer[i] = -INFINITY;

	// cudaOccupancyMaxPotentialBlockSize を用いた
	// 推奨ブロックサイズの計算
	__(cudaOccupancyMaxPotentialBlockSize(&grid_sz,
										  &block_sz,
										  kern_bitonic_sorting_step));
	launch_kernel(h_buffer, g_buffer, nrooms,
				  stream, ev1, ev2,
				  grid_sz, block_sz);
	for (grid_sz = 20; grid_sz < 120; grid_sz += 20)
	{
		for (block_sz = 32; block_sz <= 1024; block_sz += block_sz)
		{
			launch_kernel(h_buffer, g_buffer, nrooms,
						  stream, ev1, ev2,
						  grid_sz, block_sz);
		}
	}
	return 0;
}
