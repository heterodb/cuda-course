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

static int fval_comp(const void *__a, const void *__b)
{
	const float	a = *((const float *)__a);
	const float	b = *((const float *)__b);

	return (b - a);
}

int main(int argc, char *argv[])
{
	int			nitems = 4000000;
	int			nrooms = __pow2_next(nitems);
	int			max_scale = (31-__builtin_clz(nrooms-1));
	size_t		bufsz = sizeof(float) * nrooms;
	float	   *h_buffer;
	float	   *g_buffer;
	float	   *d_buffer;
	cudaStream_t stream;
	cudaEvent_t	ev1, ev2, ev3, ev4;
	float		tm1, tm2, tm3;
	struct timeval tv1, tv2;

	// Host/Deviceバッファの作成
	// 非同期メモリ転送を行うため、Managedメモリは使用しない。
	__(cudaMallocHost(&h_buffer, bufsz));
	__(cudaMalloc(&g_buffer, bufsz));
	__(cudaMallocHost(&d_buffer, bufsz));
	__(cudaStreamCreate(&stream));

	// Eventオブジェクトの作成（時間計測用）
	__(cudaEventCreate(&ev1));
	__(cudaEventCreate(&ev2));
	__(cudaEventCreate(&ev3));
	__(cudaEventCreate(&ev4));

	// ソートすべきデータの作成
	// アルゴリズムの都合上、バッファ長は 2^N でなければならないが、
	// nitems番目以降は -INFINITY を設定し、必ず nitems 番目よりも
	// 後ろにソートされるようにする。
	srand48(time(NULL));
	for (int i=0; i < nitems; i++)
		h_buffer[i] = 1000 * drand48();
	for (int i=nitems; i < nrooms; i++)
		h_buffer[i] = -INFINITY;

	// イベント1
	__(cudaEventRecord(ev1, stream));
	// streamにhost->deviceのデータ転送を投入
	__(cudaMemcpyAsync(g_buffer, h_buffer, bufsz,
					   cudaMemcpyHostToDevice,
					   stream));
	// イベント2
	__(cudaEventRecord(ev2, stream));
	// Bitonic-sortingのパラメータを変化させつつ、
	// GPU kernel functionの実行をstreamに投入
	for (int scale=0; scale <= max_scale; scale++)
	{
		for (int step=scale; step >= 0; step--)
		{
			kern_bitonic_sorting_step<<<40,320,0,stream>>>(nrooms, g_buffer,
														  scale, step);
		}
	}
	// イベント3
	__(cudaEventRecord(ev3, stream));
	// streamにdevice->hostのデータ転送を投入
	__(cudaMemcpyAsync(d_buffer, g_buffer, bufsz,
					   cudaMemcpyDeviceToHost,
					   stream));
	// イベント4
	__(cudaEventRecord(ev4, stream));
	// streamに投入済みのタスクが終了するまで待機
	__(cudaStreamSynchronize(stream));

	// CPUによるソート（比較用）
	gettimeofday(&tv1, NULL);
	qsort(h_buffer, nitems, sizeof(float), fval_comp);
	gettimeofday(&tv2, NULL);
	for (int i=0; i < nitems; i++)
	{
		if (h_buffer[i] != h_buffer[i])
		{
			printf("CPU / GPU Sorting results were not consistent at %d (%f, %f)\n",
				   i, h_buffer[i], d_buffer[i]);
			break;
		}
	}

	// ソート結果を表示
	for (int i=999800; i < 1000200; i++)
		printf("fval[%d] = % 8.4f\n", i, d_buffer[i]);
	// 実行時間を表示
	cudaEventElapsedTime(&tm1, ev1, ev2);
	cudaEventElapsedTime(&tm2, ev2, ev3);
	cudaEventElapsedTime(&tm3, ev3, ev4);
	printf("elapsed times: DMA:CPU->GPU %.3fms, GPU Kernel %.3fms DMA:GPU->CPU %.3fms\n"
		   "               CPU Quick Sort %.3fms\n",
		   tm1, tm2, tm3,
		   (double)((tv2.tv_sec  - tv1.tv_sec)  * 1000.0 +
					(tv2.tv_usec - tv1.tv_usec) / 1000.0));
	return 0;
}
