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
kern_bitonic_sorting(int nrooms, float *g_buffer)
{
	/* largest 2^N value that is smaller than or equal to 'nitems' */
	int		max_scale = (31-__clz(nrooms-1));
	int		nr_threads = (1U << max_scale);

	/* buffer length must be 2^N */
	assert((nrooms & (nrooms-1)) == 0);

	for (int scale=0; scale <= max_scale; scale++)
	{
		for (int step=scale; step >= 0; step--)
		{
			for (int tid = threadIdx.x; tid < nr_threads; tid += blockDim.x)
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
			__syncthreads();
		}
	}	
}

int main(int argc, char *argv[])
{
	int			nitems = 3000;
	int			nrooms = __pow2_next(nitems);
	size_t		bufsz = sizeof(float) * nrooms;
	float	   *h_buffer;
	float	   *g_buffer;
	float	   *d_buffer;
	cudaStream_t stream;
	cudaEvent_t	ev1, ev2, ev3, ev4;
	float		tm1, tm2, tm3;

	// Host/Deviceバッファの作成
	// 非同期メモリ転送を行うため、Managedメモリは使用しない。
	__(cudaMallocHost(&h_buffer, bufsz));
	__(cudaMalloc(&g_buffer, bufsz));
	__(cudaMallocHost(&d_buffer, bufsz));
	__(cudaStreamCreate(&stream));

	// 処理時間計測用のイベントオブジェクトの作成
	__(cudaEventCreate(&ev1));
	__(cudaEventCreate(&ev2));
	__(cudaEventCreate(&ev3));
	__(cudaEventCreate(&ev4));

	// ソートすべきデータの作成
	// アルゴリズムの都合上、バッファ長は 2^N でなければならないが、
	// nitems番目以降は -INFINITY を設定し、必ず nitems 番目よりも
	// 後ろにソートされるようにする。
	srand48(time(NULL));
	for (int i=0; i < nrooms; i++)
		h_buffer[i] = (i < nitems ? 1000 * drand48() : -INFINITY);

	// イベント1
	__(cudaEventRecord(ev1, stream));
	// streamにhost->deviceのデータ転送を投入
	__(cudaMemcpyAsync(g_buffer, h_buffer, bufsz,
					   cudaMemcpyHostToDevice,
					   stream));
	// イベント2
	__(cudaEventRecord(ev2, stream));
	// GPU kernel functionの実行をstreamに投入
	kern_bitonic_sorting<<<1,320,0,stream>>>(nrooms, g_buffer);
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
	// ソート結果を表示
	for (int i=0; i < nitems; i++)
		printf("fval[%d] = % 8.4f\n", i, d_buffer[i]);
	// 実行時間を表示
	cudaEventElapsedTime(&tm1, ev1, ev2);
	cudaEventElapsedTime(&tm2, ev2, ev3);
	cudaEventElapsedTime(&tm3, ev3, ev4);
	printf("elapsed times: DMA:CPU->GPU %.3fms, GPU Kernel %.3fms DMA:GPU->CPU %.3fms\n",
		   tm1, tm2, tm3);

	return 0;
}
