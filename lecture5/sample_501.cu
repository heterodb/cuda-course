#include "../my_common.h"

#define NITEMS		400000000UL
#define MAX_GPUS	64

__managed__ static	double	gpu_sumX[MAX_GPUS];

__global__ void
kern_compute_average(float *fp_values,
					 long unitsz,
					 int dindex)
{
	long	end = Min(unitsz * (dindex+1), NITEMS);
	long	base;
	double	sumX = 0.0;
	__shared__ double work[WARP_SIZE];

	for (base = unitsz * dindex + get_global_base();
		 base < end;
		 base += get_global_size())
	{
		long	index = base + get_local_id();
		double	fval = (index < end ? fp_values[index] : 0);

		fval += __shfl_xor_sync(0xffffffffU, fval,  1);
		fval += __shfl_xor_sync(0xffffffffU, fval,  2);
		fval += __shfl_xor_sync(0xffffffffU, fval,  4);
		fval += __shfl_xor_sync(0xffffffffU, fval,  8);
		fval += __shfl_xor_sync(0xffffffffU, fval, 16);
		if ((get_local_id() & (WARP_SIZE-1)) == 0)
			work[get_local_id() / WARP_SIZE] = fval;
		__syncthreads();

		if (get_local_id() < WARP_SIZE)
		{
			fval = work[threadIdx.x];
			fval += __shfl_xor_sync(0xffffffffU, fval,  1);
			fval += __shfl_xor_sync(0xffffffffU, fval,  2);
			fval += __shfl_xor_sync(0xffffffffU, fval,  4);
			fval += __shfl_xor_sync(0xffffffffU, fval,  8);
			fval += __shfl_xor_sync(0xffffffffU, fval, 16);

			if (get_local_id() == 0)
				sumX += fval;
		}
		__syncthreads();
	}
	if (get_local_id() == 0)
		atomicAdd(&gpu_sumX[dindex], sumX);
}

static void launch_kernels(const char *label,
						   int nr_gpus,
						   float *fp_values,
						   int *mp_count,
						   cudaStream_t *streams,
						   int drift)
{
	long		unitsz = (NITEMS + nr_gpus - 1) / nr_gpus;
	double		sumX = 0.0;
	struct timeval tv1, tv2;

	// 各GPUでは (NITEMS / Nr_GPUs) 個のデータを集計する。
	// 例えば5GPUの環境であれば、GPUあたり8000万個の集計となる。
	gettimeofday(&tv1, NULL);
	memset(gpu_sumX, 0, sizeof(gpu_sumX));
	for (int dindex=0; dindex < nr_gpus; dindex++)
	{
		int		grid_sz = mp_count[dindex];
		int		block_sz = 1024;

		__(cudaSetDevice(dindex));
		kern_compute_average<<<grid_sz, block_sz, 0, streams[dindex]>>>
			(fp_values, unitsz, (dindex + drift) % nr_gpus);
		// GPU Kernelの起動は非同期処理なので、
		// GPUの個数だけ非同期呼び出しを繰り返すことで
		// 複数のGPU Kernelを並行して動作させることができる。
	}

	// 各GPUでの処理完了を待って、集計結果を取り出す。
	for (int dindex=0; dindex < nr_gpus; dindex++)
		__(cudaStreamSynchronize(streams[dindex]));
	for (int dindex=0; dindex < nr_gpus; dindex++)
		sumX  += gpu_sumX[dindex];
	gettimeofday(&tv2, NULL);

	printf("%s trial by %d GPUs, average = %f [%.3fms]\n",
		   label,
		   nr_gpus, sumX / (double)NITEMS,
		   (double)(tv2.tv_sec  - tv1.tv_sec)  * 1000.0 +
		   (double)(tv2.tv_usec - tv1.tv_usec) / 1000.0);

}

int main(int argc, const char *argv[])
{
	cudaStream_t   *streams;
	int			   *mp_count;
	int				nr_gpus;
	float		   *fp_values;

	// システムで利用可能なGPU数を取得
	__(cudaGetDeviceCount(&nr_gpus));

	// 平均値・標準偏差を計算するためのバッファを準備
	// この初期化処理によって、物理メモリはCPUに割り当てられる
	// ことになる。
	__(cudaMallocManaged(&fp_values, sizeof(float) * NITEMS));
	for (long i=0; i < NITEMS; i++)
		fp_values[i] = drand48();

	// 非同期Streamの作成と、GPUのSM数をそれぞれ取得
	streams = (cudaStream_t *)alloca(sizeof(cudaStream_t) * nr_gpus);
	mp_count = (int *)alloca(sizeof(int) * nr_gpus);
	for (int dindex=0; dindex < nr_gpus; dindex++)
	{
		__(cudaSetDevice(dindex));
		__(cudaStreamCreate(&streams[dindex]));
		__(cudaDeviceGetAttribute(&mp_count[dindex],
								  cudaDevAttrMultiProcessorCount,
								  dindex));
	}
	
	// 1回目の実行：CPU->GPUへのメモリ移動が発生する
	launch_kernels("1st", nr_gpus, fp_values, mp_count, streams, 0);

	// 2回目の実行：GPUには既にデータがロード済み
	launch_kernels("2nd", nr_gpus, fp_values, mp_count, streams, 0);

	// 3回目の実行：GPU0->GPU1, GPU1->GPU2, ... へのデータ移動が発生する
	launch_kernels("3rd", nr_gpus, fp_values, mp_count, streams, 1);

	// 4回目の実行：既にデータ移動が発生した
	launch_kernels("4th", nr_gpus, fp_values, mp_count, streams, 1);

	// 極端なケース: GPU0に全てのデータが集まった状態で6回目を実行
	__(cudaMemPrefetchAsync(fp_values, sizeof(float) * NITEMS, 0));
	__(cudaStreamSynchronize(NULL));
	launch_kernels("5th", nr_gpus, fp_values, mp_count, streams, 0);

	return 0;
}
