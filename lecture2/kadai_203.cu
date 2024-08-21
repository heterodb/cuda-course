#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include "../my_common.h"

#define NDIMS		4000
#define NVECS		400

__managed__ double	dot_product_shared_results[NVECS * NVECS];
__managed__ double	dot_product_warp_results[NVECS * NVECS];
static 		double	dot_product_host_results[NVECS * NVECS];

// __pow2_prev: returns the largest 2^N value that is smaller than or
//              equal to the supplied 'val'
__device__ __forceinline__
static int __pow2_prev(int val)
{
	if (val == 0)
		return 0;
	return (1U<<(31-__clz(val-1)));
}

__global__ static void	dot_product_local(const float *source_vectors)
{
	const float	*x = &source_vectors[(blockIdx.x % NVECS) * NDIMS];
	const float *y = &source_vectors[(blockIdx.x / NVECS) * NDIMS];
	__shared__ double work[NDIMS];

	/*
	 * 共有メモリ上の work[] 配列を初期化
	 * ----
	 * ここでは、ベクトル各要素の積を初期値としてセットしている。
	 * SMあたりスレッド数はNDIMSよりも少ないので、ループを利用して
	 * 全てのwork[]要素が初期化できるように調整している。
	 */
	for (int index=threadIdx.x; index < NDIMS; index += blockDim.x)
		work[index] = x[index] * y[index];
	__syncthreads();

	/*
	 * 共有メモリ上の work[] 配列を用いた reduction 操作
	 * ----
	 * NDIMS=2000 に対して nscale は 1024 で初期化される。
	 *
	 * 最初のステップで、index < 1024 を担当するスレッドが
	 * work[1024]～work[1999] までの内容を自身に加算する。
	 *
	 * 次のステップで、index < 512 を担当するスレッドが、
	 * work[512]～work[1023] までの内容を自身に加算する。
	 *
	 * 以降、これを繰り返し、最終的に work[0] に全ての値が
	 * 加算される。
	 */
	for (int nscale = __pow2_prev(NDIMS); nscale > 0; nscale /= 2)
	{
		for (int index=threadIdx.x; index < nscale; index += blockDim.x)
		{
			int		buddy = nscale + index;

			if (buddy < NDIMS)
				work[index] += work[buddy];
		}
		__syncthreads();
	}
	/*
	 * Shared Memoryから、Global Memory上の結果バッファに計算結果を書き戻す。
	 */
	if (threadIdx.x == 0)
		dot_product_shared_results[blockIdx.x] = work[threadIdx.x];
}

#define WARP_SIZE	32
__global__ static void	dot_product_warp(const float *source_vectors)
{
	const float	*x = &source_vectors[(blockIdx.x % NVECS) * NDIMS];
	const float *y = &source_vectors[(blockIdx.x / NVECS) * NDIMS];
	double		sum = 0.0;
	__shared__ double work[WARP_SIZE];

	/* 共有メモリを0.0で初期化 */
	if (threadIdx.x < WARP_SIZE)
		work[threadIdx.x] = 0.0;
	__syncthreads();

	// ここに Warp-Shuffle 命令を用いた Reduction 処理

	/* 共有メモリ上で集計した内積を、結果配列に書き戻し。*/
	if (threadIdx.x == 0)
		dot_product_warp_results[blockIdx.x] = sum;
}

__host__ static double	dot_product_by_cpu(const float *x, const float *y)
{
	double	sum = 0.0;

	for (int i=0; i < NDIMS; i++)
		sum += x[i] * y[i];
	return sum;
}

int main(int argc, char *argv[])
{
	float  *source_vectors;
	struct timeval	tv1, tv2, tv3, tv4;
	int		count = 0;

	/* source_vectors[] 配列をランダムな値で初期化する。 */
	__(cudaMallocManaged(&source_vectors, sizeof(float) * NDIMS * NVECS));
	for (int i=0; i < NDIMS * NVECS; i++)
		source_vectors[i] = drand48();

	/* Shared-memory上のReduction操作によって内積を計算するGPUカーネルを起動 */
	gettimeofday(&tv1, NULL);
	__(cudaMemset(dot_product_shared_results, 0, sizeof(double) * NVECS * NVECS));
	dot_product_local<<<NVECS*NVECS,1024>>>(source_vectors);
	__(cudaDeviceSynchronize());

	/* Warp-shuffle関数を用いたReduction操作によって内積を計算するGPUカーネルを起動 */
	gettimeofday(&tv2, NULL);
	__(cudaMemset(dot_product_warp_results, 0, sizeof(double) * NVECS * NVECS));
	dot_product_warp<<<NVECS*NVECS,1024>>>(source_vectors);
	__(cudaDeviceSynchronize());

	/* CPU上の順次計算によって内積を計算する関数を呼び出し */
	gettimeofday(&tv3, NULL);
	memset(dot_product_host_results, 0, sizeof(double) * NVECS * NVECS);
	for (int i=0; i < NVECS; i++)
	{
		float  *x = &source_vectors[i * NDIMS];

		for (int j=0; j < NVECS; j++)
		{
			float  *y = &source_vectors[j * NDIMS];

			dot_product_host_results[i*NVECS + j] = dot_product_by_cpu(x, y);
		}
	}
	gettimeofday(&tv4, NULL);

	/* 実行時間のサマリを出力する */
	printf("dot product: by shared %.3fms, by warp %.3fms, by CPU %.3fms\n",
		   (double)(tv2.tv_sec  - tv1.tv_sec)  * 1000.0 +
		   (double)(tv2.tv_usec - tv1.tv_usec) / 1000.0,
		   (double)(tv3.tv_sec  - tv2.tv_sec)  * 1000.0 +
		   (double)(tv3.tv_usec - tv2.tv_usec) / 1000.0,
		   (double)(tv4.tv_sec  - tv3.tv_sec)  * 1000.0 +
		   (double)(tv4.tv_usec - tv3.tv_usec) / 1000.0);

	/*
	 * GPU(Global - Atomic)、GPU(Shared - Reduction)、CPUのそれぞれの
	 * パターンにおいて、計算結果が異なる場合にそれぞれの値を出力する。
	 */
	for (int k=0; k < NVECS * NVECS; k++)
	{
		if (dot_product_shared_results[k] != dot_product_warp_results[k])
		{
			printf("vec(%d) x vec(%d) -> by shared %.18f, by warp %.18f, by CPU %.18f\n",
				   k / NVECS,
				   k % NVECS,
				   dot_product_shared_results[k],
				   dot_product_warp_results[k],
				   dot_product_host_results[k]);
			if (++count > 20)
				break;
		}
	}
	return 0;
}
