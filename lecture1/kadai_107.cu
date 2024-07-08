#include <stdio.h>
#include <stdlib.h>
#define NITEMS		10000000

__managed__ double	gpu_sum_x = 0.0;

__global__ void my_gpu_average(float *x)
{
	int		index;

	for (index = blockDim.x * blockIdx.x + threadIdx.x;
		 index < NITEMS;
		 index += blockDim.x * gridDim.x)
		atomicAdd(&gpu_sum_x, x[index]);
}

int main(int argc, char *argv[])
{
	float  *fval;
	double	host_sum_x = 0.0;
	int		my_gpu;

	// cudaMallocManaged を使用して NITEMS * sizeof(float) バイトの
	// managed memory を割当て、fvalにそのポインタをセットする。

	/* buffer initialization */
	for (int i=0; i < NITEMS; i++)
	{
		fval[i] = 100.0 * drand48();
		host_sum_x += fval[i];
	}
	// GPUカーネルの起動前に、バッファ fval の内容が GPU に存在
	// しているべきというヒントを cudaMemPrefetchAsync を用いて
	// CUDAのランタイムに知らせる。
	/* launch GPU kernel */
	my_gpu_average<<<8,128>>>(fval);
	cudaDeviceSynchronize();

	/* fetch result */
	printf("average by CPU = %f, GPU = %f\n",
	// CPU側で集計した値（host_sum_x）と、GPU側で集計した値（gpu_sum_x）を   
	// 利用して、平均値を計算する。
	// gpu_sum_x の参照に際しては、cudaMemcpy() などは使用しない。		   
		);
	return 0;
}
