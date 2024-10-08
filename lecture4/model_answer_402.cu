#include "../my_common.h"

#define M_DIMS		2
#define NITEMS		4000000
#define N_CATEGORY	5
#define MAX_LOOPS	30
#define DIST_THRESHOLD	0.0001

__managed__ double	data[M_DIMS * NITEMS];
__managed__ int		category[NITEMS];
__managed__ double	centroid_prev[M_DIMS * N_CATEGORY];
__managed__ double	centroid_curr[M_DIMS * N_CATEGORY];
__managed__ int		centroid_nitems[N_CATEGORY];

__host__ __device__
static double distance(const double *a,
					   const double *b)
{
	double	sum = 0.0;

	for (int m=0; m < M_DIMS; m++)
	{
		double	d = a[m] - b[m];

		sum += d * d;
	}
	return sqrt(sum);
}

//クラスタ中心点を更新する
__global__ void
kern_update_centroid(double *data,
					 int    *category,
					 double *centroid_curr,
					 int    *centroid_nitems)
{
	__shared__ double __our_nitems[N_CATEGORY];
	__shared__ double __our_centroid[M_DIMS * N_CATEGORY];
	double	__my_nitems[N_CATEGORY];
	double	__my_centroid[M_DIMS * N_CATEGORY];

	// shared/private変数の初期化
	memset(__my_nitems,   0, sizeof(__my_nitems));
	memset(__my_centroid, 0, sizeof(__my_centroid));
	for (int i=get_local_id(); i <  M_DIMS * N_CATEGORY; i += get_local_size())
	{
		__our_centroid[i] = 0.0;
		if (i < N_CATEGORY)
			__our_nitems[i] = 0;
	}
	__syncthreads();

	// private変数上でクラスタ中心点を収集
	for (int i=get_global_id(); i < NITEMS; i+= get_global_size())
	{
		int		cat = category[i];

		assert(cat >= 0 && cat < N_CATEGORY);
		__my_nitems[cat] += 1;
		for (int m=0; m < M_DIMS; m++)
			__my_centroid[M_DIMS * cat + m] += data[M_DIMS * i + m];
	}
	// private変数上での集計結果を、sharedメモリに移動
	// atomic演算だが、sharedメモリ上なので10倍ほど速い
	for (int cat=0; cat < N_CATEGORY; cat++)
	{
		atomicAdd(&__our_nitems[cat], __my_nitems[cat]);
		for (int j=0; j < M_DIMS; j++)
			atomicAdd(&__our_centroid[cat * M_DIMS + j],
					  __my_centroid[cat * M_DIMS + j]);
	}
	__syncthreads();
	// 同一ブロック内の他のスレッドの実行終了を待って、
	// 共有メモリの内容をGlobalメモリに書き出す。
	for (int i=get_local_id(); i < M_DIMS * N_CATEGORY; i += get_local_size())
    {
		atomicAdd(&centroid_curr[i], __our_centroid[i]);
		if (i < N_CATEGORY)
			atomicAdd(&centroid_nitems[i], __our_nitems[i]);
	}
}

//総和(centroid_curr)+件数(centroid_nitems)を平均値に直す
__global__ void
kern_normalize_centroid(double *centroid_curr,
						int    *centroid_nitems)
{
	assert(gridDim.x == 1);
	for (int i=get_local_id(); i < M_DIMS * N_CATEGORY; i += get_local_size())
	{
		int		cat = i / M_DIMS;
		int		nitems = centroid_nitems[cat];

		assert(nitems > 0);
		if (nitems > 0)
			centroid_curr[i] /= (double)nitems;
	}
}

//各要素の属するクラスタを更新する
__global__ void
kern_update_clusters(double *data,
					 int    *category,
					 double *centroid)
{
	for (int i=get_global_id(); i < NITEMS; i += get_global_size())
	{
		int		cat = -1;
		double	shortest;
		double	dist;

		for (int k=0; k < N_CATEGORY; k++)
		{
			dist = distance(data     + M_DIMS * i,
							centroid + M_DIMS * k);
			if (cat < 0 || dist < shortest)
			{
				cat = k;
				shortest = dist;
			}
		}
		category[i] = cat;
	}
}

__host__ static void
print_one_frame(void)
{
	static int	frame_count = 0;
	static const char *colors[] = {
		"light-blue",
		"light-green",
		"yellow",
		"orange",
		"light-magenta",
		"sea-green",
		"gray",
		"salmon",
		"goldenrod",
		"cyan",
		NULL,
	};

	// Gnuplotのコマンドを出力(初回のみ)
	if (frame_count++ == 0)
	{
		printf("set terminal gif animate delay 100 optimize size 600,600\n"
			   "set out 'kadai_401_anime.gif'\n"
			   "set title 'k-means (animation)'\n"
			   "set xlabel 'X0'\n"
			   "set ylabel 'X1'\n"
			   "set palette maxcolors %d\n", N_CATEGORY+1);
		printf("set palette defined (");
		for (int i=0; i < N_CATEGORY && colors[i] != NULL; i++)
			printf("%d '%s', ", i, colors[i]);
		printf("%d 'red')\n", N_CATEGORY);
	}
	// 途中結果を出力
	printf("plot '-' with points palette pointsize 1 pointtype 7\n");
	for (int i=0; i < NITEMS; i++)
	{
		printf("%f %f %d\n",
			   data[i*M_DIMS],
			   data[i*M_DIMS+1],
			   category[i]);
	}
	for (int k=0; k < N_CATEGORY; k++)
	{
		printf("%f %f %d\n",
			   centroid_curr[k*M_DIMS],
			   centroid_curr[k*M_DIMS+1],
			   N_CATEGORY);
	}
	printf("e\n");
}

int main(int argc, const char *argv[])
{
	int		centroid_grid_sz;
	int		centroid_block_sz;
	int		normalize_grid_sz;
	int		normalize_block_sz;
	int		cluster_grid_sz;
	int		cluster_block_sz;
	int		gpu_dindex;
	int		sm_count;
	int		loop;

	// (1) data[] 配列の初期化
	//     ここでは (0.0 - 1.0) の乱数を使用しているが、
	//     外部ファイルから読み出すようにしてよい。
	srand48(time(NULL));
	for (int i=0; i < M_DIMS * NITEMS; i++)
		data[i] = drand48();

	// (2) 各データに適当なクラスタを割り当てる
	// ----------------------------------------
	for (int i=0; i < NITEMS; i++)
		category[i] = (int)((double)N_CATEGORY * drand48());

	// cudaOccupancyMaxPotentialBlockSize を用いた
	// 推奨ブロックサイズの計算
	// ここでは、グリッド数が多すぎる場合はGPUのSM数まで切り下げる。
	__(cudaGetDevice(&gpu_dindex));
	__(cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, gpu_dindex));
	__(cudaOccupancyMaxPotentialBlockSize(&centroid_grid_sz,
										  &centroid_block_sz,
										  kern_update_centroid));
	if (centroid_grid_sz > sm_count)
		centroid_grid_sz = sm_count;
	__(cudaOccupancyMaxPotentialBlockSize(&normalize_grid_sz,
										  &normalize_block_sz,
										  kern_normalize_centroid));
	if (normalize_grid_sz > 1)
		normalize_grid_sz = 1;
	__(cudaOccupancyMaxPotentialBlockSize(&cluster_grid_sz,
										  &cluster_block_sz,
										  kern_update_clusters));
	if (cluster_grid_sz > sm_count)
		cluster_grid_sz = sm_count;

	// (3) k-means法が収束するまでループ
	// ---------------------------------
	for (loop=0; loop < MAX_LOOPS; loop++)
	{
		// (4) クラスタ中心点の更新
		// ------------------------
		memset(centroid_curr,   0, sizeof(centroid_curr));
		memset(centroid_nitems, 0, sizeof(centroid_nitems));

		kern_update_centroid<<<centroid_grid_sz,
							   centroid_block_sz>>>(data,
													category,
													centroid_curr,
													centroid_nitems);
		// クラスタ中心点の標準化
		// centroid_curr(合計値) と centroid_nitems(件数) から、
		// 平均値を導出する
		kern_normalize_centroid<<<normalize_grid_sz,
								  normalize_block_sz>>>(centroid_curr,
														centroid_nitems);
		// (5) 各要素の属するクラスタを更新する
		// ------------------------------------
		kern_update_clusters<<<cluster_grid_sz,
							   cluster_block_sz>>>(data,
												   category,
												   centroid_curr);
		// GPU Kernelの実行待ち
		cudaStreamSynchronize(NULL);

		// (6) クラスタ中心点の移動距離をチェック
		// --------------------------------------
		if (loop > 0)
		{
			double	dsum = 0.0;

			for (int i=0; i < N_CATEGORY; i++)
			{
				dsum += distance(centroid_prev + i * M_DIMS,
								 centroid_curr + i * M_DIMS);
			}
			//中心点の移動が十分に小さければ終了
			if (dsum / (double)N_CATEGORY < DIST_THRESHOLD)
				break;
		}
		// (7) 一つ前のクラスタ中心点を保存
		// --------------------------------
		memcpy(centroid_prev, centroid_curr, sizeof(centroid_curr));
	}
	// (8) 最終状態を出力
	// ------------------
	printf("k-means (loops=%d)\n", loop);
	for (int cat=0; cat < N_CATEGORY; cat++)
		printf("category[%d] nitems=%d (c0=%f, c1=%f)\n",
			   cat, centroid_nitems[cat],
			   centroid_curr[cat * M_DIMS],
			   centroid_curr[cat * M_DIMS + 1]);
	return 0;
}
