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
	// centroid_curr[M_DIMS * cat] と centroid_curr[M_DIMS * cat + 1] が
	// カテゴリ(cat=[0...(N_CATEGORY-1)])のX要素とY要素
	//
	// 各要素は data[M_DIMS * index] がX要素、data[M_DIMS * index + 1]が
	// Y要素となる。
}

//総和(centroid_curr)+件数(centroid_nitems)を平均値に直す
__global__ void
kern_normalize_centroid(double *centroid_curr,
						int    *centroid_nitems)
{
	for (int i=get_global_id(); i < M_DIMS * N_CATEGORY; i += get_global_size())
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
	// 各要素は data[M_DIMS * index] がX要素、data[M_DIMS * index + 1]が
	// Y要素となり、category[index] が所属しているカテゴリとなる。
	// つまり、各要素に中心点が最も近いカテゴリを選び、category[]を更新
	// する事で、各要素の属するクラスタを更新できる。
}

__host__ static void
print_one_frame(void)
{
#if 0
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
			   "set ylabel 'X1'\n");
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
#endif
}

int main(int argc, const char *argv[])
{
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

	// (3) k-means法が収束するまでループ
	// ---------------------------------
	for (loop=0; loop < MAX_LOOPS; loop++)
	{
		// 途中経過を出力
		print_one_frame();

		// (4) クラスタ中心点の更新
		// ------------------------
		memset(centroid_curr,   0, sizeof(centroid_curr));
		memset(centroid_nitems, 0, sizeof(centroid_nitems));

		kern_update_centroid<<<...>>>(...);
		kern_normalize_centroid<<<...>>>(...);

		// (5) 各要素の属するクラスタを更新する
		// ------------------------------------
		kern_update_clusters<<<...>>>(...);

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
	print_one_frame();
	fprintf(stderr, "k-means (loops=%d)\n", loop);
	for (int cat=0; cat < N_CATEGORY; cat++)
		fprintf(stderr, "category[%d] nitems=%d (c0=%f, c1=%f)\n",
				cat, centroid_nitems[cat],
				centroid_curr[cat * M_DIMS],
				centroid_curr[cat * M_DIMS + 1]);
	return 0;
}
