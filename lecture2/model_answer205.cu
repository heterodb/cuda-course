#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include "../my_common.h"

typedef struct
{
	char	category;
	float	fval;
} my_record;

typedef struct
{
	char	category;	/* copy of category */
	int		count;		/* number of my_record */
	double	fsum;		/* sum of fval */
	char	__padding__[16];
} my_entry;

typedef struct
{
	int			nitems;
	my_entry	values[1000];
} my_summary;

#define NUM_RECORDS		1000000
__managed__ my_record	source_records[NUM_RECORDS];

#define HASH_NSLOTS		400
#define SLOT_LOCKED		(~0UL)
typedef union
{
	my_entry   *ent;
	unsigned long long int u64;
} my_hash;

__managed__ my_hash		hash_slots[HASH_NSLOTS];
__managed__ my_summary	summary;

__global__ void kern_grouping_summary(void)
{
	int		index		= blockIdx.x * blockDim.x + threadIdx.x;
	bool	jobs_done	= (index >= NUM_RECORDS);
	char	category	= (!jobs_done ? source_records[index].category : ' ');
	float	fval		= (!jobs_done ? source_records[index].fval : 0.0);

	do {
		if (!jobs_done)
		{
			my_hash		hval;

			if ((hval.u64 = atomicCAS(&hash_slots[category].u64,
									  0UL, SLOT_LOCKED)) == 0UL)
			{
				/* setup summary entry under the lock */
				int		k = atomicAdd(&summary.nitems, 1);

				summary.values[k].category = category;
				summary.values[k].count = 1;
				summary.values[k].fsum  = fval;
				__threadfence();
				/* unlock */
				hval.ent = &summary.values[k];
				atomicExch(&hash_slots[category].u64, hval.u64);
				jobs_done = true;
			}
			else if (hval.u64 != SLOT_LOCKED)
			{
				assert(hval.ent->category == category);
				atomicAdd(&hval.ent->count, 1);
				atomicAdd(&hval.ent->fsum, fval);
				jobs_done = true;
			}
		}
	} while (__syncthreads_count(!jobs_done) > 0);
}

int main(int argc, char *argv[])
{
	struct timeval	tv1, tv2;
	int		block_sz = 512;
	int		grid_sz = (NUM_RECORDS + block_sz - 1) / block_sz;

	/* initialization of the source records */
	srand(20240824);
	for (int i=0; i < NUM_RECORDS; i++)
	{
		source_records[i].category = 'A' + (26.0 * drand48());
		source_records[i].fval     = 100.0 * drand48();
	}

	/* initialization of the device working memory */
	__(cudaMemset(hash_slots, 0, sizeof(hash_slots)));
	summary.nitems = 0;

	/* grouping summary */
	gettimeofday(&tv1, NULL);
    kern_grouping_summary<<<grid_sz, block_sz>>>();
    __(cudaDeviceSynchronize());
	gettimeofday(&tv2, NULL);

	/* print results */
	printf("grouping-summary: nitems=%d, time=%.3fms\n",
		   summary.nitems,
		   (double)(tv2.tv_sec  - tv1.tv_sec)  * 1000.0 +
           (double)(tv2.tv_usec - tv1.tv_usec) / 1000.0);
	for (int i=0; i < summary.nitems; i++)
		printf("[%d] category='%c' count=%d fsum=%f\n", i,
			   summary.values[i].category,
			   summary.values[i].count,
			   summary.values[i].fsum);
	return 0;
}
