#include <stdio.h>
#include <stdlib.h>

__global__ static void my_kernel(void)
{
	if (threadIdx.x < 16)
		printf("A: thread=%u activemask=%08x\n", threadIdx.x, __activemask());
	else if (threadIdx.x % 2 == 0)
		printf("B: thread=%u activemask=%08x\n", threadIdx.x, __activemask());
	else
		printf("C: thread=%u activemask=%08x\n", threadIdx.x, __activemask());

	printf("D: thread=%u activemask=%08x\n", threadIdx.x, __activemask());
	__syncwarp();
	printf("E: thread=%u activemask=%08x\n", threadIdx.x, __activemask());
}

int main(int argc, char *argv[])
{
	my_kernel<<<1,32>>>();
    cudaDeviceSynchronize();
	return 0;
}
