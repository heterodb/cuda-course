#include "../my_common.h"
#include <fcntl.h>
#include <sys/stat.h>
#include <cufile.h>

/* error handling for cuFile APIs */
#define __CUFILE(stmt)													\
	do {                                                                \
		CUfileError_t __rc = stmt;										\
																		\
		if (IS_CUDA_ERR(__rc))											\
		{																\
			const char *errname;										\
																		\
			if (cuGetErrorName(__rc.cu_err, &errname) != CUDA_SUCCESS)	\
				errname = "Unknown CUDA Error";							\
																		\
			fprintf(stderr,												\
					"[%s:%d] failed on " #stmt " = %s\n",				\
					__FILE__, __LINE__,									\
					errname);											\
			_exit(1);													\
		}																\
		else if (IS_CUFILE_ERR(__rc.err))								\
		{																\
			fprintf(stderr,												\
					"[%s:%d] failed on " #stmt " = %s\n"				\
					__FILE__, __LINE__,									\
					CUFILE_ERRSTR(__rc.err));							\
			_exit(1);													\
		}																\
    } while(0)

/* histogram */
__managed__ static int		hist_buf[256];

__global__ void
kern_histogram_buffer(unsigned char *buffer,
					  size_t buffer_sz)
{
	__shared__ int		hist_local[256];

	for (int index=get_local_id(); index < 256; index+=get_local_size())
		hist_local[index] = 0;
	__syncthreads();

	for (int index=get_global_id(); index < buffer_sz; index += get_global_size())
	{
		int		k = buffer[index];

		assert(k >= 0 && k < 256);
		atomicAdd(&hist_local[k], 1);
	}
	__syncthreads();

	for (int k=get_local_id(); k < 256; k+=get_local_size())
		atomicAdd(&hist_buf[k], hist_local[k]);
}

int main(int argc, char *argv[])
{
	struct stat *stat_buf = (struct stat *)alloca(sizeof(struct stat) * argc);
	size_t		buffer_sz = 0;
	unsigned char *buffer;
	int			grid_sz;
	int			block_sz;

	for (int i=0; i < argc; i++)
	{
		if (stat(argv[i], &stat_buf[i]) != 0)
		{
			fprintf(stderr, "failed on stat('%s'): %m\n", argv[i]);
			return 1;
		}
		buffer_sz = Max(stat_buf[i].st_size, buffer_sz);
	}
	// cuFile (GDS) ドライバの初期化
	__CUFILE(cuFileDriverOpen());

	// GPU-Direct Storageの転送先に用いるバッファを確保し、
	// それをGDS用に割り当てる。
	__(cudaMalloc(&buffer, buffer_sz));
	__CUFILE(cuFileBufRegister(&buffer, buffer_sz, 0));

	__(cudaDeviceGetAttribute(&grid_sz, cudaDevAttrMultiProcessorCount, 0));
	block_sz = 512;
	for (int i=0; i < argc; i++)
	{
		CUfileHandle_t fhandle;
		CUfileDescr_t fdesc;
		int			rawfd;
		size_t		remained, pos;
		ssize_t		nbytes;

		rawfd = open(argv[i], O_RDONLY);
		if (rawfd < 0)
		{
			fprintf(stderr, "failed on open '%s', skipped: %m\n", argv[i]);
			continue;
		}
		memset(&fdesc, 0, sizeof(fdesc));
		fdesc.type = CU_FILE_HANDLE_TYPE_OPAQUE_FD;
		fdesc.handle.fd = rawfd;
		__CUFILE(cuFileHandleRegister(&fhandle, &fdesc));

		pos = 0;
		remained = stat_buf[i].st_size;
		while (remained > 0)
		{
			nbytes = cuFileRead(fhandle,
								buffer,
								remained,
								pos,		// file offset
								pos);		// buffer offset
			if (nbytes < 0)
			{
				if (nbytes == -1)
					fprintf(stderr, "failed on cuFileRead: %m\n");
				else
					fprintf(stderr, "failed on cuFileRead: %s\n",
							CUFILE_ERRSTR(-nbytes));
				return 1;
			}
			else if (nbytes == 0)
				break;

			pos += nbytes;
			remained -= nbytes;
		}

		// この時点で既に buffer にはファイルの内容がロードされているので、
		// GPU Kernelを起動してそれを処理する。
		memset(hist_buf, 0, sizeof(hist_buf));
		kern_histogram_buffer<<<grid_sz, block_sz>>>(buffer, pos);
		__(cudaStreamSynchronize(NULL));

		// ファイルのヒストグラムを出力
		printf("==== file: %s histogram ========\n", argv[i]);
		for (int k=0; k < 256; k+=16)
		{
			printf("% 4u % 4u % 4u % 4u % 4u % 4u % 4u % 4u    "
				   "% 4u % 4u % 4u % 4u % 4u % 4u % 4u % 4u\n",
				   hist_buf[k],    hist_buf[k+1],  hist_buf[k+2],  hist_buf[k+3],
				   hist_buf[k+4],  hist_buf[k+5],  hist_buf[k+6],  hist_buf[k+7],
				   hist_buf[k+8],  hist_buf[k+9],  hist_buf[k+10], hist_buf[k+11],
				   hist_buf[k+12], hist_buf[k+13], hist_buf[k+14], hist_buf[k+15]);
		}
		putchar('\n');
		
		cuFileHandleDeregister(fhandle);
		close(rawfd);
	}
	__CUFILE(cuFileDriverClose());

	return 0;
}
