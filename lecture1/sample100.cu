#include <stdio.h>

int main(int argc, const char *argv[])
{
	cudaDeviceProp prop;
	int		count;

	cudaGetDeviceCount(&count);
	for (int k=0; k < count; k++)
	{
		cudaGetDeviceProperties(&prop, k);

		printf("GPU%d name=%s uuid=%02x%02x%02x%02x-%02x%02x-%02x%02x-%02x%02x-%02x%02x%02x%02x%02x%02x\n",
			   k, prop.name,
			   (unsigned char)prop.uuid.bytes[0],
			   (unsigned char)prop.uuid.bytes[1],
			   (unsigned char)prop.uuid.bytes[2],
			   (unsigned char)prop.uuid.bytes[3],
			   (unsigned char)prop.uuid.bytes[4],
			   (unsigned char)prop.uuid.bytes[5],
			   (unsigned char)prop.uuid.bytes[6],
			   (unsigned char)prop.uuid.bytes[7],
			   (unsigned char)prop.uuid.bytes[8],
			   (unsigned char)prop.uuid.bytes[9],
			   (unsigned char)prop.uuid.bytes[10],
			   (unsigned char)prop.uuid.bytes[11],
			   (unsigned char)prop.uuid.bytes[12],
			   (unsigned char)prop.uuid.bytes[13],
			   (unsigned char)prop.uuid.bytes[14],
			   (unsigned char)prop.uuid.bytes[15]);
	}
	return 0;
}
