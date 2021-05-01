
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <iostream>

#define N 500000 // tuned such that kernel takes a few microseconds

__global__
void shortKernel(float* out_d, float* in_d) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < N) out_d[idx] = 1.23 * in_d[idx];
}

int main()
{
	cudaDeviceProp prop;
	int device_id;
	cudaGetDevice(&device_id);
	cudaGetDeviceProperties(&prop, device_id);
	std::cout << prop.maxThreadsPerBlock << std::endl;
	//cudaStream_t stream;
	//cudaStreamCreate(&stream);
	//cudaGraph_t graph;
	//cudaGraphExec_t instance;
	//cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
	//for (int istep = 0; istep < N; istep++) {
	//	shortKernel <<< blocks, threads, 0, stream >>> (out_d, in_d);
	//}
	//cudaStreamEndCapture(stream, &graph);
	//cudaGraphInstantiate(&instance, graph, NULL, NULL, 0);
	//cudaGraphLaunch(instance, stream);
	//cudaStreamSynchronize(stream);
}

// Helper function for using CUDA to add vectors in parallel.
//cudaError_t addWithCuda(int *a, unsigned int size)
//{
//	int *dev_a = 0;
//	cudaError_t cudaStatus;
//
//	// Choose which GPU to run on, change this on a multi-GPU system.
//	cudaStatus = cudaSetDevice(0);
//	if (cudaStatus != cudaSuccess) {
//		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
//		goto Error;
//	}
//
//	cudaStatus = cudaMalloc((void**)&dev_a, size * sizeof(int));
//	if (cudaStatus != cudaSuccess) {
//		fprintf(stderr, "cudaMalloc failed!");
//		goto Error;
//	}
//
//	// Copy input vectors from host memory to GPU buffers.
//	cudaStatus = cudaMemcpy(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice);
//	if (cudaStatus != cudaSuccess) {
//		fprintf(stderr, "cudaMemcpy failed!");
//		goto Error;
//	}
//
//	for (int i = 0; i < 100; i++)
//	{
//		// Launch a kernel on the GPU with one thread for each element.
//		addKernel <<< 1, 2 >>> (dev_a);
//
//		// Check for any errors launching the kernel
//		cudaStatus = cudaGetLastError();
//		if (cudaStatus != cudaSuccess) {
//			fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
//			goto Error;
//		}
//	}
//
//		// cudaDeviceSynchronize waits for the kernel to finish, and returns
//		// any errors encountered during the launch.
//		cudaStatus = cudaDeviceSynchronize();
//		if (cudaStatus != cudaSuccess) {
//			fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
//			goto Error;
//		}
//	// Copy output vector from GPU buffer to host memory.
//	cudaStatus = cudaMemcpy(a, dev_a, size * sizeof(int), cudaMemcpyDeviceToHost);
//	if (cudaStatus != cudaSuccess) {
//		fprintf(stderr, "cudaMemcpy failed!");
//		goto Error;
//	}
//
//Error:
//	cudaFree(dev_a);
//	
//	return cudaStatus;
//}
