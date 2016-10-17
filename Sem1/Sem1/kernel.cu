/*
============================================================================
Name        : sem1.cu
Author      : maminov
Version     :
Copyright   : copyleft
Description : CUDA compute reciprocals
============================================================================
*/

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>
#include <numeric>
#include <stdlib.h>

static void CheckCudaErrorAux(const char *, unsigned, const char *, cudaError_t);
#define CUDA_CHECK_RETURN(value) CheckCudaErrorAux(__FILE__,__LINE__, #value, value)

/**
* CUDA kernel that computes reciprocal values for a given vector
*/
__global__  void heatKernel(float *data, unsigned size, unsigned iterations, float tempPerIteration, float dx, float dt) {
	unsigned idx = blockIdx.x*blockDim.x + threadIdx.x;
	if (idx < size)
		for (int i = 0; i < iterations; ++i) {

			if (idx == 0)
				data[size + 1] += tempPerIteration;

			data[idx + 1] += (data[idx - 1 + 1] - 2 * data[idx + 1] + data[idx + 1 + 1]) * dt / (dx*dx);

			__syncthreads();
		}
}

void initialize(float *data, unsigned size)
{
	for (unsigned i = 0; i < size; ++i)
		data[i] = 0.;
}

int main(void)
{
	static const int SIM_TIME = 5;
	static const int STEPS_PER_SECOND = 10;
	static const int CYL_LENGTH = 10;
	static const int STEPS_PER_METER = 10;
	static const int WORK_SIZE = CYL_LENGTH * STEPS_PER_METER;
	static const int BLOCK_SIZE = 256;
	static const float TEMP_PER_SEC = 5.;

	float *hostData = new float[WORK_SIZE + 2];
	float *gpuData;

	initialize(hostData, WORK_SIZE + 2);

	CUDA_CHECK_RETURN(cudaMalloc((void **)&gpuData, sizeof(float)*(WORK_SIZE + 2)));
	CUDA_CHECK_RETURN(cudaMemcpy(gpuData, hostData, sizeof(float)*(WORK_SIZE + 2), cudaMemcpyHostToDevice));

	float *result = new float[WORK_SIZE + 2];

	const int blockCount = (WORK_SIZE + BLOCK_SIZE - 1) / BLOCK_SIZE;
	heatKernel <<< blockCount, BLOCK_SIZE >>> (gpuData, WORK_SIZE, SIM_TIME * STEPS_PER_SECOND, TEMP_PER_SEC / (STEPS_PER_SECOND*1.0f), CYL_LENGTH / (STEPS_PER_METER*1.0f), SIM_TIME / (STEPS_PER_SECOND*1.0f));

	CUDA_CHECK_RETURN(cudaMemcpy(result, gpuData, sizeof(float)*(WORK_SIZE + 2), cudaMemcpyDeviceToHost));


	/* Verify the results */
	for (int i = 0; i < WORK_SIZE; ++i)
		std::cout << result[i + 1] << std::endl;

	/* Free memory */
	CUDA_CHECK_RETURN(cudaFree(gpuData));
	delete[] hostData;
	delete[] result;


	return 0;
}

/**
* Check the return value of the CUDA runtime API call and exit
* the application if the call has failed.
*/
static void CheckCudaErrorAux(const char *file, unsigned line, const char *statement, cudaError_t err)
{
	if (err == cudaSuccess)
		return;
	std::cerr << statement << " returned " << cudaGetErrorString(err) << "(" << err << ") at " << file << ":" << line << std::endl;
	exit(1);
}

