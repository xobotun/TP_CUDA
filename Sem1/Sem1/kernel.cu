/*
 ============================================================================
 Name        : sem2.cu
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
#include <thrust/count.h>
#include <thrust/device_vector.h>

static void CheckCudaErrorAux (const char *, unsigned, const char *, cudaError_t);
#define CUDA_CHECK_RETURN(value) CheckCudaErrorAux(__FILE__,__LINE__, #value, value)

static const int SIM_TIME = 5;
static const int STEPS_PER_SECOND = 10;
static const int CYL_LENGTH = 10;
static const int STEPS_PER_METER = 10;
static const int WORK_SIZE = CYL_LENGTH * STEPS_PER_METER;
static const int BLOCK_SIZE = 256;
static const float ERR_LIMIT = 0.001;
static const float TEMP_PER_SEC = 5.;
static const float DX = (CYL_LENGTH * 1.) / (STEPS_PER_METER * 1.0);
static const float DT = (SIM_TIME * 1.) / (STEPS_PER_SECOND * 1.0);
static __constant__ float gpuCoefficients[WORK_SIZE * WORK_SIZE];
static float hostCoefficients[WORK_SIZE * WORK_SIZE];
const int blockCount = (WORK_SIZE + BLOCK_SIZE - 1) / BLOCK_SIZE;


__device__ int sumMultiplies(float* constants, int size, float guess) {
	unsigned idx = blockIdx.x*blockDim.x + threadIdx.x;

	float result = 0;
	for (int i = 0; i < size; ++i) {
		result += constants[idx*size + i] * guess;
	}

	return result;
}

__device__ void iterateHeat(float *data, float* constants, unsigned size, float errLimit) {
	unsigned idx = blockIdx.x*blockDim.x + threadIdx.x;

	//if (idx < size - 1) {
		float sumError = errLimit + 1;
		float prevIteration = data[idx];
		const float someCoeff = 1 / constants[idx*size + idx];
		extern __shared__ float errors[];

		while (sumError > errLimit) {
			float newIteration = someCoeff * (data[idx] - sumMultiplies(constants, size, newIteration));

			errors[idx] = fabs(prevIteration - newIteration);
			__syncthreads();

			if (idx == 0)
				for (int i = 0, sumError = 0; i < size; ++i)
					sumError += errors[i];
		}
	//}
}

/**
 * CUDA kernel that computes reciprocal values for a given vector
 */
__global__  void heatKernel(float *data, float* constants, unsigned size, unsigned iterations, float tempPerIteration, float errLimit) {
	unsigned idx = blockIdx.x*blockDim.x+threadIdx.x;
	if (idx < size - 1)
		for (int time = 0; time < iterations - 1; ++time) {
			
			iterateHeat(data, constants, size, errLimit);
		
			__syncthreads();
			if (idx == 0) {
				data[size - 1] = time * tempPerIteration;
				data[0] = 0;
			}
		}

}

void initializeData(float *data, unsigned size)
{
	for (unsigned i = 0; i < size - 1; ++i)
		data[i] = 0.;

	data[size - 1] = 5;
}

void initializeCoefficients(float *coeffs, unsigned size, float dt, float dx)
{
	const float coeff = (dt)/(dx*dx);

	for (unsigned i = 0; i < size - 1; ++i)
		for (unsigned j = 0; j < size - 1; ++j)
			coeffs[i*size + j] = 0;


	coeffs[0] = 1;

	for (unsigned i = 1; i < size - 2; ++i)
		for (unsigned j = 1; j < size - 2; ++j)
			if (i == j) {
				coeffs[i*size + j - 1] = -coeff;
				coeffs[i*size + j]     = 1 + 2 * coeff;
				coeffs[i*size + j + 1] = -coeff;
			}

	coeffs[size*size - 1] = 1;
}

int main(void) {
	float *hostData = new float[WORK_SIZE];
	float *gpuData;

	initializeData(hostData, WORK_SIZE);
	initializeCoefficients(hostCoefficients, WORK_SIZE, DT, DX);

	CUDA_CHECK_RETURN(cudaMalloc((void **)&gpuData, sizeof(float)*(WORK_SIZE)));
	CUDA_CHECK_RETURN(cudaMalloc((void **)&gpuCoefficients, sizeof(float)*(WORK_SIZE*WORK_SIZE)));
	CUDA_CHECK_RETURN(cudaMemcpy(gpuData, hostData, sizeof(float)*(WORK_SIZE), cudaMemcpyHostToDevice));
	CUDA_CHECK_RETURN(cudaMemcpyToSymbol(gpuCoefficients, hostCoefficients, sizeof(hostCoefficients)));

	float *result = new float[WORK_SIZE];

	heatKernel <<<blockCount, BLOCK_SIZE >>> (gpuData, gpuCoefficients, WORK_SIZE, SIM_TIME * STEPS_PER_SECOND, TEMP_PER_SEC / (STEPS_PER_SECOND*1.0f), ERR_LIMIT);

	CUDA_CHECK_RETURN(cudaMemcpy(result, gpuData, sizeof(float)*(WORK_SIZE), cudaMemcpyDeviceToHost));


	/* Verify the results */
	for (int i = 0; i < WORK_SIZE; ++i)
		std::cout << result[i] << std::endl;

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
static void CheckCudaErrorAux (const char *file, unsigned line, const char *statement, cudaError_t err) {
	if (err == cudaSuccess)
		return;
	std::cerr << statement<<" returned " << cudaGetErrorString(err) << "("<<err<< ") at "<<file<<":"<<line << std::endl;
	system("pause");
	exit (1);
}

