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

static void CheckCudaErrorAux (const char *, unsigned, const char *, cudaError_t);
#define CUDA_CHECK_RETURN(value) CheckCudaErrorAux(__FILE__,__LINE__, #value, value)

/**
 * CUDA kernel that computes reciprocal values for a given vector
 */
__global__  void heatKernel(float *data, float* constants, unsigned size, unsigned iterations, float tempPerIteration, float dx, float dt, float errLimit) {
	unsigned idx = blockIdx.x*blockDim.x+threadIdx.x + 1;
	// TODO: loop this segment loop for some time!
	for (int curSeg = 1; curSeg < size - 1; ++curSeg) {
		const float prevTemp = data[curSeg];
		const float someCoeff = 1 / constants[curSeg*size + curSeg];
		float sumError = errLimit + 1;

		while (fabs(sumError) >  errLimit) {


		}
		__syncthreads();
		if (idx == 0)
			data[size - 1] 	+= tempPerIteration;
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
		for (unsigned j = 0; j < size - 1; ++i)
			coeffs[i*size + j] = 0;


	coeffs[0] = 1;

	for (unsigned i = 1; i < size - 2; ++i)
		for (unsigned j = 1; j < size - 2; ++i)
			if (i == j) {
				coeffs[i*size + j - 1] = -coeff;
				coeffs[i*size + j]     = 1 + 2 * coeff;
				coeffs[i*size + j + 1] = -coeff;
			}

	coeffs[size*size - 1] = 1;
}

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
static __constant__ float *gpuCoefficients;
static float *hostCoefficients = new float[WORK_SIZE * WORK_SIZE];

int main(void)
{
	float *hostData = new float[WORK_SIZE];
	float *gpuData;
	
	initializeData(hostData, WORK_SIZE);
	initializeCoefficients(hostCoefficients, WORK_SIZE, DT, DX);

	CUDA_CHECK_RETURN(cudaMalloc((void **)&gpuData, sizeof(float)*(WORK_SIZE)));
	CUDA_CHECK_RETURN(cudaMalloc((void **)&gpuCoefficients, sizeof(float)*(WORK_SIZE*WORK_SIZE)));
	CUDA_CHECK_RETURN(cudaMemcpy(gpuData, hostData, sizeof(float)*(WORK_SIZE), cudaMemcpyHostToDevice));
	CUDA_CHECK_RETURN(cudaMemcpyToSymbol(gpuCoefficients, hostCoefficients, sizeof(float)*(WORK_SIZE*WORK_SIZE)));

	float *result = new float[WORK_SIZE];

	const int blockCount = (WORK_SIZE+BLOCK_SIZE-1)/BLOCK_SIZE;
	heatKernel<<<blockCount, BLOCK_SIZE>>> (gpuData, gpuCoefficients, WORK_SIZE, SIM_TIME * STEPS_PER_SECOND, TEMP_PER_SEC/(STEPS_PER_SECOND*1.0f), DX, DT, ERR_LIMIT);

	CUDA_CHECK_RETURN(cudaMemcpy(result, gpuData, sizeof(float)*(WORK_SIZE), cudaMemcpyDeviceToHost));


	/* Verify the results */
	for (int i=0; i < WORK_SIZE; ++i)
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
static void CheckCudaErrorAux (const char *file, unsigned line, const char *statement, cudaError_t err)
{
	if (err == cudaSuccess)
		return;
	std::cerr << statement<<" returned " << cudaGetErrorString(err) << "("<<err<< ") at "<<file<<":"<<line << std::endl;
	exit (1);
}

