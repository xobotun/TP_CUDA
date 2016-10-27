#include <stdio.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>
#include <fstream>
#include <numeric>
#include <stdlib.h>
#include <corona.h>

static void CheckCudaErrorAux(const char *, unsigned, const char *, cudaError_t);
#define CUDA_CHECK_RETURN(value) CheckCudaErrorAux(__FILE__,__LINE__, #value, value)
typedef unsigned char (*ConvolutionFunction)(unsigned char*, int, int, int, int);

__device__ __host__ unsigned char smoothe(unsigned char* pixels, int imgWidth, int imgHeight, int pixelPosX, int pixelPosY) {
	const unsigned char gaussMatrixSize = 5;
	const char gaussMatrix[] =	  { 2, 4, 5, 4, 2,
									4, 9,12, 9, 4,
									5,12,15,12, 5,
									4, 9,12, 9, 4,
									2, 4, 5, 4, 2 };

	int normalizer = 0;
	int nonNormalizedResult = 0;

	for (int dy = 0; dy <= gaussMatrixSize; ++dy)
		for (int dx = 0; dx <= gaussMatrixSize; ++dx) {
			int x = pixelPosX + dx - gaussMatrixSize/2;
			int y = pixelPosY + dy - gaussMatrixSize/2;

			if (x > 0 && y > 0 && x < imgWidth && y < imgHeight) {
				nonNormalizedResult += pixels[y*imgWidth + x] * gaussMatrix[(gaussMatrixSize*gaussMatrixSize -1) - y*imgWidth - x];
				normalizer += gaussMatrix[(gaussMatrixSize*gaussMatrixSize -1) - y*imgWidth - x];
			}
		}

	return nonNormalizedResult / normalizer;
}

__device__ __host__ unsigned char findBorders(unsigned char* pixels, int imgWidth, int imgHeight, int pixelPosX, int pixelPosY) {
	const char edgeDetectionMatrixSize = 3;
	const char horizontalEdgeDetectionMatrix[] =  { -1,  0, -1,
													-2,  0, -1,
													-1,  0, -1 };

	const char verticalEdgeDetectionMatrix[] =    { -1, -2, -1,
													-0,  0,  0,
													-1, -2, -1 };
	const unsigned char minThreshold = 100;
	const unsigned char maxThreshold = 200;

	int resultX = 0;
	int resultY = 0;

	for (int dy = 0; dy <= edgeDetectionMatrixSize; ++dy)
		for (int dx = 0; dx <= edgeDetectionMatrixSize; ++dx) {
			int x = pixelPosX + dx - edgeDetectionMatrixSize/2;
			int y = pixelPosY + dy - edgeDetectionMatrixSize/2;

			if (x > 0 && y > 0 && x < imgWidth && y < imgHeight) {
				resultX += pixels[y*imgWidth + x] * horizontalEdgeDetectionMatrix[(edgeDetectionMatrixSize*edgeDetectionMatrixSize - 1) - y*imgWidth - x];
				resultY += pixels[y*imgWidth + x] * verticalEdgeDetectionMatrix[(edgeDetectionMatrixSize*edgeDetectionMatrixSize - 1) - y*imgWidth - x];
			}
		}

	int result = roundf(sqrtf(resultX*resultX + resultY*resultY));

	if (false && result > minThreshold && result < maxThreshold)
		result = 0;

	return result;
}

void readImage(unsigned char* &pixels, int &width, int &height, std::string filename) {
	corona::Image* image = corona::OpenImage(filename.data(), corona::PF_R8G8B8);
	if (!image) {
		std::cerr << "Error while opening an image!" << std::endl;
		system("pause");
		exit(1);
	}

	width = image->getWidth();
	height = image->getHeight();
	void* coronaPixels = image->getPixels();

	pixels = new unsigned char[width * height];

	unsigned char* p = (unsigned char*)coronaPixels;
	for (int i = 0; i < width * height; ++i)
		pixels[i] = (*p++ + *p++ + *p++) / 3;
}
/*
void writeImage(unsigned char* pixels, int width, int height, std::string filename) {

	unsigned char* coronaPixels = new unsigned char[width * height * 3];
	for (int i = 0; i < width * height; i+=3)
		coronaPixels[i] = coronaPixels[i + 1] = coronaPixels[i + 2] = pixels[i / 3];


	corona::Image* image = corona::CreateImage(width, height, corona::PF_R8G8B8, coronaPixels);
	if (!image) {
		std::cerr << "Error while creating an image!" << std::endl;
		system("pause");
		exit(1);
	}

	bool success = corona::SaveImage(filename.data(), corona::FileFormat::FF_AUTODETECT, image);
	if (!success) {
		std::cerr << "Error while writing an image to disk!" << std::endl;
		system("pause");
		exit(1);
	}

	delete image;
	delete[] coronaPixels;
}
*/
void writeImage(unsigned char* pixels, int width, int height, std::string filename) {
	std::ofstream image;
	image.open(filename);

	for (int y = 0; y < height; ++y) {
		for (int x = 0; x < width - 1; ++x)
			image << (unsigned int)pixels[y*width + x] << ' ';
		image << (unsigned int)pixels[y*width + width - 1] << std::endl;
	}

	image.close();
}

int main(void) {
	unsigned char* pixels;
	int width, height;

	readImage(pixels, width, height, "img.jpg");

	// CUDA here.

	writeImage(pixels, width, height, "img.txt");

	delete[] pixels;
}

/**
* Check the return value of the CUDA runtime API call and exit
* the application if the call has failed.
*/
static void CheckCudaErrorAux(const char *file, unsigned line, const char *statement, cudaError_t err) {
	if (err == cudaSuccess)
		return;
	std::cerr << statement << " returned " << cudaGetErrorString(err) << "(" << err << ") at " << file << ":" << line << std::endl;
	system("pause");
	exit(1);
}