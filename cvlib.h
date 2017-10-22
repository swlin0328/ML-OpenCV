#include <cmath>
#include <vector>
#include <numeric>
#include <random>
#include <functional>
#include <ctime>
#include <algorithm>
#include <chrono>
#include <fstream>
#include <cassert>
#include <map>
#include <set>
#include <iostream>
#include <Windows.h>
#include "opencv2\opencv.hpp"

#define __cvlib

using namespace std;
using namespace cv;

namespace cv_lib
{
	//影像處理
	void regionExtraction(Mat& srcImage, int xRoi, int yRoi, int widthRoi, int heightRoi);

	int OTSU(Mat srcImage);

	void show_Gray_Histogram(Mat& srcImage);

	void show_RGB_Histogram(Mat& srcImage);

	void histogram_Comparison(Mat& srcImage1, Mat& srcImage2);

	Mat gamma_Transform(Mat& srcImage, float kFactor);

	Mat linear_Transform(Mat srcImage, float a, int b);

	Mat log_Transform(Mat srcImage, int c);

	Mat grayLayered(Mat& srcImage);

	void showMBitPlan(Mat srcImage);

	float calculateCurrentEntropy(Mat hist, int threshold);

	Mat maxEntropySegMentation(Mat inputImage);

	Mat Myfilter2D(Mat srcImage);

	Mat filter2D_(Mat srcImage);

	Mat DFT(Mat srcImage);

	void convolution(Mat src, Mat kernel, Mat& dst);

	Mat addSaltNoise(const Mat srcImage, int n);

	double generateGaussianNoise(double mu, double sigma);

	void myMedianBlur(Mat& src, Mat& dst, const int kSize);

	void myGaussianBlur(const Mat& src, Mat& result, int besarKernel, double delta);

	Mat guidedfilter(Mat& srcImage, Mat& srcClone, int r, double eps);

	//LUT反色
	Mat inverseColor(Mat& srcImage);

	void showManyImages(const vector<Mat>& srcImages, Size imgSize);

	//圖像比對
	//計算PSNR峰值信噪比，返回值為30~50dB，值越大越好
	double PSNR(const Mat& Img1, const Mat& Img2);

	//計算MSSIM結構相似性，返回值從0到1，值越大越好
	Scalar MSSIM(const Mat& Img1, const Mat& Img2);

	//影像讀取
	void readImgNamefromFile(char* fileName, vector<string>& imgNames);
}