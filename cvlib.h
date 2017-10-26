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

#ifndef __cv_objDetect
#define __cv_objDetect
#include "cv_objDetect.h"
#endif

using namespace std;
using namespace cv;

namespace cv_lib
{
	//影像處理
	void regionExtraction(Mat& srcImage, int xRoi, int yRoi, int widthRoi, int heightRoi);

	int OTSU(Mat srcImage);

	Mat gamma_Transform(Mat& srcImage, float kFactor);

	Mat linear_Transform(Mat srcImage, float a, int b);

	Mat log_Transform(Mat srcImage, int c);

	Mat grayLayered(Mat& srcImage);

	Mat addSaltNoise(const Mat srcImage, int n);

	double generateGaussianNoise(double mu, double sigma);

	void cacBounding(Mat src);

	void cacBoundRectRandomDirection(Mat src);

	Mat inverseColor(Mat& srcImage); //LUT反色

	//分水嶺分割
	Mat displaySegResult(Mat& segments, int numOfSegments, Mat& image);

	Mat watershedSegment(Mat& srcImage, int& noOfSegments);

	void segMerge(Mat& image, Mat& segment, int& numSeg);

	//統計分析
	void show_Gray_Histogram(Mat& srcImage);

	void show_RGB_Histogram(Mat& srcImage);

	void histogram_Comparison(Mat& srcImage1, Mat& srcImage2);

	void showMBitPlan(Mat srcImage);

	void showManyImages(const vector<Mat>& srcImages, Size imgSize);

	float calculateCurrentEntropy(Mat hist, int threshold);

	Mat maxEntropySegMentation(Mat inputImage);

	Mat DFT(Mat srcImage);

	void convolution(Mat src, Mat kernel, Mat& dst);

	vector<Mat> hsv_Analysis(Mat& srcImage);

	//二維捲積
	Mat Myfilter2D(Mat srcImage);

	Mat filter2D_(Mat srcImage);

	//濾波、銳化
	bool SobelOptaEdge(const Mat& srcImage, Mat& resultImage, int flag);

	bool SobelVerEdge(Mat srcImage, Mat& resultImage);

	Mat roberts(Mat srcImage);

	void myMedianBlur(Mat& src, Mat& dst, const int kSize);

	void myGaussianBlur(const Mat& src, Mat& result, int besarKernel, double delta);

	Mat guidedfilter(Mat& srcImage, Mat& srcClone, int r, double eps);

	//角點檢測
	Mat MoravecCorners(Mat srcImage, int kSize, int threshold); //非均勻響應，受雜訊影響

	void CornerHarris(const Mat& srcImage, Mat& result, int blockSize, int kSize, double k); //blockSize, kSize, k 角點檢測參數

	//圖像比對
	double PSNR(const Mat& Img1, const Mat& Img2);  //計算PSNR峰值信噪比，返回值30~50dB，值越大越好

	Scalar MSSIM(const Mat& Img1, const Mat& Img2); //計算MSSIM結構相似性，返回值0到1，值越大越好

	//影像讀取
	void readImgNamefromFile(char* fileName, vector<string>& imgNames);
}