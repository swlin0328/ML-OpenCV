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
#include <opencv2/opencv.hpp>
#include "opencv2/core/core.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/highgui/highgui.hpp"

#define __cv_objDetect

#ifndef __cvlib
#define __cvlib
#include "cvlib.h"
#endif

using namespace std;
using namespace cv;

namespace cv_lib
{
	//尺度變換
	void CreateScaleSpace(Mat srcImage, vector<vector<Mat>>& ScaleSpace, vector<vector<Mat>>& DoG);

	//特徵檢測
	Mat cacORBFeatureAndCompare(Mat srcImage1, Mat srcImage2); //ORB特徵匹配

	vector<Mat> calculateIntegralHOG(Mat& srcMat, int THETA); //HOG積分圖

	void calculateHOGinCell(Mat& HOGCellMat, Rect roi, vector<Mat>& integrals); //區域積分直方圖

	Mat getHOG(Point pt, vector<Mat>& integrals, int cellsize, int blocksize, int THETA);

	vector<Mat> cacHOGFeature(Mat srcImage, int cellsize = -1, int THETA = 20);

	double HaarExtract(Mat srcImage, int type, Rect roi);

	double calIntegral(Mat srcIntegral, int x, int y, int width, int height);

	Mat OLBP(Mat& srcImage);

	//人臉辨識
	Mat dectect_Skin_Color(Mat& srcImage);

	//車牌辨識
	Mat detect_License_Plate(Mat& srcImage);

	vector<Mat> extract_License_Plate(Mat& srcImage);

	vector<Mat> extract_License_Plate_by_MorphologyEx(Mat& srcGray, int width, int height);

	vector<Rect> mserGetPlate(Mat srcImage);

	Mat char_feature(Mat srcImage);
}