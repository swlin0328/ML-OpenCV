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
#include "opencv2/core/core.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2\opencv.hpp"

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

	//特徵匹配
	Mat cacORBFeatureAndCompare(Mat srcImage1, Mat srcImage2);

	//人臉辨識
	Mat dectect_Skin_Color(Mat& srcImage);

	//車牌辨識
	Mat detect_License_Plate(Mat& srcImage);

	vector<Mat> extract_License_Plate(Mat& srcImage);

	vector<Mat> extract_License_Plate_by_MorphologyEx(Mat& srcGray, int width, int height);
}