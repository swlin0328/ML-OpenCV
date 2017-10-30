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
#include <sstream>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "opencv2/core/core.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "dataManipulate.h"

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

	void detectFaces(Mat frame, CascadeClassifier face_cascade, CascadeClassifier eye_cascade);

	int detectEye(Mat& srcImage, Mat& target, Rect& eyeRect, CascadeClassifier face_cascade, CascadeClassifier eye_cascade);
	
	void trackEye(Mat& srcImage, Mat& target, Rect& eyeRect);

	//車牌辨識
	Mat detect_License_Plate(Mat& srcImage);

	vector<Mat> extract_License_Plate(Mat& srcImage);

	vector<Mat> extract_License_Plate_by_MorphologyEx(Mat& srcImg);

	vector<Rect> mserGetPlate(Mat srcImage); //Debug模式lib有bug 請使用Release Mode

	Mat char_feature(Mat srcImage);

	//特徵點鑑別
	Mat getRansacMat(const vector<DMatch>& matches, vector<KeyPoint>& keypoints1, vector<KeyPoint>& keypoints2, vector<DMatch>& outMatches, bool refineF = true);

	//影像穩定
	void videoOutput(Ptr<videostab::IFrameSource> stabFrames, string outputPath, double outputFps = 20);

	void cacStabVideo(Ptr<videostab::IFrameSource> stabFrames, string inputPath, string outputPath);

	//背景建模
	void detectBackGround(Ptr<BackgroundSubtractorKNN> pBackgroundKnn, string videoFileName);

	//運動目標檢測
	vector<Rect> get_foreground_objects(Mat scene, Ptr<BackgroundSubtractorKNN> pBackgrounndKnn, double scale, bool isFlag);
}