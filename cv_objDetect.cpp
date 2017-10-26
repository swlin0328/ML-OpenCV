/*****************************************************************************
----------------------------Warning----------------------------------------

此段程式碼僅供 林書緯本人 履歷專用作品集，未經許可請勿使用與散播
部分程式碼改自

---電子工業出版社, "OpenCV 圖像處理編程實例", 朱偉,趙春光 等編著", ISBN 978-7-121-28573-8
的C++演算法程式碼

---碁峰, "The C++ Programming Language", Bjarne Stroustrup, ISBN 978-986-347-603-0
的C++範例程式

---code by 林書緯 2017/09/26
******************************************************************************/
#include "cv_objDetect.h"

//影像辨識
namespace cv_lib
{
	Mat detect_License_Plate(Mat& srcImage)
	{
		/*
		車牌背景底色範圍
		藍色通道限定範圍 0.35 < H < 0.7, S > 0.1, I > 0.1
		黃色通道限定範圍 H < 0.4, S > 0.1, I > 0.3
		黑色通道限定範圍 I < 0.5
		白色通道限定範圍 S < 0.4, I > 0.5
		*/
		vector<Mat> hsvImage = hsv_Analysis(srcImage);
		Mat bw_blue = ((hsvImage[0] > 0.45) & (hsvImage[0] < 0.75) & (hsvImage[1] > 0.15) & (hsvImage[2] > 0.25));
		int height = bw_blue.rows;
		int width = bw_blue.cols;
		Mat bw_blue_edge = Mat::zeros(bw_blue.size(), bw_blue.type());

		Mat sobelMat;
		SobelVerEdge(srcImage, sobelMat);

		imshow("bw_blue", bw_blue);
		waitKey(0);

		for (int i = 1; i < height - 2; i++)
		{
			for (int j = 1; j < width - 2; j++)
			{
				Rect rct;
				rct.x = j - 1;
				rct.y = i - 1;
				rct.height = 3;
				rct.width = 3;

				if ((sobelMat.at<uchar>(i, j) == 255) && countNonZero(bw_blue(rct) >= 1))
				{
					bw_blue_edge.at<uchar>(i, j) = 255;
				}
			}
		}
		return bw_blue_edge;
	}

	vector<Mat> extract_License_Plate(Mat& srcImage)
	{
		Mat morph, bw_blue_edge = detect_License_Plate(srcImage);
		vector<Mat> plates;
		morphologyEx(bw_blue_edge, morph, MORPH_CLOSE, Mat::ones(2, 25, CV_8UC1));
		imshow("morphology_bw_blue_edge", bw_blue_edge);
		waitKey(0);

		vector<vector<Point>> region_contours;
		findContours(morph.clone(), region_contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE, Point(0, 0));
		vector<Rect> candidatets;
		vector<Mat> candidate_img;

		for (int n = 0; n != region_contours.size(); n++)
		{
			Rect rect = boundingRect(region_contours[n]);
			int sub = countNonZero(morph(rect));
			double ratio = double(sub) / rect.area();
			double wh_ratio = double(rect.width) / rect.height;

			if (ratio > 0.5 && wh_ratio > 2 && wh_ratio < 5 && rect.height > 12 && rect.width > 60)
			{
				imshow("rect", srcImage(rect));
				plates.push_back(srcImage(rect));
				waitKey(0);
			}
		}
		return plates;
	}

	vector<Mat> extract_License_Plate_by_MorphologyEx(Mat& srcGray, int width, int height)
	{
		Mat result;
		vector<Mat> plates;
		morphologyEx(srcGray, result, MORPH_GRADIENT, Mat(1, 2, CV_8U, Scalar(1)));
		threshold(result, result, 255 * (0.1), 255, THRESH_BINARY);

		if (width >= 400 && width < 600)
		{
			morphologyEx(result, result, MORPH_CLOSE, Mat(1, 25, CV_8U, Scalar(1)));
		}
		else if (width >= 200 && width < 300)
		{
			morphologyEx(result, result, MORPH_CLOSE, Mat(1, 20, CV_8U, Scalar(1)));
		}
		else if (width >= 600)
		{
			morphologyEx(result, result, MORPH_CLOSE, Mat(1, 28, CV_8U, Scalar(1)));
		}
		else
		{
			morphologyEx(result, result, MORPH_CLOSE, Mat(1, 15, CV_8U, Scalar(1)));
		}

		if (height >= 400 && height < 600)
		{
			morphologyEx(result, result, MORPH_CLOSE, Mat(8, 1, CV_8U, Scalar(1)));
		}
		else if (height >= 200 && height < 300)
		{
			morphologyEx(result, result, MORPH_CLOSE, Mat(6, 1, CV_8U, Scalar(1)));
		}
		else if (height >= 600)
		{
			morphologyEx(result, result, MORPH_CLOSE, Mat(10, 1, CV_8U, Scalar(1)));
		}
		else
		{
			morphologyEx(result, result, MORPH_CLOSE, Mat(4, 1, CV_8U, Scalar(1)));
		}
		vector<vector<Point>> blue_contours;
		vector<Rect> blue_rect;
		findContours(result.clone(), blue_contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE, Point(0, 0));

		for (int i = 0; i < blue_contours.size(); i++)
		{
			Rect rect = boundingRect(blue_contours[i]);
			double wh_ratio = double(rect.width) / rect.height;
			int sub = countNonZero(result(rect));
			double ratio = double(sub) / rect.area();

			if (wh_ratio > 2 && wh_ratio < 8 && rect.height > 12 && rect.width > 60 && ratio > 0.4)
			{
				blue_rect.push_back(rect);
				plates.push_back(srcGray(rect));
				imshow("rect", srcGray(rect));
				waitKey(0);
			}
		}
		imshow("result", result);
		waitKey(0);
		return plates;
	}

	Mat dectect_Skin_Color(Mat& srcImage)
	{
		/*
		YCbCr---Y為亮度，Cb為藍色分量，Cr為紅色分量
		避免光罩影響，放棄亮度通道
		膚色近似在CbCr橢圓範圍內
		*/

		Mat resultMat;
		if (srcImage.empty())
		{
			return Mat();
		}
		Mat skinMat = Mat::zeros(Size(256, 256), CV_8UC1);
		ellipse(skinMat, Point(113, 155.6), Size(23.4, 15.2), 43.0, 0.0, 360.0, Scalar(255, 255, 255), -1);
		Mat struElemen = getStructuringElement(MORPH_RECT, Size(3, 3), Point(-1, -1));
		Mat YCrCbMat;
		Mat tempMat = Mat::zeros(srcImage.size(), CV_8UC1);
		cvtColor(srcImage, YCrCbMat, CV_BGR2YCrCb);

		for (int i = 0; i < srcImage.rows; i++)
		{
			uchar* p = (uchar*)tempMat.ptr<uchar>(i);
			Vec3b* YCrCb = (Vec3b*)YCrCbMat.ptr<Vec3b>(i);
			for (int j = 0; j < srcImage.cols; j++)
			{
				if (skinMat.at<uchar>(YCrCb[j][1]), YCrCb[j][2] > 0)
				{
					p[j] = 255;
				}
			}
		}
		morphologyEx(tempMat, tempMat, MORPH_CLOSE, struElemen);

		vector<vector<Point>> contours;
		vector<vector<Point>> resContours;
		vector<Vec4i> hierarchy;
		findContours(tempMat, contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);

		for (int i = 0; i < contours.size(); i++)
		{
			if (fabs(contourArea(Mat(contours[i]))) > 1000)
			{
				resContours.push_back(contours[i]);
			}
		}
		tempMat.setTo(0);
		drawContours(tempMat, resContours, -1, Scalar(255, 0, 0), CV_FILLED);
		srcImage.copyTo(resultMat, tempMat);
		return resultMat;
	}

	void CreateScaleSpace(Mat srcImage, vector<vector<Mat>>& ScaleSpace, vector<vector<Mat>>& DoG)
	{
		Size ksize(5, 5);
		Mat srcBlurMat, up, down;

		GaussianBlur(srcImage, srcBlurMat, ksize, 0.5);
		pyrUp(srcBlurMat, up);
		up.copyTo(ScaleSpace[0][0]);
		GaussianBlur(ScaleSpace[0][0], ScaleSpace[0][0], ksize, 1.0);

		for (int i = 0; i < 4; i++)
		{
			double sigma = 1.4142135;
			for (int j = 0; j < 5+2; j++)
			{
				sigma = sigma * pow(2.0, j / 2.0);
				GaussianBlur(ScaleSpace[i][j], ScaleSpace[i][j + 1], ksize, sigma);
				DoG[i][j] = ScaleSpace[i][j] - ScaleSpace[i][j + 1];
			}

			if (i < 3)
			{
				pyrDown(ScaleSpace[i][0], down);
				down.copyTo(ScaleSpace[i + 1][0]);
			}
		}
	}

	Mat cacORBFeatureAndCompare(Mat srcImage1, Mat srcImage2)
	{
		CV_Assert(!srcImage1.empty() && !srcImage2.empty());

		Mat desciptorMat1, desciptorMat2, matchMat;
		vector<KeyPoint> keyPoint1, keyPoint2;

		Ptr<ORB> orb = ORB::create();
		orb->detect(srcImage1, keyPoint1);
		orb->detect(srcImage2, keyPoint2);
		orb->compute(srcImage1, keyPoint1, desciptorMat1);
		orb->compute(srcImage2, keyPoint2, desciptorMat2);

		BFMatcher matcher(NORM_HAMMING);
		vector<DMatch> matches;
		matcher.match(desciptorMat1, desciptorMat2, matches);
		drawMatches(srcImage1, keyPoint1, srcImage2, keyPoint2, matches, matchMat);
		imshow("Matches", matchMat);

		return matchMat;
	}
}