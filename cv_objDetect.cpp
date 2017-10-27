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

	vector<Mat> calculateIntegralHOG(Mat& srcMat, int THETA)
	{
		Mat sobelMatX, sobelMatY;
		int NBINS = 360 / THETA;
		Sobel(srcMat, sobelMatX, CV_32F, 1, 0);
		Sobel(srcMat, sobelMatY, CV_32F, 0, 1);
		vector<Mat> bins(NBINS);
		
		for (int i = 0; i < NBINS; i++)
		{
			bins[i] = Mat::zeros(srcMat.size(), CV_32F);
		}

		Mat magnMat, angleMat;
		cartToPolar(sobelMatX, sobelMatY, magnMat, angleMat, true);
		add(angleMat, Scalar(180), angleMat, angleMat < 0);
		add(angleMat, Scalar(-180), angleMat, angleMat >= 0);
		angleMat /= THETA;

		for (int y = 0; y < srcMat.rows; y++)
		{
			for (int x = 0; x < srcMat.cols; x++)
			{
				int ind = angleMat.at<float>(y, x);
				bins[ind].at<float>(y, x) += magnMat.at<float>(y, x);
			}
		}

		vector<Mat> integrals(NBINS);
		for (int i = 0; i < NBINS; i++)
		{
			integral(bins[i], integrals[i]);
		}
		return integrals;
	}

	void calculateHOGinCell(Mat& HOGCellMat, Rect roi, vector<Mat>& integrals)
	{
		int x0 = roi.x;
		int y0 = roi.y;
		int x1 = x0 + roi.width;
		int y1 = y0 + roi.height;

		for (int i = 0; i < integrals.size(); i++)
		{
			Mat integral = integrals[i];
			float a = integral.at<double>(y0, x0);
			float b = integral.at<double>(y1, x1);
			float c = integral.at<double>(y0, x1);
			float d = integral.at<double>(y1, x0);
			HOGCellMat.at<float>(0, i) = (a + b) - (c + d);
		}
	}

	Mat getHOG(Point pt, vector<Mat>& integrals, int cellsize, int blocksize, int THETA)
	{
		int R = cellsize / 2;
		if (pt.x - R < 0 || pt.y - R < 0 || pt.x + R >= integrals[0].cols || pt.y + R >= integrals[0].rows)
		{
			return Mat();
		}

		int NBINS = 360 / THETA;
		Mat hist(Size(NBINS * pow(blocksize, 2), 1), CV_32F);
		Point tl(pt.x - R, pt.y - R);
		int c = 0;

		for (int i = 0; i < blocksize; i++)
		{
			for (int j = 0; j < blocksize; j++)
			{
				Rect roi(tl, tl + Point(cellsize, cellsize));
				Mat hist_temp = hist.colRange(c, c + NBINS);
				calculateHOGinCell(hist_temp, roi, integrals);
				tl.x += cellsize;
				c += NBINS;
			}
			tl.x = pt.x - R;
			tl.y += cellsize;
		}
		normalize(hist, hist, 1, 0, NORM_L2);
		return hist;
	}

	vector<Mat> cacHOGFeature(Mat srcImage, int cellsize, int THETA)
	{
		if (cellsize == -1)
		{
			cellsize = srcImage.cols > srcImage.rows? srcImage.rows/8 : srcImage.cols/8;
		}

		Mat grayImage;
		vector<Mat> HOGMatVector;
		cvtColor(srcImage, grayImage, CV_RGB2GRAY);
		grayImage.convertTo(grayImage, CV_8UC1);

		int blocksize = 2; //block = 2*2*cellsize
		int NBINS = 360 / THETA;
		Mat HOGBlockMat(Size(NBINS, 1), CV_32F);

		vector<Mat> integrals = calculateIntegralHOG(grayImage, THETA);
		Mat image = grayImage.clone();
		image *= 0.5;

		for (int y = cellsize / 2; y < grayImage.rows; y += cellsize)
		{
			for (int x = cellsize / 2; x < grayImage.cols; x += cellsize)
			{
				Mat hist = getHOG(Point(x, y), integrals, cellsize, blocksize, THETA);
				if (hist.empty());
				{
					continue;
				}

				HOGBlockMat = Scalar(0);
				for (int i = 0; i < NBINS; i++)
				{
					for (int j = 0; j < blocksize; j++)
					{
						HOGBlockMat.at<float>(0, i) += hist.at<float>(0, i + j*NBINS);
					}
				}
				normalize(HOGBlockMat, HOGBlockMat, 1, 0, CV_L2);
				HOGMatVector.push_back(HOGBlockMat);

				Point center(x, y);
				for (int i = 0; i < NBINS; i++)
				{
					double theta = (i*THETA + 90.0) * CV_PI / 180.0;
					Point rd(cellsize*0.5*cos(theta), cellsize*0.5*sin(theta));

					Point rp = center - rd;
					Point lp = center + rd;
					line(image, rp, lp, Scalar(255 * HOGBlockMat.at<float>(0, i), 255, 255));
				}
			}
		}
		imshow("out", image);
		return HOGMatVector;
	}

	Mat OLBP(Mat& srcImage)
	{
		int nRows = srcImage.rows;
		int nCols = srcImage.cols;
		Mat resultMat(srcImage.size(), srcImage.type());

		for (int y = 1; y < nRows - 1; y++)
		{
			for (int x = 1; x < nCols - 1; x++)
			{
				uchar neighbor[8] = { 0 };
				neighbor[0] = srcImage.at<uchar>(y - 1, x - 1);
				neighbor[1] = srcImage.at<uchar>(y - 1, x);
				neighbor[2] = srcImage.at<uchar>(y - 1, x + 1);
				neighbor[3] = srcImage.at<uchar>(y, x + 1);
				neighbor[4] = srcImage.at<uchar>(y + 1, x + 1);
				neighbor[5] = srcImage.at<uchar>(y + 1, x);
				neighbor[6] = srcImage.at<uchar>(y + 1, x - 1);
				neighbor[7] = srcImage.at<uchar>(y, x - 1);

				uchar center = srcImage.at<uchar>(y, x);
				uchar temp = 0;
				for (int k = 0; k < 8; k++)
				{
					temp += (neighbor[k] >= center) * (1 << k);
				}
				resultMat.at<uchar>(y, x) = temp;
			}
			return resultMat;
		}
	}

	double HaarExtract(Mat srcImage, int type, Rect roi)
	{
		double value;
		double wh1, wh2;
		double bk1, bk2;
		int x = roi.x;
		int y = roi.y;
		int width = roi.width;
		int height = roi.height;

		Mat grayImage, integralImg;
		cvtColor(srcImage, grayImage, CV_RGB2GRAY);
		integral(grayImage, integralImg);

		switch (type)
		{
		//Haar 水平邊緣
		case 0:
			wh1 = calIntegral(integralImg, x, y, width, height);
			bk1 = calIntegral(integralImg, x + width, y, width, height);
			value = (wh1 - bk1) / static_cast<double> (width * height);
			break;
		//Haar 垂直邊緣
		case 1:
			wh1 = calIntegral(integralImg, x, y, width, height);
			bk1 = calIntegral(integralImg, x, y + height, width, height);
			value = (wh1 - bk1) / static_cast<double> (width * height);
			break;
		//Haar 水平線型
		case 2:
			wh1 = calIntegral(integralImg, x, y, width * 3, height);
			bk1 = calIntegral(integralImg, x + width, y, width, height);
			value = (wh1 - 3.0*bk1) / static_cast<double> (2.0*width * height);
			break;
		//Haar 垂直線型
		case 3:
			wh1 = calIntegral(integralImg, x, y, width, height * 3);
			bk1 = calIntegral(integralImg, x, y + height, width, height);
			value = (wh1 - 3.0*bk1) / static_cast<double> (2.0*width * height);
			break;
		//Haar 棋盤型
		case 4:
			wh1 = calIntegral(integralImg, x, y, width * 2, height * 2);
			bk1 = calIntegral(integralImg, x + width, y, width, height);
			bk2 = calIntegral(integralImg, x, y + height, width, height);
			value = (wh1 - 2.0*(bk1 + bk2)) / static_cast<double> (2.0*width * height);
			break;
		//Haar 中心包圍型
		case 5:
			wh1 = calIntegral(integralImg, x, y, width * 3, height * 3);
			bk1 = calIntegral(integralImg, x + width, y + height, width, height);
			value = (wh1 - 9.0*bk1) / static_cast<double> (8 * width * height);
			break;
		default:
			cerr << "No type!\n";
		}
		return value;
	}

	double calIntegral(Mat srcIntegral, int x, int y, int width, int height)
	{
		double term_1 = srcIntegral.at<double>(y - 1 + height, x - 1 + width);
		double term_2 = srcIntegral.at<double>(y - 1, x - 1);
		double term_3 = srcIntegral.at<double>(y - 1 + height, x - 1);
		double term_4 = srcIntegral.at<double>(y - 1, x - 1 + width);

		return (term_1 + term_2) - (term_3 + term_4);
	}

	vector<Rect> mserGetPlate(Mat srcImage)
	{
		Mat gray, gray_neg;
		cvtColor(srcImage, gray, CV_BGR2GRAY);
		gray_neg = 255 - gray;

		Ptr<MSER> regMser = MSER::create(2, 10, 5000, 0.5, 0.3);
		vector<vector<Point>> regContours;
		std::vector<cv::Rect> regRects;
		regMser->detectRegions(gray, regContours, regRects);

		Ptr<MSER> charMser = MSER::create(2, 2, 400, 0.1, 0.3);
		vector<vector<Point>> charContours;
		std::vector<cv::Rect> charRects;
		charMser->detectRegions(gray_neg, charContours, charRects);

		Mat mserMapMat = Mat::zeros(srcImage.size(), CV_8UC1);
		Mat mserNegMapMat = Mat::zeros(srcImage.size(), CV_8UC1);
		for (int i = 0 - 1; i < regContours.size(); i++)
		{
			const vector<Point>& r = regContours[i];
			for (int j = 0; j < r.size(); j++)
			{
				Point pt = r[j];
				mserMapMat.at<unsigned char>(pt) = 255;
			}
		}

		for (int i = 0; i < charContours.size(); i++)
		{
			const vector<Point>& r = charContours[i];
			for (int j = 0; j < r.size(); j++)
			{
				Point pt = r[j];
				mserNegMapMat.at<unsigned char>(pt) = 255;
			}
		}

		Mat mserResMat;
		mserResMat = mserMapMat & mserNegMapMat;
		imshow("mserMapMat", mserMapMat);
		imshow("mserNegMapMat", mserNegMapMat);
		imshow("mserResMat", mserResMat);

		Mat mserClosedMat;
		morphologyEx(mserResMat, mserClosedMat, MORPH_CLOSE, Mat::ones(1, 20, CV_8UC1));
		imshow("mserClosedMat", mserClosedMat);

		vector<vector<Point>> plate_contours;
		findContours(mserClosedMat, plate_contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE, Point(0, 0));	
		vector<Rect> candidates;

		for (int i = 0; i < plate_contours.size(); i++)
		{
			Rect rect = boundingRect(plate_contours[i]);
			double wh_ratio = rect.width / double(rect.height);
			if (rect.height > 20 && wh_ratio > 4 && wh_ratio < 7)
			{
				candidates.push_back(rect);
			}
		}
		return candidates;
	}

	Mat char_feature(Mat srcImage)
	{
		Mat gray;
		cvtColor(srcImage, gray, CV_BGR2GRAY);
		resize(gray, gray, Size(16, 32));
		equalizeHist(gray, gray);
		gray.convertTo(gray, CV_32FC1);
		gray /= norm(gray, NORM_L2);

		Mat sobel_v_kernel = (Mat_<float>(3, 3) <<
			-0.125, 0, 0.125,
			-0.250, 0, 0.250,
			-0.125, 0, 0.125);

		Mat sobel_h_kernel = (Mat_<float>(3, 3) <<
			-0.125, -0.250, -0.125,
			0, 0, 0,
			0.125, 0.250, 0.125);

		Mat h_edges, v_edges;
		filter2D(gray, h_edges, gray.type(), sobel_h_kernel, Point(-1, -1), 0, BORDER_CONSTANT);
		filter2D(gray, v_edges, gray.type(), sobel_v_kernel, Point(-1, -1), 0, BORDER_CONSTANT);

		Mat magnitude = Mat(h_edges.size(), CV_32FC1);
		Mat angle = Mat(h_edges, CV_32FC1);
		cartToPolar(v_edges, h_edges, magnitude, angle);

		Mat eight_direction[8];
		float* eight_ptr[8];
		float thre[9] = { 0, CV_PI / 4, CV_PI / 2, CV_PI * 3 / 4, CV_PI, CV_PI * 5 / 4, CV_PI * 6 / 4, CV_PI * 7 / 4, CV_PI * 2 };
		for (int i = 0; i < 8; i++)
		{
			eight_direction[i] = Mat::zeros(h_edges.size(), CV_32FC1);
			eight_ptr[i] = (float*)eight_direction[i].data;
		}

		float* ang_ptr = (float*)angle.data;
		float* mag_ptr = (float*)magnitude.data;
		for (int i = 0; i < h_edges.total(); i++, ang_ptr++, mag_ptr++)
		{
			for (int j = 0; j < 8; j++)
			{
				if ((*ang_ptr) >= thre[j] && (*ang_ptr) < thre[j + 1])
				{
					*eight_ptr[j] = *mag_ptr;
				}
				++eight_ptr[j];
			}
		}

		Mat feature = Mat::zeros(1, 112, CV_32FC1);
		float* fea_ptr = (float*)feature.data;
		//8個模長矩陣，4*2個cell，共64維特徵向量 (cellsize = 8*8 pixel)
		for (int i = 0; i < 8; i++)
		{
			for (int j = 0; j < 2; j++)
			{
				for (int k = 0; k < 4; k++)
				{
					Rect roi;
					roi.x = j * 8;
					roi.y = k * 8;
					roi.width = roi.height = 8;
					*(fea_ptr++) = sum(eight_direction[i](roi)).val[0];
				}
			}
		}

		Mat proj_row, proj_col;
		reduce(gray, proj_row, 0, CV_REDUCE_SUM);
		reduce(gray, proj_col, 1, CV_REDUCE_SUM);
		float* tp = (float*)feature.data;
		float* dd = (float*)proj_row.data;

		for (int i = 64; i < 80; i++)
		{
			tp[i] = dd[i - 64];
		}
		feature.colRange(80, 112) = proj_col.t();
		return feature;
	}
}