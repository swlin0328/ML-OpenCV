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
#include "cvlib.h"

//電腦視覺
namespace cv_lib
{
	double PSNR(const Mat& Img1, const Mat& Img2)
	{
		Mat s1;
		absdiff(Img1, Img2, s1);
		s1.convertTo(s1, CV_32F);
		s1 = s1.mul(s1);
		Scalar s = sum(s1);
		double sse = s.val[0] + s.val[1] + s.val[2];

		if (sse <= 1e-10)
		{
			return 0;
		}
		else
		{
			double mse = sse / (double)(Img1.channels() * Img1.total());
			double psnr = 10.0 * log10((255 * 255) / mse);
			return psnr;
		}
	}

	Scalar MSSIM(const Mat& Img1, const Mat& Img2)
	{
		const double C1 = 6.5025, C2 = 58.5525;
		Mat I1, I2;

		Img1.convertTo(I1, CV_32F);
		Img2.convertTo(I2, CV_32F);

		Mat I1_2 = I1.mul(I1);
		Mat I2_2 = I2.mul(I2);
		Mat I1_I2 = I1.mul(I2);

		Mat mu1, mu2;
		GaussianBlur(I1, mu1, Size(11, 11), 1.5);
		GaussianBlur(I2, mu2, Size(11, 11), 1.5);

		Mat mu1_2 = mu1.mul(mu1);
		Mat mu2_2 = mu2.mul(mu2);
		Mat mu1_mu2 = mu1.mul(mu2);
		Mat sigma1_2, sigma2_2, sigma12;

		GaussianBlur(I1_2, sigma1_2, Size(11, 11), 1.5);
		sigma1_2 -= mu1_2;
		GaussianBlur(I2_2, sigma2_2, Size(11, 11), 1.5);
		sigma2_2 -= mu2_2;
		GaussianBlur(I1_I2, sigma12, Size(11, 11), 1.5);
		sigma12 -= mu1_mu2;

		Mat t1, t2, t3;
		t1 = 2 * mu1_mu2 + C1;
		t2 = 2 * sigma12 + C2;
		t3 = t1.mul(t2);

		t1 = mu1_2 + mu2_2 + C1;
		t2 = sigma1_2 + sigma2_2 + C2;
		t1 = t1.mul(t2);

		Mat ssim_map;
		divide(t3, t1, ssim_map);
		Scalar mssim = mean(ssim_map);
		return mssim;
	}

	void regionExtraction(Mat& srcImage, int xRoi, int yRoi, int widthRoi, int heightRoi)
	{
		Mat roiImage(srcImage.rows, srcImage.cols, CV_8UC3);
		srcImage(Rect(xRoi, yRoi, widthRoi, heightRoi)).copyTo(roiImage);
		imshow("roiImage", roiImage);
		waitKey(0);
	}

	Mat inverseColor(Mat& srcImage)
	{
		int row = srcImage.rows;
		int col = srcImage.cols;
		Mat tempImage = srcImage.clone();

		if (srcImage.isContinuous() && tempImage.isContinuous())
		{
			row = 1;
			col = col * srcImage.rows * srcImage.channels();
		}

		for (int i = 0; i < row; i++)
		{
			const uchar* pSrcData = srcImage.ptr<uchar>(i);
			uchar* pResultData = tempImage.ptr<uchar>(i);

			Mat lookUpTable(1, 256, CV_8U);
			uchar* pData = lookUpTable.data;

			for (int j = 0; j < 256; j++)
			{
				pData[j] = 255 - j;
			}
			for (int j = 0; j < col; j++)
			{
				LUT(srcImage, lookUpTable, tempImage);
			}
		}
		return tempImage;
	}

	void showManyImages(const vector<Mat>& srcImages, Size imgSize)
	{
		int nNumImages = srcImages.size();
		Size nSizeWindows;

		if (nNumImages > 12)
		{
			cout << "Not More than 12 images!\n";
			return;
		}

		switch (nNumImages)
		{
		case 1: nSizeWindows = Size(1, 1); break;
		case 2: nSizeWindows = Size(2, 1); break;
		case 3:
		case 4: nSizeWindows = Size(2, 2); break;
		case 5:
		case 6: nSizeWindows = Size(3, 2); break;
		case 7:
		case 8: nSizeWindows = Size(4, 2); break;
		case 9: nSizeWindows = Size(3, 3); break;
		default: nSizeWindows = Size(4, 3); break;
		}

		int nShowImageSize = 200;
		int nSplitLineSize = 15;
		int nAroundLineSize = 50;

		const int imagesHeight = nShowImageSize * nSizeWindows.width + nAroundLineSize + (nSizeWindows.width - 1) * nSplitLineSize;
		const int imagesWidth = nShowImageSize*nSizeWindows.height + nAroundLineSize + (nSizeWindows.height - 1) * nSplitLineSize;

		cout << imagesWidth << " " << imagesHeight << endl;

		Mat showWindowImages(imagesWidth, imagesHeight, CV_8UC3, Scalar(0, 0, 0));
		int posX = (showWindowImages.cols - (nShowImageSize * nSizeWindows.width + (nSizeWindows.width - 1) * nSplitLineSize)) / 2;
		int posY = (showWindowImages.rows - (nShowImageSize * nSizeWindows.height + (nSizeWindows.height - 1) * nSplitLineSize)) / 2;

		cout << posX << " " << posY << endl;

		int tempPosX = posX;
		int tempPosY = posY;

		for (int i = 0; i < nNumImages; i++)
		{
			if ((i % nSizeWindows.width == 0) && (tempPosX != posX))
			{
				tempPosX = posX;
				tempPosY += (nSplitLineSize + nShowImageSize);
			}
			Mat tempImage = showWindowImages(Rect(tempPosX, tempPosY, nShowImageSize, nShowImageSize));
			resize(srcImages[i], tempImage, Size(nShowImageSize, nShowImageSize));
			tempPosX += (nSplitLineSize + nShowImageSize);
		}
		imshow("showWindowImages", showWindowImages);
	}

	void readImgNamefromFile(char* fileName, vector<string>& imgNames)
	{
		imgNames.clear();
		WIN32_FIND_DATA file;
		int i = 0;
		char tempFilePath[MAX_PATH + 1];
		char tempFileName[50];

		sprintf_s(tempFilePath, "%s/*", fileName);
		HANDLE handle = FindFirstFile(tempFilePath, &file);
		if (handle != INVALID_HANDLE_VALUE)
		{
			do
			{
				sprintf_s(tempFileName, "%s", fileName);
				imgNames.push_back(file.cFileName);
				imgNames[i].insert(0, tempFileName);
				i++;
			} while (FindNextFile(handle, &file));
		}
		FindClose(handle);
	}

	int OTSU(Mat srcImage)
	{
		int nCols = srcImage.cols;
		int nRows = srcImage.rows;
		int threshold = 0;
		int nSumPix[256];
		float nProDis[256];

		for (int i = 0; i < 256; i++)
		{
			nSumPix[i] = 0;
			nProDis[i] = 0;
		}

		for (int i = 0; i < nCols; i++)
		{
			for (int j = 0; j < nRows; j++)
			{
				nSumPix[(int)srcImage.at<uchar>(i, j)]++;
			}
		}

		for (int i = 0; i < 256; i++)
		{
			nProDis[i] = (float)nSumPix[i] / (nCols * nRows);
		}

		float w0, w1, u0_temp, u1_temp, u0, u1, delta_temp;
		double delta_max = 0.0;

		for (int i = 0; i < 256; i++)
		{
			w0 = w1 = u0_temp = u1_temp = u0 = u1 = delta_temp = 0;
			for (int j = 0; j < 256; j++)
			{
				if (j <= i)
				{
					w0 += nProDis[j];
					u0_temp += j * nProDis[j];
				}
			}
			u0 = u0_temp / w0;
			u1 = u1_temp / w1;
			delta_temp = (float)(w0 * w1 * pow((u0 - u1), 2));

			if (delta_temp > delta_max)
			{
				delta_max = delta_temp;
				threshold = i;
			}
		}
		return threshold;
	}

	void show_Gray_Histogram(Mat& srcImage)
	{
		if (srcImage.empty())
		{
			return;
		}
		Mat ImageGray;
		cvtColor(srcImage, ImageGray, CV_BGR2GRAY);
		const int channels[1] = { 0 };
		const int histSize[1] = { 256 };
		float pranges[2] = { 0,256 };
		const float* ranges[1] = { pranges };
		MatND hist;
		calcHist(&ImageGray, 1, channels, Mat(), hist, 1, histSize, ranges);

		int hist_w = 500;
		int hist_h = 500;
		int nHistSize = 256;

		int bin_w = cvRound((double) hist_w / nHistSize);
		Mat histImage(hist_w, hist_h, CV_8UC3, Scalar(0, 0, 0));
		normalize(hist, hist, 0, histImage.rows, NORM_MINMAX, -1, Mat());
		for (int i = 1; i < nHistSize; i++)
		{
			line(histImage, Point( bin_w * (i-1), hist_h - cvRound(hist.at<float>(i-1))), Point(bin_w * (i), hist_h - cvRound(hist.at<float>(i))), Scalar(0, 0, 255), 2, 8, 0);
		}
		imshow("histImage", histImage);
		waitKey(0);
	}

	void show_RGB_Histogram(Mat& srcImage)
	{
		if (srcImage.empty())
		{
			return;
		}
		vector<Mat> bgr_planes;
		split(srcImage, bgr_planes);

		const int channels[1] = { 0 };
		const int histSize[1] = { 256 };
		float pranges[2] = { 0,256 };
		const float* ranges[1] = { pranges };
		bool uniform = true;
		bool accumulate = false;

		Mat b_hist, g_hist, r_hist;
		calcHist(&bgr_planes[0], 1, channels, Mat(), b_hist, 1, histSize, ranges, uniform, accumulate);
		calcHist(&bgr_planes[1], 1, channels, Mat(), g_hist, 1, histSize, ranges, uniform, accumulate);
		calcHist(&bgr_planes[2], 1, channels, Mat(), r_hist, 1, histSize, ranges, uniform, accumulate);

		int hist_w = 640;
		int hist_h = 512;
		int nHistSize = 256;

		int bin_w = cvRound((double)hist_w / nHistSize);
		Mat histImage(hist_w, hist_h, CV_8UC3, Scalar(0, 0, 0));
		normalize(b_hist, b_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat());
		normalize(g_hist, g_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat());
		normalize(r_hist, r_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat());

		for (int i = 1; i < nHistSize; i++)
		{
			line(histImage, Point(bin_w * (i - 1), hist_h - cvRound(b_hist.at<float>(i - 1))), Point(bin_w * (i), hist_h - cvRound(b_hist.at<float>(i))), Scalar(255, 0, 0), 2, 8, 0);
			line(histImage, Point(bin_w * (i - 1), hist_h - cvRound(g_hist.at<float>(i - 1))), Point(bin_w * (i), hist_h - cvRound(g_hist.at<float>(i))), Scalar(0, 255, 0), 2, 8, 0);
			line(histImage, Point(bin_w * (i - 1), hist_h - cvRound(r_hist.at<float>(i - 1))), Point(bin_w * (i), hist_h - cvRound(r_hist.at<float>(i))), Scalar(0, 0, 255), 2, 8, 0);
		}
		imshow("histImage", histImage);
		waitKey(0);
	}

	void histogram_Comparison(Mat& srcImage1, Mat& srcImage2)
	{
		if (srcImage1.empty() || srcImage2.empty())
		{
			return;
		}

		Mat hsv_img1, hsv_img2;
		Mat hsv_half_down;

		cvtColor(srcImage1, hsv_img1, CV_BGR2GRAY);
		cvtColor(srcImage2, hsv_img2, CV_BGR2GRAY);
		hsv_half_down = hsv_img1(Range(hsv_img1.rows / 2, hsv_img1.rows - 1), Range(0, hsv_img1.cols - 1));

		int h_bins = 50;
		int s_bins = 60;

		int histSize[] = { h_bins, s_bins };
		float h_ranges[] = { 0, 256 };
		float s_ranges[] = { 0, 180 };
		const float* ranges[] = { h_ranges, s_ranges };
		int channels[] = { 0, 1 };
		bool uniform = true;
		bool accumulate = false;

		MatND hist_half_down, hist_img1, hist_img2;
		calcHist(&hsv_img1, 1, channels, Mat(), hist_img1, 2, histSize, ranges, uniform, accumulate);
		normalize(hist_img1, hist_img1, 0, 1, NORM_MINMAX, -1, Mat());

		calcHist(&hsv_half_down, 1, channels, Mat(), hist_half_down, 2, histSize, ranges, uniform, accumulate);
		normalize(hist_half_down, hist_half_down, 0, 1, NORM_MINMAX, -1, Mat());

		calcHist(&hsv_img2, 1, channels, Mat(), hist_img2, 2, histSize, ranges, uniform, accumulate);
		normalize(hist_img2, hist_img2, 0, 1, NORM_MINMAX, -1, Mat());

		for (int i = 0; i < 4; i++)
		{
			int compare_method = i;
			double base_img1 = compareHist(hist_img1, hist_img1, compare_method);
			double base_img2 = compareHist(hist_img1, hist_img2, compare_method);
			double base_half = compareHist(hist_img1, hist_half_down, compare_method);
			printf("Method [%d] Perfect, Base-img2, Base-Half : %f, %f, %f \n", i, base_img1, base_img2, base_half);
		}
	}

	Mat gamma_Transform(Mat& srcImage, float kFactor)
	{
		unsigned char LUT[256];
		for (int i = 0; i < 256; i++)
		{
			LUT[i] = saturate_cast<uchar>(pow(i / 255.0, kFactor) * 255.0);
		}
		Mat resultImage = srcImage.clone();

		if (srcImage.channels() == 1)
		{
			for (auto iter = resultImage.begin<uchar>(); iter != resultImage.end<uchar>(); iter++)
			{
				*iter = LUT[*iter];
			}
		}
		else
		{
			for (auto iter = resultImage.begin<Vec3b>(); iter != resultImage.end<Vec3b>(); iter++)
			{
				(*iter)[0] = LUT[(*iter)[0]];
				(*iter)[1] = LUT[(*iter)[1]];
				(*iter)[2] = LUT[(*iter)[2]];
			}
		}
		return resultImage;
	}

	Mat linear_Transform(Mat srcImage, float a, int b)
	{
		if (srcImage.empty())
		{
			cout << "No Data!" << endl;
			return Mat();
		}
		const int nRows = srcImage.rows;
		const int nCols = srcImage.cols;
		Mat resultImage = Mat::zeros(srcImage.size(), srcImage.type());

		for (int i = 0; i < nRows; i++)
		{
			for (int j = 0; j < nCols; j++)
			{
				for (int c = 0; c < 3; c++)
				{
					resultImage.at<Vec3b>(i, j)[c] = saturate_cast<uchar>(a * (srcImage.at<Vec3b>(i,j)[c]) + b);
				}
			}
		}
		return resultImage;
	}

	Mat log_Transform(Mat srcImage, int c)
	{
		if (srcImage.empty())
		{
			cout << "No Data!" << endl;
			return Mat();
		}

		Mat resultImage = Mat::zeros(srcImage.size(), srcImage.type());
		add(srcImage, Scalar(1.0), srcImage);
		srcImage.convertTo(srcImage, CV_32F);
		log(srcImage, resultImage);
		resultImage = c * resultImage;
		normalize(resultImage, resultImage, 0, 255, NORM_MINMAX);
		convertScaleAbs(resultImage, resultImage);
		return resultImage;
	}

	Mat grayLayered(Mat& srcImage)
	{
		Mat resultImage = srcImage.clone();
		int nRows = resultImage.rows;
		int nCols = resultImage.cols;
		if (resultImage.isContinuous())
		{
			nCols = nCols * nRows;
			nRows = 1;
		}

		uchar *pDataMat;
		int controlMin = 150;

		for (int i = 0; i < nRows; i++)
		{
			pDataMat = resultImage.ptr<uchar>(i);
			for (int j = 0; j < nCols; j++)
			{
				if (pDataMat[i] > controlMin)
				{
					pDataMat[i] = 255;
				}
				else
				{
					pDataMat[i] = 0;
				}
			}
		}
		return resultImage;
	}

	void showMBitPlan(Mat srcImage)
	{
		int nRows = srcImage.rows;
		int nCols = srcImage.cols;

		if (srcImage.isContinuous())
		{
			nCols = nCols * nRows;
			nRows = 1;
		}
		
		uchar *pSrcMat, *pResultMat;
		Mat resultImage = srcImage.clone();
		int pixMax = 0, pixMin = 0;
		
		for (int n = 1; n <= 8; n++)
		{
			pixMin = pow(2, n - 1);
			pixMax = pow(2, n);
			
			for (int i = 0; i < nRows; i++)
			{
				pSrcMat = srcImage.ptr<uchar>(i);
				pResultMat = resultImage.ptr<uchar>(i);
				for (int j = 0; j < nCols; j++)
				{
					if (pSrcMat[j] >= pixMin && pSrcMat[j] < pixMax)
					{
						pResultMat[i] = 255;
					}
					else
					{
						pResultMat[i] = 0;
					}
				}
			}
			char windowsName[20];
			sprintf_s(windowsName, "BitPlane %d", n);
			imshow(windowsName, resultImage);
		}
	}

	float calculateCurrentEntropy(Mat hist, int threshold)
	{
		float BackgroundSum = 0, targetSum = 0;
		const float* pDataHist = (float*)hist.ptr<float>(0);
		for (int i = 0; i < 256; i++)
		{
			if (i < threshold)
			{
				BackgroundSum += pDataHist[i];
			}
			else
			{
				targetSum += pDataHist[i];
			}
		}
		float BackgroundEntropy = 0, targetEntropy = 0;
		
		for (int i = 0; i < 256; i++)
		{
			if (i < threshold)
			{
				if (pDataHist[i] == 0)
				{
					continue;
				}
				float ratio1 = pDataHist[i] / BackgroundSum;
				BackgroundEntropy += -ratio1 * logf(ratio1);
			}
			else
			{
				if (pDataHist[i] == 0)
				{
					continue;
				}
				float ratio2 = pDataHist[i] / targetSum;
				targetEntropy += -ratio2 * logf(ratio2);
			}
		}
		return (targetEntropy + BackgroundEntropy);
	}

	Mat maxEntropySegMentation(Mat inputImage)
	{
		const int channels[1] = { 0 };
		const int histSize[1] = { 256 };
		float prange[2] = { 0, 256 };
		const float* ranges[1] = { prange };
		MatND hist;
		calcHist(&inputImage, 1, channels, Mat(), hist, 1, histSize, ranges);
		float maxentropy = 0;
		int max_index = 0;
		Mat result;

		for (int i = 0; i < 256; i++)
		{
			float cur_entropy = calculateCurrentEntropy(hist, i);
			if (cur_entropy > maxentropy)
			{
				maxentropy = cur_entropy;
				max_index = i;
			}
		}
		threshold(inputImage, result, max_index, 255, CV_THRESH_BINARY);
		return result;
	}

	void Pyramid(Mat srcImage)
	{
		if (srcImage.rows > 400 && srcImage.cols > 400)
		{
			resize(srcImage, srcImage, Size(), 0.5, 0.5);
		}
		imshow("srcImage", srcImage);

		Mat pyrDownImage, pyrUpImage;
		pyrDown(srcImage, pyrDownImage, Size(srcImage.cols / 2, srcImage.rows / 2));
		imshow("pyrDown", pyrDownImage);
		pyrUp(srcImage, pyrUpImage, Size(srcImage.cols * 2, srcImage.rows * 2));
		imshow("pyrUp", pyrUpImage);
		cvWaitKey(0);
	}

	Mat Myfilter2D(Mat srcImage)
	{
		const int nChannels = srcImage.channels();
		Mat resultImage(srcImage.size(), srcImage.type());
		
		for (int i = 1; i < srcImage.rows - 1; i++)
		{
			const uchar* previous = srcImage.ptr<uchar>(i - 1);
			const uchar* current = srcImage.ptr<uchar>(i);
			const uchar* next = srcImage.ptr<uchar>(i + 1);
			uchar* output = resultImage.ptr<uchar>(i);
			for (int j = nChannels; j < nChannels * (srcImage.cols - 1); j++)
			{
				*output++ = saturate_cast<uchar>((current[j-nChannels] + current[j+nChannels] + previous[j] + next[j])/4);
			}
		}
		resultImage.row(0).setTo(Scalar(0));
		resultImage.row(resultImage.rows-1).setTo(Scalar(0));
		resultImage.col(0).setTo(Scalar(0));
		resultImage.col(resultImage.cols - 1).setTo(Scalar(0));
		return resultImage;
	}
	
	Mat filter2D_(Mat srcImage)
	{
		Mat resultImage(srcImage.size(), srcImage.type());
		Mat kern = (Mat_<float>(3, 3) << 0, 1, 0,
										1, 0, 1,
										0, 1, 0) / 4.0;
		filter2D(srcImage, resultImage, srcImage.depth(), kern);
		return resultImage;
	}

	Mat DFT(Mat srcImage)
	{
		Mat srcGray;
		cvtColor(srcImage, srcGray, CV_RGB2GRAY);
		int nRows = getOptimalDFTSize(srcGray.rows);
		int nCols = getOptimalDFTSize(srcGray.cols);
		Mat resultImage;
		copyMakeBorder(srcGray, resultImage, 0, nRows - srcGray.rows, 0, nCols - srcGray.cols, BORDER_CONSTANT, Scalar::all(0));
		Mat planes[] = { Mat_<float>(resultImage), Mat::zeros(resultImage.size(),CV_32F) };
		Mat completeI;
		merge(planes, 2, completeI);
		dft(completeI, completeI);
		split(completeI, planes);
		magnitude(planes[0], planes[1], planes[0]);
		Mat dftResultImage = planes[0];
		dftResultImage += 1;
		log(dftResultImage, dftResultImage);
		dftResultImage = dftResultImage(Rect(0, 0, srcGray.cols, srcGray.rows));
		normalize(dftResultImage, dftResultImage, 0, 1, CV_MINMAX);
		int cx = dftResultImage.cols / 2;
		int cy = dftResultImage.rows / 2;
		Mat tmp;
		Mat q0(dftResultImage, Rect(0, 0, cx, cy));
		Mat q1(dftResultImage, Rect(cx, 0, cx, cy));
		Mat q2(dftResultImage, Rect(0, cy, cx, cy));
		Mat q3(dftResultImage, Rect(cx, cy, cx, cy));
		q0.copyTo(tmp);
		q3.copyTo(q0);
		tmp.copyTo(q3);
		q1.copyTo(tmp);
		q2.copyTo(q1);
		tmp.copyTo(q2);
		return dftResultImage;
	}

	void convolution(Mat src, Mat kernel, Mat& dst)
	{
		dst.create(abs(src.rows - kernel.rows) + 1, abs(src.cols - kernel.cols) + 1, src.type());
		Size dftSize;
		dftSize.width = getOptimalDFTSize(src.cols + kernel.cols - 1);
		dftSize.height = getOptimalDFTSize(src.rows + kernel.rows - 1);

		Mat tempA(dftSize, src.type(), Scalar::all(0));
		Mat tempB(dftSize, kernel.type(), Scalar::all(0));
		Mat roiA(tempA, Rect(0, 0, src.cols, src.rows));
		Mat roiB(tempB, Rect(0, 0, kernel.cols, kernel.rows));

		kernel.copyTo(roiB);
		dft(tempA, tempA, 0, src.rows);
		dft(tempB, tempB, 0, kernel.rows);
		mulSpectrums(tempA, tempB, tempA, DFT_COMPLEX_OUTPUT);
		dft(tempA, tempA, DFT_INVERSE + DFT_SCALE, dst.rows);
		tempA(Rect(0, 0, dst.cols, dst.rows)).copyTo(dst);
	}

	Mat addSaltNoise(const Mat srcImage, int n)
	{
		Mat resultImage = srcImage.clone();
		for (int k = 0; k < n; k++)
		{
			int i = rand() % resultImage.cols;
			int j = rand() % resultImage.rows;
			if (resultImage.channels() == 1)
			{
				resultImage.at<uchar>(i, j) = 255;
			}
			else
			{
				resultImage.at<Vec3b>(i, j)[0] = 255;
				resultImage.at<Vec3b>(i, j)[1] = 255;
				resultImage.at<Vec3b>(i, j)[2] = 255;
			}
		}
		return resultImage;
	}

	double generateGaussianNoise(double mu, double sigma)
	{
		const double epsilon = numeric_limits<double>::min();
		static double z0, z1;
		static bool flag = false;
		flag = !flag;
		if (!flag)
		{
			return z1 * sigma + mu;
		}
		double u1, u2;
		do
		{
			u1 = rand() * (1.0 / RAND_MAX);
			u2 = rand() * (1.0 / RAND_MAX);
		} while (u1 <= epsilon);
		z0 = sqrt(-2.0 * log(u1)) * cos(2 * CV_PI * u2);
		z1 = sqrt(-2.0 * log(u1)) * sin(2 * CV_PI * u2);
		return z0 * sigma + mu;
	}

	Mat addGaussianNoise(Mat& srcImage)
	{
		Mat resultImage = srcImage.clone();
		int channels = resultImage.channels();
		int nRows = resultImage.rows;
		int nCols = resultImage.cols * channels;
		
		if (resultImage.isContinuous())
		{
			nCols *= nRows;
			nRows = 1;
		}
		for (int i = 0; i < nRows; i++)
		{
			for (int j = 0; j < nCols; j++)
			{
				int val = resultImage.ptr<uchar>(i)[j] + generateGaussianNoise(2, 0.8) * 32;
				if (val < 0)
				{
					val = 0;
				}
				if (val > 255)
				{
					val = 255;
				}
				resultImage.ptr<uchar>(i)[j] = (uchar)val;
			}
		}
		return resultImage;
	}

	void myMedianBlur(Mat& src, Mat& dst, const int kSize)
	{
		dst = src.clone();
		vector<uchar> vList;
		const int nPix = (kSize * 2 + 1) * (kSize * 2 + 1);

		for (int i = kSize; i < dst.rows - kSize; i++)
		{
			for (int j = kSize; j < dst.cols - kSize; j++)
			{
				for (int pi = i - kSize; pi <= i + kSize; pi++)
				{
					for (int pj = j - kSize; pj <= j + kSize; pj++)
					{
						vList.push_back(src.at<uchar>(pi, pj));
					}
				}
				sort(vList.begin(), vList.end());
				dst.at<uchar>(i, j) = vList[nPix / 2];
				vList.clear();
			}
		}
	}

	void myGaussianBlur(const Mat& src, Mat& result, int besarKernel, double delta)
	{
		int kerR = besarKernel / 2;
		Mat kernel = Mat_<double>(besarKernel, besarKernel);
		double alpha = 1 / (2 * (22 / 7)*delta*delta);
		for (int i = -kerR; i <= kerR; i++)
		{
			for (int j = -kerR; j <= kerR; j++)
			{
				kernel.at<double>(i + kerR, j + kerR) = alpha * exp(-(((j*j) + (i*i)) / (delta* delta * 2)));
			}
		}
		result = src.clone();
		double pix;
		for (int i = kerR; i < src.rows - kerR; i++)
		{
			for (int j = kerR; j < src.cols - kerR; j++)
			{
				pix = 0;
				for (int k = -kerR; k <= kerR; k++)
				{
					for (int n = -kerR; n <= kerR; n++)
					{
						pix += src.at<uchar>(i + k, j + n) * kernel.at<double>(kerR + k, kerR + n);
					}
				}
				result.at<uchar>(i - kerR, j - kerR) = pix;
			}
		}
	}

	Mat guidedfilter(Mat& srcImage, Mat& srcClone, int r, double eps)
	{
		srcImage.convertTo(srcImage, CV_64FC1);
		srcClone.convertTo(srcClone, CV_64FC1);
		int nRows = srcImage.rows;
		int nCols = srcImage.cols;

		Mat boxResult;
		boxFilter(Mat::ones(nRows, nCols, srcImage.type()), boxResult, CV_64FC1, Size(r, r));
		Mat mean_I;
		boxFilter(srcImage, mean_I, CV_64FC1, Size(r, r));
		Mat mean_p;
		boxFilter(srcClone, mean_p, CV_64FC1, Size(r, r));
		Mat mean_Ip;
		boxFilter(srcImage.mul(srcClone), mean_Ip, CV_64FC1, Size(r, r));

		Mat cov_Ip = mean_Ip - mean_I.mul(mean_p);
		Mat mean_II;
		boxFilter(srcImage.mul(srcClone), mean_II, CV_64FC1, Size(r, r));
		Mat var_I = mean_II - mean_I.mul(mean_I);
		Mat var_Ip = mean_Ip - mean_I.mul(mean_p);
		Mat a = cov_Ip / (var_I + eps);
		Mat b = mean_p - a.mul(mean_I);

		Mat mean_a;
		boxFilter(a, mean_a, CV_64FC1, Size(r, r));
		mean_a = mean_a / boxResult;
		Mat mean_b = mean_b / boxResult;
		Mat resultMat = mean_a.mul(srcImage) + mean_b;
		return resultMat;
	}
}