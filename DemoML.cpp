#include "DemoML.h"
#include <sstream>

void Demo_Neuron()
{
	string Path{ R"(C:\Users\Acer\Desktop\ML作品集2017.10.10\測試數據集\neuron_dataset_測試完成\neuron_dataset.txt)" };
	vector<vector<double>> X, train_X, validate_X;
	vector<double> Y, train_Y, validate_Y;
	vector<string> lable{ "Iris-virginica", "Iris-versicolor", "Iris-setosa" };
	string cmd = "train";
	Linear_Algebra::make_Matrix(X, 10, 10);

	Supervise_Learning::perceptron neuron;

	//--------------------load data--------------------
	dataManipulate::load_Data_With_Bias(Path, X, Y, [&](const string& name) {if (name == lable[0])return 1.0; else return -1.0; }, cmd, 4);
	dataManipulate::train_test_split(X, Y, train_X, train_Y, validate_X, validate_Y, 0.75);

	//--------------------data rescale--------------------
	vector<vector<double>> train_Xt{ Linear_Algebra::transpose(train_X) };
	Statistics::rescale(train_Xt);
	vector<vector<double>> rescale_train_X{ Linear_Algebra::transpose(train_Xt) };

	//--------------------train--------------------
	neuron.train(rescale_train_X, train_Y);
	neuron.show_train_result();

	//--------------------validate--------------------
	for (int i = 0; i < validate_X.size(); i++)
	{
		neuron.predict_prob(validate_X[i]);
	}
	neuron.show_validate_result(validate_Y);
}

void Demo_NeuralNetwork()
{
	string train_Path{ R"(C:\Users\Acer\Desktop\ML作品集2017.10.10\測試數據集\nearalNetwork_dataset_測試完成\training_dataset.txt)" };
	string test_Path{ R"(C:\Users\Acer\Desktop\ML作品集2017.10.10\測試數據集\nearalNetwork_dataset_測試完成\test_數字三and八.txt)" };
	vector<vector<double>> train_X, validate_X;
	vector<vector<double>> train_Y, validate_Y;

	//--------------------init dataset--------------------
	Linear_Algebra::make_Matrix(train_X, 10, 25);
	Linear_Algebra::make_Matrix(train_Y, 10, 10);
	Linear_Algebra::make_Matrix(validate_X, 10, 25);
	Linear_Algebra::make_Matrix(validate_Y, 10, 10);

	Supervise_Learning::neuron_network NN(25, 10, 8, 3, 1, Supervise_Learning::activation_for_logistic, 50000, 0.5);

	//--------------------load data--------------------
	dataManipulate::load_Data_NoBias_NN(train_Path, train_X, train_Y, [](const string& name) {return 0; }, "train", 25, 10);
	dataManipulate::load_Data_NoBias_NN(test_Path, validate_X, validate_Y, [](const string& name) {return 0; }, "validate", 25, 10);
	
	//--------------------train--------------------
	NN.train(train_X, train_Y);

	//--------------------validate--------------------
	NN.predict(validate_X, validate_Y);
}

void Demo_DecisionTree()
{
	string Path{ R"(C:\Users\Acer\Desktop\ML作品集2017.10.10\測試數據集\decisionTree_dataset_測試完成\decisionTree_dataset.txt)" };
	vector<pair<map<string, string>, string>> test_dataset;
	vector<map<string, string>> X, demo_X, train_X, validate_X;
	vector<string> Y, demo_Y, train_Y, validate_Y;

	Supervise_Learning::decision_tree_id3 id3_tree;

	//--------------------load data--------------------
	dataManipulate::readData_for_tree(Path, X, Y, "train");
	demo_X = X;
	demo_Y = Y;
	dataManipulate::train_test_split(X, Y, train_X, train_Y, validate_X, validate_Y, 0.8);
	Statistics::makePair(demo_X, demo_Y, test_dataset);

	//--------------------train--------------------
	id3_tree.train(test_dataset, 10, 0.8);
	id3_tree.show_tree_struct();

	//--------------------validate--------------------
	cout << "\n\n--------------------validate--------------------\n";
	cout << "The predict result is : " << id3_tree.predict(validate_X[0]) << endl;
	cout << "The true answer is : " << validate_Y[0] << endl;
}

void Demo_Ngram()
{
	string Path{ R"(C:\Users\Acer\Desktop\ML作品集2017.10.10\測試數據集\Ngram_dataset_測試完成\Ngram_dataset.txt)" };
	string paragraph;

	//--------------------load data--------------------
	dataManipulate::readParagraph(Path, paragraph);

	//--------------------train--------------------
	NLP_lib::n_gram tri_grams(paragraph, 3);

	//--------------------create five sentence--------------------
	for (int i = 0; i < 5; i++)
	{
		string create_sentence = tri_grams.generate_using_model();
		cout << create_sentence << "\n\n";
	}
}

void Demo_Kmeans()
{
	using namespace cv;
	string Path{ R"(C:\Users\Acer\Desktop\ML作品集2017.10.10\測試數據集\Kmeans_dataset_測試完成\Kmeans_dataset.jpg)" };
	
	//--------------------load image--------------------
	Mat img = imread(Path, CV_LOAD_IMAGE_COLOR);
	Mat reduce2sixColor = img.clone();

	//--------------------train to get six clusters--------------------
	unSupervise_Learning::k_means six_color(6);
	vector<vector<double>> load_pixels;

	for (int i = 0; i < img.rows; i++)
	{
		for (int j = 0; j < img.cols; j++)
		{
			vector<double> pixel;
			for (int m = 0; m < img.channels(); m++)
			{
				pixel.push_back(img.at<Vec3b>(i, j)[m]);
			}
			load_pixels.push_back(pixel);
		}
	}
	six_color.train(load_pixels, 100);

	for (int i = 0; i < img.rows; i++)
	{
		for (int j = 0; j < img.cols; j++)
		{
			int color_index = six_color.predict(load_pixels[img.cols*i + j]);
			vector<double> color = six_color.get_centerMass(color_index);
			for (int m = 0; m < img.channels(); m++)
			{
				reduce2sixColor.at<Vec3b>(i, j)[m] = (unsigned)color[m];
			}
		}
	}
	six_color.show_num_cluster();
	cout << "The squared error is : " << six_color.squared_clustering_errors(load_pixels, 6) << "\n";
	
	//--------------------compare the original image and the image with reduced color--------------------
	namedWindow("Original picture", WINDOW_AUTOSIZE);
	imshow("Original picture", img);
	namedWindow("six_means picture", WINDOW_AUTOSIZE);
	imshow("six_means picture", reduce2sixColor);
	waitKey(0);
}

void Demo_random_forest()
{
	Supervise_Learning::random_forest forest;

	string Path{ R"(C:\Users\Acer\Desktop\ML作品集2017.10.10\測試數據集\randomForest_dataset_測試完成\randomForest_dataset.txt)" };
	vector<pair<map<string, string>, string>> test_dataset;
	vector<map<string, string>> X, demo_X, train_X, validate_X;
	vector<string> Y, demo_Y, train_Y, validate_Y;

	Supervise_Learning::decision_tree_id3 id3_tree1;
	Supervise_Learning::decision_tree_id3 id3_tree2;
	Supervise_Learning::decision_tree_id3 id3_tree3;
	Supervise_Learning::decision_tree_id3 id3_tree4;

	//--------------------load data--------------------
	dataManipulate::readData_for_tree(Path, X, Y, "train");
	demo_X = X;
	demo_Y = Y;
	dataManipulate::train_test_split(X, Y, train_X, train_Y, validate_X, validate_Y, 0.8);
	Statistics::makePair(demo_X, demo_Y, test_dataset);
	//--------------------train--------------------
	id3_tree1.pick_out_attribute("level");
	id3_tree1.train(test_dataset, 1, 0.85);

	id3_tree1.pick_out_attribute("lang");
	id3_tree2.train(test_dataset, 1, 0.85);

	id3_tree1.pick_out_attribute("tweets");
	id3_tree3.train(test_dataset, 1, 0.85);

	id3_tree1.pick_out_attribute("phd");
	id3_tree4.train(test_dataset, 1, 0.85);

	//--------------------validate--------------------
	forest.insert_tree(id3_tree1);
	forest.insert_tree(id3_tree2);
	forest.insert_tree(id3_tree3);
	forest.insert_tree(id3_tree4);
	cout << "--------------------validate--------------------\n";
	cout << "The predict result is : " << forest.predict(validate_X[0]) << endl;
	cout << "The true answer is : " << validate_Y[0] << endl;
}

void Demo_KNN()
{
	string Path{ R"(C:\Users\Acer\Desktop\ML作品集2017.10.10\測試數據集\KNN_dataset_測試完成\KNN_dataset.txt)" };
	
	vector<vector<double>> X, train_X, validate_X;
	vector<double> Y, train_Y, validate_Y;
	string cmd = "train";
	Linear_Algebra::make_Matrix(X, 10, 10);

	//--------------------load data--------------------
	dataManipulate::load_Data_With_Bias(Path, X, Y, 
		[](string label) {stringstream encoder{ label }; double val; encoder >> val; return val; }, cmd, 3, 1);
	dataManipulate::train_test_split(X, Y, train_X, train_Y, validate_X, validate_Y, 0.9);

	//--------------------data rescale--------------------
	vector<vector<double>> Xt{ Linear_Algebra::transpose(X) };
	Statistics::rescale(Xt);
	vector<vector<double>> rescale_X{ Linear_Algebra::transpose(Xt) };

	dataManipulate::train_test_split(rescale_X, Y, train_X, train_Y, validate_X, validate_Y, 0.9);
	Supervise_Learning::KNN three_neighbors(3, train_X, train_Y);

	//--------------------validate--------------------
	three_neighbors.show_validate(validate_X, validate_Y);
}

void Demo_users_internet()
{
	string Path{ R"(C:\Users\Acer\Desktop\ML作品集2017.10.10\測試數據集\user_internet_測試完成\user_dataset.txt)" };

	//--------------------load data--------------------
	NLP_lib::users_information users_net(Path);

	users_net.create_user(string{ "Jotaro" }, vector<string>{"data science", "C++", "deep learning"});
	users_net.add_friend(1, 8);
	users_net.endorse_user(1, 7);
	users_net.endorse_user(1, 4);
	users_net.endorse_user(1, 3);
	users_net.endorse_user(1, 2);
	users_net.endorse_user(2, 5);
	users_net.endorse_user(2, 3);
	users_net.endorse_user(3, 4);
	users_net.endorse_user(3, 5);
	users_net.endorse_user(4, 3);
	users_net.endorse_user(4, 5);
	users_net.endorse_user(7, 8);
	users_net.endorse_user(8, 5);
	users_net.endorse_user(8, 6);

	//--------------------train--------------------
	users_net.betweenness_centrality();
	users_net.page_rank();

	//--------------------validate--------------------
	users_net.show_training_result();
	users_net.user_similarity(1, 2);
	users_net.user_based_suggestion(1);	
}

void Demo_interest_topics()
{
	string Path{ R"(C:\Users\Acer\Desktop\ML作品集2017.10.10\測試數據集\interest_dataset_測試完成\interest_dataset.txt)" };
	vector<string> users_interests;
	string doc, line;
	
	//--------------------load data--------------------
	dataManipulate::readParagraph(Path, doc);
	istringstream is{ doc };
	while (is.peek() != EOF && getline(is, line))
	{
		string interest = "";
		auto user_data = dataManipulate::string_partition(line, ',');
		for (int i = 0; i < user_data.size(); i++)
		{
			interest += user_data[i] + " ";
		}
		users_interests.push_back(interest);
	}

	//--------------------train--------------------
	NLP_lib::K_topic_given_document K_interest(4);
	K_interest.train(users_interests);

	//--------------------validate--------------------
	K_interest.show_result(3);
}

void Demo_bottom_up_cluster()
{
	string Path{ R"(C:\Users\Acer\Desktop\ML作品集2017.10.10\測試數據集\bottomUp_dataset_測試完成\bottomUp_dataset.txt)" };

	vector<vector<double>> X, train_X, validate_X;
	vector<double> Y, train_Y, validate_Y;
	string cmd = "train";
	Linear_Algebra::make_Matrix(X, 10, 10);

	//--------------------load data--------------------
	dataManipulate::load_Data_With_Bias(Path, X, Y,
		[](string label) {stringstream encoder{ label }; double val; encoder >> val; return val; }, cmd, 3, 1);

	//--------------------data rescale--------------------
	vector<vector<double>> Xt{ Linear_Algebra::transpose(X) };
	Statistics::rescale(Xt);
	vector<vector<double>> rescale_X{ Linear_Algebra::transpose(Xt) };
	dataManipulate::train_test_split(rescale_X, Y, train_X, train_Y, validate_X, validate_Y, 0.85);

	//--------------------train--------------------
	string method = "max";
	unSupervise_Learning::bottom_up_cluster three_clusters;
	three_clusters.bottom_up(train_X, method);

	//--------------------validate--------------------
	three_clusters.predict(3, validate_X);
}

void Demo_NaiveBayesClassifier()
{
	string not_spam_Path{ R"(C:\Users\Acer\Desktop\ML作品集2017.10.10\測試數據集\Bayes_dataset_測試完成\not_spam)" };
	string spam_Path{ R"(C:\Users\Acer\Desktop\ML作品集2017.10.10\測試數據集\Bayes_dataset_測試完成\spam)" };

	vector<string> spam_X, not_spam_X;
	vector<bool> spam_label, not_spam_label;
	vector<string> mail;
	vector<bool> label;

	//--------------------load data--------------------
	dataManipulate::load_mail(not_spam_Path, "not_spam", not_spam_X, not_spam_label, false, 25);
	dataManipulate::load_mail(spam_Path, "is_spam", spam_X, spam_label, true, 25);
	for (int i = 0; i < spam_X.size(); i++)
	{
		mail.push_back(not_spam_X[i]);
		mail.push_back(spam_X[i]);
		label.push_back(not_spam_label[i]);
		label.push_back(spam_label[i]);
	}

	//--------------------data rescale--------------------
	vector<string> train_X, validate_X;
	vector<bool> train_Y, validate_Y;
	dataManipulate::train_test_split(mail, label, train_X, train_Y, validate_X, validate_Y, 0.8);

	//--------------------train--------------------
	NLP_lib::NaiveBayesClassifier spam_classifier(0.5);
	spam_classifier.train(train_X, train_Y);

	//--------------------validate--------------------
	vector<double> spam_probability = spam_classifier.predict(validate_X);
	for (int i = 0; i < spam_probability.size(); i++)
	{
		cout << "\nThe mail " << i << " is spam : " << validate_Y[i] << "\n";
		cout << "The predicted probability to be spam is " << round(10000*spam_probability[i]) / 100 << "%\n";
	}
}

void Demo_Camera()
{
	VideoCapture cap;
	// open the default camera, use something different from 0 otherwise;
	if (!cap.open(0))
	{
		return;
	}

	while (true)
	{
		Mat frame;
		cap >> frame;
		if (frame.empty()) break; // end of video stream
		imshow("Camera", frame);
		if (waitKey(10) == 27) break; // stop capturing by pressing ESC 
	}
}

void Demo_detect_car_plate()
{
	vector<Mat> srcImages;
	vector<string> srcImgPaths;
	string folderPath{ R"(C:\Users\Acer\Desktop\ML作品集2017.10.29\測試數據集\plate_recognition_測試完成)" };
	cv_lib::readImgNamefromFile(folderPath, srcImgPaths);

	for (int i = 0; i < srcImgPaths.size(); i++)
	{
		Mat img = imread(srcImgPaths[i], CV_LOAD_IMAGE_COLOR);
		srcImages.push_back(img);
	}

	cout << "extract plate by Specified Color Region \n";
	for (int i = 0; i < srcImages.size(); i++)
	{
		cv_lib::extract_License_Plate(srcImages[i]);
	}
}

void Demo_detect_car_plate_MSER()
{
	vector<Mat> srcImages;
	vector<string> srcImgPaths;
	string folderPath{ R"(C:\Users\Acer\Desktop\ML作品集2017.10.29\測試數據集\plate_recognition_測試完成)" };
	cv_lib::readImgNamefromFile(folderPath, srcImgPaths);

	for (int i = 0; i < srcImgPaths.size(); i++)
	{
		Mat img = imread(srcImgPaths[i], CV_LOAD_IMAGE_COLOR);
		srcImages.push_back(img);
	}

	cout << "extract plate by MSER \n";
	for (int i = 0; i < srcImages.size(); i++)
	{
		cv_lib::mserGetPlate(srcImages[i]);
	}
}

void Demo_detect_car_plate_Morphology()
{
	vector<Mat> srcImages;
	vector<string> srcImgPaths;
	string folderPath{ R"(C:\Users\Acer\Desktop\ML作品集2017.10.29\測試數據集\plate_recognition_測試完成)" };
	cv_lib::readImgNamefromFile(folderPath, srcImgPaths);

	for (int i = 0; i < srcImgPaths.size(); i++)
	{
		Mat img = imread(srcImgPaths[i], CV_LOAD_IMAGE_COLOR);
		srcImages.push_back(img);
	}

	cout << "extract plate by Morphology Gradient \n";
	for (int i = 0; i < srcImages.size(); i++)
	{
		cv_lib::extract_License_Plate_by_MorphologyEx(srcImages[i]);
	}
}

void Demo_dectect_Skin()
{
	vector<Mat> srcImages;
	vector<string> srcImgPaths;
	string folderPath{ R"(C:\Users\Acer\Desktop\ML作品集2017.10.29\測試數據集\face_detection_測試完成)" };
	cv_lib::readImgNamefromFile(folderPath, srcImgPaths);

	for (int i = 0; i < srcImgPaths.size(); i++)
	{
		Mat img = imread(srcImgPaths[i], CV_LOAD_IMAGE_COLOR);
		srcImages.push_back(img);
	}

	cout << "extract skin region \n";
	for (int i = 0; i < srcImages.size(); i++)
	{
		cv_lib::dectect_Skin_Color(srcImages[i]);
	}
}

void Demo_cacHOGFeature()
{
	vector<Mat> srcImages;
	vector<string> srcImgPaths;
	string folderPath{ R"(C:\Users\Acer\Desktop\ML作品集2017.10.29\測試數據集\face_detection_測試完成)" };
	cv_lib::readImgNamefromFile(folderPath, srcImgPaths);

	for (int i = 0; i < srcImgPaths.size(); i++)
	{
		Mat img = imread(srcImgPaths[i], CV_LOAD_IMAGE_COLOR);
		srcImages.push_back(img);
	}

	cout << "extract skin region \n";
	Mat face = cv_lib::dectect_Skin_Color(srcImages[0]);

	cout << "Face HOG Feature \n";
	cv_lib::cacHOGFeature(face);
}

void Demo_LPBFeature()
{
	vector<Mat> srcImages;
	vector<string> srcImgPaths;
	string folderPath{ R"(C:\Users\Acer\Desktop\ML作品集2017.10.29\測試數據集\face_detection_測試完成)" };
	cv_lib::readImgNamefromFile(folderPath, srcImgPaths);

	for (int i = 0; i < srcImgPaths.size(); i++)
	{
		Mat img = imread(srcImgPaths[i], CV_LOAD_IMAGE_COLOR);
		srcImages.push_back(img);
	}

	cout << "extract skin region \n";
	Mat face = cv_lib::dectect_Skin_Color(srcImages[0]);

	cout << "Face LPB Feature \n";
	Mat LBP_Feature = cv_lib::OLBP(face);
	cvNamedWindow("LBP_Feature", CV_WINDOW_AUTOSIZE);
	imshow("LBP_Feature", LBP_Feature);
	cvWaitKey(1000);
}

void Demo_charFeature()
{
	vector<Mat> srcImages;
	vector<string> srcImgPaths;
	string folderPath{ R"(C:\Users\Acer\Desktop\ML作品集2017.10.29\測試數據集\face_detection_測試完成)" };
	cv_lib::readImgNamefromFile(folderPath, srcImgPaths);

	for (int i = 0; i < srcImgPaths.size(); i++)
	{
		Mat img = imread(srcImgPaths[i], CV_LOAD_IMAGE_COLOR);
		srcImages.push_back(img);
	}

	cout << "extract skin region \n";
	Mat gray, face = cv_lib::dectect_Skin_Color(srcImages[0]);
	cvtColor(face, gray, COLOR_BGR2GRAY);
	
	vector<vector<Point>> regioin_contours;
	findContours(gray, regioin_contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE, Point(0, 0));
	vector<Rect> objects;

	for (int i = 0; i < regioin_contours.size(); i++)
	{
		Rect rect = boundingRect(regioin_contours[i]);
		if (rect.area() > 1000)
		{
			objects.push_back(rect);
		}
	}

	Mat char_Feature = cv_lib::char_feature(face(objects[0]));
	cv_lib::printMat(char_Feature, 3);
}

void Demo_ORB_Match()
{
	vector<Mat> srcImages;
	vector<string> srcImgPaths;
	string folderPath{ R"(C:\Users\Acer\Desktop\ML作品集2017.10.29\測試數據集\plate_recognition_測試完成)" };
	cv_lib::readImgNamefromFile(folderPath, srcImgPaths);

	for (int i = 0; i < srcImgPaths.size(); i++)
	{
		Mat img = imread(srcImgPaths[i], CV_LOAD_IMAGE_COLOR);
		srcImages.push_back(img);
	}

	cout << "Cars Matches by ORB Feature \n";
	cv_lib::cacORBFeatureAndCompare(srcImages[0], srcImages[1]);
}

void Demo_Image_Comparison()
{
	vector<Mat> srcImages;
	vector<string> srcImgPaths;
	string folderPath{ R"(C:\Users\Acer\Desktop\ML作品集2017.10.29\測試數據集\plate_recognition_測試完成)" };
	cv_lib::readImgNamefromFile(folderPath, srcImgPaths);

	for (int i = 0; i < srcImgPaths.size(); i++)
	{
		Mat img = imread(srcImgPaths[i], CV_LOAD_IMAGE_COLOR);
		srcImages.push_back(img);
	}
	
	double PSNR_Val = cv_lib::PSNR(srcImages[0], srcImages[1]);
	Scalar MSSIM_Val = cv_lib::MSSIM(srcImages[0], srcImages[1]);

	cout << "Histogram Comparison: \n";
	cv_lib::histogram_Comparison(srcImages[0], srcImages[1]);

	cout << "PSNR Comparison: \n";
	cout << PSNR_Val << endl;
	
	cout << "MSSIM Comparison: \n";
	cout << MSSIM_Val << endl;
}

void Demo_Hisogram_analysis()
{
	vector<Mat> srcImages;
	vector<string> srcImgPaths;
	string folderPath{ R"(C:\Users\Acer\Desktop\ML作品集2017.10.29\測試數據集\face_detection_測試完成)" };
	cv_lib::readImgNamefromFile(folderPath, srcImgPaths);

	for (int i = 0; i < srcImgPaths.size(); i++)
	{
		Mat img = imread(srcImgPaths[i], CV_LOAD_IMAGE_COLOR);
		srcImages.push_back(img);
	}

	cv_lib::show_Gray_Histogram(srcImages[0]);
	cv_lib::show_RGB_Histogram(srcImages[0]);
}


void Demo_MBitPlan()
{
	vector<Mat> srcImages;
	vector<string> srcImgPaths;
	string folderPath{ R"(C:\Users\Acer\Desktop\ML作品集2017.10.29\測試數據集\face_detection_測試完成)" };
	cv_lib::readImgNamefromFile(folderPath, srcImgPaths);

	for (int i = 0; i < srcImgPaths.size(); i++)
	{
		Mat img = imread(srcImgPaths[i], CV_LOAD_IMAGE_COLOR);
		srcImages.push_back(img);
	}

	cv_lib::showMBitPlan(srcImages[0]);
}