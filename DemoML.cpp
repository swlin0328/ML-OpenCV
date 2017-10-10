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