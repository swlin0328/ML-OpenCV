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
#include <queue>
#include <iostream>
#include <string>
#include <regex>
#include <sstream>

#define __dataManipulate

#pragma once#pragma once
#ifndef __Linear_Algebra
#define __Linear_Algebra
#include "Linear_Algebra.h"
#endif

#pragma once#pragma once
#ifndef __Statistics
#define __Statistics
#include "Statistics.h"
#endif

using namespace std;
namespace dataManipulate
{
	//資料讀取
	int load_Data_With_Bias(string path, vector<vector<double>>& X, vector<double>& Y, const function<double(const string&)>& encoder, string cmd, int dim = 0);

	int load_Data_NoBias_NN(string path, vector<vector<double>>& X, vector<vector<double>>& Y, const function<double(const string&)>& encoder, string cmd, int input_dim, int output_dim);

	void init_NoBias_vector(ifstream& iData, vector<vector<double>>& X, vector<vector<double>>& Y, string cmd, int input_dim, int output_dim);

	void data2vector(vector<string>& result, vector<vector<double>>& X, vector<double>& Y, const function<double(const string&)>& encoder, string cmd, int bias, int dim = 0);

	void init_vector(vector<vector<double>>& X, vector<double>& Y, const function<double(const string&)>& encoder, string cmd, int dim, vector<string> result);
	
	void readData_for_NN(ifstream& iData, vector<double>& readData, int dim);

	int readData_for_tree(string path, vector<map<string, string>>&, vector<string>&, string cmd);

	void readParagraph(string path, string& paragraph);

	//資料處理
	vector<string> string_partition(const string &source, char delim = '\n');

	void split_data(vector< pair<vector<double>, double> >& data, vector<pair<vector<double>, double>>& train, vector<pair<vector<double>, double>>& test, double trainSize = 0.8);
	
	void split_data(vector<pair<map<string, string>, string>>& data, vector<pair<map<string, string>, string>>& train, vector<pair<map<string, string>, string>>& test, double trainSize);

	void train_test_split(vector<vector<double> >& X, vector<double>& Y, vector<vector<double> >& X_train, vector<double>& Y_train, vector<vector<double> >& X_test, vector<double>& Y_test, double trainSize = 0.8);
	
	void train_test_split(vector<map<string, string>>& X, vector<string>& Y, vector<map<string, string>>& X_train, vector<string>& Y_train, vector<map<string, string>>& X_test, vector<string>& Y_test, double trainSize = 0.8);

	template<typename T>
	vector<T> bootstrap_Xi(const vector<T>& data);

	template<typename T, typename U, typename V>
	vector<U> bootstrap_statisticXi(vector<T>& data, int num_bootstrap, function<V(T)>stats_fn);

	vector<pair<vector<double>, double>> bootstrap_sample(vector<vector<double>>& X, vector<double>& Y);

	void to_lower(string word);
}