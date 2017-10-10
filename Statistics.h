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

#define __Statistics

#ifndef __Linear_Algebra
#define __Linear_Algebra
#include "Linear_Algebra.h"
#endif

#ifndef __dataManipulate
#define __dataManipulate
#include "dataManipulate.h"
#endif

using namespace std;
using namespace std::chrono;
namespace Statistics
{
	//亂數產生器
	class Rand_normal_double
	{
	public:
		Rand_normal_double(double low, double high)
		{
			unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
			std::default_random_engine generator(seed);
			std::normal_distribution<double> distribution(low, high);
			r = bind(distribution, generator);
		}
		double operator()() { return r(); }
	private:
		function<double()> r;
	};

	class Rand_uniform_Int
	{
	public:
		Rand_uniform_Int(int low, int high)
		{
			unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
			std::default_random_engine generator(seed);
			std::uniform_int_distribution<int> distribution(low, high);
			r = bind(distribution, generator);
		}
		double operator()() { return r(); }
	private:
		function<int()> r;
	};

	class Rand_uniform_double
	{
	public:
		Rand_uniform_double(double low, double high)
		{
			unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
			std::default_random_engine generator(seed);
			std::uniform_real_distribution<double> distribution(low, high);
			r = bind(distribution, generator);
		}
		double operator()() { return r(); }
	private:
		function<double()> r;
	};

	//數值統計
	vector<pair<double, double>> unique_labels(vector<pair<double, double>> labels);

	pair<pair<double, double>, bool> most_frequent_in_group(vector<pair<double, double>>& labels);

	template<typename T>
	pair<int, T> maxValue(const vector<T>& v);

	pair<int, int> maxValue(const vector<int>& v);

	pair<int, double> maxValue(vector<double>& v);

	template<typename T>
	pair<int, T> minValue(const vector<T>& v);

	pair<int, double> minValue(vector<double>& v);

	pair<int, int> minValue(vector<int>& v);

	double sum(const vector<double>& v);

	double mean(const vector<double>& v);

	double median(vector<double> v);

	double quantile(vector<double> v, float p);

	vector<double> mode(vector<double> v);

	vector<double> estimate_sample_beta(vector<pair<vector<double>, double>>& sample);

	vector<double> boostrap_standard_errors(vector<vector<double>>& X, vector<double>& Y, int num_estimate);

	vector<double> p_value(const vector<double>& w, const vector<double> sigma);

	void first_N_maxVal(vector<double>& source, deque<pair<int, double>>& result, int N);

	//數值分組
	template<typename T, typename U>
	void makePair(vector<T>& v, vector<U>& w, vector<pair<T, U> > & result);

	void makePair(vector<vector<double>>& v, vector<vector<double>>& w, vector<pair<vector<double>, vector<double>>>& result);

	void makePair(vector<map<string, string>>& v, vector<string>& w, vector<pair<map<string, string>, string>> & result);
	
	void makePair(vector<double>& v, vector<double>& w, vector<pair<double, double>>& result);

	template<typename T, typename U>
	pair<T, U> makePair(T& v, U& w);

	template<typename T, typename U>
	void unPair(vector<pair<T, U>>& source, vector<T>& v, vector<U>& w);
	
	template<typename T, typename U>
	void unPair(pair<T, U>& source, T& v, U& w);

	void unPair(vector<pair<string, bool>>& source, vector<string>& v, vector<bool>& w);

	void unPair(vector<pair<map<string, string>, string>>& source, vector<map<string, string>>& v, vector<string>& w);

	//數據預處理
	double dataRange(vector<double>& v);

	void deMean(vector<double>& v);

	double variance(vector<double>& v);

	double standard_deviation(vector<double>& v);

	double interquartile_range(vector<double>& v);

	double covariance(vector<double>& v, vector<double>& w);

	double correlation(vector<double>& v, vector<double>& w);

	void deOutlier(vector<double>& v);

	void makeVector(vector<double>& v, vector<vector<double>>& result);

	//統計函數
	int uniform_pdf(double x);

	double uniform_cdf(double x);

	double normal_pdf(double x, double mu = 0, double sigma = 1);

	double normal_cdf(double x, double mu = 0, double sigma = 1);

	double inverse_normal_cdf(double p, double mu = 0, double sigma = 1, double tolerance = 0.00001);

	int bernoulli_trail(double p);

	int binomial(int n, double p);

	double normal_probability_above(double lo, double mu = 0, double sigma = 1);

	double normal_probability_between(double lo, double hi, double mu = 0, double sigma = 1);

	double normal_probability_outside(double lo, double hi, double mu = 0, double sigma = 1);

	double normal_upper_bound(double p, double mu = 0, double sigma = 1);

	double normal_lower_bound(double p, double mu = 0, double sigma = 1);

	pair<double, double> normal_two_sided_bounds(double p, double mu = 0, double sigma = 1);

	double two_side_p_value(double x, double mu = 0, double sigma = 1);

	pair<double, double> estimate_p_sigma(int total, int occur);

	double a_b_test_statistic(int total_A, int OccurA, int total_B, int OccurB);

	template<typename T>
	vector<int> count_tp_fp_fn_tn(vector<T> lables, vector<T> predicts);

	//數值縮放
	void rescale(vector<double>& data);

	void rescale(vector<vector<double>>& data, int start_index = 1);

	//統計分析
	double accuracy(int tp, int fp, int fn, int tn);

	double precision(int tp, int fp, int fn, int tn);

	double recall(int tp, int fp, int fn, int tn);

	double f1_score(int tp, int fp, int fn, int tn);
}