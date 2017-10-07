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

#define __Linear_Algebra

#pragma once#pragma once
#ifndef __Statistics
#define __Statistics
#include "Statistics.h"
#endif

using namespace std;
using namespace std::chrono;
//線代工具
namespace Linear_Algebra
{
	//向量運算
	template<typename T, typename U>
	void vector_length_queal(const vector<T>& v, const vector<U> & w);

	template<typename T, typename U>
	void vector_length_queal(vector<T>& v, vector<U> & w);

	void vector_length_queal(vector<double>& v, vector<double> & w);

	void vector_length_queal(vector<vector<double>>& v, vector<vector<double>> & w);

	void vector_length_queal(vector<map<string, string>>& v, vector<string>& w);

	void vector_length_queal(const vector<string>& v, const vector<bool> & w);

	void vector_length_security(const vector<double>& v, const vector<double>& w);

	void vector_subtract(const vector<double>& v, const vector<double>& w, vector<double>& result);

	void vector_add(const vector<double>& v, const vector<double>& w, vector<double>& result);

	void vectors_sum(const vector<vector<double>>& vectors, vector<double>& result);

	double vector_sum(const vector<double>& vector);

	void scalar_multiply(double c, vector<double>& v);

	void vector_mean(const vector<vector<double>>& vectors, vector<double>& result);

	double dot(const vector<double>& v, const vector<double>& w);

	int dot(const vector<int>& v, const vector<int>& w);

	double sum_of_squares(const vector<double>& v);

	double magnitude(const vector<double>& v);

	double squared_distance(const vector<double>& v, const vector<double>& w);

	double distance(const vector<double>& v, const vector<double>& w);

	//向量微積分
	double difference_quotient(function<double(double)> f, const double x, double h = 0.001);

	double partial_difference_quotient(function<double(vector<double>&)> f, vector<double> v, int i, double h = 0.001);

	void estimate_gradient(function<double(vector<double>&)> f, vector<double> v, vector<double>& gradient, double h = 0.001);

	vector<double> step(vector<double> v, const vector<double>& direction, double step_size);

	vector<double> minimize_batch(function<double(vector<double>&)> target_f, const vector<double>& w_0, double tolerance = 0.001);

	vector<double> maximize_batch(function<double(vector<double>&)> target_f, const vector<double>& w_0, double tolerance = 0.001);

	template<typename T>
	vector<int> inRandomOrder(const vector<T>& data);

	vector<int> inRandomOrder(const vector<pair<vector<double>, vector<double>> >& data);

	vector<double> minimize_stochastic(function<double(vector<vector<double>>&, vector<double>&, vector<double>&)> target_f, vector<double>& w_0, vector<vector<double>>& x, vector<double>& y, double eta_0 = 0.1, int miniBatch = 10, int miniBatchFactor = 5);

	template<typename T, typename U>
	double partial_difference_quotient(function<double(vector<T>&, vector<U>&, T&)> target_f, vector<double>& v, vector<T>& X, vector<U>& Y, int i, double h = 0.001);

	template<typename T, typename U>
	void estimate_gradient(function<double(vector<T>&, vector<U>&, T&)> target_f, T& v, T& gradient, vector<T>& X, vector<U>& Y, double h = 0.001);

	template<typename T, typename U>
	double sum_square_ErrFunction(vector<T> X, vector<U> Y, T w);

	vector<double> maximize_stochastic(function<double(vector<vector<double>>&, vector<double>&, vector<double>&)> target_f, const vector<double>& w_0, vector<vector<double>>& x, vector<double>& y, double eta_0 = 0.1, int miniBatch = 10, int miniBatchFactor = 5);

	vector<double> direction(vector<double> v);

	//向量轉換
	double directional_variance_i(const vector<double>& x_i, const vector<double>& w);

	double directional_variance(const vector<vector<double>>& X, const vector<double>& w);

	void randomVector(vector<double>& w, double lo = -0.3, double hi = 0.3);

	vector<double> first_principle_component(vector<vector<double>>& X);

	vector<double> first_principle_component_sgd(vector<vector<double>>& X);

	vector<double> project(vector<double> v, vector<double> w);

	vector<double> remove_projection_from_vector(vector<double> v, vector<double> w);

	void remove_projection(vector<vector<double>>& X, vector<double> w);

	vector<vector<double>> principal_component_analysis(vector<vector<double>>& X, int num_components);

	vector<double> transform_vector(vector<double> v, vector<vector<double>> components);

	vector<vector<double>> trnsform_X(vector<vector<double>> X, vector<vector<double>> components);

	//矩陣工具
	void make_Matrix(vector<vector<double>>& matrix, int row, int col);

	vector<vector<double>> transpose(vector<vector<double>> &X);

	double linear_equation(vector<double>& w, vector<double>& X);

	double error_for_linear_regression(vector<double>& w, vector<double>& Xi, double& yi);

	double sum_of_linear_squared_errors(vector<vector<double>>& X, vector<double>& Y, vector<double>& w);

	double total_sum_of_squares(vector<double>& Y);

	double R_square(vector<double>& w, vector<vector<double>>& X, vector<double>& Y);

	vector<double> linear_regression(vector<vector<double>>& X, vector<double>& Y, vector<double>& w);
}