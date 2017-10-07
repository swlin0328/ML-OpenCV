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
#include <iomanip>

#define __Supervise_Learning

#pragma once#pragma once
#ifndef __Linear_Algebra
#define __Linear_Algebra
#include "Linear_Algebra.h"
#endif

#pragma once#pragma once
#ifndef __dataStructure
#define __dataStructure
#include "dataStructure.h"
#endif

#pragma once#pragma once
#ifndef __NLP
#define __NLP
#include "NLP.h"
#endif

using namespace std;
using namespace Linear_Algebra;
using namespace std::chrono;
namespace Supervise_Learning
{
	//訓練函數
	double activation_for_logistic(double wX);

	double activation_for_step_finction(double wX);

	double activation_for_hyperbolic(double wX);

	double square_error(vector<vector<double>>& X, vector<double>& Y, vector<double>& w, const function<double(double)>& actF);

	double square_regularization(vector<double>& w, double lamda);

	vector<double> estimate_w_with_ridge(vector<vector<double>>& X, vector<double>& Y, vector<double>& w, const function<double(double)>& actF, double lamda);

	double logistic_log_likelyhood_i(vector<double>& Xi, double& yi, vector<double>& w);

	double logistic_log_likelihood(vector<vector<double>>& X, vector<double>& Y, vector<double>& w);

	double entropy(vector<double>& class_probabilities);

	vector<double> class_probabilities(vector<pair<map<string, string>, string>>& data);

	double data_entropy(vector<pair<map<string, string>, string>>& data);

	double partition_entropy(vector<vector<pair<map<string, string>, string>>>& subsets);

	string decision_tree_classify(shared_ptr<dataStructure::tree_node>& decision_tree, map<string, string> input);

	tuple<string, int, int> find_most_common_attribute(shared_ptr<dataStructure::tree_node>& decision_tree);

	void try_make_leaf(shared_ptr<dataStructure::tree_node>& current_node, int break_point, double accuracy = 1.0);

	//監督式學習-分類器介面
	class classifier
	{
	private:
		virtual double err_inPrediction(const vector<double>& Y) = 0;
		virtual double cost_inPrediction(const vector<double>& Y) = 0;

	public:
		virtual void train(vector<vector<double>>& X, vector<double>& y) = 0;
		virtual void classify(const vector<vector<double>>& X) = 0;
		virtual void show_validate_result(const vector<double>& Y) = 0;
		virtual double predict_prob(vector<double>& X) = 0;
		virtual void show_train_result() = 0;
		virtual ~classifier() {};
	};

	//感知器
	class perceptron : public virtual classifier
	{
	private:
		double leraning_Rate;
		int n_iter;
		vector<double> w;
		vector<double> cost;
		vector<double> predict_y;
		virtual double err_inPrediction(const vector<double>& Y) override;
		virtual double cost_inPrediction(const vector<double>& Y) override;

	protected:
		void train_for_network(perceptron&, int, double);
		vector<double>& get_neuron_w(perceptron&);
		function<double(double)> actFn;

	public:
		virtual void train(vector<vector<double>>& X, vector<double>& y) override;
		virtual void classify(const vector<vector<double>>& X) override;
		virtual void show_validate_result(const vector<double>& Y) override;
		virtual void show_train_result() override;
		virtual double predict_prob(vector<double>& X) override;
		void train_clear(vector<double>& cost, vector<double>& w, int vSize);
		perceptron(double eta = 0.0001, int epoch = 500, const function<double(double)>& actFunction = activation_for_hyperbolic) :leraning_Rate(eta), n_iter(epoch), actFn(actFunction) {}
	};

	//神經網路
	class neuron_network : protected perceptron
	{
	private:
		int n_iter;
		function<double(double)> actFn;
		double learning_rate;
		vector<vector<vector<perceptron>>> NN;
		vector<perceptron> neuron_network::make_input_layer(int input_dim, int row, double eta, int epoch, const function<double(double)>& actFunction);
		vector<perceptron> neuron_network::make_hidden_layer(int row, double eta, int epoch, const function<double(double)>& actFunction);
		vector<perceptron> neuron_network::make_output_layer(int output_dim, int row, double eta, int epoch, const function<double(double)>& actFunction);

		void make_Neural_Network(vector<vector<vector<perceptron>>>& NN, int input_dim, int output_dim, int row, int col, int depth, double eta, int epoch, const function<double(double)>& actFunction);
		vector<vector<double>> feed_forward_2d(vector<double>);
		void backpropagate_2d(vector<double>&, vector<double>&, int step, int pretrain = 50, double diff_h = 0.001);

	public:
		void train(vector<vector<double>>&, vector<vector<double>>&, int pretrain = 200, double precision = 0.01, double diff_h = 0.001);
		void predict(vector<vector<double>>& X, vector<vector<double>>& Y);
		neuron_network() = delete;
		neuron_network(int input_dim, int output_dim, int row = 5, int col = 5, int depth = 1, const function<double(double)>& actFunction = activation_for_logistic, int epoch = 10000, double NN_learning_rate = 0.1, double eta = 0.0001) : n_iter(epoch), learning_rate(NN_learning_rate), actFn(actFunction) { make_Neural_Network(NN, input_dim, output_dim, row, col, depth, eta, epoch, actFunction); }
		~neuron_network() { NN.~vector(); }
	};

	//KNN模型
	template<typename T, typename U>
	int knn_classify(int k, vector<T> X, vector<U> Y, vector<T> newpoint);

	//迴歸模型
	class linear_regression : public virtual classifier
	{
	private:
		vector<double> w;
		vector<double> cost;
		vector<double> predict_y;

		virtual double err_inPrediction(const vector<double>& Y) override;
		virtual double cost_inPrediction(const vector<double>& Y) override;

	public:
		virtual void train(vector<vector<double>>& X, vector<double>& y) override;
		virtual void classify(const vector<vector<double>>& X) override;
		virtual void show_validate_result(const vector<double>& Y) override;
		virtual void show_train_result() override;
		void train_clear(vector<double>&, vector<double>&, int);
	};

	//決策樹
	class decision_tree_id3
	{
	private:
		shared_ptr<dataStructure::tree_node> root_node;
		shared_ptr<dataStructure::tree_node> build_decision_tree(vector<pair<map<string, string>, string>>&);
		void split_by_attribute(shared_ptr<dataStructure::tree_node>& current_node, int break_point, double accuracy);
		void write_the_lowest_entropy_attribute(shared_ptr<dataStructure::tree_node>& current_node);
		map<string, vector<pair<map<string, string>, string>>> partition_by(vector<pair<map<string, string>, string>>& data, string attribute);
		double partition_entropy_by(vector<pair<map<string, string>, string>>& data, string attribute);
		void BFS(shared_ptr<dataStructure::tree_node> current_node, queue<shared_ptr<dataStructure::tree_node>>& tree_queue);
		void check_node();

	public:
		decision_tree_id3(vector<pair<map<string, string>, string>>& data) { root_node = build_decision_tree(data); }
		decision_tree_id3() :root_node(shared_ptr<dataStructure::tree_node>(new dataStructure::tree_node())) {};
		void train(int break_point = 15, double accuracy = 0.85);
		void train(vector<pair<map<string, string>, string>>& data, int break_point = 15, double accuracy = 0.85);
		string predict(map<string, string>& input);
		bool pick_out_attribute(string);
		void show_tree_struct();
		void train_clear();
	};

	//隨機森林
	class random_forest
	{
	private:
		vector<decision_tree_id3> node_of_tree;

	public:
		void insert_tree(decision_tree_id3);
		string predict(map<string, string>& input);
		void train_clear();
	};
}