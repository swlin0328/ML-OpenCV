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

#define __unSupervise_Learning

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

#pragma once#pragma once
#ifndef __dataStructure
#define __dataStructure
#include "dataStructure.h"
#endif

using namespace std;
using namespace Linear_Algebra;
using namespace std::chrono;
namespace unSupervise_Learning
{
	//訓練函數
	void sort_data_by_min_distance(const vector<vector<double>>& data, vector<vector<double>>& center_mass, vector<vector<vector<double>>>& groups, int k);

	vector<vector<double>> get_values(shared_ptr<dataStructure::cluster_node>& current_node);

	double cluster_distance(shared_ptr<dataStructure::cluster_node>& cluster_node1, shared_ptr<dataStructure::cluster_node>& cluster_node2, vector<vector<double>> distance_table, const function<pair<int, double>(vector<double>&)>& distance_F);

	void init_leaf_cluster(vector<vector<double>>& data, vector<shared_ptr<dataStructure::cluster_node>>& leaf_set);

	void init_distance_table(vector<shared_ptr<dataStructure::cluster_node>>& leaf_set, vector<vector<double>>& distance_table);

	void assemble_cluster(vector<shared_ptr<dataStructure::cluster_node>>& leaf_set, vector<vector<double>>& distance_table, const function<pair<int, double>(vector<double>&)>& distance_F);

	vector<shared_ptr<dataStructure::cluster_node>> get_children(shared_ptr<dataStructure::cluster_node>& current_node);

	vector<vector<double>> get_values(shared_ptr<dataStructure::cluster_node>& current_node);

	////非監督式學習
	//k_means
	class k_means
	{
	private:
		int k;
		vector<vector<double>> means;
		void check_mass_pos(bool& check_same, vector<vector<double>>& center_mass, double precision);
		void refresh_center_mass(vector<vector<vector<double>>>& groups, vector<vector<double>>& center_mass, int k);

	public:
		k_means(int n) :k(n) { means.resize(n); };
		void show_num_cluster() { cout << "The number of cluster : " << k << " \n"; }
		double squared_clustering_errors(vector<vector<double>>& input_data, int n);

		void train(const vector<vector<double>>&, double precision = 50);
		int predict(vector<double>& input_data);
		vector<double> get_centerMass(int m) { return means[m]; }
	};

	// bottom_up_cluster
	class bottom_up_cluster
	{
	private:
		shared_ptr<dataStructure::cluster_node> root_node;
		int get_build_order(shared_ptr<dataStructure::cluster_node>& cluster_node);

	public:
		void bottom_up(vector<vector<double>>& data, const function<pair<int, double>(vector<double>&)>& distance_F);
		vector<shared_ptr<dataStructure::cluster_node>> generate_cluster(int num_cluster);
		shared_ptr<dataStructure::cluster_node> get_root_node() { return root_node; }
	};
}