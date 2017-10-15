/*****************************************************************************
----------------------------Warning----------------------------------------

此段程式碼僅供 林書緯本人 履歷專用作品集，未經許可請勿使用與散播
部分程式碼改自

---O'Reilly, "Data Science from Scratch", Joel Grus, ISBN 978-1-4979-0142-7
---博碩, "Python 機器學習", Sebastian Raschka", ISBN 978-986-434-140-5
的Python程式碼

---碁峰, "The C++ Programming Language", Bjarne Stroustrup, ISBN 978-986-347-603-0
的C++範例程式

---code by 林書緯 2017/09/26
******************************************************************************/
#include "unSupervise_Learning.h"

//非監督式學習
namespace unSupervise_Learning
{
	//訓練函數
	void k_means::train(const vector<vector<double>>& data, double precision)
	{
		Statistics::Rand_uniform_Int rand(0, data.size()-1);
		vector<vector<double>> center_mass;
		int checkcount = 0;

		for (int i = 0; i < k; i++)
		{
			center_mass.push_back(data[rand()]);
			means[i].resize(data[0].size(), 0);
		}

		while (true)
		{
			vector<vector<vector<double>>> groups;
			groups.resize(k);

			sort_data_by_min_distance(data, center_mass, groups, k);
			refresh_center_mass(groups, center_mass, k);
			vector<vector<double>> aaa = means;////////////
			bool check_same = true;
			check_mass_pos(check_same, center_mass, precision);
			if (check_same) { return; }
			means = center_mass;
		}
	}

	void k_means::check_mass_pos(bool& check_same, vector<vector<double>>& center_mass, double precision)
	{
		for (int i = 0; i < k; i++)
		{
			for (int j = 0; j < center_mass[i].size(); j++)
			{
				if (distance(center_mass[i],means[i]) > precision)
				{
					check_same = false;
				}
			}
		}
	}

	void k_means::refresh_center_mass(vector<vector<vector<double>>>& groups, vector<vector<double>>& center_mass, int k)
	{
		for (int i = 0; i < k; i++)
		{
			vector<double> center_mass_refresh;
			center_mass_refresh.resize(groups[i].front().size(), 0);
			vector_mean(groups[i], center_mass_refresh);
			center_mass[i] = center_mass_refresh;
		}
	}

	void sort_data_by_min_distance(const vector<vector<double>>& data, vector<vector<double>>& center_mass, vector<vector<vector<double>>>& groups, int k)
	{
		for (int j = 0; j < data.size(); j++)
		{
			vector<double> distance;
			for (int i = 0; i < k; i++)
			{
				distance.push_back(Linear_Algebra::distance(data[j], center_mass[i]));
			}
			int group_index = Statistics::minValue(distance).first;
			groups[group_index].push_back(data[j]);
		}
	}

	double k_means::squared_clustering_errors(vector<vector<double>>& input_data, int n)
	{
		k_means test_clusters(n);
		test_clusters.train(input_data);
		auto test_center_mass = test_clusters.means;
		double squared_err = 0;

		for (int i = 0; i < input_data.size(); i++)
		{
			int cluster = test_clusters.predict(input_data[i]);
			squared_err += squared_distance(input_data[i], test_center_mass[cluster]);
		}

		return squared_err;
	}

	int k_means::predict(vector<double>& input_data)
	{
		vector<double> distance;
		for (int i = 0; i < k; i++)
		{
			distance.push_back(Linear_Algebra::distance(means[i], input_data));
		}

		int pred_cluster = Statistics::minValue(distance).first;
		return pred_cluster;
	};

	double bottom_up_cluster::cluster_distance(shared_ptr<dataStructure::cluster_node>& cluster_node1, shared_ptr<dataStructure::cluster_node>& cluster_node2, vector<vector<double>>& distance_table, const function<pair<int, double>(const vector<double>&)>& distance_F)
	{
		vector<double> distances;
		double find_distance_between_clusters;
		vector<shared_ptr<dataStructure::cluster_node>> subcluster1, subcluster2;
		search_subnode_by_DFS(cluster_node1, subcluster1);
		search_subnode_by_DFS(cluster_node2, subcluster2);

		for (int i = 0; i < subcluster1.size(); i++)
		{
			for (int j = 0; j < subcluster2.size(); j++)
			{
				find_distance_between_clusters = distance_table[(subcluster1[i]->label) - 1][(subcluster2[j]->label) - 1];
				distances.push_back(find_distance_between_clusters);
			}
		}
		return distance_F(distances).second;
	}

	void init_leaf_cluster(vector<vector<double>>& data, vector<shared_ptr<dataStructure::cluster_node>>& leaf_set)
	{
		for (int i = 0; i < data.size(); i++)
		{
			shared_ptr<dataStructure::cluster_node> leaf_node{ new dataStructure::cluster_node(data[i]) };
			leaf_set.push_back(leaf_node);
		}
	}

	void init_distance_table(vector<shared_ptr<dataStructure::cluster_node>>& leaf_set, vector<vector<double>>& distance_table)
	{
		for (int i = 0; i < leaf_set.size(); i++)
		{
			vector<double> distance_between_clusters(leaf_set.size(), numeric_limits<double>::max());
			for (int j = 0; j < leaf_set.size(); j++)
			{
				if (i != j)
				{
					distance_between_clusters[j] = Linear_Algebra::distance(leaf_set[i]->data, leaf_set[j]->data);
				}
			}
			distance_table.push_back(distance_between_clusters);
		}
	}

	void bottom_up_cluster::assemble_cluster(vector<shared_ptr<dataStructure::cluster_node>>& leaf_set, vector<vector<double>>& distance_table, string method)
	{
		while (leaf_set.size() > 1)
		{
			vector<double> distance;
			for (int i = 0; i < leaf_set.size(); i++)
			{
				for (int j = 0; j < leaf_set.size(); j++)
				{
					double point_distance = numeric_limits<double>::max();
					if (i != j)
					{
						point_distance = cluster_distance(leaf_set[i], leaf_set[j], distance_table, distance_F);
					}
					distance.push_back(point_distance);
				}
			}
			int distance_index = Statistics::minValue<double>(distance).first;

			for (int i = 0; i < leaf_set.size(); i++)
			{
				bool is_clusterd = false;
				for (int j = 0; j < leaf_set.size(); j++)
				{
					if ( i * leaf_set.size() + j == distance_index )
					{
						shared_ptr<dataStructure::cluster_node> leaf_node{ new dataStructure::cluster_node(leaf_set[i], leaf_set[j]) };

						leaf_node->order = leaf_node->get_num_build_node();

						swap(leaf_set.back(), leaf_set[j]);
						leaf_set.pop_back();

						swap(leaf_set.back(), leaf_set[i]);
						leaf_set.pop_back();

						leaf_set.push_back(leaf_node);
						is_clusterd = true;
						break;
					}
				}
				if(is_clusterd) { break; }
			}
			cout << "already classified to " << leaf_set.size() << " clusters\n";
		}
	}

	void bottom_up_cluster::search_subnode_by_DFS(shared_ptr<dataStructure::cluster_node> current_node, vector<shared_ptr<dataStructure::cluster_node>>& leaf_set)
	{
		if (current_node->is_leaf)
		{
			leaf_set.push_back(current_node);
			return;
		}

		for (int i = 0; i < current_node->child_nodes.size(); i++)
		{
			search_subnode_by_DFS(current_node->child_nodes[i], leaf_set);
		}
	}

	void bottom_up_cluster::bottom_up(vector<vector<double>>& data, string method)
	{
		dataManipulate::to_lower(method);
		if (method == "max")
		{
			distance_F = Statistics::maxValue<double>;
		}
		else
		{
			distance_F = Statistics::minValue<double>;
		}

		vector<shared_ptr<dataStructure::cluster_node>> leaf_set;
		init_leaf_cluster(data, leaf_set);
		vector<vector<double>> distance_table;
		init_distance_table(leaf_set, distance_table);
		assemble_cluster(leaf_set, distance_table, method);
		root_node = leaf_set.front();
	};

	int bottom_up_cluster::get_build_order(shared_ptr<dataStructure::cluster_node>& cluster_node)
	{
		return abs((cluster_node->order) - cluster_node->get_num_build_node());
	};

	vector<shared_ptr<dataStructure::cluster_node>> bottom_up_cluster::generate_cluster(int num_cluster)
	{
		vector<shared_ptr<dataStructure::cluster_node>> cluster_set;
		cluster_set.push_back(root_node);

		while (cluster_set.size() != num_cluster)
		{
			vector<int> num_of_order;
			for (int i = 0; i < cluster_set.size(); i++)
			{
				num_of_order.push_back(get_build_order(cluster_set[i]));
			}
			int min_index = Statistics::minValue<int>(num_of_order).first;

			shared_ptr<dataStructure::cluster_node> sub_cluster = cluster_set[min_index];

			for (int i = 0; i < sub_cluster->child_nodes.size(); i++)
			{
				cluster_set.push_back(sub_cluster->child_nodes[i]);
			}
			swap(cluster_set.back(), cluster_set[min_index]);
			cluster_set.pop_back();
		}
		return cluster_set;
	};

	void bottom_up_cluster::predict(int num_cluster, vector<vector<double>>& data)
	{
		auto clusters_set = generate_cluster(num_cluster);

		for (int i = 0; i < data.size(); i++)
		{
			vector<double> distance_to_cluster;
			for (int j = 0; j < clusters_set.size(); j++)
			{
				vector<double> distance_to_each_points;
				vector<shared_ptr<dataStructure::cluster_node>> subnode;
				search_subnode_by_DFS(clusters_set[j], subnode);

				for (int m = 0; m < subnode.size(); m++)
				{
					double distance = Linear_Algebra::distance(subnode[m]->data, data[i]);
					distance_to_each_points.push_back(distance);
				}
				double result = distance_F(distance_to_each_points).second;
				distance_to_cluster.push_back(result);
			}
			int cluster_index = Statistics::minValue<double>(distance_to_cluster).first;
			cout << "The data point " << i << " is belonging to the cluster : " << cluster_index << "\n";
		}
	}

	vector<shared_ptr<dataStructure::cluster_node>> get_children(shared_ptr<dataStructure::cluster_node>& current_node)
	{
		if (current_node->is_leaf)
		{
			cerr << "a leaf cluster has no children \n";
		}
		return current_node->child_nodes;
	}

	vector<vector<double>> get_values(shared_ptr<dataStructure::cluster_node>& current_node)
	{
		vector<vector<double>> value_set;

		if (current_node->is_leaf)
		{
			value_set.push_back(current_node->data);
		}
		else
		{
			for (int i = 0; i < current_node->child_nodes.size(); i++)
			{
				value_set.push_back(current_node->child_nodes[i]->data);
			}
		}
		return value_set;
	}
}