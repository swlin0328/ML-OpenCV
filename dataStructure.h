#include <string>
#include <vector>
#include <map>
#include <memory>
#include <set>

#define __dataStructure

using namespace std;
namespace dataStructure
{
	//決策樹資料結構
	struct tree_node
	{
		tree_node(vector<pair<map<string, string>, string>>& data) : dataset(data) { }
		tree_node() {};
		string node_name = "root";
		double child_entropy = numeric_limits<double>::max();
		int depth = 0;
		bool is_leaf = false;

		string classified_attribute;
		set<string> used_attributes;
		vector<shared_ptr<tree_node>> child_nodes;
		vector<pair<map<string, string>, string>> dataset;
	};

	//集群使用資料結構
	struct cluster_build_count
	{
		static int count;
		static int leaf_created;
	};

	struct cluster_node
	{
		cluster_build_count build_count;
		int order = 0;
		int label = 0;
		bool is_leaf = true;
		vector<double> data;
		vector<shared_ptr<cluster_node>> child_nodes;
		int get_num_build_node() { return build_count.count; }
		cluster_node(vector<double>& point) : data(point) 
		{ 
			label = ++build_count.leaf_created;
		}
		cluster_node(shared_ptr<cluster_node> node1, shared_ptr<cluster_node> node2)
		{
			child_nodes.push_back(node1);
			child_nodes.push_back(node2);
			if (!(node1->is_leaf))
			{
				for (int i = 0; i < node1->child_nodes.size(); i++)
				{
					child_nodes.push_back(node1->child_nodes[i]);
				}
			}
			if (!(node2->is_leaf))
			{
				for (int i = 0; i < node2->child_nodes.size(); i++)
				{
					child_nodes.push_back(node2->child_nodes[i]);
				}
			}
			build_count.count++;
		};
	};

	//使用者網路資料結構
	struct user_build_count
	{
		static int user_created;
	};

	struct user_information
	{
		string user_name;
		int user_id;
		vector<string> interest;
		set<shared_ptr<user_information>> friendship;
		map<int, vector<vector<int>>> shortest_paths_to;
		double betweenness_centrality;
		double page_rank;
		vector<int> endorsed_by;
		vector<int> endorses;

		user_build_count build_information;
		user_information() = delete;
		user_information(string& name, vector<string>& interests) : user_name(name), interest(interests)
		{
			user_id = ++build_information.user_created; 
		}
	};
};