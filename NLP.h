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
#include <regex>
#include <sstream>

#define __NLP

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
#ifndef __Statistics
#define __Statistics
#include "Statistics.h"
#endif

#ifndef __dataManipulate
#define __dataManipulate
#include "dataManipulate.h"
#endif

using namespace std;
using namespace Linear_Algebra;
using namespace std::chrono;
namespace NLP_lib
{
	//訓練函數
	set<string> tokenize(string text);

	map<string, vector<int>> count_words(vector<pair<string, bool>>& training_set);

	void count_word(map<string, vector<int>>& countTable, string word, bool isSpam);

	vector<pair<string, vector<double>>> word_probabilities(map<string, vector<int>>& count_table, int total_spams, int total_non_spams, int k = 0.5);

	double spam_probability(vector<pair<string, vector<double>>> word_probs, string message);

	void count_word(map<string, int>& countTable, string word);

	pair<int, string> most_common_word(map<string, int>& count_table, vector<string>& answer, vector<int>& count);

	string fix_unicode(string& text);

	string extract_word(string& word);

	int document_words_length(string& text);

	int distict_word_count(string& document);

	bool check_friend_in_paths(shared_ptr<dataStructure::user_information> user_friend, vector<vector<int>>& result_paths_to_user);

	void init_start_point(queue<pair<shared_ptr<dataStructure::user_information>, shared_ptr<dataStructure::user_information>>>& frontier, shared_ptr<dataStructure::user_information>& from_user);

	bool is_same_path(vector<int>& path1, vector<int>& path2);

	double cosine_similarity(vector<int>& v, vector<int>& w);

	//NLP分類器
	/*
	simplified Bayes
	P(V,S) = P(V|S) * P(S) = P(S|V) * P(V)
	P(S|V) = P(V|S) * P(S) / [P(V|S) * P(S) + P (V|-S) P(-S)]
	*/
	//貝氏分類器
	class NaiveBayesClassifier
	{
	private:
		double leraning_Constant;
		vector<pair<string, vector<double>>> word_probs;

	public:
		void train(const vector<string>& X, const vector<bool>& y);
		vector<double> predict(const vector<string>& X);
		NaiveBayesClassifier(double k = 0.5) :leraning_Constant(k) {}
	};

	//n-gram模型
	class n_gram
	{
	private:
		int n;
		map<string, vector<string>> model_result;
		map<string, vector<string>> n_grams(const string& paragraph, int n);

	public:
		n_gram(string text, int k) : n(k) { model_result = n_grams(text, n); }
		n_gram() = delete;
		string generate_using_model();
	};

	//主題模型化
	/*
	主題數量 K
	文件d 主題n 機率
	主題n 單詞w 機率

	隨機挑選主題--> 隨機挑選單詞
	*/
	class K_topic_given_document
	{
	private:
		vector<vector<int>> document_topic_count; //每個文件中，每個主題出現次數
		vector<map<string, int>> topic_word_count; //每個主題中，每個單詞出現的次數
		vector<int> topic_words_count; //每個主題中，單詞的總數量
		vector<vector<int>> documents_topics;
		vector<int> document_length;
		int K;
		vector<int> W;

		int sample_from(vector<double>& weights);
		double p_word_given_topic(int document_index, int k_topic, string& word, double beta = 0.1);
		double p_topic_given_document(int document_index, int k_topic, double alpha = 0.1);
		vector<vector<int>> init_topic_to_each_word(vector<string>& documents);
		double topic_weight(int document_index, string& word, int k_topic);
		int choose_new_topic(int document_index, string& word);

	public:
		K_topic_given_document(int n = 5) : K(n) 
		{
			topic_words_count.resize(K, 0);
			topic_word_count.resize(K);
		}
		K_topic_given_document() = delete;
		void K_topic_given_document::train(vector<string>& documents, int epoch = 2000);
		void show_result(int n = 2);
	};

	//使用者網路分類
	class users_information
	{
	private:
		vector<shared_ptr<dataStructure::user_information>> users;
		map<int, shared_ptr<dataStructure::user_information>> user_table;
		set<string> interest_set;

		vector<int> make_user_interest_vector(int user_id);
		void shortest_paths_from(shared_ptr<dataStructure::user_information> from_user);
		bool add_path_and_node_if_new_path(vector<int>& new_path_to_user, vector<vector<int>>& old_paths_to_user, vector<vector<int>>& result_paths_to_user, int min_path_length, queue<pair<shared_ptr<dataStructure::user_information>, shared_ptr<dataStructure::user_information>>>& frontier);
		void users_information::add_node_to_queue(int user_id, vector<vector<int>>& result_paths_to_user, queue<pair<shared_ptr<dataStructure::user_information>, shared_ptr<dataStructure::user_information>>>& frontier);
		void show_centrality();
		void show_page_rank();

	public:
		users_information() {};
		users_information(string path);
		void create_user(string& name, vector<string>& interests);
		void add_friend(int user_id, int friend_id);
		void endorse_user(int source_id, int target_id);

		double user_similarity(int user_id1, int user_id2);
		void betweenness_centrality();
		void page_rank(double damping = 0.85, int n_iters = 100);
		void show_training_result();
		vector<pair<int, double>> most_similar_users(int user_id);
		vector<pair<string, double>> user_based_suggestion(int user_id, int num_interest = 5);
	};
}