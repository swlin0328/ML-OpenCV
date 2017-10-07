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
#include "NLP.h"

//自然語言
namespace NLP_lib
{
	set<string> tokenize(string text)
	{
		istringstream iData{ text };
		string line, word;
		set<string> unique_word_set;
		regex word_pattern{ "[a-z0-9A-Z]+ .?" };

		while (getline(iData, line) && iData.peek() != EOF)
		{
			dataManipulate::to_lower(line);

			istringstream is{ line };
			while (is >> word && is.peek() != EOF)
			{
				if (regex_match(word.begin(), word.end(), word_pattern))
				{
					unique_word_set.insert(word);
				}
			}
		}
		return unique_word_set;
	}

	map<string, vector<int>> count_words(vector<pair<string,bool>>& training_set)
	{
		vector<string> text;
		vector<bool> isSpam;
		Statistics::unPair(training_set, text, isSpam);

		map<string, vector<int>> countTable;
		for (int i = 0; i < text.size(); i++)
		{
			set<string> unique_word = tokenize(text[i]);
			for (auto ptr = unique_word.begin(); ptr != unique_word.end(); ptr++)
			{
				count_word(countTable, *ptr, isSpam[i]);
			}
		}
		return countTable;
	}

	void count_word(map<string, vector<int>>& countTable, string word, bool isSpam)
	{
		if (countTable.find(word) != countTable.end())
		{
			countTable[word][isSpam]++;
		}
		else
		{
			countTable[word][isSpam] = 1;
		}
	}

	void count_word(map<string, int>& countTable, string word)
	{
		if (countTable.find(word) != countTable.end())
		{
			countTable[word]++;
		}
		else
		{
			countTable[word] = 1;
		}
	}

	pair<int, string> most_common_word(map<string, int>& count_table, vector<string>& answer, vector<int>& count)
	{
		for (auto iter = count_table.begin(); iter != count_table.end(); iter++)
		{
			count.push_back((*iter).second);
			answer.push_back((*iter).first);
		}

		int max_index = Statistics::maxValue(count).first;
		return pair<int, string>(count[max_index], answer[max_index]);
	}

	vector<pair<string, vector<double>>> word_probabilities(map<string, vector<int>>& count_table, int total_spams, int total_non_spams, int k)
	{
		vector<pair<string,vector<double>>> word_probability;
		for (auto iter = count_table.begin(); iter != count_table.end(); iter++)
		{
			string word = (*iter).first;
			vector<int> frequency = (*iter).second;
			vector<double> probability;
			double spam_probability, non_spam_probability;
			
			spam_probability = (frequency[1] + k) / (total_spams + 2 * k);
			non_spam_probability = (frequency[0] + k) / (total_non_spams + 2 * k);
			//spam_probability[is_Spam]
			probability.push_back(non_spam_probability);
			probability.push_back(spam_probability);
			word_probability.push_back(pair<string, vector<double>>{word, probability});
		}
		return word_probability;
	}

	double spam_probability(vector<pair<string, vector<double>>> word_probs, string message)
	{
		auto message_words = tokenize(message);
		double log_prob_spam = 0, log_pro_not_spam = 0;
		double spam_probability, non_spam_probability;
		
		for(int i = 0; i < word_probs.size(); i++)
		{
			string word;
			double p_spam, p_not_spam;

			word = word_probs[i].first;
			p_spam = word_probs[i].second[1];
			p_not_spam = word_probs[i].second[0];

			if (message_words.find(word) != message_words.end())
			{
				log_prob_spam += log(p_spam);
				log_pro_not_spam += log(p_not_spam);
			}
			else
			{
				log_prob_spam += log(1 - p_spam);
				log_pro_not_spam += log(1 - p_not_spam);
			}
		}
		spam_probability = exp(log_prob_spam);
		non_spam_probability = exp(log_pro_not_spam);
		return spam_probability / (spam_probability + non_spam_probability);
	}

	string fix_unicode(string& text)
	{
		return regex_replace(text, regex("\u2019"), "'");
	}

	string extract_word(string& word)
	{
		regex word_pattern{ R"([\w'’]+\.?)" };
		if (regex_match(word.begin(), word.end(), word_pattern))
		{
			fix_unicode(word);
			return word;
		}
		return string("");
	}

	map<string, vector<string>> n_gram:: n_grams(const string& paragraph, int n)
	{
		istringstream word_separate{ paragraph };
		string word_candidate;
		vector<string> words;
		map<string, vector<string>> word_pair;

		while (word_separate >> word_candidate && word_separate.peek() != EOF)
		{
			word_candidate = extract_word(word_candidate);
			if (word_candidate != "")
			{
				if(word_candidate.back()!='.')
				{
					words.push_back(word_candidate);
				}
				else
				{
					words.push_back(word_candidate.substr(0, word_candidate.size()-1));
					words.push_back(string{word_candidate.back()});
				}
			}
		}

		for (int i = 0; i < words.size() - n; i++)
		{
			string first_word = words[i];
			string next_words = "";
			for (int j = i + 1; j < i + 1 + n; j++)
			{
				if (words[j] != ".")
				{
					next_words += (" " + words[j]);
				}
				else
				{
					next_words += (words[j]);
				}
			}
			word_pair[first_word].push_back(next_words);
		}
		return word_pair;
	}

	int document_words_length(string& text)
	{
		istringstream iData{ text };
		string line, word;
		regex word_pattern{ "[a-z0-9A-Z]+ .?" };
		int count = 0;

		while (getline(iData, line) && iData.peek() != EOF)
		{
			istringstream is{ line };
			while (is >> word && is.peek() != EOF)
			{
				if (regex_match(word.begin(), word.end(), word_pattern))
				{
					count++;
				}
			}
		}
		return count;
	}

	int distict_word_count(string& document)
	{
		set<string> distinct_word = tokenize(document);
		return distinct_word.size();
	}

	bool check_friend_in_paths(shared_ptr<dataStructure::user_information> user_friend, vector<vector<int>>& result_paths_to_user)
	{
		vector<int>::iterator check;
		bool in_path = false;

		for (int i = 0; i < result_paths_to_user.size(); i++)
		{
			check = find(result_paths_to_user[i].begin(), result_paths_to_user[i].end(), user_friend->user_id);
			if (check != result_paths_to_user[i].end())
			{
				in_path = true;
			}
		}
		return in_path;
	}

	void init_start_point(queue<pair<shared_ptr<dataStructure::user_information>, shared_ptr<dataStructure::user_information>>>& frontier, shared_ptr<dataStructure::user_information>& from_user)
	{
		for (auto iter = from_user->friendship.begin(); iter != from_user->friendship.end(); iter++)
		{
			pair<shared_ptr<dataStructure::user_information>, shared_ptr<dataStructure::user_information>> friendship;
			friendship.first = from_user;
			friendship.second = *iter;

			frontier.push(friendship);
		}
	}

	bool is_same_path(vector<int>& path1, vector<int>& path2)
	{
		if (path1.size() != path2.size())
		{
			return false;
		}
		for (int i = 0; i < path1.size(); i++)
		{
			if (path1[i] != path2[i])
			{
				return false;
			}
		}
		return true;
	}

	void add_path_if_new_and_smaller(vector<int>& new_path_to_user, vector<vector<int>>& old_paths_to_user, vector<vector<int>>& result_paths_to_user, int min_path_length)
	{
		if (new_path_to_user.size() <= min_path_length)
		{
			bool is_old_path = false;
			for (int j = 0; j < old_paths_to_user.size(); j++)
			{
				is_old_path = is_same_path(new_path_to_user, old_paths_to_user[j]);
			}
			if (!is_old_path)
			{
				result_paths_to_user.push_back(new_path_to_user);
			}
		}
	}

	double cosine_similarity(vector<int>& v, vector<int>& w)
	{
		return dot(v, w) / sqrt(dot(v, v) * dot(w, w));
	}

	//訓練函數
	string n_gram::generate_using_model()
	{
		string prev(".");
		string result = "";
		string current, last_word;
		int rand, randIndex, word_index;
		Statistics::Rand_uniform_Int randInt(0, 10000);

		while (true)
		{
			rand = randInt();
			randIndex = rand % model_result[prev].size();
			current = model_result[prev][randIndex];
			result += current;

			word_index = current.find_last_of(" ");
			last_word = current.substr(word_index + 1);
			if (last_word.back() == '.')
			{
				break;
			}
			prev = last_word;
		}
		int first_index = result.find_first_not_of(" ");
		result = result.substr(first_index);
		return result;
	}

	int K_topic_given_document::sample_from(vector<double>& weights)
	{
		Statistics::Rand_uniform_double rand_gen(0, 1);
		double total = Statistics::sum(weights);
		double rnd = total * rand_gen();

		for (int i = 0; i < weights.size(); i++)
		{
			rnd -= weights[i];
			if (rnd <= 0)
			{
				return i;
			}
		}
		return 0;
	}

	double K_topic_given_document::p_word_given_topic(int document_index, int k_topic, string& word, double beta)
	{
		return (topic_word_count[k_topic][word] + beta) / (topic_words_count[k_topic] + W[document_index] * beta);
	}

	double K_topic_given_document::p_topic_given_document(int document_index, int k_topic, double alpha)
	{
		int doc_length = document_length[document_index];
		return (document_topic_count[document_index][k_topic] + alpha) / (doc_length + K * alpha);
	}

	vector<vector<int>> K_topic_given_document::init_topic_to_each_word(vector<string>& documents)
	{
		Statistics::Rand_uniform_Int randInt(0, K-1);

		for (int j = 0; j < documents.size(); j++)
		{
			int total_words = NLP_lib::document_words_length(documents[j]);
			document_length.push_back(total_words);

			vector<int> paragraph_topics;
			for (int i = 0; i < total_words; i++)
			{
				paragraph_topics.push_back(randInt());
			}
			documents_topics.push_back(paragraph_topics);
			//順便計算每組文章 Unique 單詞數
			W.push_back(NLP_lib::distict_word_count(documents[j]));
		}
		return documents_topics;
	}

	double K_topic_given_document::topic_weight(int document_index, string& word, int k_topic)
	{
		return p_word_given_topic(document_index, k_topic, word) *  p_topic_given_document(document_index, k_topic);
	}

	int K_topic_given_document::choose_new_topic(int document_index, string& word)
	{
		vector<double> new_topic_weight;
		for (int i = 0; i < K; i++)
		{
			new_topic_weight.push_back(topic_weight(document_index, word, i));
		}
		return sample_from(new_topic_weight);
	}


	void K_topic_given_document::train(vector<string>& documents, int epoch)
	{
		vector<vector<int>> topic_of_word = init_topic_to_each_word(documents); //隨機指定主題到每個單詞
		vector<vector<string>> documents_word;

		for (int i = 0; i < documents.size(); i++)
		{
			dataManipulate::to_lower(documents[i]);
			istringstream iData{ documents[i] };
			vector<string> paragraph_word;
			regex word_pattern{ "[a-z0-9A-Z]+ .?" };
			string word;
			vector<int> topic_count;
			topic_count.resize(K, 0);

			for (int j = 0; j < document_length[i]; j++)
			{
				int topic_index = topic_of_word[i][j];
				topic_count[topic_index]++;
				topic_words_count[topic_index]++;

				while (!regex_match(word.begin(), word.end(), word_pattern) && iData.peek() != EOF)
				{
					iData >> word;
				}
				paragraph_word.push_back(word);
				topic_word_count[topic_index][word]++;
			}
			documents_word.push_back(paragraph_word);
			document_topic_count.push_back(topic_count);
		}
		for (int i = 0; i < epoch; i++)
		{
			for (int j = 0; j < documents.size(); j++)
			{
				for (int w = 0; w < document_length[j]; w++)
				{
					string word = documents_word[j][w];
					int topic_index = topic_of_word[j][w];
					int new_topic = choose_new_topic(j, documents_word[j][w]);

					document_topic_count[j][topic_index]--;
					topic_word_count[topic_index][word]--;
					topic_words_count[topic_index]--;

					document_topic_count[j][new_topic]++;
					topic_word_count[new_topic][word]++;
					topic_words_count[new_topic]++;
				}
			}
		}
	}

	void K_topic_given_document::show_result(int n)
	{
		//分組結果 --> 顯示前n項
		vector<vector<pair<string, int>>> topic_group;
		for (int i = 0; i < topic_word_count.size(); i++)
		{
			vector<pair<string, int>> classify;
			for (auto iter = topic_word_count[i].begin(); iter != topic_word_count[i].end(); iter++)
			{
				pair<string, int> data(iter->first, iter->second);
				classify.push_back(data);
			}
			sort(classify.begin(), classify.end(), [](pair<string, int> x1, pair<string, int> x2) { return x1.second > x2.second; });
			topic_group.push_back(classify);
		}
		for (int i = 0; i < topic_group.size(); i++)
		{
			for (int j = 0; j < n; j++)
			{
				cout << "主題 " << i << " : 項目-> " << topic_group[i][j].first << " 票數-> " << topic_group[i][j].second << "\n";
			}
		}
	}

	void users_information::create_user(string& name, vector<string>& interest)
	{
		for (int i = 0; i < interest.size(); i++)
		{
			dataManipulate::to_lower(interest[i]);
			interest_set.insert(interest[i]);
		}

		dataStructure::user_information user(name, interest);
		shared_ptr<dataStructure::user_information> user_ptr{ &user };
		user_table[user.user_id] = user_ptr;
		users.push_back(user_ptr);
	}

	void users_information::add_friend(int user_id, int friend_id)
	{
		shared_ptr<dataStructure::user_information> user_ptr = user_table[user_id];
		shared_ptr<dataStructure::user_information> friend_ptr = user_table[friend_id];

		user_ptr->friendship.insert(friend_ptr);
		friend_ptr->friendship.insert(user_ptr);
	}

	void users_information::endorse_user(int source_id, int target_id)
	{
		user_table[source_id]->endorses.push_back(target_id);
		user_table[target_id]->endorsed_by.push_back(source_id);
	}

	void users_information::betweenness_centrality()
	{
		for (int i = 0; i < users.size(); i++)
		{
			users[i]->betweenness_centrality = 0.0;
		}

		for (int i = 0; i < users.size(); i++)
		{
			int source_id = users[i]->user_id;

			for (auto iter = users[i]->shortest_paths_to.begin(); iter != users[i]->shortest_paths_to.end(); iter++)
			{
				int target_id = iter->first;
				if (source_id < target_id)
				{
					int num_paths = iter->second.size();
					double contrib = 1 / num_paths;
					for (int j = 0; j < iter->second.size(); j++)
					{
						for (int k = 0; k < iter->second[j].size(); k++)
						{
							int user_id = iter->second[j][k];
							if (user_id != source_id && user_id != target_id)
							{
								user_table[user_id]->betweenness_centrality += contrib;
							}
						}
					}
				}
			}
		}
	}

	void users_information::shortest_paths_from(shared_ptr<dataStructure::user_information> from_user)
	{
		queue<pair<shared_ptr<dataStructure::user_information>, shared_ptr<dataStructure::user_information>>> frontier;
		map<int, vector<vector<int>>> shortest_paths_to;

		NLP_lib::init_start_point(frontier, from_user);
		while (!frontier.empty())
		{
			pair<shared_ptr<dataStructure::user_information>, shared_ptr<dataStructure::user_information>> friendship;
			shared_ptr<dataStructure::user_information> prev_user = frontier.front().first;
			shared_ptr<dataStructure::user_information> user = frontier.front().second;
			frontier.pop();

			int min_path_length;
			vector<vector<int>> paths_to_prev_user = shortest_paths_to[prev_user->user_id];
			vector<vector<int>> old_paths_to_user, new_paths_to_user, result_paths_to_user;
			for (int i = 0; i < paths_to_prev_user.size(); i++)
			{
				paths_to_prev_user[i].push_back(user->user_id);
				new_paths_to_user.push_back(paths_to_prev_user[i]);
			}

			new_paths_to_user.push_back(vector<int>{user->user_id});
			old_paths_to_user = shortest_paths_to[user->user_id];
			result_paths_to_user = old_paths_to_user;

			if (old_paths_to_user.size())
			{
				vector<int> path_length;
				for (int i = 0; i < old_paths_to_user.size(); i++)
				{
					path_length.push_back(old_paths_to_user[i].size());
				}
				min_path_length = Statistics::minValue(path_length).second;
			}
			else
			{
				min_path_length = numeric_limits<int>::max();
			}

			for (int i = 0; i < new_paths_to_user.size(); i++)
			{
				NLP_lib::add_path_if_new_and_smaller(new_paths_to_user[i], old_paths_to_user, result_paths_to_user, min_path_length);
			}

			for (auto iter = user->friendship.begin(); iter != user->friendship.end(); iter++)
			{
				bool is_friend_in_path = NLP_lib::check_friend_in_paths(*iter, result_paths_to_user);
				if (!is_friend_in_path)
				{
					frontier.push(pair<shared_ptr<dataStructure::user_information>, shared_ptr<dataStructure::user_information>>(user, *iter));
				}
			}
			shortest_paths_to[user->user_id] = result_paths_to_user;
		}
		from_user->shortest_paths_to = shortest_paths_to;
	}

	void users_information::page_rank(double damping, int n_iters)
	{
		int num_users = users.size();
		int round;
		double base_rank = (1 - damping) / num_users;
		double init_rank = 1 / num_users;
		vector<map<int, double>> vote_table;
		vote_table.resize(2); //投票箱(初始、結果)

		for (int i = 0; i < num_users; i++)
		{
			users[i]->page_rank = base_rank;
			vote_table[0][users[i]->user_id] = init_rank - base_rank;
			vote_table[1][users[i]->user_id] = 0;
		}

		for (int i = 0; i < n_iters; i++)
		{
			round = i % 2;
			for (int j = 0; j < num_users; j++)
			{
				double link_rank = vote_table[round][users[j]->user_id] / users[j]->endorses.size();
				for (int k = 0; k < users[j]->endorses.size(); k++)
				{
					vote_table[1 - round][users[j]->endorses[k]] += link_rank;
				}
				vote_table[round][users[j]->user_id] = 0;
			}
		}

		for (auto iter = vote_table[1 - round].begin(); iter != vote_table[1 - round].end(); iter++)
		{
			int user_id = (*iter).first;
			double score = (*iter).second;
			users[user_id]->page_rank += score;
		}
	}

	vector<int> users_information::make_user_interest_vector(int user_id)
	{
		auto user_interest = user_table[user_id]->interest;
		map<string, int> hash_table;
		vector<int> interest_vector;
		interest_vector.resize(interest_set.size(), 0);
		int index = 0;

		for (auto iter = interest_set.begin(); iter != interest_set.end(); iter++)
		{
			hash_table[*iter] = index++;
		}

		for (int i = 0; i < user_interest.size(); i++)
		{
			int interest_index = hash_table[user_interest[i]];
			interest_vector[interest_index] = 1;
		}
		return interest_vector;
	}

	double users_information::user_similarity(int user_id1, int user_id2)
	{
		vector<int> interest_vector1 = make_user_interest_vector(user_id1);
		vector<int> interest_vector2 = make_user_interest_vector(user_id2);

		return NLP_lib::cosine_similarity(interest_vector1, interest_vector2);
	}

	vector<pair<int, double>> users_information::most_similar_users(int user_id)
	{
		vector<pair<int, double>> similar_table;
		double similarity;

		for (int i = 0; i < users.size(); i++)
		{
			if (users[i]->user_id != user_id)
			{
				similarity = user_similarity(user_id, users[i]->user_id);
				similar_table.push_back(pair<int, double>(users[i]->user_id, similarity));
			}
			else
			{
				similar_table.push_back(pair<int, double>(users[i]->user_id, 0.0));
			}
		}
		sort(similar_table.begin(), similar_table.end(), [](pair<int, double> user1, pair<int, double>& user2) {return user1.second > user2.second; });
		return similar_table;
	}

	vector<pair<string, double>> users_information::user_based_suggestion(int user_id, int num_interest)
	{
		//使用者:分數
		vector<pair<int, double>> similar_table = most_similar_users(user_id);
		//興趣編號:前N個最高分興趣
		deque<pair<int, double>> result;
		//興趣編號
		int interest_index = 0;
		//興趣編號:興趣名稱
		map<int, string> interest_table;
		//興趣名稱:分數
		vector<pair<string, double>> recommend_interest;
		vector<double> recommend_vector;
		recommend_vector.resize(interest_set.size(), 0);

		for (int i = 0; i < similar_table.size(); i++)
		{
			vector<int> interest_vector;
			make_user_interest_vector(similar_table[i].first);

			for (int j = 0; j < recommend_vector.size(); j++)
			{
				recommend_vector[j] += interest_vector[j] * similar_table[i].second;
			}
		}
		Statistics::first_N_maxVal(recommend_vector, result, num_interest);

		for (auto iter = interest_set.begin(); iter != interest_set.end(); iter++)
		{
			interest_table[interest_index++] = *iter;
		}

		for (int i = 0; i < num_interest; i++)
		{
			string interested_name = interest_table[result.at(i).first];
			double score = result.at(i).second;
			recommend_interest.push_back(pair<string, double>(interested_name, score));
		}
		return recommend_interest;
	}

	void NaiveBayesClassifier::train(const vector<string>& X, const vector<bool>& y)
	{
		vector<pair<string, bool>> training_set;
		vector_length_queal(X, y);
		int num_spams = 0, num_non_spams = 0;

		for (int i = 0; i < X.size(); i++)
		{
			training_set.push_back(make_pair(X[i], y[i]));
		}

		for (int i = 0; i < y.size(); i++)
		{
			if (y[i]) { num_spams += 1; }
		}

		num_non_spams = training_set.size() - num_spams;
		auto word_counts = NLP_lib::count_words(training_set);
		word_probs = NLP_lib::word_probabilities(word_counts, num_spams, num_non_spams, leraning_Constant);
	}

	vector<double> NaiveBayesClassifier::predict(const vector<string>& X)
	{
		vector<double> spam_probs;
		for (int i = 0; i < X.size(); i++)
		{
			spam_probs.push_back(NLP_lib::spam_probability(word_probs, X[i]));
		}
		return spam_probs;
	}
}