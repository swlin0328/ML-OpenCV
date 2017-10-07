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
#include "Supervise_Learning.h"

//監督式學習
namespace Supervise_Learning
{
	//訓練函數
	void perceptron::train(vector<vector<double>>& X, vector<double>& y)
	{
		train_clear(cost, w, X[0].size());
		Linear_Algebra::randomVector(w);

		for (int i = 0; i < n_iter; i++)
		{
			vector<double> gradient;

			estimate_gradient(
			[&](vector<double> w_0)
			{double output{ 0 }, err{ 0 };
			return square_error(X, y, w_0, actFn); }, w, gradient);

			for (int k = 0; k < w.size(); k++)
			{
				w[k] -= leraning_Rate * gradient[k];
			}
			cost.push_back(square_error(X, y, w, actFn));
			classify(X);
		}
	}

	void perceptron::train_for_network(perceptron& neuron, int index, double delta)
	{
		neuron.w[index] += delta;
	}

	vector<double>& perceptron::get_neuron_w(perceptron& neuron)
	{
		return neuron.w;
	}

	void perceptron::classify(const vector<vector<double>>& X)
	{
		predict_y.clear();
		predict_y.resize(0);
		for (int i = 0; i < X.size(); i++)
		{
			double value = actFn(dot(X[i], w));
			if (actFn(dot(X[i], w)) > 0.5)
			{
				predict_y.push_back(1);
			}
			else
			{
				predict_y.push_back(-1);
			}
		}
	}

	void perceptron::show_train_result()
	{
		cout << "Number of epoch: " << n_iter << " , the cost values for the each epoch in training\n";
		for (int i = 0; i < n_iter; i++)
		{
			if ((i+1) % 50 == 0)
			{
				cout << "The epoch " << i+1 << " of the cost is: " << cost[i] << "\n";
			}
		}
	}

	double perceptron::predict_prob(vector<double>& input)
	{
		return dot(w, input);
	}

	void perceptron::show_validate_result(const vector<double>& Y)
	{
		double num = err_inPrediction(Y);
		double accurate = 1 - (num / Y.size());
		cout << "The predict accurate: " << setprecision(4) << 100 * accurate << " %\n";
	}

	double perceptron::err_inPrediction(const vector<double>& Y)
	{
		int num = 0;
		for (int i = 0; i < Y.size(); i++)
		{
			int predict = predict_y[i] / abs(predict_y[i]);
			if (predict != Y[i])
			{
				num += 1;
			}
		}
		return num;
	}

	double perceptron::cost_inPrediction(const vector<double>& Y)
	{
		double cost{ 0 };
		for (int i = 0; i < Y.size(); i++)
		{
			cost += pow((Y[i] - predict_y[i]), 2);
		}
		return cost;
	}

	double activation_for_logistic(double wX)
	{
		double amplifier = 1.0;
		return (1 / (1 + exp(-amplifier * wX)));
	}

	double activation_for_step_finction(double wX)
	{
		return (wX / abs(wX));
	}

	double activation_for_hyperbolic(double wX)
	{
		return tanh(wX);
	}

	double square_error(vector<vector<double>>& X, vector<double>& Y, vector<double>& w, const function<double(double)>& actF)
	{
		double errValue = 0;
		for (int i = 0; i < X.size(); i++)
		{
			errValue += pow((Y[i] - actF(dot(w, X[i]))), 2);
		}
		return errValue;
	}

	double square_regularization(vector<double>& w, double lamda)
	{
		return (lamda * dot(w, w));
	}

	vector<double> estimate_w_with_ridge(vector<vector<double>>& X, vector<double>& Y, vector<double>& w, const function<double(double)>& actF, double lamda)
	{
		w.resize(X[0].size());
		randomVector(w);
		return minimize_stochastic([&](vector<vector<double>>& X_data, vector<double>& Y_data, vector<double>& w_0)
		{
			return (square_error(X_data, Y_data, w_0, actF) + square_regularization(w_0, lamda));
		}, w, X, Y);
	}

	template<typename T>
	double majority_vote(vector<T> labels)
	{
		//labels 需先以距離排列處理
		pair<T, bool> most_frequent = Statistics::most_frequent_in_group(labels);

		if (most_frequent.second == true)
		{
			return most_frequent.first;
		}
		else
		{
			labels.pop_back();
			return majority_vote(labels);
		}
	}

	template<typename T, typename U>
	int knn_classify(int k, vector<T> X, vector<U> Y, vector<T> newpoint)
	{
		vector<double> distance_Group;

		for (int i = 0; i < X.size(); i++)
		{
			distance_Group.push_back(Linear_Algebra::distance(X[i], newpoint));
		}

		vector<pair<double, double>> neighbor = make_pair(distance_Group, Y);
		sort(neighbor.begin(), neighbor.end(), [](pair<double, double>& p1, pair<double, double>& p2) {
			return (p1.first < p2.first);)};

		vector< pair<double, double> > k_nearert_lables;
		for (int i = 0; i < k; i++)
		{
			k_nearert_lables.push_back(neighbor[i]);
		}
		return majority_vote(k_nearert_lables);
	}

	void linear_regression::train(vector<vector<double>>& X, vector<double>& y)
	{
		train_clear(cost, w, X[0].size());
		w = Linear_Algebra::linear_regression(X, y, w);
	}

	void linear_regression::classify(const vector<vector<double>>& X)
	{
		predict_y.clear();
		predict_y.resize(0);
		for (int i = 0; i < X.size(); i++)
		{
			predict_y.push_back(dot(X[i], w));
			cout << "The predict values: " << predict_y[i] << "\n";
		}
	}

	void linear_regression::show_validate_result(const vector<double>& Y)
	{
		cout << "The error in prediction: " << setprecision(4) << err_inPrediction(Y) << " \n";
		cout << "The square error in prediction: " << setprecision(4) << cost_inPrediction(Y) << " \n";
	}

	double linear_regression::err_inPrediction(const vector<double>& Y)
	{
		double result = 0;
		for (int i = 0; i < Y.size(); i++)
		{
			result += Y[i] - predict_y[i];
		}
		return result;
	}

	double linear_regression::cost_inPrediction(const vector<double>& Y)
	{
		double cost{ 0 };
		for (int i = 0; i < Y.size(); i++)
		{
			cost += pow((Y[i] - predict_y[i]), 2);
		}
		return cost;
	}

	void linear_regression::show_train_result()
	{
		cout << "The square error : " << cost[0] << " , in the last training\n";
	}

	void perceptron::train_clear(vector<double>& cost, vector<double>& w, int vSize)
	{
		cost.clear();
		w.clear();
		cost.resize(0);
		w.resize(vSize, 0);
	}

	void linear_regression::train_clear(vector<double>& cost, vector<double>& w, int vSize)
	{
		cost.clear();
		w.clear();
		cost.resize(0);
		w.resize(vSize, 0);
	}

	double logistic_log_likelyhood_i(vector<double>& Xi, double& yi, vector<double>& w)
	{
		if (yi == 1)
		{
			return log(activation_for_logistic(dot(w, Xi)));
		}
		else
		{
			return log(1 - activation_for_logistic(dot(w, Xi)));
		}
	}

	double logistic_log_likelihood(vector<vector<double>>& X, vector<double>& Y, vector<double>& w)
	{
		double sum = 0;
		for (int i = 0; i < X.size(); i++)
		{
			sum += logistic_log_likelyhood_i(X[i], Y[i], w);
		}
		return sum;
	}

	double entropy(vector<double>& class_probabilities)
	{
		double sum = 0;
		for (int i = 0; i < class_probabilities.size(); i++)
		{
			if (class_probabilities[i])
			{
				sum += -class_probabilities[i] * log(class_probabilities[i]);
			}
		}
		return sum;
	}

	vector<double> class_probabilities(vector<pair<map<string, string>, string>>& data)
	{
		map<string, int> count_table;
		vector<double> class_probabilities;
		int total_count = data.size();

		for (int i = 0; i < total_count; i++)
		{
			string label = data[i].second;
			NLP_lib::count_word(count_table, label);
		}
		for (auto ptr = count_table.begin(); ptr != count_table.end(); ptr++)
		{
			class_probabilities.push_back((*ptr).second / total_count);
		}
		return class_probabilities;
	}

	double data_entropy(vector<pair<map<string, string>, string>>& data)
	{
		vector<double> probabilities{ class_probabilities(data) };
		return entropy(probabilities);
	}

	double partition_entropy(vector<vector<pair<map<string, string>, string>>>& subsets)
	{
		double total_subsets_entropy = 0, total_count = 0;
		for (int i = 0; i < subsets.size(); i++)
		{
			total_count += subsets[i].size();
		}
		for (int i = 0; i < subsets.size(); i++)
		{
			total_subsets_entropy += data_entropy(subsets[i]) * (subsets[i].size() / total_count);
		}
		return total_subsets_entropy;
	}

	map<string, vector<pair<map<string, string>, string>>> decision_tree_id3::partition_by(vector<pair<map<string, string>, string>>& data, string attribute)
	{
		map<string, vector<pair<map<string, string>, string>>> groups;
		for (int i = 0; i < data.size(); i++)
		{
			string key = data[i].first[attribute];
			groups[key].push_back(data[i]);
		}
		return groups;
	}

	double  decision_tree_id3::partition_entropy_by(vector<pair<map<string, string>, string>>& data, string attribute)
	{
		map<string, vector<pair<map<string, string>, string>>> partitions = partition_by(data, attribute);
		vector<vector<pair<map<string, string>, string>>> subsets;

		for (auto iter = partitions.begin(); iter != partitions.end(); iter++)
		{
			subsets.push_back((*iter).second);
		}
		return partition_entropy(subsets);
	}

	void decision_tree_id3::write_the_lowest_entropy_attribute(shared_ptr<dataStructure::tree_node>& current_node)
	{
		vector<string> attributes;
		for (auto iter = current_node->dataset[0].first.begin(); iter != current_node->dataset[0].first.end(); iter++)
		{
			attributes.push_back((*iter).first);
		}
		vector<double> group_entropy;
		for (int i = 0; i < attributes.size(); i++)
		{
			double entropy = numeric_limits<double>::max();
			set<string> used_attrubute = current_node->used_attributes;
			if (used_attrubute.insert(attributes[i]).second)
			{
				entropy = partition_entropy_by(current_node->dataset, attributes[i]);
			}
			group_entropy.push_back(entropy);
		}
		auto min_entropy = Statistics::minValue(group_entropy);

		current_node->used_attributes.insert(attributes[min_entropy.first]);
		current_node->child_entropy = min_entropy.second;
		current_node->classified_attribute = attributes[min_entropy.first];
	}

	void decision_tree_id3::split_by_attribute(shared_ptr<dataStructure::tree_node>& current_node, int break_point, double accuracy)
	{
		try_make_leaf(current_node, break_point, accuracy);
		if (current_node->is_leaf){return;}

		write_the_lowest_entropy_attribute(current_node);
		auto groups = partition_by(current_node->dataset, current_node->classified_attribute);
		for (auto iter = groups.begin(); iter != groups.end(); iter++)
		{
			shared_ptr<dataStructure::tree_node> newnode(new dataStructure::tree_node((*iter).second));
			newnode->node_name = (*iter).first;
			newnode->used_attributes = current_node->used_attributes;
			newnode->depth = (current_node->depth) + 1;
			current_node->child_nodes.push_back(move(newnode));
		}
		for (int i = 0; i < current_node->child_nodes.size(); i++)
		{
			split_by_attribute(current_node->child_nodes[i], break_point, accuracy);
		}
	}

	void decision_tree_id3::train(int break_point, double accuracy)
	{
		check_node();
		split_by_attribute(root_node, break_point, accuracy);
	}

	void decision_tree_id3::train(vector<pair<map<string, string>, string>>& data, int break_point, double accuracy)
	{
		train_clear();
		root_node->dataset = data;
		check_node();
		split_by_attribute(root_node, break_point, accuracy);
	}

	string decision_tree_id3::predict(map<string, string>& input)
	{
		check_node();
		return decision_tree_classify(root_node, input);
	}

	string decision_tree_classify(shared_ptr<dataStructure::tree_node>& decision_tree, map<string, string> input)
	{
		if (decision_tree->is_leaf)
		{
			auto result = find_most_common_attribute(decision_tree);
			return get<0>(result);
		}

		string pred_attribute = (decision_tree->classified_attribute);

		for (int i = 0; i < decision_tree->child_nodes.size(); i++)
		{
			if (decision_tree->child_nodes[i]->node_name == input[pred_attribute])
			{
				return decision_tree_classify(decision_tree->child_nodes[i], input);
			}
		}
		return get<0>(find_most_common_attribute(decision_tree));
	}

	void decision_tree_id3::check_node()
	{
		if (root_node == nullptr)
		{
			decision_tree_id3 new_tree;
			this->root_node = move(new_tree.root_node);
		}

		if (root_node == nullptr || root_node->dataset.size() == 0)
		{
			cerr << "null tree \n";
		}
	}

	shared_ptr<dataStructure::tree_node> decision_tree_id3::build_decision_tree(vector<pair<map<string, string>, string>>& data)
	{
		shared_ptr<dataStructure::tree_node> root_of_decision_tree;
		root_of_decision_tree->dataset = data;
		return root_of_decision_tree;
	}

	tuple<string, int, int> find_most_common_attribute(shared_ptr<dataStructure::tree_node>& decision_tree)
	{
		map<string, int> count_table;
		vector<string> answer;
		vector<int> count;
		for (int i = 0; i < decision_tree->dataset.size(); i++)
		{
			NLP_lib::count_word(count_table, decision_tree->dataset[i].second);
		}

		pair<int, string> answer_frequency = NLP_lib::most_common_word(count_table, answer, count);

		return tuple<string, int, int>(answer_frequency.second, answer_frequency.first, answer.size());
	}

	void try_make_leaf(shared_ptr<dataStructure::tree_node>& current_node, int break_point, double accuracy)
	{
		auto result = find_most_common_attribute(current_node);
		int num_input = current_node->dataset.size();

		if (current_node->depth == break_point)
		{
			current_node->is_leaf = true;
			current_node->classified_attribute = get<0>(result);
		}
		if (get<1>(result) > num_input * accuracy)
		{
			current_node->is_leaf = true;
			current_node->classified_attribute = get<0>(result);
		}
		if (current_node->used_attributes.size() >= get<2>(result))
		{
			current_node->is_leaf = true;
			current_node->classified_attribute = get<0>(result);
		}
	}

	bool decision_tree_id3::pick_out_attribute(string attribute)
	{
		return root_node->used_attributes.insert(attribute).second;
	}

	void decision_tree_id3::train_clear()
	{
		root_node.~shared_ptr();
		decision_tree_id3 new_tree;
		this->root_node = move(new_tree.root_node);
	}

	void decision_tree_id3::show_tree_struct()
	{
		queue<shared_ptr<dataStructure::tree_node>> tree_queue;
		BFS(root_node, tree_queue);
	}

	void decision_tree_id3::BFS(shared_ptr<dataStructure::tree_node> current_node, queue<shared_ptr<dataStructure::tree_node>>& tree_queue)
	{
		if (current_node->is_leaf)
		{
			cout <<"---------------This node is leaf---------------\n";
			cout << "Current depth : " << current_node->depth << "\n";
			cout << "Current answer : " << current_node->node_name << "\n";
			cout << "Next classified attribute : " << current_node->classified_attribute << "\n";
			cout << "-----------------------------------------------\n";
			return;
		}

		for (int i = 0; i < current_node->child_nodes.size(); i++)
		{
			tree_queue.push(current_node->child_nodes[i]);
		}
		cout << "Current depth : " << current_node->depth << "\n";
		cout << "Current answer : " << current_node->node_name << "\n";
		cout << "Next classified attribute : " << current_node->classified_attribute << "\n";

		while (!tree_queue.empty())
		{
			shared_ptr<dataStructure::tree_node>& tree_search = tree_queue.front();
			tree_queue.pop();
			BFS(tree_search, tree_queue);
		}
	}

	void random_forest::insert_tree(decision_tree_id3 tree)
	{
		node_of_tree.push_back(move(tree));
	}

	void random_forest::train_clear()
	{
		for (int i = 0; i < node_of_tree.size(); i++)
		{
			node_of_tree[i].train_clear();
		}
	}

	string random_forest::predict(map<string, string>& input)
	{
		map<string, int> count_table;
		vector<string> answer;
		vector<int> count;

		for (int i = 0; i < node_of_tree.size(); i++)
		{
			NLP_lib::count_word(count_table, node_of_tree[i].predict(input));
		}
		pair<int, string> answer_frequency = NLP_lib::most_common_word(count_table, answer, count);
		return answer_frequency.second;
	}

	void neuron_network::make_Neural_Network(vector<vector<vector<perceptron>>>& NN, int input_dim, int output_dim, int row, int col, int depth, double eta, int epoch, const function<double(double)>& actFunction)
	{
		vector<vector<vector<perceptron>>> network_3d;

		for (int k = 0; k < depth; k++)
		{
			vector<vector<perceptron>> network_2d;
			network_2d.push_back(make_input_layer(input_dim, row, eta, epoch, actFunction));

			//hidden_layer
			for (int j = 1; j < col - 1; j++)
			{
				network_2d.push_back(make_hidden_layer(row, eta, epoch, actFunction));
			}
			network_2d.push_back(make_output_layer(output_dim, row, eta, epoch, actFunction));

			network_3d.push_back(network_2d);
		}
		NN = move(network_3d);
	}

	vector<perceptron> neuron_network::make_input_layer(int input_dim, int row, double eta, int epoch, const function<double(double)>& actFunction)
	{
		vector<perceptron> input_layer;
		for (int i = 0; i < row; i++)
		{
			perceptron neuron(eta, epoch, actFunction);
			vector<double>& w = get_neuron_w(neuron);
			w.resize(input_dim, 0);
			randomVector(w, -0.5, 0.5);
			input_layer.push_back(neuron);
		}
		return input_layer;
	}

	vector<perceptron> neuron_network::make_hidden_layer(int row, double eta, int epoch, const function<double(double)>& actFunction)
	{
		vector<perceptron> hidden_layer;
		for (int i = 0; i < row; i++)
		{
			perceptron neuron(eta, epoch, actFunction);
			vector<double>& w = get_neuron_w(neuron);
			w.resize(row + 1, 0);
			randomVector(w, -0.5, 0.5);
			hidden_layer.push_back(neuron);
		}
		return  hidden_layer;
	}

	vector<perceptron> neuron_network::make_output_layer(int output_dim, int row, double eta, int epoch, const function<double(double)>& actFunction)
	{
		vector<perceptron> output_layer;
		for (int i = 0; i < output_dim; i++)
		{
			perceptron neuron(eta, epoch, actFunction);
			vector<double>& w = get_neuron_w(neuron);
			w.resize(row + 1, 0);
			randomVector(w, -0.5, 0.5);
			output_layer.push_back(neuron);
		}
		return output_layer;
	}

	vector<vector<double>> neuron_network::feed_forward_2d(vector<double> input_vector)
	{
		vector<vector<double>> output_each_layers;
		output_each_layers.push_back(input_vector);

		for (int i = 0; i < NN.size(); i++)
		{
			for (int j = 0; j < NN[i].size(); j++)
			{
				vector<double> score_output, actFn_output;
				if (j < NN[i].size() - 1)
				{
					score_output.push_back(1);
					actFn_output.push_back(1);
				}

				for (int neuron = 0; neuron < NN[i][j].size(); neuron++)
				{
					double score = dot(get_neuron_w(NN[i][j][neuron]), input_vector);
					score_output.push_back(score);
					actFn_output.push_back(actFn(score));
				}
				output_each_layers.push_back(score_output);
				input_vector = actFn_output;
			}
		}
		return output_each_layers;
	}

	void neuron_network::backpropagate_2d(vector<double>& input_vector, vector<double>& target, int step, int pretrain, double diff_h)
	{
		vector<vector<double>> outputs{ feed_forward_2d(input_vector) };
		vector<vector<double>> layers_deltas;
		vector<double> output_deltas;
		int num_output_layer = outputs.size() -1;
		int num_NN_layer = outputs.size() - 2;
		vector_length_queal(outputs[num_output_layer], target);
		//outlayer
		for (int i = 0; i < target.size(); i++)
		{
			double diff_actFn = (actFn(outputs[num_output_layer][i] + diff_h) - actFn(outputs[num_output_layer][i])) / diff_h;
			output_deltas.push_back(diff_actFn * (actFn(outputs[num_output_layer][i]) - target[i]));
		}
		layers_deltas.push_back(output_deltas);

		for (int i = 0; i < NN[0][num_NN_layer].size(); i++)
		{
			for (int j = 0; j < get_neuron_w(NN[0][num_NN_layer][i]).size(); j++)
			{
				double partial_gradient = learning_rate * output_deltas[i] * actFn(outputs[num_output_layer - 1][j]);
				if (step < pretrain || step > pretrain*(num_NN_layer + 1))
				{
					train_for_network(NN[0][num_NN_layer][i], j, -partial_gradient);
				}
			}
		}

		//hidden layer
		for (int num_layer = num_output_layer; num_layer > 1; num_layer--)
		{
			int NN_layer = num_layer - 1;
			vector<double> hidden_deltas;

			for (int i = 1; i < outputs[num_layer - 1].size(); i++)
			{
				double hidden_delta = 0;
				double diff_actFn = (actFn(outputs[num_layer - 1][i] + diff_h) - actFn(outputs[num_layer - 1][i])) / diff_h;
				for (int k = 0; k < layers_deltas[num_output_layer - num_layer].size(); k++)
				{
					hidden_delta += (diff_actFn * layers_deltas[num_output_layer - num_layer][k] * get_neuron_w(NN[0][NN_layer][k]).at(i));
				}
				hidden_deltas.push_back(hidden_delta);
			}
			layers_deltas.push_back(hidden_deltas);

			for (int i = 0; i < NN[0][NN_layer - 1].size(); i++)
			{
				for (int j = 0; j < get_neuron_w(NN[0][NN_layer - 1][i]).size(); j++)
				{
					double partial_gradient;
					if (num_layer - 2 > 0)
					{
						partial_gradient = learning_rate * hidden_deltas[i] * actFn(outputs[num_layer - 2][j]);
					}
					else
					{
						partial_gradient = learning_rate * hidden_deltas[i] * outputs[num_layer - 2][j];
					}
					if (step > pretrain*(num_NN_layer + 1) || ((num_NN_layer - NN_layer + 1)*pretrain > step && (num_NN_layer - NN_layer)*pretrain < step))
					{
						train_for_network(NN[0][NN_layer - 1][i], j, -partial_gradient);
					}
				}
			}
		}
	}

	void neuron_network::train(vector<vector<double>>& input, vector<vector<double>>& target, int pretrain, double precision, double diff_h)
	{
		vector<pair<vector<double>, vector<double>>> dataset;
		Statistics::makePair(input, target, dataset);
		auto rand_index_set = inRandomOrder(dataset);
		int count = 1, check_count = 1;

		for (int step = 0; step < n_iter; step++)
		{
			for (int index = 0; index < dataset.size(); index++)
			{
				backpropagate_2d(dataset[index].first, dataset[index].second, step, pretrain);
			}

			count++;
			check_count++;
			if (count % 200 == 0)
			{
				learning_rate *= 0.96;
				if (check_count % 1000 == 0)
				{
					Statistics::Rand_uniform_Int rand_gen(0, rand_index_set.size()-1);
					int index = rand_index_set[rand_gen()];

					vector<vector<double>> check_result = feed_forward_2d(dataset[index].first);
					vector<double> result = check_result.back();
					for (int i = 0; i < result.size(); i++)
					{
						result[i] = actFn(result[i]);
					}

					vector_subtract(result, dataset[index].second, result);
					if (sum_of_squares(result) < precision) { break; }
				}
			}
		}
	}

	void neuron_network::predict(vector<vector<double>>& validata_X, vector<vector<double>>& validate_Y)
	{
		for (int i = 0; i < validata_X.size(); i++)
		{
			vector<vector<double>> output_each_layers = feed_forward_2d(validata_X[i]);
			vector<double> P_predict;
			int last_layer = output_each_layers.size() - 1;

			cout << "The predict probability for " << i << "th data is\n";
			for (int j = 0; j < output_each_layers[last_layer].size(); j++)
			{
				P_predict.push_back(actFn(output_each_layers[last_layer][j]));
			}
			for (int j = 0; j < P_predict.size(); j++)
			{
				scalar_multiply(1 / vector_sum(P_predict), P_predict);
				if (P_predict[j] < 0.01) { P_predict[j] = 0; }
				cout << P_predict[j] << "  ";
			}
			cout << "\n";
			cout << "The true answer for " << i << "th data is\n";
			for (int j = 0; j < validate_Y[i].size(); j++)
			{
				cout << validate_Y[i][j] << "  ";
			}
			cout << "\n\n\n";
		}
	}
}