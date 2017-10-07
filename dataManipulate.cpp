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
#include "dataManipulate.h"

namespace dataManipulate
{
	int load_Data_With_Bias(string path, vector<vector<double>>& X, vector<double>& Y, const function<double(const string&)>& encoder, string cmd, int dim)
	{
		ifstream iData(path, ios::in);
		int count = 0;
		string line;
		getline(iData, line);
		vector<string> readString = string_partition(line, ',');

		if (dim == 0)
		{
			dim = readString.size() - 1;
		}

		init_vector(X, Y, encoder, cmd, dim, readString);

		while (getline(iData, line) && iData.peek() != EOF)
		{
			vector<string> readData = string_partition(line, ',');
			data2vector(readData, X, Y, encoder, cmd, 1, dim);
			count++;
		}
		iData.close();
		return count;
	}

	void data2vector(vector<string>& result, vector<vector<double>>& X, vector<double>& Y, const function<double(const string&)>& encoder, string cmd, int bias, int dim)
	{
		vector<double> Xi;
		int Xi_size = dim + bias;
		Xi.reserve(Xi_size);
		Xi.push_back(1.0);
		X.push_back(Xi);
		int X_size = X.size();

		for (int i = 0; i < dim; i++)
		{
			istringstream is(result[i]);
			double val;
			is >> val;
			X[X_size-1].push_back(val);
		}

		to_lower(cmd);
		if (cmd != "test" )
		{
			Y.push_back(encoder(result[result.size() - 1]));
		}
	}

	void init_vector(vector<vector<double>>& X, vector<double>& Y, const function<double(const string&)>& encoder, string cmd, int dim, vector<string> result)
	{
		X[0].push_back(1.0);
		for (int i = 0; i < dim; i++)
		{
			istringstream is(result[i]);
			double val;
			string label;
			if (!isalpha(is.peek()))
			{
				is >> val;
				X[0].push_back(val);
			}
			else
			{
				is >> label;
				X[0].push_back(encoder(label));
			}
		}
		to_lower(cmd);
		if (cmd != "test")
		{
			Y.push_back(encoder(result[result.size() - 1]));
		}
	}

	void init_NoBias_vector(ifstream& iData, vector<vector<double>>& X, vector<vector<double>>& Y, string cmd, int input_dim, int output_dim)
	{
		vector<double> readData;
		readData_for_NN(iData, readData, input_dim);
		X[0] = readData;

		to_lower(cmd);
		if (cmd != "test")
		{
			vector<double> outData;
			readData_for_NN(iData, outData, output_dim);
			Y[0] = outData;
		}
	}

	int load_Data_NoBias_NN(string path, vector<vector<double>>& X, vector<vector<double>>& Y, const function<double(const string&)>& encoder, string cmd, int input_dim, int output_dim)
	{
		ifstream iData(path, ios::in);
		int count = 0;
		string line;
		
		init_NoBias_vector(iData, X, Y, cmd, input_dim, output_dim);
		while (iData.peek() != EOF)
		{
			vector<double> readData;
			readData_for_NN(iData, readData, input_dim);
			X.push_back(readData);

			to_lower(cmd);
			if (cmd != "test")
			{
				vector<double> outData;
				readData_for_NN(iData, outData, output_dim);
				Y.push_back(outData);
			}
			count++;
		}
		iData.close();
		return count;
	}

	void readData_for_NN(ifstream& iData, vector<double>& readData, int dim)
	{
		string line;
		while (readData.size() < dim && getline(iData, line))
		{
			vector<string> tempData = string_partition(line, ',');
			for (int i = 0; i < tempData.size(); i++)
			{
				istringstream is(tempData[i]);
				double val;
				if (!isalpha(is.peek()))
				{
					is >> val;
					readData.push_back(val);
				}
				else
				{
					cerr << "Not a number!";
				}
			}
		}
	}

	vector<string> string_partition(const string &source, char delim)
	{
		vector<string> result;
		stringstream ss;
		ss.str(source);
		string item;

		while (getline(ss, item, delim)) 
		{
			result.push_back(item);
		}
		return result;
	}

	void split_data(vector<pair<map<string, string>, string>>& data, vector<pair<map<string, string>, string>>& train, vector<pair<map<string, string>, string>>& test, double trainSize)
	{
		Statistics::Rand_uniform_double ranDouble(0, 1);
		for (int i = 0; i < data.size(); i++)
		{
			if (ranDouble() < trainSize)
			{
				train.push_back(data[i]);
			}
			else
			{
				test.push_back(data[i]);
			}
		}
	}

	void split_data(vector< pair<vector<double>, double> >& data, vector<pair<vector<double>, double>>& train, vector<pair<vector<double>, double>>& test, double trainSize)
	{
		Statistics::Rand_uniform_double ranDouble(0, 1);
		for (int i = 0; i < data.size(); i++)
		{
			if (ranDouble() < trainSize)
			{
				train.push_back(data[i]);
			}
			else
			{
				test.push_back(data[i]);
			}
		}
	}

	void train_test_split(vector<vector<double> >& X, vector<double>& Y, vector<vector<double> >& X_train, vector<double>& Y_train, vector<vector<double> >& X_test, vector<double>& Y_test, double trainSize)
	{
		Linear_Algebra::vector_length_queal(Y, X);
		vector<pair<vector<double>, double>> X_y_combine, train, test;
		double safty_factor = 1.2;

		X_train.reserve(X.size() * safty_factor * trainSize), X_test.reserve(X.size() * (safty_factor - trainSize));
		Y_train.reserve(Y.size() * safty_factor * trainSize), Y_test.reserve(Y.size() * (safty_factor - trainSize));
		
		for (int i = 0; i < X.size(); i++)
		{
			X_y_combine.push_back(move(pair<vector<double>, double> {X[i], Y[i]}));
		}

		split_data(X_y_combine, train, test, trainSize);
			
		Statistics::unPair(train, X_train, Y_train);
		Statistics::unPair(test, X_test, Y_test);
	}

	void train_test_split(vector<map<string,string>>& X, vector<string>& Y, vector<map<string, string>>& X_train, vector<string>& Y_train, vector<map<string, string>>& X_test, vector<string>& Y_test, double trainSize)
	{
		Linear_Algebra::vector_length_queal(X, Y);
		vector<pair<map<string, string>, string>> X_y_combine, train, test;
		double safty_factor = 1.2;

		X_train.reserve(X.size() * safty_factor * trainSize), X_test.reserve(X.size() * (safty_factor - trainSize));
		Y_train.reserve(Y.size() * safty_factor * trainSize), Y_test.reserve(Y.size() * (safty_factor - trainSize));

		for (int i = 0; i < X.size(); i++)
		{
			X_y_combine.push_back(move(pair<map<string, string>, string> {X[i], Y[i]}));
		}

		split_data(X_y_combine, train, test, trainSize);

		Statistics::unPair(train, X_train, Y_train);
		Statistics::unPair(test, X_test, Y_test);
	}

	template<typename T>
	vector<T> bootstrap_Xi(const vector<T>& data)
	{
		Statistics::Rand_uniform_Int ranInt(0, data.size()-1);
		vector<T> bootstrap_data;

		for (int i = 0; i < data.size(); i++)
		{
			bootstrap_data.push_back(data[ranInt()]);
		}
		return bootstrap_data;
	}

	template<typename T, typename U, typename V>
	vector<U> bootstrap_statisticXi(vector<T>& data, int num_bootstrap, function<V(T)>stats_fn)
	{
		vector<U> bootstrap_statistic_result;
		for (int i = 0; i < num_bootstrap; i++)
		{
			vector<T> boostrap = bootstrap_Xi(data);
			bootstrap_statistic_result.push_back(stats_fn(boostrap));
		}
		return bootstrap_statistic_result;
	}

	vector<pair<vector<double>, double>> bootstrap_sample(vector<vector<double>>& X, vector<double>& Y)
	{
		vector<pair<vector<double>, double>> bootstrap_data;
		Statistics::Rand_uniform_Int ranInt(0, X.size() -1);

		for (int i = 0; i < X.size(); i++)
		{
			int randNum = ranInt();
			bootstrap_data.push_back(pair<vector<double>, double>{X[i], Y[i]});
		}
		return bootstrap_data;
	}

	void to_lower(string word)
	{
		transform(word.begin(), word.end(), word.begin(), tolower);
	}

	int readData_for_tree(string path, vector<map<string, string>>& X, vector<string>& Y, string cmd)
	{
		ifstream iData(path, ios::in);
		int count = 0;
		string line;

		while (getline(iData, line) && iData.peek() != EOF)
		{
			if (line == "") { continue; }
			map<string, string> X_dict;
			vector<string> readData = string_partition(line, ',');
			to_lower(line);

			for (int i = 0; i < readData.size(); i++)
			{
				vector<string> X_data = string_partition(readData[i], ':');
				X_dict[X_data[0]] = X_data[1];
			}
			to_lower(cmd);

			if (cmd == "train")
			{
				getline(iData, line);
				to_lower(line);
				Y.push_back(line);
			}
			X.push_back(X_dict);
			count++;
		}
		iData.close();
		return count;
	}

	void readParagraph(string path, string& paragraph)
	{
		ifstream iData(path, ios::in);
		string line;
		paragraph = "";
		while (iData.peek() != EOF)
		{
			getline(iData, line);
			paragraph += (line + "\n");
		}
	}
}