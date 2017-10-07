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
#include "Linear_Algebra.h"

//線代工具
namespace Linear_Algebra
{
	template<typename T, typename U>
	void vector_length_queal(const vector<T>& v, const vector<U> & w)
	{
		assert(v.size() == w.size());
	}

	template<typename T, typename U>
	void vector_length_queal(vector<T>& v, vector<U> & w)
	{
		assert(v.size() == w.size());
	}

	void vector_length_queal(vector<vector<double>>& v, vector<vector<double>> & w)
	{
		assert(v.size() == w.size());
	}

	void vector_length_queal(vector<double>& v, vector<double> & w)
	{
		assert(v.size() == w.size());
	}

	void vector_length_queal(vector<map<string, string>>& v, vector<string>& w)
	{
		assert(v.size() == w.size());
	}

	void vector_length_queal(const vector<string>& v, const vector<bool> & w)
	{
		assert(v.size() == w.size());
	}

	void vector_length_security(const vector<double>& v, const vector<double>& w)
	{
		assert(v.capacity() >= w.size());
	}

	void vector_subtract(const vector<double>& v, const vector<double>& w, vector<double>& result)
	{
		vector_length_queal(v, w);
		vector_length_security(result, w);

		for (int i = 0; i < v.size(); i++)
		{
			result[i] = v[i] - w[i];
		}
	}

	void vector_add(const vector<double>& v, const vector<double>& w, vector<double>& result)
	{
		vector_length_queal(v, w);
		vector_length_security(result, w);

		for (int i = 0; i < v.size(); i++)
		{
			result[i] += v[i] + w[i];
		}
	}

	void vectors_sum(const vector<vector<double>>& vectors, vector<double>& result)
	{
		size_t vsnum = vectors.size(), vnum = vectors[0].size();
		vector_length_security(result, vectors[0]);

		if (vsnum % 2 == 1)
		{
			vsnum -= 1;
			vector_add(vectors[vsnum], vector<double>(vnum, 0), result);
		}

		for (int i = 0; i < vsnum; i += 2)
		{
			vector_length_queal(vectors[i], vectors[i + 1]);
			vector_add(vectors[i], vectors[i + 1], result);
		}
	}

	double vector_sum(const vector<double>& vec)
	{
		double sum = 0;
		for (int i = 0; i < vec.size(); i++)
		{
			sum += vec[i];
		}
		return sum;
	}

	void scalar_multiply(double c, vector<double>& v)
	{
		for (int i = 0; i < v.size(); i++)
		{
			v[i] *= c;
		}
	}

	void vector_mean(const vector<vector<double>>& vectors, vector<double>& result)
	{
		size_t vsnum = vectors.size();
		vector_length_security(result, vectors[0]);
		
		vectors_sum(vectors, result);
		scalar_multiply(1.0/vsnum, result);
	}

	double dot(const vector<double>& v, const vector<double>& w)
	{
		vector_length_queal(v, w);
		double sum = 0;

		for (int i = 0; i < v.size(); i++)
		{
			sum += v[i] * w[i];
		}

		return sum;
	}

	int dot(const vector<int>& v, const vector<int>& w)
	{
		vector_length_queal(v, w);
		int sum = 0;

		for (int i = 0; i < v.size(); i++)
		{
			sum += v[i] * w[i];
		}

		return sum;
	}

	double sum_of_squares(const vector<double>& v)
	{
		return dot(v, v);
	}

	double magnitude(const vector<double>& v)
	{
		return sqrt(sum_of_squares(v));
	}

	double squared_distance(const vector<double>& v, const vector<double>& w)
	{
		vector<double> result;
		result.resize(v.size(), 0);
		vector_subtract(v, w, result);
		return sum_of_squares(result);
	}

	double distance(const vector<double>& v, const vector<double>& w)
	{
		return sqrt(squared_distance(v, w));
	}
	
	double difference_quotient(function<double(double)> f, const double x, double h)
	{
		return (f(x + h) - f(x)) / h;
	}

	double partial_difference_quotient(function<double(vector<double>&)> f, vector<double> w, int i, double h)
	{
		vector<double> v = w;
		v[i] += h;
		return (f(v) - f(w)) / h;
	}

	void estimate_gradient(function<double(vector<double>&)> f, vector<double> v, vector<double>& gradient, double h)
	{
		vector<double> result(v.size(), 0);

		for (int i = 0; i < v.size(); i++)
		{
			result[i] = partial_difference_quotient(f, v, i, h);
		}
		gradient = result;
	}

	vector<double> step(vector<double> v, const vector<double>& direction, double step_size)
	{
		vector_length_queal(v, direction);

		for (int i = 0; i < direction.size(); i++)
		{
			v[i] = v[i] + step_size * direction[i];
		}
		return v;
	}

	vector<double> minimize_batch(function<double(vector<double>&)> target_f, const vector<double>& w_0, double tolerance)
	{
		const vector<double> step_size({1000, 100, 10, 1, 0.1, 0.01, 0.001, 0.0001, 0.00001, 0.1 * tolerance }) ;
		const int stepSize = step_size.size();
		vector<double> gradient;
		vector<double> w = w_0;
		double result{ target_f(w) };

		while (true)
		{
			estimate_gradient(target_f, w, gradient);
			vector<double> nextPossibleValue(stepSize, 0);
			vector<vector<double>> next_Possible_w(stepSize);

			for (int i = 0; i < step_size.size(); i++)
			{
				vector<double> temp_w;
				temp_w = step(w, gradient, -step_size[i]);

				nextPossibleValue[i] = target_f(temp_w);
				next_Possible_w.push_back(temp_w);
			}
			
			pair<int, double> next_result{ Statistics::minValue(nextPossibleValue) };
			if (abs(result - next_result.second) < tolerance)
			{
				return next_Possible_w[next_result.first];
			}
			w = next_Possible_w[next_result.first];
			result = nextPossibleValue[next_result.first];
		}
	}

	vector<double> maximize_batch(function<double(vector<double>&)> target_f, const vector<double>& w_0, double tolerance)
	{
		const vector<double> step_size({ 1000, 100, 10, 1, 0.1, 0.01, 0.001, 0.0001, 0.00001, 0.1 * tolerance });
		const int stepSize = step_size.size();
		vector<double> gradient;
		vector<double> w = w_0;
		double result{ target_f(w) };

		while (true)
		{
			estimate_gradient(target_f, w, gradient);
			vector<double> nextPossibleValue(stepSize, 0);
			vector<vector<double>> next_Possible_w(stepSize);

			for (int i = 0; i < step_size.size(); i++)
			{
				vector<double> temp_w;
				temp_w = step(w, gradient, step_size[i]);

				nextPossibleValue[i] = target_f(temp_w);
				next_Possible_w.push_back(temp_w);
			}

			pair<int, double> next_result{ Statistics::maxValue(nextPossibleValue) };
			if (abs(result - next_result.second) < tolerance)
			{
				return next_Possible_w[next_result.first];
			}
			w = next_Possible_w[next_result.first];
			result = nextPossibleValue[next_result.first];
		}
	}

	template<typename T>
	vector<int> inRandomOrder(const vector<T>& data)
	{
		vector<int> indexes(data.size());
		for (int i = 0; i < data.size(); i++)
		{
			indexes.push_back(i);
		}
		unsigned seed = (unsigned)time(NULL);
		shuffle(indexes.begin(), indexes.end(), std::default_random_engine(seed));
		return indexes;
	}

	vector<int> inRandomOrder(const vector<pair<vector<double>, vector<double>> >& data)
	{
		vector<int> indexes;
		for (int i = 0; i < data.size(); i++)
		{
			indexes.push_back(i);
		}
		unsigned seed = (unsigned)time(NULL);
		shuffle(indexes.begin(), indexes.end(), std::default_random_engine(seed));
		return indexes;
	}

	//T==vector<double>   U==double
	vector<double> minimize_stochastic(function<double(vector<vector<double>>&, vector<double>&, vector<double>&)> target_f, vector<double>& w_0, vector<vector<double>>& x, vector<double>& y, double eta_0, int miniBatch, int miniBatchFactor)
	{
		vector<pair<vector<double>, double>> data;
		vector<double> w = w_0, min_w;
		double eta = eta_0, min_value = numeric_limits<double>::max();
		int iterations_with_no_improvement = 0;
		Statistics::Rand_uniform_Int randomInt(0, data.size() -1);
		
		for (int i = 0; i < x.size(); i++)
		{
			data.push_back(move(make_pair(x[i], y[i])));
		}

		while (iterations_with_no_improvement < 120)
		{
			double value = 0;
			for (int i = 0; i < data.size(); i++)
			{
				value += target_f(x, y, w_0);
			}

			if (value < min_value)
			{
				min_w = w;
				min_value = value;
				iterations_with_no_improvement = 0;
				eta = eta_0;
			}
			else
			{
				iterations_with_no_improvement += 1;
				eta *= 0.9;
			}

			for (int i = 0; i < miniBatchFactor*data.size()/miniBatch ; i++)
			{
				vector<int> indexes{ inRandomOrder(data) };
				vector<vector<double>> X_i;
				vector<double> y_i;

				for (int i = 0; i < miniBatch; i++)
				{
					pair<vector<double>, double> randata{data[indexes[randomInt()]]};
					X_i.push_back(randata.first);
					y_i.push_back(randata.second);
				}

				vector<double> gradient_i;
				estimate_gradient(target_f, w, gradient_i, X_i, y_i);
				scalar_multiply(-eta, gradient_i);
				vector_subtract(w, gradient_i, w);
			}
		}
		return min_w;
	}

	template<typename T, typename U>
	double partial_difference_quotient(function<double(vector<T>&, vector<U>&, T&)> target_f, vector<double>& v, vector<T>& X, vector<U>& Y, int i, double h)
	{
		vector<double> w = v;
		w[i] += h;
		return (target_f(X, Y, w) - target_f(X, Y ,v)) / h;
	}

	template<typename T, typename U>
	void estimate_gradient(function<double(vector<T>&, vector<U>&, T&)> target_f, T& v, T& gradient, vector<T>& X, vector<U>& Y, double h)
	{
		vector<double> result(v.size(), 0);

		for (int i = 0; i < v.size(); i++)
		{
			result[i] = partial_difference_quotient(target_f, v, X, Y, i, h);
		}

		gradient = result;
	}

	template<typename T, typename U>
	double sum_square_ErrFunction(vector<T> X, vector<U> Y, T w)
	{
		double sum = 0;
		for (int i = 0; i < X.size(); i++)
		{
			for (int j = 0; j < Y.size; j++)
			{
				sum += pow((Y[i] - dot(w, x[i])), 2);
			}
		}
		return sum;
	}

	vector<double> maximize_stochastic(function<double(vector<vector<double>>&, vector<double>&, vector<double>&)> target_f, vector<double>& w_0, vector<vector<double>>& x, vector<double>& y, double eta_0 , int miniBatch, int miniBatchFactor)
	{
		vector<pair<vector<double>, double>> data;
		vector<double> w = w_0, max_w;
		double eta = eta_0, max_value = numeric_limits<double>::min();
		int iterations_with_no_improvement = 0;
		Statistics::Rand_uniform_Int randomInt(0, data.size() -1);

		for (int i = 0; i < x.size(); i++)
		{
			data.push_back(move(make_pair(x[i], y[i])));
		}

		while (iterations_with_no_improvement < 120)
		{
			double value = 0;
			for (int i = 0; i < data.size(); i++)
			{
				value += target_f(x, y, w_0);
			}

			if (value > max_value)
			{
				max_w = w;
				max_value = value;
				iterations_with_no_improvement = 0;
				eta = eta_0;
			}
			else
			{
				iterations_with_no_improvement += 1;
				eta *= 0.9;
			}

			for (int i = 0; i < data.size(); i++)
			{
				vector<int> indexes{ inRandomOrder(data) };
				vector<vector<double>> X_i;
				vector<double> y_i;

				for (int i = 0; i < miniBatch; i++)
				{
					int randnum = indexes[randomInt()];
					X_i.push_back(data[randnum].first);
					y_i.push_back(data[randnum].second);
				}

				vector<double> gradient_i;
				estimate_gradient(target_f, w, gradient_i, X_i, y_i);
				scalar_multiply(eta, gradient_i);
				vector_add(w, gradient_i, w);
			}
		}
		return max_w;
	}

	vector<double> direction(vector<double> v)
	{
		double mag = magnitude(v);

		for (int i = 0; i < v.size(); i++)
		{
			v[i] /= mag;
		}
		return v;
	}

	double directional_variance_i(const vector<double>& x_i, const vector<double>& w)
	{
		return pow(dot(x_i, direction(w)), 2);
	}

	double directional_variance(const vector<vector<double>>& X, const vector<double>& w)
	{
		double sum = 0;
		for (int i = 0; i < X.size(); i++)
		{
			sum += directional_variance_i(X[i], w);
		}
		return sum;
	}

	void randomVector(vector<double>& w, double lo, double hi)
	{
		Statistics::Rand_uniform_double Random(lo, hi);
		for (int i = 0; i < w.size(); i++)
		{
			w[i] = Random();
		}
	}

	vector<double> first_principle_component(vector<vector<double>>& X)
	{
		const vector<double> guessVec(X.at(0).size(), 0);
		vector<double> unscale_w;
		unscale_w = maximize_batch(
			[&](vector<double>& w)
			{ return directional_variance(X, w); }, guessVec);

		return direction(unscale_w);
	}

	vector<double> first_principle_component_sgd(vector<vector<double>>& X)
	{
		vector<double> guessVec(X.at(0).size(), 0);
		vector<double> useless(X.at(0).size(), 0);
		vector<double> unscale_w;
		unscale_w = maximize_stochastic(
					[](vector<vector<double>>& X, vector<double>& useless, vector<double>& w)
					{ return directional_variance(X, w); }, guessVec, X, useless, 0.1, 10, 5);
		
		return direction(unscale_w);
	}

	vector<double> project(vector<double> v, vector<double> w)
	{
		double projection_length = dot(v, w);
		scalar_multiply(projection_length, w);
		return w;
	}

	vector<double> remove_projection_from_vector(vector<double> v, vector<double> w)
	{
		vector_subtract(v, project(v, w), v);
		return v;
	}

	void remove_projection(vector<vector<double>>& X, vector<double> w)
	{
		for (int i = 0; i < X.size(); i++)
		{
			remove_projection_from_vector(X[i], w);
		}
	}

	vector<vector<double>> principal_component_analysis(vector<vector<double>>& X, int num_components)
	{
		vector<vector<double>> components;
		for (int i = 0; i < num_components; i++)
		{
			components.push_back(first_principle_component(X));
			remove_projection(X, components[i]);
		}
		return components;
	}

	vector<double> transform_vector(vector<double> v, vector<vector<double>> components)
	{
		vector<double> transformVec;
		for (int i = 0; i < components.size(); i++)
		{
			transformVec.push_back(dot(v, components[i]));
		}
		return transformVec;
	}

	vector<vector<double>> trnsform_X(vector<vector<double>> X, vector<vector<double>> components)
	{
		vector<vector<double>> trnsform_Matrix;

		for (int i = 0; i < X.size(); i++)
		{
			trnsform_Matrix.push_back(transform_vector(X[i], components));
		}
		return trnsform_Matrix;
	}

	void make_Matrix(vector<vector<double>>& matrix, int row, int col)
	{
		vector<double> Xi;
		Xi.reserve(col);
		matrix.push_back(Xi);
		matrix.reserve(row);
	}

	vector<vector<double>> transpose(vector<vector<double>> &X)
	{
		vector<vector<double>> Xt;
		for (int j = 0; j < X[0].size(); j++)
		{
			vector<double> transpose;
			for (int i = 0; i < X.size(); i++)
			{
				transpose.push_back(X[i][j]);
			}
			Xt.push_back(transpose);
		}
		return Xt;
	}

	double linear_equation(vector<double>& w, vector<double>& X)
	{
		return dot(w, X);
	}

	double error_for_linear_regression(vector<double>& w, vector<double>& Xi, double& yi)
	{
		return yi - linear_equation(w ,Xi);
	}

	double sum_of_linear_squared_errors(vector<vector<double>>& X, vector<double>& Y, vector<double>& w)
	{
		double sum = 0;
		vector_length_queal(Y, X);
		for (int i = 0; i < X.size(); i++)
		{
			sum += pow(error_for_linear_regression(w, X[i], Y[i]), 2);
		}
		return sum;
	}

	double total_sum_of_squares(vector<double>& Y)
	{
		double sum = 0;
		Statistics::deMean(Y);
		for (int i = 0; i < Y.size(); i++)
		{
			sum += Y[i] * Y[i];
		}
		return sum;
	}

	double R_square(vector<double>& w, vector<vector<double>>& X, vector<double>& Y)
	{
		return (1.0 - (sum_of_linear_squared_errors(X, Y, w)) / total_sum_of_squares(Y));
	}

	vector<double> linear_regression(vector<vector<double>>& X, vector<double>& Y, vector<double>& w)
	{
		w.resize(X[0].size());
		randomVector(w);
		return minimize_stochastic(sum_of_linear_squared_errors, w, X, Y);
	}
}