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
#include "Statistics.h"

//統計模型
using namespace Linear_Algebra;

namespace Statistics
{
	template<typename T>
	vector<T> unique_labels(vector<T> labels)
	{
		for (int i = 0; i < labels.size(); i++)
		{
			labels /= abs(labels);
		}
		sort(labels.begin(), labels.end());
		unique(labels.begin(), labels.end());
		return labels;
	}

	template<typename T>
	pair<T, bool> most_frequent_in_group(vector<T> labels)
	{
		vector<T> clone_group = labels;
		vector<T> unique_group = unique_group(labels);
		vector<int> count_group;

		for (int i = 0; i < unique_group.size(), i++)
		{
			int frequency = count(clone_group.begin(), clone_group.end(), unique_group[i]);
			count_group.push_back(frequency);
		}
		pair<int, double> find_most_frequent = maxValue(count_group);
		sort(count_group.begin(), count_group.end());
		if (count_group[0] != count_group[1])
		{
			pair<T, bool> result{ unique_group[find_most_frequent.first], true };
		}
		else
		{
			pair<T, bool> result{ unique_group[find_most_frequent.first], false };
		}
		return result;
	}

	template<typename T>
	pair<int, T> maxValue(const vector<T>& v)
	{
		T Max = v[0];
		int index = 0;

		for (int i = 1; i < v.size(); i++)
		{
			if (Max < v[i])
			{
				Max = v[i];
				index = i;
			}
		}
		return makePair<int, T>(index, Max);
	}

	pair<int, int> maxValue(const vector<int>& v)
	{
		int Max = v[0];
		int index = 0;

		for (int i = 1; i < v.size(); i++)
		{
			if (Max < v[i])
			{
				Max = v[i];
				index = i;
			}
		}
		return makePair<int, int>(index, Max);
	}

	pair<int, double> maxValue(vector<double>& v)
	{
		double Max = v[0];
		int index = 0;

		for (int i = 1; i < v.size(); i++)
		{
			if (Max < v[i])
			{
				Max = v[i];
				index = i;
			}
		}
		return makePair<int, double>(index, Max);
	}

	template<typename T>
	pair<int,T> minValue(const vector<T>& v)
	{
		T Min = v[0];
		int index = 0;

		for (int i = 1; i < v.size(); i++)
		{
			if (Min > v[i])
			{
				Min = v[i];
				index = i;
			}
		}
		return makePair<int, T>(index, Min);
	}

	pair<int, double> minValue(vector<double>& v)
	{
		double Min = v[0];
		int index = 0;

		for (int i = 1; i < v.size(); i++)
		{
			if (Min > v[i])
			{
				Min = v[i];
				index = i;
			}
		}
		return makePair<int, double>(index, Min);
	}

	pair<int, int> minValue(vector<int>& v)
	{
		int Min = v[0];
		int index = 0;

		for (int i = 1; i < v.size(); i++)
		{
			if (Min > v[i])
			{
				Min = v[i];
				index = i;
			}
		}
		return makePair<int, int>(index, Min);
	}

	double sum(const vector<double>& v)
	{
		double sum = v[0];

		for (int i = 1; i < v.size(); i++)
		{
			sum += v[i];
		}
		return sum;
	}

	double mean(const vector<double>& v)
	{
		return sum(v) / v.size();
	}

	double median(vector<double> v)
	{
		size_t vsize = v.size();

		size_t midpoint = vsize / 2;

		sort(v.begin(), v.end());
		if (vsize % 2 == 1)
		{
			return v[midpoint];
		}
		else
		{
			size_t lo = midpoint - 1;
			size_t hi = midpoint;
			return (v[lo] + v[hi]) / 2;
		}
	}

	double quantile(vector<double> v, float p)
	{
		int p_index = floor(p*v.size());
		sort(v.begin(), v.end());

		return v[p_index];
	}

	vector<double> mode(vector<double> v)
	{
		assert(v.size() > 0);
		double maxval = v[0], nextval = v[0], mode_candidate = v[0];
		int maxcount = 1, nextcount = 0, candidate_count = 1;
		vector<double> result;
		sort(v.begin(), v.end());

		for (int i = 1; i < v.size(); i++)
		{
			if (maxval == v[i])
			{
				maxcount++;
			}
			else if (nextval != v[i])
			{
				nextval = v[i];
				nextcount = 1;
			}
			else
			{
				nextcount++;
				if (nextcount > maxcount)
				{
					maxval = v[i];
					maxcount = nextcount;
				}
			}
		}
		for (int i = 1; i < v.size(); i++)
		{
			if (mode_candidate == v[i])
			{
				candidate_count++;
				if (candidate_count == maxcount && mode_candidate != maxval)
				{
					result.push_back(mode_candidate);
				}
			}
			else
			{
				mode_candidate = v[i];
				candidate_count = 1;
			}
		}
		return result;
	}

	template<typename T, typename U>
	void makePair(vector<T>& v, vector<U>& w, vector<pair<T, U> > & result)
	{
		vector_length_queal(v, w);
		for (int i = 0; i < v.size(); i++)
		{
			result.push_back(pair<T, U>{v[i], w[i]});
		}
	}

	void makePair(vector<vector<double>>& v, vector<vector<double>>& w, vector<pair<vector<double>, vector<double>>> & result)
	{
		vector_length_queal(v, w);
		for (int i = 0; i < v.size(); i++)
		{
			result.push_back(pair<vector<double>, vector<double>>{v[i], w[i]});
		}
	}

	void makePair(vector<map<string, string>>& v, vector<string>& w, vector<pair<map<string, string>, string>> & result)
	{
		vector_length_queal(v, w);
		for (int i = 0; i < v.size(); i++)
		{
			result.push_back(pair<map<string, string>, string>{v[i], w[i]});
		}
	}
	

	template<typename T, typename U>
	pair<T, U> makePair(T& v, U& w)
	{
		return pair<T, U>{v, w};
	}

	template<typename T, typename U>
	void unPair(vector<pair<T, U>>& source, vector<T>& v, vector<U>& w)
	{
		for (int i = 0; i < source.size(); i++)
		{
			v.push_back(source[i].first);
			w.push_back(source[i].second);
		}
	}

	void unPair(vector<pair<string, bool>>& source, vector<string>& v, vector<bool>& w)
	{
		for (int i = 0; i < source.size(); i++)
		{
			v.push_back(source[i].first);
			w.push_back(source[i].second);
		}
	}

	void unPair(vector<pair<map<string, string>, string>>& source, vector<map<string, string>>& v, vector<string>& w)
	{
		for (int i = 0; i < source.size(); i++)
		{
			v.push_back(source[i].first);
			w.push_back(source[i].second);
		}
	}

	template<typename T, typename U>
	void unPair(pair<T, U>& source, T& v, U& w)
	{
		v = source.first;
		w = source.second;
	}

	double dataRange(vector<double>& v)
	{
		return (maxValue(v).second - minValue(v).second);
	}

	void deMean(vector<double>& v)
	{
		double v_mean = mean(v);
		for (int i = 0; i < v.size(); i++)
		{
			v[i] -= v_mean;
		}
	}

	double variance(vector<double>& v)
	{
		deMean(v);
		return sum_of_squares(v) / (v.size() -1 );
	}

	double standard_deviation(vector<double>& v)
	{
		return sqrt(variance(v));
	}

	double interquartile_range(vector<double>& v)
	{
		return quantile(v, 0.75) - quantile (v, 0.25);
	}

	double covariance(vector<double>& v, vector<double>& w)
	{
		vector_length_queal(v, w);
		deMean(v);
		deMean(w);
		return dot(v, w) / (v.size() - 1);
	}

	double correlation(vector<double>& v, vector<double>& w)
	{
		double sigma_v = standard_deviation(v);
		double sigma_w = standard_deviation(w);
		if (sigma_v > 0 && sigma_w > 0)
		{
			return covariance(v, w) / (sigma_v * sigma_w);
		}
		else
		{
			return 0;
		}
	}

	void deOutlier(vector<double>& v)
	{
		double midVal = median(v);
		double IQR = interquartile_range(v);
		double sigma_v = standard_deviation(v);
		double hi_Filter, lo_Filter;
		queue<double> temp;

		if (3*IQR > 6*sigma_v)
		{
			hi_Filter = midVal + 1.5 * IQR;
			lo_Filter = midVal - 1.5 * IQR;
		}
		else
		{
			hi_Filter = midVal + 3 * sigma_v;
			lo_Filter = midVal - 3 * sigma_v;
		}
		
		for (int i = 0; i < v.size(); i++)
		{
			if (v[i] > lo_Filter && v[i] < hi_Filter)
			{
				temp.push(v[i]);
			}
		}
		v.clear();
		for (int i = 0; i < temp.size(); i++)
		{
			v[i] = temp.front();
			temp.pop();
		}
	}

	void makeVector(vector<double>& v, vector<vector<double>>& result)
	{
		vector_length_queal(v, result);

		for (int i = 0; i < v.size(); i++)
		{
			result[i].push_back(v[i]);
		}
	}

	int uniform_pdf(double x)
	{
		if (x >= 0 && x < 1)
		{
			return 1;
		}
		else
		{
			return 0;
		}
	}

	double uniform_cdf(double x)
	{
		if (x < 0)
		{
			return 0;
		}
		else if (x < 1)
		{
			return x;
		}
		else
		{
			return 1;
		}
	}

	double normal_pdf(double x, double mu, double sigma)
	{
		double sqrt_two_pi = sqrt(2 * 3.1415926535897);
		double sigma_square = pow(sigma, 2);
		double deMean_square = pow((x - mu), 2);
		double constant = sqrt_two_pi * sigma;
		
		return exp(-deMean_square) / (2 * sigma_square * constant);
	}

	double normal_cdf(double x, double mu, double sigma)
	{
		double deMean = (x - mu);
		double mod_sigma = sqrt(2) * sigma;

		return ( 1 + erf( deMean / mod_sigma ) ) / 2;
	}

	double inverse_normal_cdf(double p, double mu, double sigma, double tolerance)
	{
		if (mu != 0 || sigma != 1)
		{
			return mu + sigma * inverse_normal_cdf(p, tolerance = tolerance);
		}

		double low_val = -10, low_p = 0;
		double high_val = 10, high_p = 1;
		double mid_val, mid_p;

		while ((high_val - low_val) > tolerance)
		{
			mid_val = (low_val + high_val) / 2;
			mid_p = normal_cdf(mid_val);

			if (mid_p < p)
			{
				low_val = mid_val;
				low_p = mid_p;
			}
			else if (mid_p > p)
			{
				high_val = mid_val;
				high_p = mid_p;
			}
			else
			{
				break;
			}
		}
		return mid_val;
	}

	int bernoulli_trail(double p)
	{
		Rand_normal_double random(0, 1);
		if (random() < p)
		{
			return 1;
		}
		else
		{
			return 0;
		}
	}

	int binomial(int n, double p)
	{
		int sum = 0;
		for (int i = 0; i < n; i++)
		{
			sum += bernoulli_trail(p);
		}
		return sum;
	}

	double normal_probability_above(double lo, double mu, double sigma)
	{
		return 1 - normal_cdf(lo, mu, sigma);
	}

	double normal_probability_between(double lo, double hi, double mu, double sigma)
	{
		return normal_cdf(hi, mu, sigma) - normal_cdf(lo, mu, sigma);
	}

	double normal_probability_outside(double lo, double hi, double mu, double sigma)
	{
		return 1 - normal_probability_between(lo, hi, mu, sigma);
	}

	double normal_upper_bound(double p, double mu, double sigma)
	{
		return inverse_normal_cdf(p, mu, sigma);
	}

	double normal_lower_bound(double p, double mu, double sigma)
	{
		return inverse_normal_cdf(1 - p, mu, sigma);
	}

	pair<double, double> normal_two_sided_bounds(double p, double mu, double sigma)
	{
		double tail_probability = (1 - p) / 2;
		double upper_bound = normal_lower_bound(tail_probability, mu, sigma);
		double lower_bound = normal_upper_bound(tail_probability, mu, sigma);
		pair<double, double> lo_hi_bound(lower_bound,upper_bound);

		return pair<double, double> {lower_bound, upper_bound};
	}

	double two_side_p_value(double x, double mu, double sigma)
	{
		if (x >= mu)
		{
			return normal_probability_above(x, mu, sigma);
		}
		else
		{
			return normal_cdf(x, mu, sigma);
		}
	}

	pair<double, double> estimate_p_sigma(int total, int occur)
	{
		double p = occur / total;
		double sigma = sqrt(p * (1 - p) / total);

		return pair<double, double>{p, sigma};
	}

	double a_b_test_statistic(int total_A, int OccurA, int total_B, int OccurB)
	{
		vector<pair<double, double>> result;
		vector<double> P, sigma;

		result.push_back(estimate_p_sigma(total_A, OccurA));
		result.push_back(estimate_p_sigma(total_B, OccurB));
		
		for (int i = 0; i < result.size(); i++)
		{
			unPair(result, P, sigma);
		}
		return (P[1] - P[0]) / sqrt(pow(P[0], 2) + pow(P[1], 2));
	}

	void rescale(vector<double>& data)
	{
		double average = mean(data), sigma = standard_deviation(data);
		if (sigma > 0)
		{
			deMean(data);
			for (int i = 0; i < data.size(); i++)
			{
				data[i] /= sigma;
			}
		}
	}

	void rescale(vector<vector<double>>& data, int start_index)
	{
		for (int j = start_index; j < data.size(); j++)
		{
			double average = mean(data[j]), sigma = standard_deviation(data[j]);
			if (sigma > 0)
			{
				deMean(data[j]);
				for (int i = 0; i < data.size(); i++)
				{
					data[j][i] /= sigma;
				}
			}
		}
	}

	template<typename T>
	vector<int> count_tp_fp_fn_tn(vector<T> lables, vector<T> predicts)
	{
		vector<int> count(4, 0);
		vector_length_queal(lables, predicts);
		for (int i = 0; i < lables; i++)
		{
			if (lables[i] == predicts[i] && predicts[i] == 1)
			{
				count[0]++;
			}
			else if (lables[i] == predicts[i] && predicts[i] == 0)
			{
				count[3]++;
			}
			else if (lables[i] != predicts[i] && predicts[i] == 1)
			{
				count[1]++;
			}
			else
			{
				count[2]++;
			}
		}
	}

	double accuracy(int tp, int fp, int fn, int tn)
	{
		int correct = tp + tn;
		int total = tp + fp + fn + tn;
		double result = correct / total;

		return result;
	}

	double precision(int tp, int fp, int fn, int tn)
	{
		int total_positive = tp + fp;
		double result = tp / total_positive;
		return result;
	}

	double recall(int tp, int fp, int fn, int tn)
	{
		int target = tp + fn;
		double result = tp / target;
		return result;
	}

	double f1_score(int tp, int fp, int fn, int tn)
	{
		double p = precision(tp, fp, fn, tn);
		double r = recall(tp, fp, fn, tn);
		return 2 * p * r / (p + r);
	}


	vector<double> estimate_sample_beta(vector<pair<vector<double>, double>>& sample)
	{
		vector<vector<double>> sample_X;
		vector<double> sample_Y, w;
		Statistics::unPair(sample, sample_X, sample_Y);

		return linear_regression(sample_X, sample_Y, w);
	}

	vector<double> boostrap_standard_errors(vector<vector<double>>& X, vector<double>& Y, int num_estimate)
	{
		vector<vector<double>> w_estimates;
		vector<double> w_standard_errors;
		for (int i = 0; i < num_estimate; i++)
		{
			vector<pair<vector<double>, double>> boot_sample = dataManipulate::bootstrap_sample(X, Y);
			w_estimates.push_back(estimate_sample_beta(boot_sample));
		}
		vector<vector<double>> wt_estimates = transpose(w_estimates);
		for (int i = 0; i < wt_estimates.size(); i++)
		{
			w_standard_errors.push_back(Statistics::standard_deviation(wt_estimates[i]));
		}
		return w_standard_errors;
	}

	vector<double> p_value(vector<double>& w, vector<double> sigma)
	{
		vector<double> p_result;
		for (int i = 0; i < w.size(); i++)
		{
			double p = Statistics::normal_cdf(w[i] / sigma[i]);
			if (w[i] > 0)
			{
				p_result.push_back(2 * (1 - p));
			}
			else
			{
				p_result.push_back(2 * p);
			}
		}
		return  p_result;
	}

	void first_N_maxVal(vector<double>& source, deque<pair<int, double>>& result, int N)
	{
		for (int i = 0; i < source.size(); i++)
		{
			deque<pair<int, double>> temp_queue;

			if (result.size() > N)
			{
				while (source[i] >= result[result.size() - 1].second)
				{
					auto end_interest = result.back();
					temp_queue.push_back(end_interest);
				}
				result.push_back(pair<int, double>(i, source[i]));

				while (result.size() <= N)
				{
					result.push_back(temp_queue.back());
					temp_queue.pop_back();
				}
			}
			else
			{
				if (result.size() > 0)
				{
					while (source[i] >= result[result.size() - 1].second && result.size() > 0)
					{
						auto end_interest = result.back();
						temp_queue.push_back(end_interest);
					}
					result.push_back(pair<int, double>(i, source[i]));

					while (result.size() <= N)
					{
						result.push_back(temp_queue.back());
						temp_queue.pop_back();
					}
				}
			}
		}
	}
}
