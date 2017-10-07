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
#include "cvlib.h"
#include "DemoML.h"
//#include "opencv2\opencv.hpp";
//using namespace cv;
using namespace std;

int main(void)
{
	///Demo_Neuron();
	Demo_NeuralNetwork();
	//Demo_DecisionTree();
	//Demo_Ngram();

	//圖像計算，約跑3~5分鐘
	//Demo_Kmeans();

	//Demo_random_forest();
	system("pause");
	return 0;
}