#include<iostream>
#include<opencv2\opencv.hpp>
#include <opencv2/ml/ml.hpp>
#include "opencv2/xfeatures2d.hpp"
#include "opencv2/features2d.hpp"
using namespace cv::ml;
using namespace std;
using namespace cv;
using namespace cv::xfeatures2d;
#pragma comment (lib, "opencv_objdetect320.lib")
int main()
{
	int Width = 28; int Height = 28;
	const int step = 7;
	HOGDescriptor hog(Size(Width, Height), Size(14, 14), Size(step, step), Size(step, step), 9);
	int DescriptorDim = 324;
	Ptr<SVM> svm = SVM::create();
	bool TRAIN =0;
	if (TRAIN)//训练过程
	{
		//读取训练样本路径
		vector<string> file_name;
		vector<int> file_label;
		system("cmd /c dir E:\\LVT\\地铁\\train\\positivephh\\*.bmp /b >D:\\allfile.txt");//正样本位置
		ifstream in("D:\\allfile.txt");
		if (in) // 有该文件  
		{
			string line;//每一张图片的名称
			while (getline(in, line)) // line中不包括每行的换行符
			{
				const string imagename = "E:\\LVT\\地铁\\train\\positivephh\\" + line;//正样本图片路径
				cout << imagename << endl;
				file_name.push_back(imagename);
				file_label.push_back(1);
			}
		}
		system("cmd /c dir E:\\LVT\\地铁\\train\\negativephh\\*.bmp /b >D:\\allfile.txt");//负样本位置
		ifstream in1("D:\\allfile.txt");
		if (in1) // 有该文件  
		{
			string line;//每一张图片的名称
			while (getline(in1, line)) // line中不包括每行的换行符
			{
				const string imagename = "E:\\LVT\\地铁\\train\\negativephh\\" + line;//负样本图片路径
				cout << imagename << endl;
				file_name.push_back(imagename);
				file_label.push_back(-1);
			}
		}
		Mat sampleFeatureMat;//特征矩阵
		Mat sampleLabelMat;//标签矩阵
		sampleFeatureMat = Mat::zeros(file_name.size(), DescriptorDim, CV_32FC1);
		sampleLabelMat = Mat::zeros(file_name.size(), 1, CV_32SC1);//sampleLabelMat的数据类型必须为有符号整数型
		for (int i = 0; i < file_name.size(); i++)//遍历所有图像
		{
			Mat src = imread(file_name[i], 0);
			//blur(src, src, Size(3,3));
			resize(src, src, Size(Width, Height));
			vector<float> descriptors;
			hog.compute(src, descriptors, Size(step, step));//计算HOG描述子，检测窗口移动步长(8,8)

			//将计算好的HOG描述子复制到样本特征矩阵sampleFeatureMat
			for (int j = 0; j < DescriptorDim; j++)
				sampleFeatureMat.at<float>(i, j) = descriptors[j];
			sampleLabelMat.at<int>(i, 0) = file_label[i];
		}
		//sampleLabelMat.convertTo(sampleLabelMat, CV_32SC1);
		///////////////////////////////////使用SVM分类器训练///////////////////////////////////////////////////    

		//设置参数
		svm->setType(SVM::C_SVC);//指定SVM的类型
		svm->setGamma(1);
		svm->setC(1);
		//svm->setGamma(1);
		//svm->setC(10);
		svm->setKernel(SVM::RBF);//SVM的内核类型
		svm->setTermCriteria(TermCriteria(cv::TermCriteria::MAX_ITER + cv::TermCriteria::EPS, 2000, 1e-3));

		cout << "Starting training..." << endl;
		//使用SVM学习 
		svm->train(sampleFeatureMat, cv::ml::ROW_SAMPLE, sampleLabelMat);
		//svm->trainAuto(trainDataSet);
		// 如果想让部分参数不进行优化，可以使用*cv::ml::ParamGrid::create(0, 0, 0)代替相应Grid默认值，如下不对C进行优化：  
		//svm->trainAuto(trainDataSet, 10, *cv::ml::ParamGrid::create(0, 0, 0));  
		cout << "Finishing training..." << endl;
		//保存分类器
		svm->save("D:\\SVM_HOG.xml");
	}
	//"D:\\VS13programdyy\\lv1\\lv1\\SVM_HOG.xml"dyy
	else//识别过程
	{
		svm = SVM::load("D:\\SVM_HOG.xml");

		//读取训练样本路径
		vector<string> file_name;
		vector<int> file_label;
		system("cmd /c dir E:\\LVT\\地铁\\test\\positivexyy1\\*.bmp /b >D:\\allfile.txt");//把图片名全部保存到txt中
		ifstream in("D:\\allfile.txt");
		int pos_num = 0;//正样本个数
		int neg_num = 0;//负样本个数
		int count_true = 0;//识别正确的个数
		int count_pos_true = 0;//识别正样本正确个数
		int count_neg_true = 0;//识别负样本正确个数
		if (in) // 有该文件  
		{
			string line;//每一张图片的名称
			while (getline(in, line)) // line中不包括每行的换行符
			{
				const string imagename = "E:\\LVT\\地铁\\test\\positivexyy1\\" + line;//图片路径
				cout << imagename << endl;
				file_name.push_back(imagename);
				file_label.push_back(1);
				pos_num++;
			}
		}
		system("cmd /c dir E:\\LVT\\地铁\\test\\negativexyy1\\*.bmp /b >D:\\allfile.txt");//把图片名全部保存到txt中
		ifstream in1("D:\\allfile.txt");
		if (in1) // 有该文件  
		{
			string line;//每一张图片的名称
			while (getline(in1, line)) // line中不包括每行的换行符
			{
				const string imagename = "E:\\LVT\\地铁\\test\\negativexyy1\\" + line;//图片路径
				cout << imagename << endl;
				file_name.push_back(imagename);
				file_label.push_back(-1);
				neg_num++;
			}
		}

		for (int i = 0; i < file_name.size(); i++)//遍历所有图像
		{
			Mat src_ = imread(file_name[i], 0);
			Mat src = src_.clone();

			resize(src, src, Size(Width, Height));

			vector<float> descriptors;
			hog.compute(src, descriptors, Size(step, step));
			Mat sampleMat = Mat::zeros(1, DescriptorDim, CV_32FC1);

			for (int i = 0; i<DescriptorDim; i++)
				sampleMat.at<float>(0, i) = descriptors[i];
			float response = svm->predict(sampleMat);

			cout << "预测:" << response << "真实:" << file_label[i] << endl;
			if (abs(response - file_label[i]) < 0.5)
			{
				count_true++;
				if (file_label[i] > 0)
					count_pos_true++;
				if (file_label[i] < 0)
					count_neg_true++;
			}
			else
			{
				int nToken = file_name[i].find("cam");
				string img_name;
				if (nToken > 0)
				{
					img_name = file_name[i].substr(nToken, file_name[i].length());//部件名称
				}
				if (file_label[i] > 0)//正样本分成了负样本
				{
					imwrite("E:\\LVT\\data\\errP2Nxyy1\\" + img_name, src_);
				}
				if (file_label[i] < 0)//负样本分成了正样本
				{
					imwrite("E:\\LVT\\data\\errN2Pxyy1\\" + img_name, src_);
				}
			}
		}
		cout << "总体识别率：" << (float)count_true / file_name.size() << endl;
		cout << "正样本识别率：" << (float)count_pos_true / pos_num << endl;
		cout << "负样本识别率：" << (float)count_neg_true / neg_num << endl;

		getchar();
	}

	return 1;
}