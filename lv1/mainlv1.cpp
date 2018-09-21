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
	if (TRAIN)//ѵ������
	{
		//��ȡѵ������·��
		vector<string> file_name;
		vector<int> file_label;
		system("cmd /c dir E:\\LVT\\����\\train\\positivephh\\*.bmp /b >D:\\allfile.txt");//������λ��
		ifstream in("D:\\allfile.txt");
		if (in) // �и��ļ�  
		{
			string line;//ÿһ��ͼƬ������
			while (getline(in, line)) // line�в�����ÿ�еĻ��з�
			{
				const string imagename = "E:\\LVT\\����\\train\\positivephh\\" + line;//������ͼƬ·��
				cout << imagename << endl;
				file_name.push_back(imagename);
				file_label.push_back(1);
			}
		}
		system("cmd /c dir E:\\LVT\\����\\train\\negativephh\\*.bmp /b >D:\\allfile.txt");//������λ��
		ifstream in1("D:\\allfile.txt");
		if (in1) // �и��ļ�  
		{
			string line;//ÿһ��ͼƬ������
			while (getline(in1, line)) // line�в�����ÿ�еĻ��з�
			{
				const string imagename = "E:\\LVT\\����\\train\\negativephh\\" + line;//������ͼƬ·��
				cout << imagename << endl;
				file_name.push_back(imagename);
				file_label.push_back(-1);
			}
		}
		Mat sampleFeatureMat;//��������
		Mat sampleLabelMat;//��ǩ����
		sampleFeatureMat = Mat::zeros(file_name.size(), DescriptorDim, CV_32FC1);
		sampleLabelMat = Mat::zeros(file_name.size(), 1, CV_32SC1);//sampleLabelMat���������ͱ���Ϊ�з���������
		for (int i = 0; i < file_name.size(); i++)//��������ͼ��
		{
			Mat src = imread(file_name[i], 0);
			//blur(src, src, Size(3,3));
			resize(src, src, Size(Width, Height));
			vector<float> descriptors;
			hog.compute(src, descriptors, Size(step, step));//����HOG�����ӣ���ⴰ���ƶ�����(8,8)

			//������õ�HOG�����Ӹ��Ƶ�������������sampleFeatureMat
			for (int j = 0; j < DescriptorDim; j++)
				sampleFeatureMat.at<float>(i, j) = descriptors[j];
			sampleLabelMat.at<int>(i, 0) = file_label[i];
		}
		//sampleLabelMat.convertTo(sampleLabelMat, CV_32SC1);
		///////////////////////////////////ʹ��SVM������ѵ��///////////////////////////////////////////////////    

		//���ò���
		svm->setType(SVM::C_SVC);//ָ��SVM������
		svm->setGamma(1);
		svm->setC(1);
		//svm->setGamma(1);
		//svm->setC(10);
		svm->setKernel(SVM::RBF);//SVM���ں�����
		svm->setTermCriteria(TermCriteria(cv::TermCriteria::MAX_ITER + cv::TermCriteria::EPS, 2000, 1e-3));

		cout << "Starting training..." << endl;
		//ʹ��SVMѧϰ 
		svm->train(sampleFeatureMat, cv::ml::ROW_SAMPLE, sampleLabelMat);
		//svm->trainAuto(trainDataSet);
		// ������ò��ֲ����������Ż�������ʹ��*cv::ml::ParamGrid::create(0, 0, 0)������ӦGridĬ��ֵ�����²���C�����Ż���  
		//svm->trainAuto(trainDataSet, 10, *cv::ml::ParamGrid::create(0, 0, 0));  
		cout << "Finishing training..." << endl;
		//���������
		svm->save("D:\\SVM_HOG.xml");
	}
	//"D:\\VS13programdyy\\lv1\\lv1\\SVM_HOG.xml"dyy
	else//ʶ�����
	{
		svm = SVM::load("D:\\SVM_HOG.xml");

		//��ȡѵ������·��
		vector<string> file_name;
		vector<int> file_label;
		system("cmd /c dir E:\\LVT\\����\\test\\positivexyy1\\*.bmp /b >D:\\allfile.txt");//��ͼƬ��ȫ�����浽txt��
		ifstream in("D:\\allfile.txt");
		int pos_num = 0;//����������
		int neg_num = 0;//����������
		int count_true = 0;//ʶ����ȷ�ĸ���
		int count_pos_true = 0;//ʶ����������ȷ����
		int count_neg_true = 0;//ʶ��������ȷ����
		if (in) // �и��ļ�  
		{
			string line;//ÿһ��ͼƬ������
			while (getline(in, line)) // line�в�����ÿ�еĻ��з�
			{
				const string imagename = "E:\\LVT\\����\\test\\positivexyy1\\" + line;//ͼƬ·��
				cout << imagename << endl;
				file_name.push_back(imagename);
				file_label.push_back(1);
				pos_num++;
			}
		}
		system("cmd /c dir E:\\LVT\\����\\test\\negativexyy1\\*.bmp /b >D:\\allfile.txt");//��ͼƬ��ȫ�����浽txt��
		ifstream in1("D:\\allfile.txt");
		if (in1) // �и��ļ�  
		{
			string line;//ÿһ��ͼƬ������
			while (getline(in1, line)) // line�в�����ÿ�еĻ��з�
			{
				const string imagename = "E:\\LVT\\����\\test\\negativexyy1\\" + line;//ͼƬ·��
				cout << imagename << endl;
				file_name.push_back(imagename);
				file_label.push_back(-1);
				neg_num++;
			}
		}

		for (int i = 0; i < file_name.size(); i++)//��������ͼ��
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

			cout << "Ԥ��:" << response << "��ʵ:" << file_label[i] << endl;
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
					img_name = file_name[i].substr(nToken, file_name[i].length());//��������
				}
				if (file_label[i] > 0)//�������ֳ��˸�����
				{
					imwrite("E:\\LVT\\data\\errP2Nxyy1\\" + img_name, src_);
				}
				if (file_label[i] < 0)//�������ֳ���������
				{
					imwrite("E:\\LVT\\data\\errN2Pxyy1\\" + img_name, src_);
				}
			}
		}
		cout << "����ʶ���ʣ�" << (float)count_true / file_name.size() << endl;
		cout << "������ʶ���ʣ�" << (float)count_pos_true / pos_num << endl;
		cout << "������ʶ���ʣ�" << (float)count_neg_true / neg_num << endl;

		getchar();
	}

	return 1;
}