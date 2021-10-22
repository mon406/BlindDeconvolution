#ifndef __INCLUDED_H_MakeKernel__
#define __INCLUDED_H_MakeKernel__

#include "main.h"

/* �萔 */
const double PI = 3.141592653589793;
const int filetersize_cols = 20;
const int filetersize_rows = 20;

/* �֐� */
void KernelImage_Visualization(Mat& KernelImage);	// �J�[�l���摜�̉���
void KernelImage_Visualization_double(Mat& KernelImage);
void KernelMat_Normalization(Mat& inoutKernel);		// �J�[�l���摜�̐��K��


/* �N���X */
class KERNEL {
private:
	int x, y;
	int index;
public:
	int cols;		// �摜�̕�
	int rows;		// �摜�̍���
	int size;		// �摜�̑��s�N�Z����
	double sum;		// ��f�l�̑��a
	Mat Kernel;						// �J�[�l��
	Mat Kernel_normalized;			// �J�[�l��(���a�Ŋ���)

	KERNEL();						// ������
	KERNEL(int);
	void display();					// �J�[�l�����\��
	void display_detail();
	void inverse_normalization();	// �J�[�l��(���a�ł�����)
	void visualization(Mat&);		// �J�[�l���摜�̉���
	void copy(KERNEL&);
	void resize_copy(double&, KERNEL&);
	void resize_copy(double, double, KERNEL&);
};
KERNEL::KERNEL() {
	cols = filetersize_cols;
	rows = filetersize_rows;
	size = cols * rows;
	sum = 0.0;
	Kernel = Mat_<double>(filetersize_rows, filetersize_cols);
	Kernel_normalized = Mat_<double>(filetersize_rows, filetersize_cols);
}
KERNEL::KERNEL(int initialized_index) {
	if (initialized_index == 0) {
		Kernel = (Mat_<double>(filetersize_rows, filetersize_cols) 		// 20*20 (line)
			<< 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			0, 0, 0, 0, 0, 0, 20, 10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			0, 0, 0, 0, 0, 30, 50, 30, 10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			0, 0, 0, 0, 10, 40, 100, 40, 20, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			0, 0, 0, 0, 20, 50, 100, 30, 10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			0, 0, 0, 0, 10, 20, 40, 100, 30, 10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			0, 0, 0, 0, 10, 20, 30, 100, 40, 20, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			0, 0, 0, 0, 0, 10, 20, 40, 100, 30, 10, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			0, 0, 0, 0, 0, 0, 10, 30, 100, 30, 20, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			0, 0, 0, 0, 0, 0, 0, 20, 40, 90, 30, 20, 0, 0, 0, 0, 0, 0, 0, 0,
			0, 0, 0, 0, 0, 0, 0, 10, 20, 40, 80, 40, 10, 0, 0, 0, 0, 0, 0, 0,
			0, 0, 0, 0, 0, 0, 0, 0, 10, 20, 40, 90, 30, 10, 0, 0, 0, 0, 0, 0,
			0, 0, 0, 0, 0, 0, 0, 0, 0, 10, 20, 90, 40, 10, 0, 0, 0, 0, 0, 0,
			0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 10, 20, 100, 30, 10, 0, 0, 20, 10, 0,
			0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 20, 100, 50, 10, 10, 40, 90, 20, 0,
			0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 10, 30, 100, 60, 30, 100, 40, 20, 0,
			0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 10, 30, 100, 100, 40, 20, 0, 0,
			0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 10, 30, 40, 30, 10, 0, 0,
			0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 10, 20, 10, 0, 0, 0,
			0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
		cout << " KERNEL(int 0) set" << endl;		// �m�F�p
	}
	else if (initialized_index == 1) {
		Kernel = (Mat_<double>(filetersize_rows, filetersize_cols) 		// 20*20 (PSF)
			<< 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			0, 0, 0, 0, 0, 0, 0, 0, 10, 15, 15, 10, 0, 0, 0, 0, 0, 0, 0, 0,
			0, 0, 0, 0, 0, 0, 0, 20, 30, 40, 40, 30, 20, 0, 0, 0, 0, 0, 0, 0,
			0, 0, 0, 0, 0, 0, 20, 50, 60, 70, 70, 60, 50, 20, 0, 0, 0, 0, 0, 0,
			0, 0, 0, 0, 0, 10, 30, 60, 80, 90, 90, 80, 60, 30, 10, 0, 0, 0, 0, 0,
			0, 0, 0, 0, 0, 15, 40, 70, 90, 100, 100, 90, 70, 40, 15, 0, 0, 0, 0, 0,
			0, 0, 0, 0, 0, 15, 40, 70, 90, 100, 100, 90, 70, 40, 15, 0, 0, 0, 0, 0,
			0, 0, 0, 0, 0, 10, 30, 60, 80, 90, 90, 80, 60, 30, 10, 0, 0, 0, 0, 0,
			0, 0, 0, 0, 0, 0, 20, 50, 60, 70, 70, 60, 50, 20, 0, 0, 0, 0, 0, 0,
			0, 0, 0, 0, 0, 0, 0, 20, 30, 40, 40, 30, 20, 0, 0, 0, 0, 0, 0, 0,
			0, 0, 0, 0, 0, 0, 0, 0, 10, 15, 15, 10, 0, 0, 0, 0, 0, 0, 0, 0,
			0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
		cout << " KERNEL(int 1) set" << endl;		// �m�F�p
	}
	else if (initialized_index == 2) {
		Kernel = (Mat_<double>(filetersize_rows, filetersize_cols) 		// 20*20 (most zero)
			<< 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			0, 0, 0, 0, 0, 0, 0, 0, 0, 10, 10, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			0, 0, 0, 0, 0, 0, 0, 0, 0, 20, 20, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			0, 0, 0, 0, 0, 0, 0, 10, 20, 40, 40, 20, 10, 0, 0, 0, 0, 0, 0, 0,
			0, 0, 0, 0, 0, 0, 0, 10, 20, 40, 40, 20, 10, 0, 0, 0, 0, 0, 0, 0,
			0, 0, 0, 0, 0, 0, 0, 0, 0, 20, 20, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			0, 0, 0, 0, 0, 0, 0, 0, 0, 10, 10, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
		cout << " KERNEL(int 2) set" << endl;		// �m�F�p
	}
	else {
		Kernel = (Mat_<double>(filetersize_rows, filetersize_cols) 		// 20*20 (almost zero)
			<< 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			0, 0, 0, 0, 0, 0, 0, 0, 0, 50, 50, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			0, 0, 0, 0, 0, 0, 0, 0, 0, 50, 50, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
		cout << " KERNEL(others) set" << endl;		// �m�F�p
	}
	cols = Kernel.cols;
	rows = Kernel.rows;
	size = cols * rows;


	sum = 0.0;
#pragma omp parallel for private(x)
	for (y = 0; y < rows; y++) {
		for (x = 0; x < cols; x++) {
			sum += (double)Kernel.at<double>(y, x);
		}
	}
	cout << " sum = " << (double)sum << endl;

	Kernel_normalized = Kernel / (double)sum;
}
void KERNEL::display() {
	cout << "##############################" << endl;
	cout << " cols = " << (int)cols << endl;
	cout << " rows = " << (int)rows << endl;
	cout << " type = " << (
		Kernel.type() == CV_8UC1 ? "CV_8UC1" :
		Kernel.type() == CV_64FC1 ? "CV_64FC1" :
		"other"
		) << endl;
	cout << " sum = " << (double)sum << endl;
	cout << "##############################" << endl;
	cout << endl;
}
void KERNEL::display_detail() {
	cout << "##############################" << endl;
	cout << " cols = " << (int)cols << endl;
	cout << " rows = " << (int)rows << endl;
	cout << " type = " << (
		Kernel.type() == CV_8UC1 ? "CV_8UC1" :
		Kernel.type() == CV_64FC1 ? "CV_64FC1" :
		"other"
		) << endl;
	cout << " sum = " << (double)sum << endl;
	cout << endl;
	cout << Kernel << endl;
	cout << endl;
	cout << Kernel_normalized << endl;
	cout << "##############################" << endl;
	cout << endl;
}
void KERNEL::inverse_normalization() {
	double calc_sum = 0.0;	// ���a���v�Z
	double calc_num;
#pragma omp parallel for private(x)
	for (y = 0; y < rows; y++) {
		for (x = 0; x < cols; x++) {
			calc_num = (double)Kernel_normalized.at<double>(y, x);
			calc_sum += calc_num;
		}
	}
	sum = (double)calc_sum;
	//cout << " sum = " << (double)sum << endl;	 // �m�F�p

	if (sum != 0) {
		double multi_sum = 1.0 / (double)sum;		// ���a���P�ɂȂ�悤�ȌW��
#pragma omp parallel for private(x)
		for (y = 0; y < rows; y++) {
			for (x = 0; x < cols; x++) {
				calc_sum = Kernel_normalized.at<double>(y, x);
				calc_sum *= (double)multi_sum;
				Kernel_normalized.at<double>(y, x) = calc_sum;
			}
		}
		normalize(Kernel_normalized, Kernel, 0, 100, NORM_MINMAX);
	}
	else {
		Kernel = Mat::zeros(Kernel.size(), CV_64FC1);
		Kernel_normalized = Mat::zeros(Kernel_normalized.size(), CV_64FC1);
		cout << "WARNING! KERNEL::inverse_normalization() : sum=0" << endl;
	}
}
void KERNEL::visualization(Mat& outputIMG) {
	if (sum != 0) {
		KernelImage_Visualization(Kernel);
		//KernelImage_Visualization_double(Kernel);
	}
	else { cout << "WARNING! KERNEL::visualization() : sum=0" << endl; }
	Kernel.copyTo(outputIMG);
}
void KERNEL::copy(KERNEL& inputKERNEL) {
	cols = (int)inputKERNEL.cols;
	rows = (int)inputKERNEL.rows;
	size = cols * rows;
	sum = (double)inputKERNEL.sum;
	inputKERNEL.Kernel.copyTo(Kernel);
	inputKERNEL.Kernel_normalized.copyTo(Kernel_normalized);
}
void KERNEL::resize_copy(double& resize_factor, KERNEL& inputKERNEL) {
	//display_detail();

	Mat tmpKernel;
	resize(inputKERNEL.Kernel, tmpKernel, Size(), resize_factor, resize_factor);
	tmpKernel.copyTo(Kernel);
	resize(inputKERNEL.Kernel_normalized, tmpKernel, Size(), resize_factor, resize_factor);
	tmpKernel.copyTo(Kernel_normalized);

	cols = (int)tmpKernel.cols;
	rows = (int)tmpKernel.rows;
	size = cols * rows;

	inverse_normalization();
	//display_detail();

	if (sum == 0) { cout << "WARNING! KERNEL::resize_copy() : sum=0" << endl; }
}
void KERNEL::resize_copy(double resize_factor_X, double resize_factor_Y, KERNEL& inputKERNEL) {
	//display_detail();

	Mat tmpKernel;
	resize(inputKERNEL.Kernel, tmpKernel, Size(), resize_factor_X, resize_factor_Y);
	tmpKernel.copyTo(Kernel);
	resize(inputKERNEL.Kernel_normalized, tmpKernel, Size(), resize_factor_X, resize_factor_Y);
	tmpKernel.copyTo(Kernel_normalized);

	cols = (int)tmpKernel.cols;
	rows = (int)tmpKernel.rows;
	size = cols * rows;

	inverse_normalization();
	//display_detail();

	if (sum == 0) { cout << "WARNING! KERNEL::resize_copy() : sum=0" << endl; }
}


/* �֐� */
// �J�[�l���摜�̉���
void KernelImage_Visualization(Mat& KernelImage) {
	Mat KernelImage_tmp = Mat(KernelImage.size(), CV_8UC1);
	int xx, yy;
	int Kernel_index;
	double Kernel_tmp;

#pragma omp parallel for private(xx)
	for (yy = 0; yy < KernelImage.rows; yy++) {
		for (xx = 0; xx < KernelImage.cols; xx++) {
			Kernel_index = yy * KernelImage.cols + xx;
			Kernel_tmp = (double)KernelImage.at<double>(yy, xx);

			Kernel_tmp *= 2;	// ���₷�����邽�߂Ɂ~2
			if (Kernel_tmp > MAX_INTENSE) { Kernel_tmp = MAX_INTENSE; }

			KernelImage_tmp.data[Kernel_index] = (uchar)Kernel_tmp;
		}
	}
	KernelImage_tmp.convertTo(KernelImage, CV_8UC1);
}
void KernelImage_Visualization_double(Mat& KernelImage) {
	Mat KernelImage_tmp = Mat(KernelImage.size(), CV_64FC1);
	int xx, yy;
	int Kernel_index;
	double Kernel_tmp;

#pragma omp parallel for private(xx)
	for (yy = 0; yy < KernelImage.rows; yy++) {
		for (xx = 0; xx < KernelImage.cols; xx++) {
			Kernel_index = yy * KernelImage.cols + xx;
			Kernel_tmp = (double)KernelImage.at<double>(yy, xx);

			Kernel_tmp *= 2;	// ���₷�����邽�߂Ɂ~2
			if (Kernel_tmp > MAX_INTENSE) { Kernel_tmp = MAX_INTENSE; }

			KernelImage_tmp.at<double>(yy, xx) = (double)Kernel_tmp;
		}
	}
	KernelImage_tmp.copyTo(KernelImage);
}

// �J�[�l���摜�̐��K��
void KernelMat_Normalization(Mat& inoutKernel) {
	int x, y;
	double calc_sum = 0.0;	// ���a���v�Z
#pragma omp parallel for private(x)
	for (y = 0; y < inoutKernel.rows; y++) {
		for (x = 0; x < inoutKernel.cols; x++) {
			calc_sum = (double)inoutKernel.at<double>(y, x);
			calc_sum += calc_sum;
		}
	}
	double multi_sum = 1.0 / (double)calc_sum;		// ���a���P�ɂȂ�悤�ȌW��
	//cout << "   multi_sum = " << (double)multi_sum << " : calc_sum = " << (double)calc_sum << endl;	// �m�F�p
#pragma omp parallel for private(x)
	for (y = 0; y < inoutKernel.rows; y++) {
		for (x = 0; x < inoutKernel.cols; x++) {
			calc_sum = (double)inoutKernel.at<double>(y, x);
			calc_sum *= multi_sum;
			inoutKernel.at<double>(y, x) = calc_sum;
		}
	}
	//normalize(inoutKernel, inoutKernel, 0, 1, NORM_MINMAX);
}

#endif