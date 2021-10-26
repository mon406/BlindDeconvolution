#ifndef __INCLUDED_H_CalculateVector__
#define __INCLUDED_H_CalculateVector__

#include "main.h"

/* 定義する関数 */
double multi_vector(Mat& srcImg, Mat& srcImg2);							// 1次元ベクトルの乗算(内積)
void multi_matrix_vector(Mat& srcImg, Mat& dft_srcImg2, Mat& dstImg);	// 2次元ベクトルと1次元ベクトルの乗算

void make_matrix_A(Mat& srcImg, double& img_size, double& lambda, double& myu, Mat& dstImg);
void make_matrix_A2(Mat& srcImg, int& kernel_size, double& parameter, Mat& dstImg);


/* 関数 */
// 1次元ベクトルの乗算(内積)
double multi_vector(Mat& srcImg, Mat& srcImg2) {
	double Result = 0.0;
	int x, y;
	double in, in2, out;

	if (srcImg.cols != srcImg2.cols || srcImg.rows != srcImg2.rows) { cout << "ERROR! multi_vector() : Can't translate because of wrong sizes." << endl; }
	if (srcImg.channels() == 1) {
#pragma omp parallel for private(x)
		for (y = 0; y < srcImg.rows; y++) {
			for (x = 0; x < srcImg.cols; x++) {
				in = srcImg.at<double>(y, x);
				in2 = srcImg2.at<double>(y, x);
				out = in * in2;
				Result += out;
			}
		}
	}
	else { cout << "ERROR! multi_vector() : Can't translate because of wrong channels." << endl; }

	return Result;
}

// 2次元ベクトルと1次元ベクトルの乗算
void multi_matrix_vector(Mat& srcImg, Mat& dft_srcImg2, Mat& dstImg) {
	int Msize = dft_srcImg2.rows;
	int Nsize = dft_srcImg2.cols;
	// srcImg's DFT
	Mat dft_srcImg = Mat::zeros(Msize, Nsize, CV_64FC2);
	Mat planes2[] = { Mat_<double>(srcImg), Mat::zeros(srcImg.size(), CV_64F) };
	Mat srcImg_sub;
	merge(planes2, 2, srcImg_sub);
	copyMakeBorder(srcImg_sub, dft_srcImg, 0, Msize - srcImg.rows, 0, Nsize - srcImg.cols, BORDER_CONSTANT, (0.0, 0.0));
	dft(dft_srcImg, dft_srcImg, 0, dft_srcImg.rows);

	Mat Img = Mat::zeros(srcImg.size(), CV_64FC1);
	Mat Img2;
	int x, y;
	double in, in2, out;

	if (srcImg.channels() == 1 && dft_srcImg2.channels() == 2) {
		// 計算
		mulSpectrums(dft_srcImg2, dft_srcImg, Img2, 0, false);
		//inverseDFT
		dft(Img2, Img2, cv::DFT_INVERSE + cv::DFT_SCALE | cv::DFT_REAL_OUTPUT);
		//copyMakeBorder(Img2, Img2, srcImg.rows / 2, srcImg.rows / 2, srcImg.cols / 2, srcImg.cols / 2, BORDER_WRAP);
		Img = Img2(Rect(0, 0, srcImg.cols, srcImg.rows));
	}
	else { cout << "ERROR! multi_matrix_vector() : Can't translate because of wrong channels." << endl; }

	Img.copyTo(dstImg);
}

void make_matrix_A(Mat& srcImg, double& img_size, double& lambda, double& myu, Mat& dstImg) {
	int x, y, c;
	int total = srcImg.cols * srcImg.rows;
	Mat ConvMatrix = Mat::zeros(Size(img_size, total), CV_64FC1);

#pragma omp parallel for private(x)
	for (y = 0; y < srcImg.rows; y++) {
		for (x = 0; x < srcImg.cols; x++) {
			ConvMatrix.at<double>(y, x) = 0.0;
		}
	}


}
void make_matrix_A2(Mat& srcImg, int& kernel_size, double& parameter, Mat& dstImg) {
	int x, y, c;
	int total = srcImg.cols * srcImg.rows;
	Mat ConvMatrix = Mat::zeros(Size(kernel_size * 2, total), CV_64FC1);
	Mat TransConvMatrix = Mat::zeros(Size(total, kernel_size * 2), CV_64FC1);

	double plus_parameter = (double)parameter / 2.0;
	cout << "  make_matrix_A2() : plus_parameter = " << (double)plus_parameter << endl;	// 確認用

	int input_x = 0, input_y = 0;
	double input_num;
	int now_x, now_y;
	int half_kernel_size = (int)(kernel_size / 2) - 1;
#pragma omp parallel for private(x)
	for (y = 0; y < srcImg.rows; y++) {
		for (x = 0; x < srcImg.cols; x++) {
			input_x = 0;
			for (int yy = 0; yy < kernel_size; yy++) {
				for (int xx = 0; xx < kernel_size; xx++) {
					now_x = x + xx - half_kernel_size;
					now_y = y + yy - half_kernel_size;
					if(now_x < 0 || now_x >= srcImg.cols || now_y < 0 || now_y >= srcImg.rows){ input_num = 0.0; }
					else { input_num = srcImg.at<double>(now_y, now_x); }
					ConvMatrix.at<double>(input_y, input_x) = input_num;
					TransConvMatrix.at<double>(input_x, input_y) = input_num;
					input_x++;
				}
			}
			input_y++;
		}
	}
	//checkMat_detail(ConvMatrix);

	double Trans = 0.0;
#pragma omp parallel
	for (y = 0; y < ConvMatrix.rows * ConvMatrix.cols; y++) {
		Trans += (double)((double)ConvMatrix.data[y] * (double)TransConvMatrix.data[y] + plus_parameter);
	}
	cout << "  make_matrix_A2() : Trans = " << (double)Trans << endl;	// 確認用

	ConvMatrix.copyTo(dstImg);
}


#endif