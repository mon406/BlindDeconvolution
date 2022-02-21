#ifndef __INCLUDED_H_Calculate_Complex_Number__
#define __INCLUDED_H_Calculate_Complex_Number__

#include "main.h"

/* 定義する関数 */
void multi_Vec2d(Vec2d& answer, Vec2d& complex1, Vec2d& complex2);	// ２つの複素数の乗算
void divi_Vec2d(Vec2d& answer, Vec2d& complex1, Vec2d& complex2);	// ２つの複素数の除算

void reciprocal_complex_Mat(Mat& srcImg, Mat& dstImg);	// 2次元ベクトルの逆数
void abs_complex_Mat(Mat& srcImg, Mat& dstImg);			// 2次元ベクトルの大きさ
void abs_pow_complex_Mat(Mat& srcImg, Mat& dstImg);		// 2次元ベクトルの大きさの２乗

void wiener_filter(Mat& srcImg, Mat& dstImg, Mat& parameterImg);	// 2次元ベクトルの逆数(ウィーナ・フィルター)


/* 関数 */

// 複素数の掛け算
void multi_Vec2d(Vec2d& answer, Vec2d& complex1, Vec2d& complex2) {
	Vec2d result = { 0.0, 0.0 };

	result[0] = ((double)complex1[0] * (double)complex2[0]) - ((double)complex1[1] * (double)complex2[1]);
	result[1] = ((double)complex1[1] * (double)complex2[0]) + ((double)complex1[0] * (double)complex2[1]);

	answer = result;
}
// 複素数の割り算
void divi_Vec2d(Vec2d& answer, Vec2d& complex1, Vec2d& complex2) {
	Vec2d result = { 0.0, 0.0 };
	double denom_num = 0.0;

	Vec2d div_complex = { 0.0, 0.0 };
	if (complex2[0] != 0 || complex2[1] != 0) {
		if (complex2[0] == 0) { denom_num = (double)pow(complex2[1], 2); }
		else if (complex2[1] == 0) { denom_num = (double)pow(complex2[0], 2); }
		else { denom_num = (double)pow(complex2[0], 2) + (double)pow(complex2[1], 2); }

		if (denom_num != 0) {
			div_complex = complex2;
			div_complex[1] = -1.0 * (double)div_complex[1];
			div_complex[0] /= (double)denom_num;
			div_complex[1] /= (double)denom_num;
		}

		result[0] = ((double)complex1[0] * (double)div_complex[0]) - ((double)complex1[1] * (double)div_complex[1]);
		result[1] = ((double)complex1[1] * (double)div_complex[0]) + ((double)complex1[0] * (double)div_complex[1]);
	}
	else { cout << "WARNING! divi_complex_2() : Can't divide 0." << endl; }

	answer = result;
}

// 2次元ベクトルの逆数
void reciprocal_complex_Mat(Mat& srcImg, Mat& dstImg) {
	Mat Img = Mat::zeros(srcImg.size(), CV_64FC2);
	int x, y;
	Vec2d in, out;
	Vec2d one = { 1.0, 0.0 };

	if (srcImg.channels() == 2) {
#pragma omp parallel for private(x)
		for (y = 0; y < srcImg.rows; y++) {
			for (x = 0; x < srcImg.cols; x++) {
				in = srcImg.at<Vec2d>(y, x);
				divi_Vec2d(out, one, in);
				Img.at<Vec2d>(y, x) = out;
			}
		}

		Img.copyTo(dstImg);
	}
	else { cout << "ERROR! reciprocal_complex() : Can't translate because of wrong channels." << endl; }
}

// 2次元ベクトルの大きさ
void abs_complex_Mat(Mat& srcImg, Mat& dstImg) {
	Mat Img = Mat::zeros(srcImg.size(), CV_64FC1);

	if (srcImg.channels() == 2) {
		Mat planes[2];
		split(srcImg, planes);					// planes[0] = Re(DFT(SRC)), planes[1] = Im(DFT(SRC))
		magnitude(planes[0], planes[1], Img);	// Img = sqrt( planes[0]^2 + planes[1]^2 )

		Img.copyTo(dstImg);
	}
	else { cout << "ERROR! abs_complex() : Can't translate because of wrong channels." << endl; }
}
// 2次元ベクトルの大きさの２乗
void abs_pow_complex_Mat(Mat& srcImg, Mat& dstImg) {
	Mat Img = Mat::zeros(srcImg.size(), CV_64FC1);
	Mat Img2 = Mat::zeros(srcImg.size(), CV_64FC2);

	if (srcImg.channels() == 2) {
		Mat planes[2];
		split(srcImg, planes);					// planes[0] = Re(DFT(SRC)), planes[1] = Im(DFT(SRC))
		magnitude(planes[0], planes[1], Img);	// Img = sqrt( planes[0]^2 + planes[1]^2 )

		double pow_calc;
		Vec2d pow_out;
#pragma omp parallel for private(x)
		for (int y = 0; y < Img.rows; y++) {
			for (int x = 0; x < Img.cols; x++) {
				pow_calc = (double)Img.at<double>(y, x);
				pow_calc = (double)pow(pow_calc, 2);	// Imgの2乗
				pow_out = { pow_calc, 0.0 };
				Img2.at<Vec2d>(y, x) = pow_out;
			}
		}

		Img2.copyTo(dstImg);
	}
	else { cout << "ERROR! abs_pow_complex() : Can't translate because of wrong channels." << endl; }
}

// 2次元ベクトルの逆数(ウィーナ・フィルター)
void wiener_filter(Mat& srcImg, Mat& dstImg, Mat& parameterImg) {
	Mat Img = Mat::zeros(srcImg.size(), CV_64FC2);

	Vec2d Filter_Parameter;
	int x, y;
	Vec2d abs_in, semi_in, out;
	Vec2d one = { 1.0, 0.0 };

	Mat AbsPowMat;
	abs_pow_complex_Mat(srcImg, AbsPowMat);

	if (srcImg.channels() != 2) { cout << "ERROR! wiener_filter() : Can't translate because of wrong channels." << endl; }
	else if (srcImg.rows != parameterImg.rows || srcImg.cols != parameterImg.cols) { cout << "ERROR! wiener_filter() : Can't translate because of wrong sizes." << endl; }
	else {
		//cout << " wiener_filter() : srcImg.rows = " << srcImg.rows << ", srcImg.cols = " << srcImg.cols << endl;	// 確認用
#pragma omp parallel for private(x)
		for (y = 0; y < srcImg.rows; y++) {
			for (x = 0; x < srcImg.cols; x++) {
				abs_in = AbsPowMat.at<Vec2d>(y, x);
				Filter_Parameter = parameterImg.at<Vec2d>(y, x);
				semi_in = abs_in + Filter_Parameter;
				divi_Vec2d(out, one, semi_in);
				Img.at<Vec2d>(y, x) = out;
				//cout << " " << out << " <- " << srcImg.at<Vec2d>(y, x) << endl;	// 確認用
			}
		}
	}

	mulSpectrums(Img, srcImg, Img, 0, true);	// 複素共役
	Img.copyTo(dstImg);
}


#endif