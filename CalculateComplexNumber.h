#ifndef __INCLUDED_H_CalculateComplexNumber__
#define __INCLUDED_H_CalculateComplexNumber__

#include "main.h"

/* ’è‹`‚·‚éŠÖ” */
void multi_complex_2(Vec2d& answer, Vec2d& complex1, Vec2d& complex2);	// ‚Q‚Â‚Ì•¡‘f”‚ÌæZ
void divi_complex_2(Vec2d& answer, Vec2d& complex1, Vec2d& complex2);	// ‚Q‚Â‚Ì•¡‘f”‚ÌœZ

void multi_complex(Mat& srcImg, Mat& srcImg2, Mat& dstImg);			// 2ŸŒ³ƒxƒNƒgƒ‹‚ÌæZ
void divi_complex(Mat& srcImg, Mat& srcImg2, Mat& dstImg);			// 2ŸŒ³ƒxƒNƒgƒ‹‚ÌœZ
void mluti_divi_complex(Mat& srcImg, Mat& srcImg2, Mat& dstImg);	// 2ŸŒ³ƒxƒNƒgƒ‹‚ÌæœZ

void reciprocal_complex(Mat& srcImg, Mat& dstImg);					// 2ŸŒ³ƒxƒNƒgƒ‹‚Ì‹t”
void complex_conjugate(Mat& srcImg, Mat& dstImg);					// 2ŸŒ³ƒxƒNƒgƒ‹(•¡‘f”)‚Ì•¡‘f‹¤–ğ


/* ŠÖ” */
// •¡‘f”‚ÌŠ|‚¯Z
void multi_complex_2(Vec2d& answer, Vec2d& complex1, Vec2d& complex2) {
	Vec2d result = { 0.0, 0.0 };

	result[0] = ((double)complex1[0] * (double)complex2[0]) - ((double)complex1[1] * (double)complex2[1]);
	result[1] = ((double)complex1[1] * (double)complex2[0]) + ((double)complex1[0] * (double)complex2[1]);

	answer = result;
}
// •¡‘f”‚ÌŠ„‚èZ
void divi_complex_2(Vec2d& answer, Vec2d& complex1, Vec2d& complex2) {
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

// 2ŸŒ³ƒxƒNƒgƒ‹‚ÌæZ
void multi_complex(Mat& srcImg, Mat& srcImg2, Mat& dstImg) {
	Mat Img = Mat::zeros(srcImg.size(), CV_64FC2);
	int x, y;
	Vec2d in, in2, out;

	if (srcImg.cols != srcImg2.cols || srcImg.rows != srcImg2.rows) { cout << "ERROR! multi_complex() : Can't translate because of wrong sizes." << endl; }
	if (srcImg.channels() == 2) {
#pragma omp parallel for private(x)
		for (y = 0; y < srcImg.rows; y++) {
			for (x = 0; x < srcImg.cols; x++) {
				in = srcImg.at<Vec2d>(y, x);
				in2 = srcImg2.at<Vec2d>(y, x);
				multi_complex_2(out, in, in2);
				Img.at<Vec2d>(y, x) = out;
			}
		}

		Img.copyTo(dstImg);
	}
	else { cout << "ERROR! multi_complex() : Can't translate because of wrong channels." << endl; }
}
// 2ŸŒ³ƒxƒNƒgƒ‹‚ÌœZ
void divi_complex(Mat& srcImg, Mat& srcImg2, Mat& dstImg) {
	Mat Img = Mat::zeros(srcImg.size(), CV_64FC2);
	int x, y;
	Vec2d in, in2, out;

	if (srcImg.cols != srcImg2.cols || srcImg.rows != srcImg2.rows) { cout << "ERROR! multi_complex() : Can't translate because of wrong sizes." << endl; }
	if (srcImg.channels() == 2) {
#pragma omp parallel for private(x)
		for (y = 0; y < srcImg.rows; y++) {
			for (x = 0; x < srcImg.cols; x++) {
				in = srcImg.at<Vec2d>(y, x);
				in2 = srcImg2.at<Vec2d>(y, x);
				divi_complex_2(out, in, in2);
				Img.at<Vec2d>(y, x) = out;
			}
		}

		Img.copyTo(dstImg);
	}
	else { cout << "ERROR! divi_complex() : Can't translate because of wrong channels." << endl; }
}
// 2ŸŒ³ƒxƒNƒgƒ‹‚ÌæœZ
void mluti_divi_complex(Mat& srcImg, Mat& srcImg2, Mat& dstImg) {
	Mat Img = Mat::zeros(srcImg.size(), CV_64FC2);
	int x, y;
	Vec2d in, in2, out;
	Vec2d zero = { 0.0, 0.0 };
	Vec2d error_num = { 0.0, 0.0 };
	double all_num = (double)srcImg.rows * (double)srcImg.cols;
	Vec2d all_number = { all_num, 0.0 };

	if (srcImg.cols != srcImg2.cols || srcImg.rows != srcImg2.rows) { cout << "ERROR! multi_complex() : Can't translate because of wrong sizes." << endl; }
	if (srcImg.channels() == 2) {
#pragma omp parallel for private(x)
		for (y = 0; y < srcImg.rows; y++) {
			for (x = 0; x < srcImg.cols; x++) {
				in = srcImg.at<Vec2d>(y, x);
				in2 = srcImg2.at<Vec2d>(y, x);
				multi_complex_2(out, in, in2);
				divi_complex_2(out, out, in2);
				Img.at<Vec2d>(y, x) = out;
				//cout << " in = "<< in << "  =>  out = " << out << endl;
				error_num += out - in;
				//if (error_num != zero) { cout << " in = " << in << "  =>  out = " << out << endl; }
				//cout << " error = " << error_num << endl;
			}
		}
		divi_complex_2(error_num, error_num, all_number);
		//cout << " sum error = " << error_num << endl;

		Img.copyTo(dstImg);
	}
	else { cout << "ERROR! mluti_divi_complex() : Can't translate because of wrong channels." << endl; }
}

// 2ŸŒ³ƒxƒNƒgƒ‹‚Ì‹t”
void reciprocal_complex(Mat& srcImg, Mat& dstImg) {
	Mat Img = Mat::zeros(srcImg.size(), CV_64FC2);
	int x, y;
	Vec2d in, out;
	Vec2d one = { 1.0, 0.0 };

	if (srcImg.channels() == 2) {
#pragma omp parallel for private(x)
		for (y = 0; y < srcImg.rows; y++) {
			for (x = 0; x < srcImg.cols; x++) {
				in = srcImg.at<Vec2d>(y, x);
				divi_complex_2(out, one, in);
				Img.at<Vec2d>(y, x) = out;
			}
		}

		Img.copyTo(dstImg);
	}
	else { cout << "ERROR! reciprocal_complex() : Can't translate because of wrong channels." << endl; }
}

// 2ŸŒ³ƒxƒNƒgƒ‹(•¡‘f”)‚Ì•¡‘f‹¤–ğ
void complex_conjugate(Mat& srcImg, Mat& dstImg) {
	Mat Img = Mat::zeros(srcImg.size(), CV_64FC2);
	int x, y;
	Vec2d in, out;

	if (srcImg.channels() == 2) {
#pragma omp parallel for private(x)
		for (y = 0; y < srcImg.rows; y++) {
			for (x = 0; x < srcImg.cols; x++) {
				in = srcImg.at<Vec2d>(y, x);
				out[0] = in[0];
				out[1] = -in[1];
				Img.at<Vec2d>(y, x) = out;
			}
		}

		Img.copyTo(dstImg);
	}
	else { cout << "ERROR! complex_conjugate() : Can't calculate because of wrong channels." << endl; }
}


#endif