#ifndef __INCLUDED_H_DiscreteFourierTransform__
#define __INCLUDED_H_DiscreteFourierTransform__

#include "main.h"
#include "CalculateComplexNumber.h"

/* 定義する関数 */
void visualbule_complex(Mat& srcImg, Mat& dstImg);		// 複素画像の描画
void abs_complex(Mat& srcImg, Mat& dstImg);			// 2次元ベクトルの大きさ
void abs_pow_complex(Mat& srcImg, Mat& dstImg);		// 2次元ベクトルの大きさの２乗

void LowPathFilter(Mat& srcImg, Mat& dstImg, double Theredhold);	// 低周波数のみのフィルタに変更 (ローパスフィルター)

void wiener_filter(Mat& srcImg, Mat& dstImg, Mat& parameterImg);	// 2次元ベクトルの逆数(ウィーナ・フィルター)


/* 関数 */
// 複素画像の描画
void visualbule_complex(Mat& srcImg, Mat& dstImg) {
	Mat Img;
	srcImg.copyTo(Img);

	if (srcImg.channels() == 2) {
		/* 絶対値を計算してlogスケールにする */
		/*   => log(1 + sqrt( Re(DFT(SRC))^2 + Im(DFT(SRC))^2 )) */
		Mat planes[2];
		split(Img, planes);								// planes[0] = Re(DFT(SRC)), planes[1] = Im(DFT(SRC))
		magnitude(planes[0], planes[1], planes[0]);		// planes[0] = magnitude
		Mat mag = planes[0];
		mag += Scalar::all(1);
		// 結果の値は大きいものと小さいものが混ざっているのでlogを適用して抑制する
		log(mag, mag);

		/* 画像を表示可能にする */
		// 行・列が奇数の場合、クロップスする
		mag = mag(Rect(0, 0, mag.cols & -2, mag.rows & -2));
		// 画像の中央に原点が来るように象限を入れ替える
		int cx = mag.cols / 2;
		int cy = mag.rows / 2;
		Mat q0(mag, Rect(0, 0, cx, cy));	// 左上(第二象限)
		Mat q1(mag, Rect(cx, 0, cx, cy));	// 右上(第一象限)
		Mat q2(mag, Rect(0, cy, cx, cy));	// 左下(第三象限)
		Mat q3(mag, Rect(cx, cy, cx, cy));	// 右下(第四象限)

		Mat tmpImg;
		// 入れ替え(左上と右下)
		q0.copyTo(tmpImg);
		q3.copyTo(q0);
		tmpImg.copyTo(q3);
		// 入れ替え(右上と左下)
		q1.copyTo(tmpImg);
		q2.copyTo(q1);
		tmpImg.copyTo(q2);

		// 見ることができる値に変換(double[0,1]に変換)
		normalize(mag, mag, 0, 1, NORM_MINMAX);

		mag.convertTo(dstImg, CV_8U, 255);
	}
	else { cout << "ERROR! visualbule_complex() : Can't translate because of wrong channels. " << srcImg.channels() << endl; }
}


// 2次元ベクトルの大きさ
void abs_complex(Mat& srcImg, Mat& dstImg) {
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
void abs_pow_complex(Mat& srcImg, Mat& dstImg) {
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


// 低周波数のみのフィルタに変更 (ローパスフィルター)
void LowPathFilter(Mat& srcImg, Mat& dstImg, double Theredhold) {
	Mat Img;
	srcImg.copyTo(Img);
	int x, y, chan;
	double threshold;
	if (Theredhold != 0) { threshold = (double)Theredhold; }
	else {
		threshold = srcImg.cols / 2.0;
		if (threshold > (srcImg.rows / 2.0)) { threshold = srcImg.rows / 2.0; }
	}
	threshold = pow(threshold, 2);
	cout << "  LowPathFilter : threshold = " << threshold << endl;	// 確認
	double now_num1, now_num2, now_num3, now_num4;

	if (srcImg.channels() == 2) {
		Mat planes[2];
		split(Img, planes);

		// 低周波数のみ
#pragma omp parallel for private(x)
		for (y = 0; y < planes[0].rows; y++) {
			for (x = 0; x < planes[0].cols; x++) {
				now_num1 = (double)pow(x, 2);
				now_num2 = (double)pow((x - planes[0].cols), 2);
				now_num3 = (double)pow(y, 2);
				now_num4 = (double)pow((y - planes[0].rows), 2);
				if (now_num1 < threshold || now_num2 < threshold || now_num3 < threshold || now_num4 < threshold) {
					if (now_num1 < threshold) {
						planes[0].at<double>(y, x) = (double)planes[0].at<double>(y, x) / (double)now_num1;
						planes[1].at<double>(y, x) = (double)planes[1].at<double>(y, x) / (double)now_num1;
					}
					else if (now_num2 < threshold) {
						planes[0].at<double>(y, x) = (double)planes[0].at<double>(y, x) / (double)now_num2;
						planes[1].at<double>(y, x) = (double)planes[1].at<double>(y, x) / (double)now_num2;
					}
					else if (now_num3 < threshold) {
						planes[0].at<double>(y, x) = (double)planes[0].at<double>(y, x) / (double)now_num3;
						planes[1].at<double>(y, x) = (double)planes[1].at<double>(y, x) / (double)now_num3;
					}
					else if (now_num4 < threshold) {
						planes[0].at<double>(y, x) = (double)planes[0].at<double>(y, x) / (double)now_num4;
						planes[1].at<double>(y, x) = (double)planes[1].at<double>(y, x) / (double)now_num4;
					}
				}
			}
		}

		merge(planes, 2, dstImg);
	}
	else { cout << "ERROR! LowPassFilter() : Can't translate because of wrong channels." << endl; }
	cout << endl;
}

// 2次元ベクトルの逆数(ウィーナ・フィルター)
void wiener_filter(Mat& srcImg, Mat& dstImg, Mat& parameterImg) {
	Mat Img = Mat::zeros(srcImg.size(), CV_64FC2);

	Vec2d Filter_Parameter;
	int x, y;
	Vec2d abs_in, semi_in, out;
	Vec2d one = { 1.0, 0.0 };

	Mat AbsPowMat;
	abs_pow_complex(srcImg, AbsPowMat);

	if (srcImg.channels() != 2) { cout << "ERROR! wiener_filter() : Can't translate because of wrong channels." << endl; }
	else if (srcImg.rows != parameterImg.rows || srcImg.cols != parameterImg.cols) { cout << "ERROR! wiener_filter() : Can't translate because of wrong sizes." << endl; }
	else {
#pragma omp parallel for private(x)
		for (y = 0; y < srcImg.rows; y++) {
			for (x = 0; x < srcImg.cols; x++) {
				abs_in = AbsPowMat.at<Vec2d>(y, x);
				Filter_Parameter = parameterImg.at<Vec2d>(y, x);
				semi_in = abs_in + Filter_Parameter;
				divi_complex_2(out, one, semi_in);
				Img.at<Vec2d>(y, x) = out;
				//cout << " " << out << " <- " << srcImg.at<Vec2d>(y, x) << endl;	// 確認用
			}
		}
	}

	mulSpectrums(Img, srcImg, Img, 0, true);	// 複素共役
	Img.copyTo(dstImg);
}


#endif