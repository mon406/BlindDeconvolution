#ifndef __INCLUDED_H_ConjugateGradient__
#define __INCLUDED_H_ConjugateGradient__

#include "main.h"
#include "CalculateVector.h"
#include "CalculateComplexNumber.h"
#include "DiscreteFourierTransform.h"


/* 定義する関数 */
void CG_method(Mat& Ax, Mat& x, Mat& b, double& penalty);
void CG_method(Mat& Ax0, Mat& Ax1, Mat& Ax2, Mat& x0, Mat& x1, Mat& x2, Mat& b0, Mat& b1, Mat& b2);
void calc_Ap_FFT(Mat& Ax, Mat& x, Mat& p, Mat& dst);


/* 関数 */
void CG_method(Mat& Ax, Mat& x, Mat& b, double& penalty) {
	/* 設定値 */
	int MaxIteration = 10;
	double ErrorThreshold = 1.0e-04;
	double incr_Parameter = 2.0, decr_Parameter = 2.0, blanc_Parameter = 10.0;

	Mat Residual = Mat::zeros(x.size(), CV_64FC1);		// 残差ベクトル
	Mat Perpendicular = Mat::zeros(x.size(), CV_64FC1);	// 探索方向ベクトル
	double Alpha = 0.0;
	double Beta = 0.0;

	Mat New_x;
	x.copyTo(New_x);
	Mat AP = Mat::zeros(x.size(), CV_64FC1);
	Mat New_Residual;

	if (Ax.cols != x.cols || Ax.rows != x.rows || Ax.cols != b.cols || Ax.rows != b.rows) { cout << "ERROR! CG_method() : Can't translate because of wrong sizes." << endl; }
	else if (Ax.type() != CV_64FC1 || x.type() != CV_64FC1 || b.type() != CV_64FC1) { cout << "ERROR! CG_method() : Can't translate because of wrong channels." << endl; }
	else {
		/* 初期値設定 */
		Residual = b - Ax;
		Perpendicular = Residual;

		double energy = 0.0;
		/* 反復法 */
		for (int Iteration = 0; Iteration < MaxIteration; Iteration++) {
			/* Calculate Alpha */
			double Numerator = multi_vector(Residual, Perpendicular);		// ベクトルの内積
			calc_Ap_FFT(Ax, x, Perpendicular, AP);
			double Denominator = multi_vector(Perpendicular, AP);
			if (Denominator != 0) { Alpha = (double)(Numerator / Denominator); }
			else { cout << "WARNING! CG_method() : Can't calculate Alpha because of Denominator = 0" << endl; break; }

			/* Calculate x */
			New_x = x + Alpha * Perpendicular;

			/* Calculate Residual */
			New_Residual = Residual - Alpha * AP;
			energy = (double)norm(New_Residual);
			energy /= (double)((double)New_Residual.cols * (double)New_Residual.rows);
			//cout << "   energy = " << (double)energy << endl;	// 確認用
			if (energy < ErrorThreshold) { break; }

			/* Calculate Beta */
			double Numerator2 = multi_vector(New_Residual, New_Residual);		// ベクトルの内積
			double Denominator2 = multi_vector(Residual, Residual);
			if (Denominator2 != 0) { Beta = (double)(Numerator2 / Denominator2); }
			else { cout << "WARNING! CG_method() : Can't calculate Beta because of Denominator2 = 0" << endl; break; }

			/* Calculate Perpendicular */
			Perpendicular = New_Residual + Beta * Perpendicular;

			New_Residual.copyTo(Residual);
			New_x.copyTo(x);
			//double main_diff = (double)norm(New_Residual);
			//double sub_diff = (double)norm(Perpendicular) * (double)blanc_Parameter;
			////cout << "   main_diff = " << (double)main_diff << " , sub_diff = " << (double)sub_diff << endl;	// 確認用
			//if (main_diff > sub_diff) { penalty *= incr_Parameter; }
			//else if (main_diff < sub_diff) { penalty /= decr_Parameter; }
		}
		cout << "   energy = " << (double)energy << endl;	// 確認用
	}

	New_x.copyTo(x);
}

void CG_method(Mat& Ax0, Mat& Ax1, Mat& Ax2, Mat& x0, Mat& x1, Mat& x2, Mat& b0, Mat& b1, Mat& b2) {
	/* 設定値 */
	int MaxIteration = 100;
	double ErrorThreshold = 1.0e-04;

	Mat Residual[3] = { Mat::zeros(x0.size(), CV_64FC1), Mat::zeros(x1.size(), CV_64FC1), Mat::zeros(x2.size(), CV_64FC1) };		// 残差ベクトル
	Mat Perpendicular[3] = { Mat::zeros(x0.size(), CV_64FC1), Mat::zeros(x1.size(), CV_64FC1), Mat::zeros(x2.size(), CV_64FC1) };	// 探索方向ベクトル
	double Alpha[3] = { 0.0, 0.0, 0.0 };
	double Beta[3] = { 0.0, 0.0, 0.0 };

	Mat New_x[3];
	x0.copyTo(New_x[0]);
	x1.copyTo(New_x[1]);
	x2.copyTo(New_x[2]);
	Mat AP[3] = { Mat::zeros(x0.size(), CV_64FC1), Mat::zeros(x1.size(), CV_64FC1), Mat::zeros(x2.size(), CV_64FC1) };
	Mat New_Residual[3];

	if (Ax1.cols != x1.cols || Ax1.rows != x1.rows || Ax1.cols != b1.cols || Ax1.rows != b1.rows) { cout << "ERROR! CG_method() : Can't translate because of wrong sizes." << endl; }
	else if (Ax1.type() != CV_64FC1 || x1.type() != CV_64FC1 || b1.type() != CV_64FC1) { cout << "ERROR! CG_method() : Can't translate because of wrong channels." << endl; }
	else {
		int c;
		Mat Ax[3], x[3];
		Ax0.copyTo(Ax[0]);
		Ax1.copyTo(Ax[1]);
		Ax2.copyTo(Ax[2]);
		x0.copyTo(x[0]);
		x1.copyTo(x[1]);
		x2.copyTo(x[2]);

		/* 初期値設定 */
		Residual[0] = b0 - Ax0;
		Residual[1] = b1 - Ax1;
		Residual[2] = b2 - Ax2;
		double before_color_mean[3], after_color_mean[3];
		for (c = 0; c < 3; c++) {
			before_color_mean[c] = (double)mean(x[c])[0];
			Residual[c].copyTo(Perpendicular[c]);
		}

		/* 反復法 */
		for (int Iteration = 0; Iteration < MaxIteration; Iteration++) {
			double energy = 0.0;
			//cout << "  " << (int)Iteration;	// 確認用
			for (c = 0; c < 3; c++) {
				/* Calculate Alpha */
				double Numerator = multi_vector(Residual[c], Perpendicular[c]);		// ベクトルの内積
				calc_Ap_FFT(Ax[c], x[c], Perpendicular[c], AP[c]);
				double Denominator = multi_vector(Perpendicular[c], AP[c]);
				Alpha[c] = (double)(Numerator / Denominator);

				/* Calculate x */
				New_x[c] = x[c] + Alpha[c] * Perpendicular[c];

				/* Calculate Residual */
				New_Residual[c] = Residual[c] - Alpha[c] * AP[c];
				energy = (double)norm(New_Residual[c]);
				energy /= (double)((double)New_Residual[c].cols * (double)New_Residual[c].rows);
				//cout << " :: " << (int)c << " : energy = " << (double)energy;	// 確認用
				if (energy < ErrorThreshold || Iteration == MaxIteration - 1) {
					cout << "  " << (int)Iteration << " : " << (int)c << " : energy = " << (double)energy << endl;	// 確認用
					break;
				}

				/* Calculate Beta */
				double Numerator2 = multi_vector(New_Residual[c], New_Residual[c]);		// ベクトルの内積
				double Denominator2 = multi_vector(Residual[c], Residual[c]);
				Beta[c] = (double)(Numerator2 / Denominator2);

				/* Calculate Perpendicular */
				Perpendicular[c] = New_Residual[c] + Beta[c] * Perpendicular[c];

				New_Residual[c].copyTo(Residual[c]);
				New_x[c].copyTo(x[c]);
			}
			//cout << endl;	// 確認用
		}

		for (c = 0; c < 3; c++) {
			after_color_mean[c] = (double)mean(New_x[c])[0];
			New_x[c] *= (double)(before_color_mean[c] / after_color_mean[c]);
			//normalize(New_x[c], New_x[c], 0, 255, NORM_MINMAX);
		}
	}

	New_x[0].copyTo(x0);
	New_x[1].copyTo(x1);
	New_x[2].copyTo(x2);
}

void calc_Ap_FFT(Mat& Ax, Mat& x, Mat& p, Mat& dst) {
	/* 画像をCV_64Fに変換(前処理) */
	Mat double_Ax, double_x, double_p;
	Ax.convertTo(double_Ax, CV_64FC1);
	x.convertTo(double_x, CV_64FC1);
	p.convertTo(double_p, CV_64FC1);

	// DFT変換のサイズを計算
	int Mplus = double_Ax.rows + double_x.rows;
	int Nplus = double_Ax.cols + double_x.cols;
	int Msize = getOptimalDFTSize(Mplus);
	int Nsize = getOptimalDFTSize(Nplus);
	//cout << "  FFT Size  : (" << Mplus << "," << Nplus << ") => (" << Msize << "," << Nsize << ")" << endl;	// 確認

	Mat double_Ax_sub, double_x_sub, double_p_sub;
	Mat planes_Ax[] = { Mat_<double>(double_Ax), Mat::zeros(double_Ax.size(), CV_64F) };
	merge(planes_Ax, 2, double_Ax_sub);
	Mat planes_x[] = { Mat_<double>(double_x), Mat::zeros(double_x.size(), CV_64F) };
	merge(planes_x, 2, double_x_sub);
	Mat planes_p[] = { Mat_<double>(double_p), Mat::zeros(double_p.size(), CV_64F) };
	merge(planes_p, 2, double_p_sub);

	/* DFT */
	Mat dft_double_Ax = Mat::zeros(Msize, Nsize, CV_64FC2);
	Mat dft_double_x = Mat::zeros(Msize, Nsize, CV_64FC2);
	Mat dft_double_p = Mat::zeros(Msize, Nsize, CV_64FC2);
	Mat dft_doubleQuantImg[3] = { Mat::zeros(Msize, Nsize, CV_64FC2), Mat::zeros(Msize, Nsize, CV_64FC2), Mat::zeros(Msize, Nsize, CV_64FC2) };
	copyMakeBorder(double_Ax_sub, dft_double_Ax, double_x.rows / 2, Msize - Mplus + double_x.rows / 2, double_x.cols / 2, Nsize - Nplus + double_x.cols / 2, BORDER_CONSTANT, (0.0, 0.0));
	dft(dft_double_Ax, dft_double_Ax);
	copyMakeBorder(double_x_sub, dft_double_x, double_Ax.rows / 2, Msize - Mplus + double_Ax.rows / 2, double_Ax.cols / 2, Nsize - Nplus + double_Ax.cols / 2, BORDER_REPLICATE);
	dft(dft_double_x, dft_double_x);
	copyMakeBorder(double_p_sub, dft_double_p, double_Ax.rows / 2, Msize - Mplus + double_Ax.rows / 2, double_Ax.cols / 2, Nsize - Nplus + double_Ax.cols / 2, BORDER_REPLICATE);
	dft(dft_double_p, dft_double_p);

	/* Apを求める */
	Mat dft_Ap, dft_A;
	divi_complex(dft_double_Ax, dft_double_x, dft_A);
	mulSpectrums(dft_A, dft_double_p, dft_Ap, 0, false);

	/* inverseDFT */
	Mat Ap;
	dft(dft_Ap, dft_Ap, cv::DFT_INVERSE + cv::DFT_SCALE | cv::DFT_REAL_OUTPUT);
	//dft_Ap.convertTo(Image_dst_deblurred2, CV_8UC1);	// 確認
	Ap = dft_Ap(Rect(double_x.cols / 2, double_x.rows / 2, double_Ax.cols, double_Ax.rows));

	Ap.copyTo(dst);
}


#endif