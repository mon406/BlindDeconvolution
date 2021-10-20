#ifndef __INCLUDED_H_ConjugateGradient__
#define __INCLUDED_H_ConjugateGradient__

#include "main.h"
#include "CalculateVector.h"
#include "CalculateComplexNumber.h"
#include "DiscreteFourierTransform.h"


/* ��`����֐� */
void CG_method(Mat& Ax, Mat& x, Mat& b);
void CG_method(Mat& Ax0, Mat& Ax1, Mat& Ax2, Mat& x0, Mat& x1, Mat& x2, Mat& b0, Mat& b1, Mat& b2);
void calc_Ap_FFT(Mat& Ax, Mat& x, Mat& p, Mat& dst);


/* �֐� */
void CG_method(Mat& Ax, Mat& x, Mat& b) {
	/* �ݒ�l */
	int MaxIteration = 100;
	double ErrorThreshold = 1.0e-04;

	Mat Residual = Mat::zeros(x.size(), CV_64FC1);		// �c���x�N�g��
	Mat Perpendicular = Mat::zeros(x.size(), CV_64FC1);	// �T�������x�N�g��
	double Alpha = 0.0;
	double Beta = 0.0;

	Mat New_x;
	x.copyTo(New_x);
	Mat AP = Mat::zeros(x.size(), CV_64FC1);
	Mat New_Residual;

	if (Ax.cols != x.cols || Ax.rows != x.rows || Ax.cols != b.cols || Ax.rows != b.rows) { cout << "ERROR! CG_method() : Can't translate because of wrong sizes." << endl; }
	else if (Ax.type() != CV_64FC1 || x.type() != CV_64FC1 || b.type() != CV_64FC1) { cout << "ERROR! CG_method() : Can't translate because of wrong channels." << endl; }
	else {
		/* �����l�ݒ� */
		Residual = b - Ax;
		Perpendicular = Residual;

		double energy = 0.0;
		/* �����@ */
		for (int Iteration = 0; Iteration < MaxIteration; Iteration++) {
			/* Calculate Alpha */
			double Numerator = multi_vector(Residual, Perpendicular);		// �x�N�g���̓���
			calc_Ap_FFT(Ax, x, Perpendicular, AP);
			double Denominator = multi_vector(Perpendicular, AP);
			Alpha = (double)(Numerator / Denominator);

			/* Calculate x */
			New_x = x + Alpha * Perpendicular;

			/* Calculate Residual */
			New_Residual = Residual - Alpha * AP;
			energy = (double)norm(New_Residual);
			energy /= (double)((double)New_Residual.cols * (double)New_Residual.rows);
			if (energy < ErrorThreshold) { break; }

			/* Calculate Beta */
			double Numerator2 = multi_vector(New_Residual, New_Residual);		// �x�N�g���̓���
			double Denominator2 = multi_vector(Residual, Residual);
			Beta = (double)(Numerator2 / Denominator2);

			/* Calculate Perpendicular */
			Perpendicular = New_Residual + Beta * Perpendicular;

			New_Residual.copyTo(Residual);
			New_x.copyTo(x);
		}
	}

	New_x.copyTo(x);
}

void CG_method(Mat& Ax0, Mat& Ax1, Mat& Ax2, Mat& x0, Mat& x1, Mat& x2, Mat& b0, Mat& b1, Mat& b2) {
	/* �ݒ�l */
	int MaxIteration = 100;
	double ErrorThreshold = 1.0e-04;

	Mat Residual[3] = { Mat::zeros(x0.size(), CV_64FC1), Mat::zeros(x1.size(), CV_64FC1), Mat::zeros(x2.size(), CV_64FC1) };		// �c���x�N�g��
	Mat Perpendicular[3] = { Mat::zeros(x0.size(), CV_64FC1), Mat::zeros(x1.size(), CV_64FC1), Mat::zeros(x2.size(), CV_64FC1) };	// �T�������x�N�g��
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

		/* �����l�ݒ� */
		Residual[0] = b0 - Ax0;
		Residual[1] = b1 - Ax1;
		Residual[2] = b2 - Ax2;
		for (c = 0; c < 3; c++) {
			Residual[c].copyTo(Perpendicular[c]);
		}

		double energy = 0.0;
		/* �����@ */
		//for (int Iteration = 0; Iteration < MaxIteration; Iteration++) {
		//	for (c = 0; c < 3; c++) {
		//		/* Calculate Alpha */
		//		double Numerator = multi_vector(Residual, Perpendicular);		// �x�N�g���̓���
		//		calc_Ap_FFT(Ax[c], x[c], Perpendicular, AP);
		//		double Denominator = multi_vector(Perpendicular, AP);
		//		Alpha = (double)(Numerator / Denominator);

		//		/* Calculate x */
		//		New_x = x + Alpha * Perpendicular;

		//		/* Calculate Residual */
		//		New_Residual = Residual - Alpha * AP;
		//		energy = (double)norm(New_Residual);
		//		energy /= (double)((double)New_Residual.cols * (double)New_Residual.rows);
		//		if (energy < ErrorThreshold) { break; }

		//		/* Calculate Beta */
		//		double Numerator2 = multi_vector(New_Residual, New_Residual);		// �x�N�g���̓���
		//		double Denominator2 = multi_vector(Residual, Residual);
		//		Beta = (double)(Numerator2 / Denominator2);

		//		/* Calculate Perpendicular */
		//		Perpendicular = New_Residual + Beta * Perpendicular;

		//		New_Residual.copyTo(Residual);
		//		New_x.copyTo(x);
		//	}
		//}
	}

	//New_x.copyTo(x);
}

void calc_Ap_FFT(Mat& Ax, Mat& x, Mat& p, Mat& dst) {
	/* �摜��CV_64F�ɕϊ�(�O����) */
	Mat double_Ax, double_x, double_p;
	Ax.convertTo(double_Ax, CV_64FC1);
	x.convertTo(double_x, CV_64FC1);
	p.convertTo(double_p, CV_64FC1);

	// DFT�ϊ��̃T�C�Y���v�Z
	int Mplus = double_Ax.rows + double_x.rows;
	int Nplus = double_Ax.cols + double_x.cols;
	int Msize = getOptimalDFTSize(Mplus);
	int Nsize = getOptimalDFTSize(Nplus);
	//cout << "  FFT Size  : (" << Mplus << "," << Nplus << ") => (" << Msize << "," << Nsize << ")" << endl;	// �m�F

	/* DFT */
	Mat dft_double_Ax = Mat::zeros(Msize, Nsize, CV_64FC2);
	Mat dft_double_x = Mat::zeros(Msize, Nsize, CV_64FC2);
	Mat dft_double_p = Mat::zeros(Msize, Nsize, CV_64FC2);
	Mat dft_doubleQuantImg[3] = { Mat::zeros(Msize, Nsize, CV_64FC2), Mat::zeros(Msize, Nsize, CV_64FC2), Mat::zeros(Msize, Nsize, CV_64FC2) };
	copyMakeBorder(double_Ax, dft_double_Ax, double_x.rows / 2, Msize - Mplus + double_x.rows / 2, double_x.cols / 2, Nsize - Nplus + double_x.cols / 2, BORDER_CONSTANT, (0.0, 0.0));
	dft(dft_double_Ax, dft_double_Ax);
	copyMakeBorder(double_x, dft_double_x, double_Ax.rows / 2, Msize - Mplus + double_Ax.rows / 2, double_Ax.cols / 2, Nsize - Nplus + double_Ax.cols / 2, BORDER_REPLICATE);
	dft(dft_double_x, dft_double_x);
	copyMakeBorder(double_p, dft_double_p, double_Ax.rows / 2, Msize - Mplus + double_Ax.rows / 2, double_Ax.cols / 2, Nsize - Nplus + double_Ax.cols / 2, BORDER_REPLICATE);
	dft(dft_double_p, dft_double_p);

	/* Ap�����߂� */
	Mat dft_Ap, dft_A;
	divi_complex(dft_double_Ax, dft_double_x, dft_A);
	mulSpectrums(dft_A, dft_double_p, dft_Ap, 0, false);

	/* inverseDFT */
	Mat Ap;
	dft(dft_Ap, dft_Ap, cv::DFT_INVERSE + cv::DFT_SCALE | cv::DFT_REAL_OUTPUT);
	Ap = dft_Ap(Rect(double_x.cols / 2, double_x.rows / 2, double_Ax.cols, double_Ax.rows));

	Ap.copyTo(dst);
}


#endif