#pragma once

/* �g�p�f�B���N�g���w��y�ђ�` */
#define _CRT_SECURE_NO_WARNINGS
#define _USE_MATH_DEFINES
#include <iostream>
#include <cmath>
#include <cstdlib>
#include <vector>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <time.h>

using namespace std;
using namespace cv;
string win_src = "src";				// ���͉摜�E�B���h�E
string win_dst = "dst";				// �o�͉摜�E�B���h�E
string win_dst2 = "dst_kernel";
string win_dst3 = "dst_deblurred";
string win_dst4 = "dst_deblurred2";

/* ���o�͉摜 */
Mat Image_src;				// ���͉摜
Mat Image_src_gray;			// ���͉摜(�O���[�X�P�[��)
//Mat Image_src_hist;		// ���̓q�X�g�O�����摜
//Mat Image_dst_hist;		// �o�̓q�X�g�O�����摜
Mat Image_dst;				// �o�͉摜(�ڂ��摜)
Mat Image_dst_deblurred;	// �o�͕�C�摜(��C�摜)
Mat Image_dst_deblurred2;
Mat Image_kernel_original;	// �J�[�l��(�t�B���^�[)
Mat Image_kernel;

/* �萔 */
int MAX_INTENSE = 255;	// �ő�F�l
int WIDTH;				// ���͉摜�̉����i�s�N�Z�����j
int HEIGHT;				// ���͉摜�̏c���i�s�N�Z�����j
int MAX_DATA;			// ���͉摜�̑��s�N�Z����

/* �ϐ� */
Mat GammaMat;

/* ���o�͊֐� */
void Input_Image();			// �摜�̓���
void Output_Image();		// �摜�̏o��


/* �֐��錾 */
void checkMat(Mat& checkImg);								// Mat�֐��̃T�C�Y�ƌ^�̊m�F
void SSIMcalc(double& ssim, Mat& image_1, Mat& image_2);
void Evaluation_MSE_PSNR_SSIM(Mat& Original, Mat& Inpaint);	// MSE&PSNR&SSIM�ɂ��摜�]��
void DrawHist(Mat& targetImg, Mat& image_hist);				// �q�X�g�O�����v�Z&�`��


/* �֐���` */
// Mat�֐��̊m�F
void checkMat(Mat& checkImg) {
	cout << "##############################" << endl;
	cout << " cols = " << checkImg.cols << endl;
	cout << " rows = " << checkImg.rows << endl;
	cout << " type = " << (
		checkImg.type() == CV_8UC1 ? "CV_8UC1" :
		checkImg.type() == CV_8UC3 ? "CV_8UC3" :
		checkImg.type() == CV_64FC1 ? "CV_64FC1" :
		checkImg.type() == CV_64FC2 ? "CV_64FC2" :
		checkImg.type() == CV_64FC3 ? "CV_64FC3" :
		"other"
		) << endl;
	cout << "##############################" << endl;
	cout << endl;
}
void checkMat_detail(Mat& checkImg) {
	cout << "##############################" << endl;
	cout << " cols = " << checkImg.cols << endl;
	cout << " rows = " << checkImg.rows << endl;
	cout << " type = " << (
		checkImg.type() == CV_8UC1 ? "CV_8UC1" :
		checkImg.type() == CV_8UC3 ? "CV_8UC3" :
		checkImg.type() == CV_64FC1 ? "CV_64FC1" :
		checkImg.type() == CV_64FC2 ? "CV_64FC2" :
		checkImg.type() == CV_64FC3 ? "CV_64FC3" :
		"other"
		) << endl;
	cout << "##############################" << endl;
	cout << checkImg << endl;
	cout << endl;
}

// MSE&PSNR&SSIM�ɂ��摜�]��
void Evaluation_MSE_PSNR_SSIM(Mat& Original, Mat& Inpaint) {
	double MSE = 0.0, PSNR = 0.0, SSIM = 0.0;
	Mat beforeIMG, afterIMG;
	Original.copyTo(beforeIMG);
	Inpaint.copyTo(afterIMG);

	double MSE_sum = 0.0;	// MSE�l
	double image_cost;		// ��f�l�̍���
	int compare_size = 1, color_ind;
	int occ_pix_count = 0;

	if (Original.size() != Inpaint.size()) {
		cout << "ERROR! MSE_PSNR_SSIM() : Can't calculate because of wrong size." << endl;
		cout << "       => " << Original.size() << " & " << Inpaint.size() << endl;
	}
	else if (Original.channels() != Inpaint.channels()) { cout << "ERROR! MSE_PSNR_SSIM() : Can't calculate because of wrong channels." << endl; }
	else {
		if (Original.channels() == 3) {
			/* MSE�v�Z(RGB) */
			for (int i = 0; i < Original.rows; i++) {
				for (int j = 0; j < Original.cols; j++) {
					image_cost = 0.0;
					color_ind = i * Original.cols * 3 + j * 3;
					for (int k = 0; k < 3; k++) {
						image_cost += pow((double)Inpaint.data[color_ind] - (double)Original.data[color_ind], 2.0);
						color_ind++;
					}
					MSE_sum += (double)image_cost;
					occ_pix_count++;
				}
			}
			compare_size = occ_pix_count * 3;
		}
		else if (Original.channels() == 1) {
			/* MSE�v�Z(Grayscale) */
			for (int i = 0; i < Original.rows; i++) {
				for (int j = 0; j < Original.cols; j++) {
					image_cost = 0.0;
					color_ind = i * Original.cols + j;
					image_cost += pow((double)Inpaint.data[color_ind] - (double)Original.data[color_ind], 2.0);
					MSE_sum += (double)image_cost;
					occ_pix_count++;
				}
			}
			compare_size = occ_pix_count;
		}
		MSE = (double)MSE_sum / (double)compare_size;

		/* PSNR�v�Z */
		if (MSE == 0) { PSNR = -1; }
		else { PSNR = 20 * (double)log10(MAX_INTENSE) - 10 * (double)log10(MSE); }

		/* SSIM�v�Z */
		SSIMcalc(SSIM, beforeIMG, afterIMG);

		/* �]�����ʕ\�� */
		cout << "--- �]�� ------------------------------------------" << endl;
		cout << " MSE  : " << MSE << endl;
		if (PSNR >= 0) { cout << " PSNR : " << PSNR << endl; }
		else { cout << " PSNR : inf" << endl; }
		cout << " SSIM : " << SSIM << endl;
		cout << "---------------------------------------------------" << endl;
	}
	cout << endl;
}
// SSIM�Z�o
void SSIMcalc(double& ssim, Mat& image_1, Mat& image_2) {
	const double C1 = pow(0.01 * 255, 2), C2 = pow(0.03 * 255, 2);

	Mat I1, I2;
	image_1.convertTo(I1, CV_32F);	// cannot calculate on one byte large values
	image_2.convertTo(I2, CV_32F);
	Mat I2_2 = I2.mul(I2);			// I2^2
	Mat I1_2 = I1.mul(I1);			// I1^2
	Mat I1_I2 = I1.mul(I2);			// I1 * I2

	Mat mu1, mu2;   // PRELIMINARY COMPUTING
	GaussianBlur(I1, mu1, Size(11, 11), 1.5);
	GaussianBlur(I2, mu2, Size(11, 11), 1.5);
	Mat mu1_2 = mu1.mul(mu1);
	Mat mu2_2 = mu2.mul(mu2);
	Mat mu1_mu2 = mu1.mul(mu2);

	Mat sigma1_2, sigma2_2, sigma12;
	GaussianBlur(I1_2, sigma1_2, Size(11, 11), 1.5);
	sigma1_2 -= mu1_2;
	GaussianBlur(I2_2, sigma2_2, Size(11, 11), 1.5);
	sigma2_2 -= mu2_2;
	GaussianBlur(I1_I2, sigma12, Size(11, 11), 1.5);
	sigma12 -= mu1_mu2;

	Mat t1, t2, t3;
	t1 = 2 * mu1_mu2 + C1;
	t2 = 2 * sigma12 + C2;
	t3 = t1.mul(t2);              // t3 = ((2*mu1_mu2 + C1).*(2*sigma12 + C2))
	t1 = mu1_2 + mu2_2 + C1;
	t2 = sigma1_2 + sigma2_2 + C2;
	t1 = t1.mul(t2);               // t1 =((mu1_2 + mu2_2 + C1).*(sigma1_2 + sigma2_2 + C2))

	Mat ssim_map;
	divide(t3, t1, ssim_map);      // ssim_map =  t3./t1;
	Scalar mssim = mean(ssim_map); // mssim = average of ssim map

	/* SSIM����(RGB or Gray) */
	double SSIM;
	SSIM = (double)mssim[0] + (double)mssim[1] + (double)mssim[2];
	if (image_1.channels() == 3) { SSIM = (double)SSIM / 3.0; }
	ssim = (double)SSIM;
}

// �q�X�g�O�����v�Z&�`��
void DrawHist(Mat& targetImg, Mat& image_hist) {
	if (targetImg.channels() == 3) {
		Mat channels[3];
		int cha_index, ha_index;
		uchar Gray;
		for (int channel = 0; channel < 3; channel++) {
			channels[channel] = Mat(Size(targetImg.cols, targetImg.rows), CV_8UC1);
			for (int i = 0; i < targetImg.rows; i++) {
				for (int j = 0; j < targetImg.cols; j++) {
					cha_index = i * targetImg.cols * 3 + j * 3 + channel;
					ha_index = i * targetImg.cols + j;
					Gray = (uchar)targetImg.data[cha_index];
					channels[channel].data[ha_index] = Gray;
				}
			}
		}

		/* �ϐ��錾 */
		Mat R, G, B;
		int hist_size = 256;
		float range[] = { 0, 256 };
		const float* hist_range = range;

		/* ��f���𐔂��� */
		calcHist(&channels[0], 1, 0, Mat(), B, 1, &hist_size, &hist_range);
		calcHist(&channels[1], 1, 0, Mat(), G, 1, &hist_size, &hist_range);
		calcHist(&channels[2], 1, 0, Mat(), R, 1, &hist_size, &hist_range);

		/* �m�F�i�q�X�g�O���������Œ�̂��߁j*/
		int MAX_COUNT = 0;
		double Min_count[3], Max_count[3];
		for (int ch = 0; ch < 3; ch++) {
			if (ch == 0) { minMaxLoc(B, &Min_count[ch], &Max_count[ch]); }
			else if (ch == 1) { minMaxLoc(G, &Min_count[ch], &Max_count[ch]); }
			else if (ch == 2) { minMaxLoc(R, &Min_count[ch], &Max_count[ch]); }
			if (Max_count[ch] > MAX_COUNT) {
				MAX_COUNT = (int)Max_count[ch];
			}
		}
		//MAX_COUNT = 80000;

		/* �q�X�g�O���������p�̉摜���쐬 */
		image_hist = Mat(Size(276, 320), CV_8UC3, Scalar(255, 255, 255));

		/* �w�i��`��i���₷�����邽�߂Ƀq�X�g�O���������̔w�i���O���[�ɂ���j */
		for (int i = 0; i < 3; i++) {
			rectangle(image_hist, Point(10, 10 + 100 * i), Point(265, 100 + 100 * i), Scalar(230, 230, 230), -1);
		}

		for (int i = 0; i < 256; i++) {
			line(image_hist, Point(10 + i, 100), Point(10 + i, 100 - (int)((float)(R.at<float>(i) / MAX_COUNT) * 80)), Scalar(0, 0, 255), 1, 8, 0);
			line(image_hist, Point(10 + i, 200), Point(10 + i, 200 - (int)((float)(G.at<float>(i) / MAX_COUNT) * 80)), Scalar(0, 255, 0), 1, 8, 0);
			line(image_hist, Point(10 + i, 300), Point(10 + i, 300 - (int)((float)(B.at<float>(i) / MAX_COUNT) * 80)), Scalar(255, 0, 0), 1, 8, 0);

			if (i % 10 == 0) {		// ����10�����C��������
				line(image_hist, Point(10 + i, 100), Point(10 + i, 10),
					Scalar(170, 170, 170), 1, 8, 0);
				line(image_hist, Point(10 + i, 200), Point(10 + i, 110),
					Scalar(170, 170, 170), 1, 8, 0);
				line(image_hist, Point(10 + i, 300), Point(10 + i, 210),
					Scalar(170, 170, 170), 1, 8, 0);

				if (i % 50 == 0) {	// ����50���Z�����C��������
					line(image_hist, Point(10 + i, 100), Point(10 + i, 10),
						Scalar(50, 50, 50), 1, 8, 0);
					line(image_hist, Point(10 + i, 200), Point(10 + i, 110),
						Scalar(50, 50, 50), 1, 8, 0);
					line(image_hist, Point(10 + i, 300), Point(10 + i, 210),
						Scalar(50, 50, 50), 1, 8, 0);
				}
			}
		}

		/* �q�X�g�O�������\�� */
		cout << "--- �q�X�g�O������� -------------------------" << endl;
		cout << " RGB�摜" << endl;
		cout << " MAX_COUNT : " << MAX_COUNT << endl;
		cout << "----------------------------------------------" << endl;
	}
	else if (targetImg.channels() == 1) {
		Mat channels;
		int ha_index;
		uchar Gray;
		channels = Mat(Size(targetImg.cols, targetImg.rows), CV_8UC1);
		for (int i = 0; i < targetImg.rows; i++) {
			for (int j = 0; j < targetImg.cols; j++) {
				ha_index = i * targetImg.cols + j;
				Gray = (uchar)targetImg.data[ha_index];
				channels.data[ha_index] = Gray;
			}
		}

		/* �ϐ��錾 */
		Mat GRAY;
		int hist_size = 256;
		float range[] = { 0, 256 };
		const float* hist_range = range;

		/* ��f���𐔂��� */
		calcHist(&channels, 1, 0, Mat(), GRAY, 1, &hist_size, &hist_range);

		/* �m�F�i�q�X�g�O���������Œ�̂��߁j*/
		int MAX_COUNT = 0;
		double Min_count, Max_count;
		minMaxLoc(GRAY, &Min_count, &Max_count);
		if (Max_count > MAX_COUNT) {
			MAX_COUNT = (int)Max_count;
		}
		//MAX_COUNT = 80000;

		/* �q�X�g�O���������p�̉摜���쐬 */
		image_hist = Mat(Size(276, 120), CV_8UC3, Scalar(255, 255, 255));

		/* �w�i��`��i���₷�����邽�߂Ƀq�X�g�O���������̔w�i���O���[�ɂ���j */
		rectangle(image_hist, Point(10, 10), Point(265, 100), Scalar(230, 230, 230), -1);

		for (int i = 0; i < 256; i++) {
			line(image_hist, Point(10 + i, 100), Point(10 + i, 100 - (int)((float)(GRAY.at<float>(i) / MAX_COUNT) * 80)), Scalar(0, 0, 0), 1, 8, 0);

			if (i % 10 == 0) {		// ����10�����C��������
				line(image_hist, Point(10 + i, 100), Point(10 + i, 10),
					Scalar(170, 170, 170), 1, 8, 0);

				if (i % 50 == 0) {	// ����50���Z�����C��������
					line(image_hist, Point(10 + i, 100), Point(10 + i, 10),
						Scalar(50, 50, 50), 1, 8, 0);
				}
			}
		}

		/* �q�X�g�O�������\�� */
		cout << "--- �q�X�g�O������� ------------------------------" << endl;
		cout << " GrayScale�摜" << endl;
		cout << " MAX_COUNT : " << MAX_COUNT << endl;
		cout << "---------------------------------------------------" << endl;
	}
	else {
		cout << "ERROR! drawHist_Color()  :  Can't draw Histgram because of its channel." << endl;
	}
	cout << endl;
}