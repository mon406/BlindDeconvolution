#ifndef __INCLUDED_H_DiscreteFourierTransform__
#define __INCLUDED_H_DiscreteFourierTransform__

#include "main.h"
#include "CalculateComplexNumber.h"

/* ��`����֐� */
void visualbule_complex(Mat& srcImg, Mat& dstImg);		// ���f�摜�̕`��
void abs_complex(Mat& srcImg, Mat& dstImg);			// 2�����x�N�g���̑傫��
void abs_pow_complex(Mat& srcImg, Mat& dstImg);		// 2�����x�N�g���̑傫���̂Q��

void LowPathFilter(Mat& srcImg, Mat& dstImg, double Theredhold);	// ����g���݂̂̃t�B���^�ɕύX (���[�p�X�t�B���^�[)

void wiener_filter(Mat& srcImg, Mat& dstImg, Mat& parameterImg);	// 2�����x�N�g���̋t��(�E�B�[�i�E�t�B���^�[)


/* �֐� */
// ���f�摜�̕`��
void visualbule_complex(Mat& srcImg, Mat& dstImg) {
	Mat Img;
	srcImg.copyTo(Img);

	if (srcImg.channels() == 2) {
		/* ��Βl���v�Z����log�X�P�[���ɂ��� */
		/*   => log(1 + sqrt( Re(DFT(SRC))^2 + Im(DFT(SRC))^2 )) */
		Mat planes[2];
		split(Img, planes);								// planes[0] = Re(DFT(SRC)), planes[1] = Im(DFT(SRC))
		magnitude(planes[0], planes[1], planes[0]);		// planes[0] = magnitude
		Mat mag = planes[0];
		mag += Scalar::all(1);
		// ���ʂ̒l�͑傫�����̂Ə��������̂��������Ă���̂�log��K�p���ė}������
		log(mag, mag);

		/* �摜��\���\�ɂ��� */
		// �s�E�񂪊�̏ꍇ�A�N���b�v�X����
		mag = mag(Rect(0, 0, mag.cols & -2, mag.rows & -2));
		// �摜�̒����Ɍ��_������悤�ɏی������ւ���
		int cx = mag.cols / 2;
		int cy = mag.rows / 2;
		Mat q0(mag, Rect(0, 0, cx, cy));	// ����(���ی�)
		Mat q1(mag, Rect(cx, 0, cx, cy));	// �E��(���ی�)
		Mat q2(mag, Rect(0, cy, cx, cy));	// ����(��O�ی�)
		Mat q3(mag, Rect(cx, cy, cx, cy));	// �E��(��l�ی�)

		Mat tmpImg;
		// ����ւ�(����ƉE��)
		q0.copyTo(tmpImg);
		q3.copyTo(q0);
		tmpImg.copyTo(q3);
		// ����ւ�(�E��ƍ���)
		q1.copyTo(tmpImg);
		q2.copyTo(q1);
		tmpImg.copyTo(q2);

		// ���邱�Ƃ��ł���l�ɕϊ�(double[0,1]�ɕϊ�)
		normalize(mag, mag, 0, 1, NORM_MINMAX);

		mag.convertTo(dstImg, CV_8U, 255);
	}
	else { cout << "ERROR! visualbule_complex() : Can't translate because of wrong channels. " << srcImg.channels() << endl; }
}


// 2�����x�N�g���̑傫��
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
// 2�����x�N�g���̑傫���̂Q��
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
				pow_calc = (double)pow(pow_calc, 2);	// Img��2��
				pow_out = { pow_calc, 0.0 };
				Img2.at<Vec2d>(y, x) = pow_out;
			}
		}

		Img2.copyTo(dstImg);
	}
	else { cout << "ERROR! abs_pow_complex() : Can't translate because of wrong channels." << endl; }
}


// ����g���݂̂̃t�B���^�ɕύX (���[�p�X�t�B���^�[)
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
	cout << "  LowPathFilter : threshold = " << threshold << endl;	// �m�F
	double now_num1, now_num2, now_num3, now_num4;

	if (srcImg.channels() == 2) {
		Mat planes[2];
		split(Img, planes);

		// ����g���̂�
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

// 2�����x�N�g���̋t��(�E�B�[�i�E�t�B���^�[)
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
				//cout << " " << out << " <- " << srcImg.at<Vec2d>(y, x) << endl;	// �m�F�p
			}
		}
	}

	mulSpectrums(Img, srcImg, Img, 0, true);	// ���f����
	Img.copyTo(dstImg);
}


#endif