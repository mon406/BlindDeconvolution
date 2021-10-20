#ifndef __INCLUDED_H_BlindDeconvolution__
#define __INCLUDED_H_BlindDeconvolution__

#include "main.h"
#include "Quantized_Image.h"
#include "MakeKernel.h"
#include "CalculateComplexNumber.h"
#include "DiscreteFourierTransform.h"

/* �萔 */
int MAX_Iteration = 10;	// �ő唽����

/* �p�����[�^ */
double Myu = 0.4e-03;
double Rambda = 0.4e-03;
double Tau = 1.0e-03;

/* �֐� */
double CostCalculateLeast(int X, int Y, Mat& Quant_Img, Mat& Now_Img, Mat& contrast_Img);


/* Blind_Deconvolution �N���X */
class Blind_Deconvolution {
private:
	int x, y, c, pyr;
	int index;
public:
	int XSIZE;			// �摜�̕�
	int YSIZE;			// �摜�̍���
	int MAX_PIX;		// �摜�̑��s�N�Z����
	vector<int> X_SIZE;
	vector<int> Y_SIZE;
	vector<int> K_X_SIZE;
	vector<int> K_Y_SIZE;
	double ResizeFactor = 0.75;	// �s���~�b�h�̏k���v�f
	int PYRAMID_NUM = 3;		// �s���~�b�h�K�w��

	vector<Mat> Img;
	vector<QuantMatDouble> QuantImg;
	vector<KERNEL> Kernel;
	vector<Mat> BlurrImg;
	vector<Mat> TrueImg;	// ���摜

	Blind_Deconvolution();	// ������
	void deblurring(Mat&, Mat&, KERNEL&);
	void initialization(Mat&, Mat&, KERNEL&);
	void UpdateQuantizedImage(Mat&, QuantMatDouble&);
	//void UpdateImage(Mat&, Mat&, KERNEL&, Mat&);
	//void UpdateImage_check(Mat&, Mat&, KERNEL&, Mat&);
	//void UpdateKarnel(KERNEL&, Mat&, Mat&);
	//void UpdateKarnel_check(KERNEL&, Mat&, Mat&);
	void Upsampling(int);
};
Blind_Deconvolution::Blind_Deconvolution() {
	XSIZE = WIDTH;
	YSIZE = HEIGHT;
	MAX_PIX = XSIZE * YSIZE;
	Img.clear();
	QuantImg.clear();
	Kernel.clear();
	BlurrImg.clear();
	TrueImg.clear();

	X_SIZE.clear();
	X_SIZE.push_back(XSIZE);
	Y_SIZE.clear();
	Y_SIZE.push_back(YSIZE);
	K_X_SIZE.clear();
	K_Y_SIZE.clear();
}
void Blind_Deconvolution::deblurring(Mat& Img_true, Mat& Img_inoutput, KERNEL& Kernel_inoutput) {
	/* ������ */
	cout << "Initialization..." << endl;		// ���s�m�F�p
	initialization(Img_true, Img_inoutput, Kernel_inoutput);

	/* �ڂ����� */
	for (pyr = PYRAMID_NUM; pyr >= 3; pyr--) {
		cout << "Deconvolting in " << (int)pyr << endl;		// ���s�m�F�p

		for (int i = 0; i < MAX_Iteration; i++) {
			/* Update x~ */
			cout << " Update QuantImg... " << endl;				// ���s�m�F�p
			UpdateQuantizedImage(Img[pyr], QuantImg[pyr]);

			/* Update x */
			cout << " Update Img... " << endl;					// ���s�m�F�p
			//UpdateImage(Img[pyr], QuantImg[pyr].QMat, Kernel[pyr], BlurrImg[pyr]);
			//UpdateImage_check(Img[pyr], QuantImg[pyr].QMat, Kernel[pyr], BlurrImg[pyr]);

			/*if (i == 1) {
				break;
			}*/

			/* Update k */
			cout << " Update Karnel... " << endl;				// ���s�m�F�p
			//UpdateKarnel(Kernel[pyr], QuantImg[pyr].QMat, BlurrImg[pyr]);
			//UpdateKarnel_check(Kernel[pyr], QuantImg[pyr].QMat, BlurrImg[pyr]);

			if (i == 0) {
				break;
			}
		}

		/* Upsample */
		if (pyr != 0) {
			cout << " Upsample..." << endl;						// ���s�m�F�p
			Upsampling(pyr);
		}
		//pyr++;	// �m�F�p

		/* �o�� */
		Img[pyr].convertTo(Img_inoutput, CV_8UC3);
		Kernel_inoutput.copy(Kernel[pyr]);
		//for (int pyr_index = 0; pyr_index < pyr; pyr_index++) {
		//	resize(Image_kernel_original, Image_kernel_original, Size(), ResizeFactor, ResizeFactor);		// �m�F�p
		//}
		QuantImg[pyr].QMat.convertTo(Image_dst_deblurred2, CV_8UC3);		// �m�F�p
		BlurrImg[pyr].convertTo(Image_dst, CV_8UC3);		// �m�F�p
		TrueImg[pyr].convertTo(Img_true, CV_8UC3);
	}
	cout << endl;

	/* �o�͊m�F */
	//int check_pyr = 0;
	//Img[check_pyr].convertTo(Img_inoutput, CV_8UC3);
	//Kernel_inoutput.copy(Kernel[check_pyr]);
	//QuantImg[check_pyr].QMat.convertTo(Image_dst_deblurred2, CV_8UC3);		// �m�F�p
	//BlurrImg[check_pyr].convertTo(Image_dst, CV_8UC3);
	//TrueImg[check_pyr].convertTo(Img_true, CV_8UC3);
}
void Blind_Deconvolution::initialization(Mat& Img_true, Mat& Img_input, KERNEL& Kernel_input) {
	XSIZE = Img_input.cols;
	YSIZE = Img_input.rows;
	MAX_PIX = XSIZE * YSIZE;
	Img.clear();
	QuantImg.clear();
	Kernel.clear();
	BlurrImg.clear();
	TrueImg.clear();

	X_SIZE.clear();
	X_SIZE.push_back(XSIZE);
	Y_SIZE.clear();
	Y_SIZE.push_back(YSIZE);
	K_X_SIZE.clear();
	K_Y_SIZE.clear();

	Img.push_back(Img_input);
	BlurrImg.push_back(Img_input);
	TrueImg.push_back(Img_true);
	QuantMatDouble quantMat = QuantMatDouble(10, Img_input, 0);
	quantMat.quantedQMat();
	QuantImg.push_back(quantMat);
	Kernel.push_back(Kernel_input);
	//Kernel[0].display_detail();	// �m�F�p
	cout << " [0] : " << Img[0].size() << endl;		// ���s�m�F�p
	cout << "     : (" << Kernel[0].rows << "," << Kernel[0].cols << ")" << endl;	// ���s�m�F�p
	int pyr_next = 0;
	Mat Img_tmp;
	QuantMatDouble QuantImg_tmp;
	KERNEL Karnel_tmp;
	Mat TrueImg_tmp;
	for (pyr = 0; pyr < PYRAMID_NUM; pyr++) {
		resize(Img[pyr], Img_tmp, Size(), ResizeFactor, ResizeFactor);
		resize(TrueImg[pyr], TrueImg_tmp, Size(), ResizeFactor, ResizeFactor);
		QuantImg_tmp = QuantMatDouble(10, Img_tmp);
		QuantImg_tmp.quantedQMat();
		Karnel_tmp = KERNEL();
		Karnel_tmp.resize_copy(ResizeFactor, Kernel[pyr]);
		Img.push_back(Img_tmp);
		BlurrImg.push_back(Img_tmp);
		TrueImg.push_back(TrueImg_tmp);
		QuantImg.push_back(QuantImg_tmp);
		Kernel.push_back(Karnel_tmp);
		//Kernel[pyr + 1].display_detail();	// �m�F�p
		pyr_next = pyr + 1;
		cout << " [" << (int)(pyr + 1) << "] : " << Img[pyr_next].size() << endl;	// �m�F�p
		cout << "     : " << Kernel[pyr_next].Kernel.size() << endl;	// �m�F�p
		X_SIZE.push_back(Img[pyr_next].cols);
		Y_SIZE.push_back(Img[pyr_next].rows);
		K_X_SIZE.push_back(Kernel[pyr_next].cols);
		K_Y_SIZE.push_back(Kernel[pyr_next].rows);
	}
	cout << endl;
}
void Blind_Deconvolution::UpdateQuantizedImage(Mat& Img_Now, QuantMatDouble& QuantImg_Now) {
	int Iteration_Number = 1;
	double Error = 1.0e-04;

	QuantImg_Now.quantedQMat();
	/* Optimizing MRF using BP */
	Mat NewQuantImg;
	QuantImg_Now.QMat.copyTo(NewQuantImg);
	Mat BeforeQuantImg;
	QuantImg_Now.QMat.copyTo(BeforeQuantImg);
	Mat QuantAfter1, QuantAfter2;
	NewQuantImg.copyTo(QuantAfter1);
	NewQuantImg.copyTo(QuantAfter2);

	double diff[3], min_diff, energy;
	double candidate_color[2];
	double before_color;
	Mat color_Img, gray_Img, contrust;
	Img_Now.convertTo(color_Img, CV_8UC3);
	cvtColor(color_Img, gray_Img, COLOR_BGR2GRAY);
	Laplacian(gray_Img, contrust, CV_64F, 3);
	convertScaleAbs(contrust, contrust, 1, 0);
	contrust.convertTo(contrust, CV_64FC1);
	for (int i = 0; i < Iteration_Number; i++) {
		cout << "  iterate:" << i << endl;	// �m�F�p
#pragma omp parallel for private(x)
		for (y = 0; y < QuantImg_Now.rows; y++) {
			for (x = 0; x < QuantImg_Now.cols; x++) {
				index = (y * QuantImg_Now.cols + x) * 3;
				Vec3d color = BeforeQuantImg.at<Vec3d>(y, x);
				//cout << "   " << color << endl;	// �m�F�p
				//cout << "   [" << (double)BeforeQuantImg.data[index + 0] << ", " << (double)BeforeQuantImg.data[index + 1] << ", " << (double)BeforeQuantImg.data[index + 2] << "]" << endl;	// �m�F�p
				//cout << "   [" << (double)BeforeQuantImg.at<Vec3d>(y, x)[0] << ", " << (double)BeforeQuantImg.at<Vec3d>(y, x)[1] << ", " << (double)BeforeQuantImg.at<Vec3d>(y, x)[2] << "]" << endl;	// �m�F�p
				//cout << endl;
				for (c = 0; c < 3; c++) {
					//before_color = (double)BeforeQuantImg.data[index + c];
					before_color = (double)color[c];
					QuantImg_Now.searchUpDown(before_color, candidate_color[0], candidate_color[1]);
					//cout << before_color << " " << candidate_color[0] << " " << candidate_color[1] << endl;	// �m�F�p

					diff[0] = CostCalculateLeast(x, y, NewQuantImg, Img_Now, contrust);
					if (before_color != candidate_color[0]) {
						QuantAfter1.data[index] = candidate_color[0];
						diff[1] = CostCalculateLeast(x, y, QuantAfter1, Img_Now, contrust);
					}
					else { diff[1] = diff[0]; }
					if (before_color != candidate_color[1]) {
						QuantAfter2.data[index] = candidate_color[1];
						diff[2] = CostCalculateLeast(x, y, QuantAfter2, Img_Now, contrust);
					}
					else { diff[2] = diff[0]; }
					//cout << "   diff : " << diff[0] << "," << diff[1] << "," << diff[2] << endl;	// �m�F�p

					min_diff = diff[0];
					if (diff[1] < min_diff) {
						min_diff = diff[1];
						//NewQuantImg.data[index] = candidate_color[0];
						color[c] = candidate_color[0];
						//cout << "    up" << endl;	// �m�F�p
					}
					if (diff[2] < min_diff) {
						min_diff = diff[1];
						//NewQuantImg.data[index] = candidate_color[1];
						color[c] = candidate_color[1];
						//cout << "    down" << endl;	// �m�F�p
					}

					//cout << "  " << (int)NewQuantImg.data[index];	// �m�F�p
				}

				NewQuantImg.at<Vec3d>(y, x) = color;
				//cout << "   ->" << color << endl;	// �m�F�p
			}
		}

		energy = (double)norm(NewQuantImg, BeforeQuantImg, NORM_L2) / (double)(MAX_PIX * 3.0);
		cout << "  " << (int)i << " : energy = " << (double)energy << endl;	// �m�F�p
		if (energy < Error) { break; }
		NewQuantImg.copyTo(BeforeQuantImg);
	}
	//NewQuantImg.convertTo(Image_dst_deblurred2, CV_8UC3);		// �m�F�p

	/*QuantMatDouble QuantImage_tmp = QuantMatDouble(10, NewQuantImg);
	QuantImage_tmp.quantedQMat();
	QuantImage_tmp.QMat.copyTo(NewQuantImg);*/

	NewQuantImg.copyTo(QuantImg_Now.QMat);
}

void Blind_Deconvolution::Upsampling(int before_pyrLEVEL) {
	int after_pyrLEVEL = before_pyrLEVEL - 1;
	Mat imgNEXT;
	Kernel[after_pyrLEVEL] = KERNEL();
	double ReResizeFactor = 1.0 / ResizeFactor;
	//resize(Img[before_pyrLEVEL], imgNEXT, Size(), ReResizeFactor, ReResizeFactor);
	resize(Img[before_pyrLEVEL], imgNEXT, Size(), (double)X_SIZE[after_pyrLEVEL] / (double)Img[before_pyrLEVEL].cols, (double)Y_SIZE[after_pyrLEVEL] / (double)Img[before_pyrLEVEL].rows);
	imgNEXT.copyTo(Img[after_pyrLEVEL]);
	QuantMatDouble q_imgNEXT;
	QuantImg[after_pyrLEVEL] = QuantMatDouble(10, imgNEXT);
	QuantImg[after_pyrLEVEL].quantedQMat();
	//Kernel[after_pyrLEVEL].resize_copy(ReResizeFactor, Kernel[before_pyrLEVEL]);
	Kernel[after_pyrLEVEL].resize_copy((double)K_X_SIZE[after_pyrLEVEL] / (double)Kernel[before_pyrLEVEL].cols, (double)K_Y_SIZE[after_pyrLEVEL] / (double)Kernel[before_pyrLEVEL].rows, Kernel[before_pyrLEVEL]);
}


double CostCalculateLeast(int X, int Y, Mat& Quant_Img, Mat& Now_Img, Mat& contrast_Img) {
	double Cost = 0.0;
	int Max_Pix = Quant_Img.cols * Quant_Img.rows * 3;

	int adject;
	double diff, norm, Uorm = 0.0;
	double cout_tmp = 0.0;
	double now[3], adj;
	int Costindex = (Y * Quant_Img.cols + X) * 3;
	for (int c = 0; c < 3; c++) {
		now[c] = Quant_Img.data[Costindex + c];
	}
	double wight;
	if (X > 0) {
		for (int c = 0; c < 3; c++) {
			Costindex = (Y * Quant_Img.cols + (X - 1)) * 3;
			adj = Quant_Img.data[Costindex + c];
			if (now[c] != adj) {
				wight = (double)contrast_Img.at<double>(Y, X - 1);
				if (wight != 0) { wight = 1.0 / (double)wight; }
				cout_tmp += wight;

				diff = (double)abs(adj - now[c]);
				norm = (double)pow(diff, 2);
				Uorm += norm;
			}
		}
	}
	if (X < Quant_Img.cols - 1) {
		for (int c = 0; c < 3; c++) {
			Costindex = (Y * Quant_Img.cols + (X + 1)) * 3;
			adj = Quant_Img.data[Costindex + c];
			if (now[c] != adj) {
				wight = (double)contrast_Img.at<double>(Y, X + 1);
				if (wight != 0) { wight = 1.0 / (double)wight; }
				cout_tmp += wight;

				diff = (double)abs(adj - now[c]);
				norm = (double)pow(diff, 2);
				Uorm += norm;
			}
		}
	}
	if (Y > 0) {
		for (int c = 0; c < 3; c++) {
			Costindex = ((Y - 1) * Quant_Img.cols + X) * 3;
			adj = Quant_Img.data[Costindex + c];
			if (now[c] != adj) {
				wight = (double)contrast_Img.at<double>(Y - 1, X);
				if (wight != 0) { wight = 1.0 / (double)wight; }
				cout_tmp += wight;

				diff = (double)abs(adj - now[c]);
				norm = (double)pow(diff, 2);
				Uorm += norm;
			}
		}
	}
	if (Y < Quant_Img.rows - 1) {
		for (int c = 0; c < 3; c++) {
			Costindex = ((Y + 1) * Quant_Img.cols + X) * 3;
			adj = Quant_Img.data[Costindex + c];
			if (now[c] != adj) {
				wight = (double)contrast_Img.at<double>(Y + 1, X);
				if (wight != 0) { wight = 1.0 / (double)wight; }
				cout_tmp += wight;

				diff = (double)abs(adj - now[c]);
				norm = (double)pow(diff, 2);
				Uorm += norm;
			}
		}
	}
	Uorm /= (double)Max_Pix;
	Cost = Myu * Uorm;
	cout_tmp /= (double)Max_Pix;
	Cost += cout_tmp;

	return Cost;
}


#endif