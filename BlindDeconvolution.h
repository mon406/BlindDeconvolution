#ifndef __INCLUDED_H_BlindDeconvolution__
#define __INCLUDED_H_BlindDeconvolution__

#include "main.h"
#include "Quantized_Image.h"
#include "MakeKernel.h"
#include "CalculateComplexNumber.h"
#include "DiscreteFourierTransform.h"
#include "CalculateVector.h"
#include "ConjugateGradient.h"

/* 定数 */
int MAX_Iteration = 10;	// 最大反復回数

/* パラメータ */
double Myu = 0.4e-03;
double Rambda = 0.4e-03;
double Tau = 1.0e-03;

/* 関数 */
double CostCalculateLeast(int X, int Y, Mat& Quant_Img, Mat& Now_Img, Mat& contrast_Img);


/* Blind_Deconvolution クラス */
class Blind_Deconvolution {
private:
	int x, y, c, pyr;
	int index;
public:
	int XSIZE;			// 画像の幅
	int YSIZE;			// 画像の高さ
	int MAX_PIX;		// 画像の総ピクセル数
	vector<int> X_SIZE;
	vector<int> Y_SIZE;
	vector<int> K_X_SIZE;
	vector<int> K_Y_SIZE;
	double ResizeFactor = 0.75;	// ピラミッドの縮小要素
	int PYRAMID_NUM = 3;		// ピラミッド階層数

	vector<Mat> Img;
	vector<QuantMatDouble> QuantImg;
	vector<KERNEL> Kernel;
	vector<Mat> BlurrImg;
	vector<Mat> TrueImg;	// 元画像

	Blind_Deconvolution();	// 初期化
	void deblurring(Mat&, Mat&, KERNEL&);
	void initialization(Mat&, Mat&, KERNEL&);
	void UpdateQuantizedImage(Mat&, QuantMatDouble&);
	void UpdateImage(Mat&, Mat&, KERNEL&, Mat&);
	void UpdateImage_check(Mat&, Mat&, KERNEL&, Mat&);
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
	/* 初期化 */
	cout << "Initialization..." << endl;		// 実行確認用
	initialization(Img_true, Img_inoutput, Kernel_inoutput);

	/* ぼけ除去 */
	for (pyr = PYRAMID_NUM; pyr >= 3; pyr--) {
		cout << "Deconvolting in " << (int)pyr << endl;		// 実行確認用

		for (int i = 0; i < MAX_Iteration; i++) {
			/* Update x~ */
			cout << " Update QuantImg... " << endl;				// 実行確認用
			//UpdateQuantizedImage(Img[pyr], QuantImg[pyr]);

			/* Update x */
			cout << " Update Img... " << endl;					// 実行確認用
			//UpdateImage(Img[pyr], QuantImg[pyr].QMat, Kernel[pyr], BlurrImg[pyr]);
			UpdateImage_check(Img[pyr], QuantImg[pyr].QMat, Kernel[pyr], BlurrImg[pyr]);

			/*if (i == 1) {
				break;
			}*/

			/* Update k */
			cout << " Update Karnel... " << endl;				// 実行確認用
			//UpdateKarnel(Kernel[pyr], QuantImg[pyr].QMat, BlurrImg[pyr]);
			//UpdateKarnel_check(Kernel[pyr], QuantImg[pyr].QMat, BlurrImg[pyr]);

			if (i == 0) {
				break;
			}
		}

		/* Upsample */
		if (pyr != 0) {
			cout << " Upsample..." << endl;						// 実行確認用
			Upsampling(pyr);
		}
		//pyr++;	// 確認用

		/* 出力 */
		Img[pyr].convertTo(Img_inoutput, CV_8UC3);
		Kernel_inoutput.copy(Kernel[pyr]);
		//for (int pyr_index = 0; pyr_index < pyr; pyr_index++) {
		//	resize(Image_kernel_original, Image_kernel_original, Size(), ResizeFactor, ResizeFactor);		// 確認用
		//}
		QuantImg[pyr].QMat.convertTo(Image_dst_deblurred2, CV_8UC3);		// 確認用
		BlurrImg[pyr].convertTo(Image_dst, CV_8UC3);		// 確認用
		TrueImg[pyr].convertTo(Img_true, CV_8UC3);
	}
	cout << endl;

	/* 出力確認 */
	//int check_pyr = 0;
	//Img[check_pyr].convertTo(Img_inoutput, CV_8UC3);
	//Kernel_inoutput.copy(Kernel[check_pyr]);
	//QuantImg[check_pyr].QMat.convertTo(Image_dst_deblurred2, CV_8UC3);		// 確認用
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
	//Kernel[0].display_detail();	// 確認用
	cout << " [0] : " << Img[0].size() << endl;		// 実行確認用
	cout << "     : (" << Kernel[0].rows << "," << Kernel[0].cols << ")" << endl;	// 実行確認用
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
		//Kernel[pyr + 1].display_detail();	// 確認用
		pyr_next = pyr + 1;
		cout << " [" << (int)(pyr + 1) << "] : " << Img[pyr_next].size() << endl;	// 確認用
		cout << "     : " << Kernel[pyr_next].Kernel.size() << endl;	// 確認用
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
		cout << "  iterate:" << i << endl;	// 確認用
#pragma omp parallel for private(x)
		for (y = 0; y < QuantImg_Now.rows; y++) {
			for (x = 0; x < QuantImg_Now.cols; x++) {
				index = (y * QuantImg_Now.cols + x) * 3;
				Vec3d color = BeforeQuantImg.at<Vec3d>(y, x);
				//cout << "   " << color << endl;	// 確認用
				//cout << "   [" << (double)BeforeQuantImg.data[index + 0] << ", " << (double)BeforeQuantImg.data[index + 1] << ", " << (double)BeforeQuantImg.data[index + 2] << "]" << endl;	// 確認用
				//cout << "   [" << (double)BeforeQuantImg.at<Vec3d>(y, x)[0] << ", " << (double)BeforeQuantImg.at<Vec3d>(y, x)[1] << ", " << (double)BeforeQuantImg.at<Vec3d>(y, x)[2] << "]" << endl;	// 確認用
				//cout << endl;
				for (c = 0; c < 3; c++) {
					//before_color = (double)BeforeQuantImg.data[index + c];
					before_color = (double)color[c];
					QuantImg_Now.searchUpDown(before_color, candidate_color[0], candidate_color[1]);
					//cout << before_color << " " << candidate_color[0] << " " << candidate_color[1] << endl;	// 確認用

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
					//cout << "   diff : " << diff[0] << "," << diff[1] << "," << diff[2] << endl;	// 確認用

					min_diff = diff[0];
					if (diff[1] < min_diff) {
						min_diff = diff[1];
						//NewQuantImg.data[index] = candidate_color[0];
						color[c] = candidate_color[0];
						//cout << "    up" << endl;	// 確認用
					}
					if (diff[2] < min_diff) {
						min_diff = diff[1];
						//NewQuantImg.data[index] = candidate_color[1];
						color[c] = candidate_color[1];
						//cout << "    down" << endl;	// 確認用
					}

					//cout << "  " << (int)NewQuantImg.data[index];	// 確認用
				}

				NewQuantImg.at<Vec3d>(y, x) = color;
				//cout << "   ->" << color << endl;	// 確認用
			}
		}

		energy = (double)norm(NewQuantImg, BeforeQuantImg, NORM_L2) / (double)(MAX_PIX * 3.0);
		cout << "  " << (int)i << " : energy = " << (double)energy << endl;	// 確認用
		if (energy < Error) { break; }
		NewQuantImg.copyTo(BeforeQuantImg);
	}
	//NewQuantImg.convertTo(Image_dst_deblurred2, CV_8UC3);		// 確認用

	/*QuantMatDouble QuantImage_tmp = QuantMatDouble(10, NewQuantImg);
	QuantImage_tmp.quantedQMat();
	QuantImage_tmp.QMat.copyTo(NewQuantImg);*/

	NewQuantImg.copyTo(QuantImg_Now.QMat);
}
void Blind_Deconvolution::UpdateImage(Mat& Img_Now, Mat& QuantImg_Now, KERNEL& Karnel_Now, Mat& BlurrImg_Now) {
	//QuantImg_Now.convertTo(Image_dst_deblurred2, CV_8UC3);		// 確認用
	/* Optimizing x by X~,k using FFT */
	Mat grad_h = (Mat_<double>(3, 3)	// 3*3
		<< -1, 0, 1,
		-2, 0, 2,
		-1, 0, 1);
	Mat grad_v = (Mat_<double>(3, 3)	// 3*3
		<< -1, -2, -1,
		0, 0, 0,
		1, 2, 1);

	/* 画像をCV_64Fに変換(前処理) */
	// カーネル
	Mat doubleKernel;
	Karnel_Now.Kernel_normalized.copyTo(doubleKernel);
	// ぼけ画像
	Mat BlurredImg;
	BlurrImg_Now.copyTo(BlurredImg);
	// 量子化画像
	Mat QuantImg;
	QuantImg_Now.copyTo(QuantImg);
	// 3つのチャネルB, G, Rに分離 (OpenCVではデフォルトでB, G, Rの順)
	Mat doubleBlurredImg_sub[3] = { Mat::zeros(BlurredImg.size(), CV_64F), Mat::zeros(BlurredImg.size(), CV_64F), Mat::zeros(BlurredImg.size(), CV_64F) };
	split(BlurredImg, doubleBlurredImg_sub);
	Mat doubleQuantImg_sub[3] = { Mat::zeros(QuantImg.size(), CV_64F), Mat::zeros(QuantImg.size(), CV_64F), Mat::zeros(QuantImg.size(), CV_64F) };
	split(QuantImg, doubleQuantImg_sub);
	Mat doubleBlurredImg[3];
	Mat doubleQuantImg[3];
	for (c = 0; c < 3; c++) {
		Mat planes_BI[] = { Mat_<double>(doubleBlurredImg_sub[c]), Mat::zeros(doubleBlurredImg_sub[c].size(), CV_64F) };
		merge(planes_BI, 2, doubleBlurredImg[c]);
		Mat planes_QI[] = { Mat_<double>(doubleQuantImg_sub[c]), Mat::zeros(doubleQuantImg_sub[c].size(), CV_64F) };
		merge(planes_QI, 2, doubleQuantImg[c]);
	}

	// DFT変換のサイズを計算
	int Mplus = BlurredImg.rows + doubleKernel.rows;
	int Nplus = BlurredImg.cols + doubleKernel.cols;
	int Msize = getOptimalDFTSize(Mplus);
	int Nsize = getOptimalDFTSize(Nplus);
	//cout << "  FFT Size  : (" << Mplus << "," << Nplus << ") => (" << Msize << "," << Nsize << ")" << endl;	// 確認

	/* DFT */
	// フィルター
	Mat Grad_h, Grad_v;
	Mat planes_h[] = { Mat_<double>(grad_h), Mat::zeros(grad_h.size(), CV_64F) };
	merge(planes_h, 2, Grad_h);
	Mat planes_v[] = { Mat_<double>(grad_v), Mat::zeros(grad_v.size(), CV_64F) };
	merge(planes_v, 2, Grad_v);
	Mat dft_H = Mat::zeros(Msize, Nsize, CV_64FC2);
	copyMakeBorder(Grad_h, dft_H, 0, Msize - Grad_h.rows, 0, Nsize - Grad_h.cols, BORDER_CONSTANT, (0.0, 0.0));
	dft(dft_H, dft_H, 0, dft_H.rows);
	//visualbule_complex(dft_H, Image_dst_deblurred2);	// 確認
	Mat dft_V = Mat::zeros(Msize, Nsize, CV_64FC2);
	copyMakeBorder(Grad_v, dft_V, 0, Msize - Grad_v.rows, 0, Nsize - Grad_v.cols, BORDER_CONSTANT, (0.0, 0.0));
	dft(dft_V, dft_V, 0, dft_V.rows);
	//visualbule_complex(dft_V, Image_dst_deblurred2);	// 確認
	// カーネル
	Mat dft_Kernel = Mat::zeros(Msize, Nsize, CV_64FC2);
	Mat planes_K[] = { Mat_<double>(doubleKernel), Mat::zeros(doubleKernel.size(), CV_64F) };
	merge(planes_K, 2, doubleKernel);
	copyMakeBorder(doubleKernel, dft_Kernel, 0, Msize - doubleKernel.rows, 0, Nsize - doubleKernel.cols, BORDER_CONSTANT, (0.0, 0.0));
	dft(dft_Kernel, dft_Kernel, 0, dft_Kernel.rows);
	//visualbule_complex(dft_Kernel, Image_dst_deblurred2);	// 確認
	// ぼけ画像＆量子化画像
	Mat dft_doubleBlurredImg[3] = { Mat::zeros(Msize, Nsize, CV_64FC2), Mat::zeros(Msize, Nsize, CV_64FC2), Mat::zeros(Msize, Nsize, CV_64FC2) };
	Mat dft_doubleQuantImg[3] = { Mat::zeros(Msize, Nsize, CV_64FC2), Mat::zeros(Msize, Nsize, CV_64FC2), Mat::zeros(Msize, Nsize, CV_64FC2) };
	for (c = 0; c < 3; c++) {
		copyMakeBorder(doubleBlurredImg[c], dft_doubleBlurredImg[c], doubleKernel.rows / 2, Msize - Mplus + doubleKernel.rows / 2, doubleKernel.cols / 2, Nsize - Nplus + doubleKernel.cols / 2, BORDER_REPLICATE);
		dft(dft_doubleBlurredImg[c], dft_doubleBlurredImg[c]);
		copyMakeBorder(doubleQuantImg[c], dft_doubleQuantImg[c], doubleKernel.rows / 2, Msize - Mplus + doubleKernel.rows / 2, doubleKernel.cols / 2, Nsize - Nplus + doubleKernel.cols / 2, BORDER_REPLICATE);
		dft(dft_doubleQuantImg[c], dft_doubleQuantImg[c]);
	}
	//visualbule_complex(dft_doubleBlurredImg_2[0], Image_dst_deblurred2);	// 確認
	//visualbule_complex(dft_doubleQuantImg_2[0], Image_dst_deblurred2);	// 確認

	/* ぼけ除去画像を求める */
	Mat dft_doubleNewImg[3];
	for (c = 0; c < 3; c++) {
		mulSpectrums(dft_doubleBlurredImg[c], dft_Kernel, dft_doubleNewImg[c], 0, true);	// 複素共役
	}
	Mat denom_K, denom_H, denom_V;
	abs_pow_complex(dft_Kernel, denom_K);	// 2次元ベクトルの大きさの２乗
	abs_pow_complex(dft_H, denom_H);
	abs_pow_complex(dft_V, denom_V);

	Vec2d complexRambda = { Rambda , 0.0 }, complexMyu = { Myu, 0.0 };
	Vec2d number, number1;
	Vec2d denom, denom1, denom2;
	for (c = 0; c < 3; c++) {
#pragma omp parallel for private(x)
		for (y = 0; y < Msize; y++) {
			for (x = 0; x < Nsize; x++) {
				number = dft_doubleNewImg[c].at<Vec2d>(y, x);
				number1 = dft_doubleQuantImg[c].at<Vec2d>(y, x);
				multi_complex_2(number1, complexMyu, number1);
				number = number + number1;

				denom = denom_K.at<Vec2d>(y, x);
				denom1 = denom_H.at<Vec2d>(y, x);
				denom2 = denom_V.at<Vec2d>(y, x);
				denom1 = denom1 + denom2;
				multi_complex_2(denom1, complexRambda, denom1);
				denom = denom + denom1 + complexMyu;

				divi_complex_2(number, number, denom);
				dft_doubleNewImg[c].at<Vec2d>(y, x) = number;
				//cout << " " << dft_doubleNewImg[c].at<Vec2d>(y, x) << " = " << number << endl;	// 確認用
			}
		}
	}
	//visualbule_complex(dft_doubleNewImg[0], Image_dst_deblurred2);	// 確認

	/* inverseDFT */
	Mat doubleNewImg[3];
	for (c = 0; c < 3; c++) {
		dft(dft_doubleNewImg[c], dft_doubleNewImg[c], cv::DFT_INVERSE + cv::DFT_SCALE | cv::DFT_REAL_OUTPUT);
		//doubleNewImg[c] = dft_doubleNewImg[c](Rect(doubleKernel.cols / 2, doubleKernel.rows / 2, BlurredImg.cols, BlurredImg.rows));
		doubleNewImg[c] = dft_doubleNewImg[c](Rect(0, 0, BlurredImg.cols, BlurredImg.rows));
	}
	Mat NewImg;
	merge(doubleNewImg, 3, NewImg);

	NewImg.copyTo(Img_Now);
}
void Blind_Deconvolution::UpdateImage_check(Mat& Img_Now, Mat& QuantImg_Now, KERNEL& Karnel_Now, Mat& BlurrImg_Now) {
	/* Optimizing x by X~,k using FFT */
	Mat grad_h = (Mat_<double>(3, 3)	// 3*3
		<< -1, 0, 1,
		-2, 0, 2,
		-1, 0, 1);
	Mat grad_v = (Mat_<double>(3, 3)	// 3*3
		<< -1, -2, -1,
		0, 0, 0,
		1, 2, 1);

	/* 画像をCV_64Fに変換(前処理) */
	// カーネル
	Mat doubleKernel;
	Karnel_Now.Kernel_normalized.copyTo(doubleKernel);
	// ぼけ画像
	Mat BlurredImg;
	BlurrImg_Now.convertTo(BlurredImg, CV_64FC3);
	// 量子化画像
	Mat QuantImg;
	QuantImg_Now.convertTo(QuantImg, CV_64FC3);
	// 3つのチャネルB, G, Rに分離 (OpenCVではデフォルトでB, G, Rの順)
	Mat doubleBlurredImg_sub[3] = { Mat::zeros(BlurredImg.size(), CV_64F), Mat::zeros(BlurredImg.size(), CV_64F), Mat::zeros(BlurredImg.size(), CV_64F) };
	split(BlurredImg, doubleBlurredImg_sub);
	Mat doubleQuantImg_sub[3] = { Mat::zeros(QuantImg.size(), CV_64F), Mat::zeros(QuantImg.size(), CV_64F), Mat::zeros(QuantImg.size(), CV_64F) };
	split(QuantImg, doubleQuantImg_sub);
	Mat doubleBlurredImg[3];
	Mat doubleQuantImg[3];
	for (c = 0; c < 3; c++) {
		Mat planes_BI[] = { Mat_<double>(doubleBlurredImg_sub[c]), Mat::zeros(doubleBlurredImg_sub[c].size(), CV_64F) };
		merge(planes_BI, 2, doubleBlurredImg[c]);
		Mat planes_QI[] = { Mat_<double>(doubleQuantImg_sub[c]), Mat::zeros(doubleQuantImg_sub[c].size(), CV_64F) };
		merge(planes_QI, 2, doubleQuantImg[c]);
	}

	// DFT変換のサイズを計算
	int Mplus = BlurredImg.rows + doubleKernel.rows;
	int Nplus = BlurredImg.cols + doubleKernel.cols;
	int Msize = getOptimalDFTSize(Mplus);
	int Nsize = getOptimalDFTSize(Nplus);
	//cout << "  FFT Size  : (" << Mplus << "," << Nplus << ") => (" << Msize << "," << Nsize << ")" << endl;	// 確認

	/* DFT */
	// フィルター
	Mat Grad_h, Grad_v;
	Mat planes_h[] = { Mat_<double>(grad_h), Mat::zeros(grad_h.size(), CV_64F) };
	merge(planes_h, 2, Grad_h);
	Mat planes_v[] = { Mat_<double>(grad_v), Mat::zeros(grad_v.size(), CV_64F) };
	merge(planes_v, 2, Grad_v);
	Mat dft_H = Mat::zeros(Msize, Nsize, CV_64FC2);
	copyMakeBorder(Grad_h, dft_H, 0, Msize - Grad_h.rows, 0, Nsize - Grad_h.cols, BORDER_CONSTANT, (0.0, 0.0));
	dft(dft_H, dft_H, 0, dft_H.rows);
	//visualbule_complex(dft_H, Image_dst_deblurred2);	// 確認
	Mat dft_V = Mat::zeros(Msize, Nsize, CV_64FC2);
	copyMakeBorder(Grad_v, dft_V, 0, Msize - Grad_v.rows, 0, Nsize - Grad_v.cols, BORDER_CONSTANT, (0.0, 0.0));
	dft(dft_V, dft_V, 0, dft_V.rows);
	//visualbule_complex(dft_V, Image_dst_deblurred2);	// 確認
	// カーネル
	Mat dft_Kernel = Mat::zeros(Msize, Nsize, CV_64FC2);
	Mat planes_K[] = { Mat_<double>(doubleKernel), Mat::zeros(doubleKernel.size(), CV_64F) };
	merge(planes_K, 2, doubleKernel);
	copyMakeBorder(doubleKernel, dft_Kernel, 0, Msize - doubleKernel.rows, 0, Nsize - doubleKernel.cols, BORDER_CONSTANT, (0.0, 0.0));
	dft(dft_Kernel, dft_Kernel, 0, dft_Kernel.rows);
	//visualbule_complex(dft_Kernel, Image_dst_deblurred2);	// 確認
	// ぼけ画像＆量子化画像
	Mat dft_doubleBlurredImg[3] = { Mat::zeros(Msize, Nsize, CV_64FC2), Mat::zeros(Msize, Nsize, CV_64FC2), Mat::zeros(Msize, Nsize, CV_64FC2) };
	Mat dft_doubleQuantImg[3] = { Mat::zeros(Msize, Nsize, CV_64FC2), Mat::zeros(Msize, Nsize, CV_64FC2), Mat::zeros(Msize, Nsize, CV_64FC2) };
	for (c = 0; c < 3; c++) {
		copyMakeBorder(doubleBlurredImg[c], dft_doubleBlurredImg[c], doubleKernel.rows / 2, Msize - Mplus + doubleKernel.rows / 2, doubleKernel.cols / 2, Nsize - Nplus + doubleKernel.cols / 2, BORDER_REPLICATE);
		dft(dft_doubleBlurredImg[c], dft_doubleBlurredImg[c]);
		copyMakeBorder(doubleQuantImg[c], dft_doubleQuantImg[c], doubleKernel.rows / 2, Msize - Mplus + doubleKernel.rows / 2, doubleKernel.cols / 2, Nsize - Nplus + doubleKernel.cols / 2, BORDER_REPLICATE);
		dft(dft_doubleQuantImg[c], dft_doubleQuantImg[c]);
	}
	//visualbule_complex(dft_doubleBlurredImg_2[0], Image_dst_deblurred2);	// 確認
	//visualbule_complex(dft_doubleQuantImg_2[0], Image_dst_deblurred2);	// 確認


	/* ぼけ除去画像を求める */
	/* Axとbを求める */
	Mat dft_doubleNewImg[3], dft_doubleNewImg1[3], dft_doubleNewImg2[3];
	for (c = 0; c < 3; c++) {
		mulSpectrums(dft_doubleBlurredImg[c], dft_Kernel, dft_doubleNewImg2[c], 0, true);	// 複素共役
	}
	Mat denom_K, denom_H, denom_V;
	abs_pow_complex(dft_Kernel, denom_K);	// 2次元ベクトルの大きさの２乗
	abs_pow_complex(dft_H, denom_H);
	abs_pow_complex(dft_V, denom_V);

	Vec2d complexRambda = { Rambda , 0.0 }, complexMyu = { Myu, 0.0 };
	Vec2d number, number1;
	Vec2d denom, denom1, denom2;
	for (c = 0; c < 3; c++) {
		dft_doubleNewImg[c] = Mat::zeros(Msize, Nsize, CV_64FC2);

#pragma omp parallel for private(x)
		for (y = 0; y < Msize; y++) {
			for (x = 0; x < Nsize; x++) {
				number = dft_doubleNewImg2[c].at<Vec2d>(y, x);
				number1 = dft_doubleQuantImg[c].at<Vec2d>(y, x);
				multi_complex_2(number1, complexMyu, number1);
				number = number + number1;
				dft_doubleNewImg2[c].at<Vec2d>(y, x) = number;

				denom = denom_K.at<Vec2d>(y, x);
				denom1 = denom_H.at<Vec2d>(y, x);
				denom2 = denom_V.at<Vec2d>(y, x);
				denom1 = denom1 + denom2;
				multi_complex_2(denom1, complexRambda, denom1);
				denom = denom + denom1 + complexMyu;
				dft_doubleNewImg[c].at<Vec2d>(y, x) = denom;
			}
		}
		mulSpectrums(dft_doubleNewImg[c], dft_Kernel, dft_doubleNewImg1[c], 0, false);
	}
	//visualbule_complex(dft_doubleNewImg[0], Image_dst_deblurred2);	// 確認

	/* inverseDFT */
	Mat doubleNewImg1[3], doubleNewImg2[3];
	for (c = 0; c < 3; c++) {
		dft(dft_doubleNewImg1[c], dft_doubleNewImg1[c], cv::DFT_INVERSE + cv::DFT_SCALE | cv::DFT_REAL_OUTPUT);
		//doubleNewImg1[c] = dft_doubleNewImg1[c](Rect(doubleKernel.cols / 2, doubleKernel.rows / 2, BlurredImg.cols, BlurredImg.rows));
		doubleNewImg1[c] = dft_doubleNewImg1[c](Rect(0, 0, BlurredImg.cols, BlurredImg.rows));
		dft(dft_doubleNewImg2[c], dft_doubleNewImg2[c], cv::DFT_INVERSE + cv::DFT_SCALE | cv::DFT_REAL_OUTPUT);
		//doubleNewImg2[c] = dft_doubleNewImg2[c](Rect(doubleKernel.cols / 2, doubleKernel.rows / 2, BlurredImg.cols, BlurredImg.rows));
		doubleNewImg2[c] = dft_doubleNewImg2[c](Rect(0, 0, BlurredImg.cols, BlurredImg.rows));
	}
	//doubleNewImg1[c].convertTo(Image_dst_deblurred2, CV_8UC1);	// 確認
	//doubleNewImg2[c].convertTo(Image_dst_deblurred2, CV_8UC1);	// 確認

	/* CG method */
	int Iterate_Num = 100;
	double ERROR_END_NUM = 1.0e-04;

	/* 初期値設定 */
	Mat NextX[3], LastX[3];		// 初期値
	Mat Residual[3] = { Mat::zeros(BlurredImg.size(), CV_64F), Mat::zeros(BlurredImg.size(), CV_64F), Mat::zeros(BlurredImg.size(), CV_64F) };	// 残差ベクトル
	Mat P_base[3] = { Mat::zeros(BlurredImg.size(), CV_64F), Mat::zeros(BlurredImg.size(), CV_64F), Mat::zeros(BlurredImg.size(), CV_64F) };	// 探索方向ベクトル
	for (c = 0; c < 3; c++) {
		doubleBlurredImg_sub[c].convertTo(LastX[c], CV_64F);
		doubleBlurredImg_sub[c].convertTo(NextX[c], CV_64F);

		Mat Mat_tmp = doubleNewImg2[c] - doubleNewImg1[c];
		Mat_tmp.copyTo(Residual[c]);
		Mat_tmp.copyTo(P_base[c]);
	}
	//P_base[0].convertTo(Image_dst_deblurred2, CV_8UC1);	// 確認

	Mat Alpha[3] = { Mat::zeros(BlurredImg.size(), CV_64F), Mat::zeros(BlurredImg.size(), CV_64F), Mat::zeros(BlurredImg.size(), CV_64F) };
	Mat Beta[3] = { Mat::zeros(BlurredImg.size(), CV_64F), Mat::zeros(BlurredImg.size(), CV_64F), Mat::zeros(BlurredImg.size(), CV_64F) };
	double ALPHA = 0.0, BETA = 0.0;
	double energy_ave, energy[3];
	Mat doubleNewImg[3];
	double nextX;
	//double KernelSize = (double)Karnel_Now.cols * (double)Karnel_Now.rows;
	for (c = 0; c < 3; c++) {
		//make_matrix_A(doubleBlurredImg[3], KernelSize, Rambda, doubleNewImg[3]);
		double color_mean = (double)mean(doubleBlurredImg[c])[0];
		cout << "   color_mean = " << (double)color_mean << endl;	// 確認用

		for (int i_number = 0; i_number < Iterate_Num; i_number++) {
			// Calculate ALPHA
			double Numerator, Denominator;
			Numerator = multi_vector(P_base[c], Residual[c]);		// ベクトルの内積
			multi_matrix_vector(P_base[c], dft_doubleNewImg[c], doubleNewImg[c]);
			//if (i_number == 1) { doubleNewImg[c].convertTo(Image_dst_deblurred2, CV_8UC1); checkMat_detail(doubleNewImg[c]); }	// 確認
			Denominator = multi_vector(P_base[c], doubleNewImg[c]);
			ALPHA = (double)(Numerator / Denominator);
			//cout << "  ALPHA = " << (double)ALPHA << endl;	// 確認用

			// Calculate Image
#pragma omp parallel for private(x)
			for (y = 0; y < BlurredImg.rows; y++) {
				for (x = 0; x < BlurredImg.cols; x++) {
					//cout << "  " << (double)NextX[c].at<double>(y, x) << " => ";	// 確認用
					nextX = LastX[c].at<double>(y, x) + (ALPHA * P_base[c].at<double>(y, x));
					NextX[c].at<double>(y, x) = (double)nextX;
					//cout << (double)NextX[c].at<double>(y, x) << " = " << (double)nextX << endl;	// 確認用
				}
			}

			Mat Residual_before[3];
			// Calculate Residual
			Residual[c].copyTo(Residual_before[c]);
			Residual[c] = Residual[c] - ALPHA * doubleNewImg[c];

			//			energy[c] = 0.0;
			//#pragma omp parallel for private(x)
			//			for (y = 0; y < Residual[c].rows; y++) {
			//				for (x = 0; x < Residual[c].cols; x++) {
			//					energy[c] += (double)Residual[c].at<double>(y, x);
			//				}
			//			}
			energy[c] = (double)norm(Residual[c]);
			//energy[c] = (double)mean(Residual[c])[0];
			energy[c] /= (double)((double)Residual[c].cols * (double)Residual[c].rows);
			//cout << "  " << (int)i_number << " : energy = " << (double)energy[c] << endl;	// 確認用
			if (energy[c] < ERROR_END_NUM) {
				cout << "  " << (int)c << " : " << (int)i_number << " : energy = " << (double)energy[c] << endl;	// 確認用
				break;
			}

			// Calculate BETA
			double Numerator2, Denominator2;
			Numerator2 = multi_vector(Residual_before[c], Residual_before[c]);
			Denominator2 = multi_vector(Residual[c], Residual[c]);
			BETA = (double)(Numerator2 / Denominator2);
			//cout << "  BETA = " << (double)BETA << endl;	// 確認用

			// Calculate P_base
#pragma omp parallel for private(x)
			for (y = 0; y < BlurredImg.rows; y++) {
				for (x = 0; x < BlurredImg.cols; x++) {
					P_base[c].at<double>(y, x) = Residual[c].at<double>(y, x) + (BETA * P_base[c].at<double>(y, x));
				}
			}

			//normalize(NextX[c], NextX[c], 0, 255, NORM_MINMAX);
			NextX[c].copyTo(LastX[c]);
			//if (i_number == 0) { NextX[0].convertTo(Image_dst_deblurred2, CV_8UC1); }	// 確認
		}

		double before_color_mean = (double)mean(NextX[c])[0];
		cout << "   before_color_mean = " << (double)before_color_mean << endl;	// 確認用
		Mat tmp;
		normalize(NextX[c], tmp, 0, 255, NORM_MINMAX);
		double after_color_mean = (double)mean(tmp)[0];
		//NextX[c] = NextX[c] + (color_mean - after_color_mean);
		//NextX[c] = NextX[c] + (color_mean - before_color_mean);
		//normalize(NextX[c], NextX[c], 0, 255, NORM_MINMAX);
		NextX[c] = NextX[c] * (double)(color_mean / before_color_mean);
		cout << "   diff_color_mean = " << (double)(color_mean - after_color_mean) << endl;	// 確認用
		cout << "  " << (int)c << " : energy = " << (double)energy[c] << endl;	// 確認用
	}
	//NextX[0].convertTo(Image_dst_deblurred2, CV_8UC1);	// 確認
	//checkMat_detail(NextX[0]);	// 確認

	Mat NewImg;
	merge(NextX, 3, NewImg);
	//NewImg.convertTo(Image_dst_deblurred2, CV_8UC1);	// 確認
	double E = (double)norm(BlurredImg, NewImg, NORM_L2) / (double)(BlurredImg.cols * BlurredImg.rows * 3.0);
	cout << "  E = " << (double)E << endl;	// 確認用

	NewImg.copyTo(Img_Now);
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