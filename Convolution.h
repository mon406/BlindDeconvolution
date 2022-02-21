#ifndef __INCLUDED_H_Convolution__
#define __INCLUDED_H_Convolution__

#include "main.h"
#include "Fourier_Transform_Mat.h"
#include "Calculate_Complex_Number.h"


/*--- Convolutionクラス -----------------------------------------------
	DFTを用いて画像にカーネルを畳み込むクラス (FTMatクラスを使用)

	TrueImage:			真画像 (入力用)
	ConvImage:			畳み込み画像 (出力用)
	KernelFilter:		カーネル (出力用)
	KernelFastOrder:	カーネル指定値 (入力用)
-----------------------------------------------------------------------*/
class Convolution {
private:
	int c = 0;
public:
	FTMat3D TrueImage;		// 真画像 (カラー)
	FTMat3D ConvImage;		// 畳み込み画像
	KERNEL KernelFilter;	// カーネル

	Convolution();																	// 初期化
	void convolved(Mat& True, Mat& Output, KERNEL& Kernel, int KernelFastOrder);	// カーネルの畳み込み(ぼけ画像作成)
	void Evaluatuion();																// 画像比較評価
};
Convolution::Convolution() {
	TrueImage = FTMat3D();
	ConvImage = FTMat3D();
	KernelFilter = KERNEL();
}
void Convolution::convolved(Mat& True, Mat& Output, KERNEL& Kernel, int KernelFastOrder) {
	/* 前処理 */
	// FTMatクラスに変換
	TrueImage = FTMat3D(True);
	ConvImage = FTMat3D();
	KernelFilter = KERNEL(KernelFastOrder);

	// DFT変換のサイズを計算
	int Mplus = True.rows + KernelFilter.rows;
	int Nplus = True.cols + KernelFilter.cols;
	int Msize = getOptimalDFTSize(Mplus);
	int Nsize = getOptimalDFTSize(Nplus);
	cout << " FFT Size  : (" << Nplus << "," << Mplus << ") => (" << Nsize << "," << Msize << ")" << endl;	// 確認
	// インデックスを指定して1次元ベクトルに変換
	TrueImage.toVector(1, 0, 1, Nsize, Msize);
	KernelFilter.toVector(1, 1, 0, Nsize, Msize);
	// DFT変換
	TrueImage.DFT();
	KernelFilter.DFT();

	/* ぼけ画像を求める */
	Mat dft_ConvolvedImg[3];
	for (c = 0; c < 3; c++) {
		mulSpectrums(TrueImage.FT_Mat[c].dft_ImgVec, KernelFilter.dft_ImgVec, dft_ConvolvedImg[c], 0, false);
	}

	/* 出力 */
	ConvImage = FTMat3D(dft_ConvolvedImg[0], dft_ConvolvedImg[1], dft_ConvolvedImg[2]);
	ConvImage.settingB(1, 0, 1, Nsize, Msize);
	ConvImage.settingAverageColor(TrueImage);
	// inverseDFT変換
	ConvImage.iDFT();
	// 2次元ベクトルに変換
	ConvImage.toMatrix(1, 0, True.cols, True.rows);

	KernelFilter.copyTo(Kernel);
	ConvImage.output(Output);
}
void Convolution::Evaluatuion() {
	Mat True_Img, Convolved_Img;
	TrueImage.output(True_Img);
	ConvImage.output(Convolved_Img);

	cout << "ぼけ画像 と 真画像 (ガウスノイズなし)" << endl;
	Evaluation_MSE_PSNR_SSIM(True_Img, Convolved_Img);
}


/*--- Deconvolutionクラス ---------------------------------------------
	DFTを用いて画像にカーネルを逆畳み込みするクラス (FTMatクラスを使用)
	インバースフィルターによるデコンボリューション
	BlurrImage:			ぼけ画像 (入力用)
	DeconvImage:		逆畳み込み画像 (出力用)
	KernelFilter:		カーネル (入力用)
-----------------------------------------------------------------------*/
class Deconvolution {
private:
	int c = 0;
public:
	FTMat3D BlurrImage;		// ぼけ画像
	FTMat3D DeconvImage;	// 逆畳み込み画像
	KERNEL KernelFilter;	// カーネル

	Deconvolution();											// 初期化
	void deconvolved(Mat& Blurr, Mat& Output, KERNEL& Kernel);	// 単純(理論的)逆畳み込み
	void Evaluatuion(Mat& True);								// 画像比較評価
};
Deconvolution::Deconvolution() {
	BlurrImage = FTMat3D();
	DeconvImage = FTMat3D();
	KernelFilter = KERNEL();
}
void Deconvolution::deconvolved(Mat& Blurr, Mat& Output, KERNEL& Kernel) {
	/* 前処理 */
	// FTMatクラスに変換
	BlurrImage = FTMat3D(Blurr);
	DeconvImage = FTMat3D();
	KernelFilter = KERNEL();
	Kernel.copyTo(KernelFilter);

	// DFT変換のサイズを計算
	int Mplus = Blurr.rows + KernelFilter.rows;
	int Nplus = Blurr.cols + KernelFilter.cols;
	int Msize = getOptimalDFTSize(Mplus);
	int Nsize = getOptimalDFTSize(Nplus);
	cout << " FFT Size  : (" << Nplus << "," << Mplus << ") => (" << Nsize << "," << Msize << ")" << endl;	// 確認
	// インデックスを指定して1次元ベクトルに変換
	BlurrImage.toVector(1, 0, 1, Nsize, Msize);
	KernelFilter.toVector(1, 1, 0, Nsize, Msize);
	// DFT変換
	BlurrImage.DFT();
	KernelFilter.DFT();

	/* ぼけ除去画像(真の元画像)を求める */
	Mat dft_DeconvolvedImg[3];
	Mat abs_pow_KernelFilter, denom_KernelFilter;
	abs_pow_complex_Mat(KernelFilter.dft_ImgVec, abs_pow_KernelFilter);		// 2次元ベクトルの大きさの２乗
	reciprocal_complex_Mat(abs_pow_KernelFilter, denom_KernelFilter);		// 2次元ベクトルの逆数
	for (c = 0; c < 3; c++) {
		mulSpectrums(BlurrImage.FT_Mat[c].dft_ImgVec, KernelFilter.dft_ImgVec, dft_DeconvolvedImg[c], 0, true);	// 複素共役
		mulSpectrums(dft_DeconvolvedImg[c], denom_KernelFilter, dft_DeconvolvedImg[c], 0, false);
	}

	/* 出力 */
	DeconvImage = FTMat3D(dft_DeconvolvedImg[0], dft_DeconvolvedImg[1], dft_DeconvolvedImg[2]);
	DeconvImage.settingB(1, 0, 1, Nsize, Msize);
	DeconvImage.settingAverageColor(BlurrImage);
	// inverseDFT変換
	DeconvImage.iDFT();
	// 2次元ベクトルに変換
	DeconvImage.toMatrix(1, 0, Blurr.cols, Blurr.rows);

	DeconvImage.output(Output);
}
void Deconvolution::Evaluatuion(Mat& True) {
	Mat Deconvolved_Img;
	DeconvImage.output(Deconvolved_Img);

	cout << "ぼけ除去画像 と 真画像" << endl;
	Evaluation_MSE_PSNR_SSIM(True, Deconvolved_Img);
}

/*--- Deconvolution_WFクラス ------------------------------------------
	Deconvolutionクラスを継承した逆畳み込みクラス (FTMatクラスを使用)
	ウィーナ・フィルターによるデコンボリューション
	TrueImage:			真画像 (入力用)
	BlurrImage:			ぼけ画像 (入力用)
	DeconvImage:		逆畳み込み画像 (出力用)
	KernelFilter:		カーネル (入力用)
-----------------------------------------------------------------------*/
class Deconvolution_WF : public Deconvolution {
private:
	int x = 0, y = 0, c = 0;
	int index = 0;
public:
	FTMat3D TrueImage;		// 真画像 (カラー)
	//FTMat3D BlurrImage;	// ぼけ画像
	//FTMat3D DeconvImage;	// 逆畳み込み画像
	//KERNEL KernelFilter;	// カーネル
	Mat dft_TrueConstant;		// ウィーナ・フィルターの真の係数

	Deconvolution_WF();													// 初期化
	void calcWienerFilterConstant(Convolution& InputData, Mat& Blurr);	// FFTの誤差より真の係数を求める
	void deconvolved_WF(Mat& Blurr, Mat& Output, KERNEL& Kernel);		// 逆畳み込み (ウィーナ・フィルター)
};
Deconvolution_WF::Deconvolution_WF() {
	TrueImage = FTMat3D();
	BlurrImage = FTMat3D();
	DeconvImage = FTMat3D();
	KernelFilter = KERNEL();
	dft_TrueConstant.release();
}
void Deconvolution_WF::calcWienerFilterConstant(Convolution& InputData, Mat& Blurr) {
	/* ConvolutionのFTMatクラスを用いる*/
	BlurrImage = FTMat3D(Blurr);
	InputData.TrueImage.copyTo(TrueImage);
	InputData.KernelFilter.copyTo(KernelFilter);

	// DFT変換のサイズを計算
	int Mplus = TrueImage.FT_Mat[0].ImgMat.rows + KernelFilter.rows;
	int Nplus = TrueImage.FT_Mat[0].ImgMat.cols + KernelFilter.cols;
	int Msize = getOptimalDFTSize(Mplus);
	int Nsize = getOptimalDFTSize(Nplus);
	cout << " FFT Size  : (" << TrueImage.FT_Mat[0].ImgMat.cols << "+" << KernelFilter.cols << ", " << TrueImage.FT_Mat[0].ImgMat.rows << "+" << KernelFilter.rows << ")" << endl;	// 確認
	cout << " FFT Size  : (" << Nplus << "," << Mplus << ") => (" << Nsize << "," << Msize << ")" << endl;	// 確認
	// インデックスを指定して1次元ベクトルに変換
	BlurrImage.toVector(1, 0, 1, Nsize, Msize);
	TrueImage.toVector(1, 0, 1, Nsize, Msize);
	KernelFilter.toVector(1, 1, 0, Nsize, Msize);
	// DFT変換
	BlurrImage.DFT();
	TrueImage.DFT();
	KernelFilter.DFT();

	/* 真の係数を計算 */
	Mat dft_TrueConstant_RGB[3];
	Mat denom_dft_TrueImage[3];
	Mat dft_DeconvolvedImg_tmp1[3], dft_DeconvolvedImg_tmp2[3];
	Mat abs_pow_KernelFilter;
	abs_pow_complex_Mat(KernelFilter.dft_ImgVec, abs_pow_KernelFilter);		// 2次元ベクトルの大きさの２乗
	for (c = 0; c < 3; c++) {
		mulSpectrums(BlurrImage.FT_Mat[c].dft_ImgVec, KernelFilter.dft_ImgVec, dft_DeconvolvedImg_tmp1[c], 0, true);	// 複素共役
		reciprocal_complex_Mat(TrueImage.FT_Mat[c].dft_ImgVec, denom_dft_TrueImage[c]);									// 2次元ベクトルの逆数
		mulSpectrums(dft_DeconvolvedImg_tmp1[c], denom_dft_TrueImage[c], dft_DeconvolvedImg_tmp2[c], 0, false);

		dft_TrueConstant_RGB[c] = Mat::zeros(1, Nsize * Msize, CV_64FC2);
		dft_TrueConstant_RGB[c] = dft_DeconvolvedImg_tmp2[c] - abs_pow_KernelFilter;
	}

	// 真の係数のMat確認
	dft_TrueConstant = Mat::zeros(1, Nsize * Msize, CV_64FC2);
	Vec2d ave_number[3], ave_num;
	Vec2d All_Num = { (double)Nsize * (double)Msize, 0.0 };
	Vec2d Color_Num = { 3.0, 0.0 };
	Vec2d Error = { 0.0, 0.0 };
#pragma omp parallel for private(y, x, index)
	for (c = 0; c < 3; c++) {
		ave_number[c] = { 0.0, 0.0 };
		index = 0;
		for (y = 0; y < Msize; y++) {
			for (x = 0; x < Nsize; x++) {
				ave_num = dft_TrueConstant_RGB[c].at<Vec2d>(0, index);
				ave_number[c] += ave_num;
				divi_Vec2d(ave_num, ave_num, Color_Num);
				dft_TrueConstant.at<Vec2d>(0, index) += ave_num;
				index++;
			}
		}
		divi_Vec2d(ave_number[c], ave_number[c], All_Num);
		cout << "  " << c << " : " << ave_number[c] << endl;	// 確認用

		Error += ave_number[c];
	}
	divi_Vec2d(Error, Error, Color_Num);
	cout << " Error = " << Error << endl;	// 確認用
	cout << endl;
	//checkMat_detail(dft_TrueConstant);	// 確認用
}
void Deconvolution_WF::deconvolved_WF(Mat& Blurr, Mat& Output, KERNEL& Kernel) {
	/* 前処理 */
	// FTMatクラスに変換
	BlurrImage = FTMat3D(Blurr);
	DeconvImage = FTMat3D();
	KernelFilter = KERNEL();
	Kernel.copyTo(KernelFilter);

	// DFT変換のサイズを計算
	int Mplus = Blurr.rows + KernelFilter.rows;
	int Nplus = Blurr.cols + KernelFilter.cols;
	int Msize = getOptimalDFTSize(Mplus);
	int Nsize = getOptimalDFTSize(Nplus);
	cout << " FFT Size  : (" << Nplus << "," << Mplus << ") => (" << Nsize << "," << Msize << ")" << endl;	// 確認
	// インデックスを指定して1次元ベクトルに変換
	BlurrImage.toVector(1, 0, 1, Nsize, Msize);
	KernelFilter.toVector(1, 1, 0, Nsize, Msize);
	// DFT変換
	BlurrImage.DFT();
	KernelFilter.DFT();

	/* ぼけ除去画像(真の元画像)を求める */
	Mat dft_DeconvolvedImg[3];
	Mat denom_WienerFilter;
	wiener_filter(KernelFilter.dft_ImgVec, denom_WienerFilter, dft_TrueConstant);	// 2次元ベクトルの逆数(ウィーナ・フィルター)
	for (c = 0; c < 3; c++) {
		mulSpectrums(BlurrImage.FT_Mat[c].dft_ImgVec, denom_WienerFilter, dft_DeconvolvedImg[c], 0, false);
	}

	/* 出力 */
	DeconvImage = FTMat3D(dft_DeconvolvedImg[0], dft_DeconvolvedImg[1], dft_DeconvolvedImg[2]);
	DeconvImage.settingB(1, 0, 1, Nsize, Msize);
	DeconvImage.settingAverageColor(BlurrImage);
	// inverseDFT変換
	DeconvImage.iDFT();
	// 2次元ベクトルに変換
	DeconvImage.toMatrix(1, 0, Blurr.cols, Blurr.rows);

	DeconvImage.output(Output);
}


#endif