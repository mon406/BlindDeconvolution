#include "main.h"
#include "MakeKernel.h"
#include "CalculateComplexNumber.h"
#include "DiscreteFourierTransform.h"
#include "BlindDeconvolution.h"

/* 関数(後定義) */
void ConvolvedImage(Mat& InputImage, Mat& OutputImage, KERNEL& KernelImage);								// カーネル画像を用いた畳み込み
void DeconvolvedImage(Mat& InputImage, Mat& OutputImage, KERNEL& KernelImage, Mat& FilterParameter);		// カーネルを用いた逆畳み込み(wiener filter)
void DeconvolvedImage_simple(Mat& InputImage, Mat& OutputImage1, Mat& OutputImage2, KERNEL& KernelImage);	// カーネルを用いた逆畳み込み(inverse filter)
void DeconvolvedImage_simple_true(Mat& TrueImage, Mat& InputImage, Mat& OutputParameter, KERNEL& KernelImage);	// FFTの誤差より真の係数を求める


int main() {
	/* 画像の入力 */
	Input_Image();
	Image_src.copyTo(Image_dst);
	Image_dst.copyTo(Image_dst_deblurred);
	Image_dst.copyTo(Image_dst_deblurred2);

	clock_t start, end;	// 処理時間表示用
	start = clock();
	//--- 画像処理 -------------------------------------------------------------------------------
	/* カーネル生成 */
	cout << "カーネル生成..." << endl;			// 実行確認用
	KERNEL kernel = KERNEL(0);
	kernel.Kernel.copyTo(Image_kernel_original);
	//kernel.display_detail();		// 確認用

	/* ぼけ画像生成 */
	cout << "ぼけ画像生成..." << endl;			// 実行確認用
	ConvolvedImage(Image_src, Image_dst, kernel);				// I = x ** k
	// ノイズを考慮するための係数を決定
	Mat GammaMat;
	DeconvolvedImage_simple_true(Image_src, Image_dst, GammaMat, kernel);
	// ガウスノイズ付加
	GaussianBlur(Image_dst, Image_dst, Size(11, 11), 1, 1);		// I = I + n
	Image_dst.copyTo(Image_dst_deblurred);
	Image_dst.copyTo(Image_dst_deblurred2);

	/* 量子化画像チェック */
	//QuantMatDouble QuantizedImageD = QuantMatDouble(10, Image_src);		// Quantize x
	//QuantizedImageD.quantedQMat();
	//QuantizedImageD.QMat.convertTo(Image_dst_deblurred2, CV_8UC3);
	//ConvolvedImage(Image_dst_deblurred2, Image_dst_deblurred2, kernel);				// I = x ** k
	//GaussianBlur(Image_dst_deblurred2, Image_dst_deblurred2, Size(11, 11), 1, 1);	// I = I + n


	/* ぼけ除去 */
	cout << "ぼけ除去処理..." << endl;			// 実行確認用
	//DeconvolvedImage_simple(Image_dst, Image_dst_deblurred, Image_dst_deblurred2, kernel);	// inverse filter
	DeconvolvedImage(Image_dst, Image_dst_deblurred2, kernel, GammaMat);					// wiener filter

	kernel = KERNEL(0);
	Blind_Deconvolution BlindDeconvolusion = Blind_Deconvolution();
	BlindDeconvolusion.deblurring(Image_src, Image_dst_deblurred, kernel);

	//--------------------------------------------------------------------------------------------
	end = clock();
	double time_difference = (double)end - (double)start;
	const double time = time_difference / CLOCKS_PER_SEC * 1000.0;
	cout << "time : " << time << " [ms]" << endl;
	cout << endl;

	/* カーネル可視化 */
	KernelImage_Visualization(Image_kernel_original);
	//KernelImage_Visualization_double(Image_kernel_original);
	kernel.inverse_normalization();
	kernel.visualization(Image_kernel);
	//kernel.display_detail();	// 確認用

	/* 画像の評価 */
	cout << "ぼけ画像 と 真画像" << endl;				// 実行確認用
	Evaluation_MSE_PSNR_SSIM(Image_src, Image_dst);
	cout << "ぼけ除去画像 と 真画像" << endl;			// 実行確認用
	Evaluation_MSE_PSNR_SSIM(Image_src, Image_dst_deblurred);
	cout << "ぼけ除去画像2 と 真画像" << endl;			// 実行確認用
	Evaluation_MSE_PSNR_SSIM(Image_src, Image_dst_deblurred2);
	cout << "推定カーネル と 真のカーネル" << endl;		// 実行確認用
	Evaluation_MSE_PSNR_SSIM(Image_kernel, Image_kernel_original);

	//Image_kernel.convertTo(Image_kernel, CV_8UC1);
	//Image_kernel_original.convertTo(Image_kernel_original, CV_8UC1);

	/* ヒストグラム作成 */
	//DrawHist(Image_src, Image_src_hist);
	//DrawHist(Image_dst, Image_dst_hist);

	/* 画像の出力 */
	Output_Image();

	return 0;
}

// 画像の入力
void Input_Image() {
	string file_src = "C:\\Users\\mon25\\Desktop\\BlindDeconvolution\\src.jpg";		// 入力画像のファイル名
	//string file_src = "C:\\Users\\Yuki Momma\\Desktop\\BlindDeconvolution\\src.jpg";	// 入力画像のファイル名
	Image_src = imread(file_src, 1);		// 入力画像（カラー）の読み込み
	Image_src_gray = imread(file_src, 0);	// 入力画像（グレースケール）の読み込み

	/* パラメータ定義 */
	WIDTH = Image_src.cols;
	HEIGHT = Image_src.rows;
	MAX_DATA = WIDTH * HEIGHT;
	cout << "INPUT : WIDTH = " << WIDTH << " , HEIGHT = " << HEIGHT << endl;
	cout << endl;

	Image_dst = Mat(Size(WIDTH, HEIGHT), CV_8UC3);	// 出力画像（カラー）の初期化
	Image_dst_deblurred = Mat(Size(WIDTH, HEIGHT), CV_8UC3);
	Image_dst_deblurred2 = Mat(Size(WIDTH, HEIGHT), CV_8UC3);
}
// 画像の出力
void Output_Image() {
	string file_dst = "C:\\Users\\mon25\\Desktop\\BlindDeconvolution\\dst.jpg";		// 出力画像のファイル名
	//string file_dst2 = "C:\\Users\\mon25\\Desktop\\BlindDeconvolution\\dst_hist.jpg";
	//string file_dst3 = "C:\\Users\\mon25\\Desktop\\BlindDeconvolution\\src_hist.jpg";
	string file_dst4 = "C:\\Users\\mon25\\Desktop\\BlindDeconvolution\\dst_kernel.jpg";
	string file_dst5 = "C:\\Users\\mon25\\Desktop\\BlindDeconvolution\\dst_deblurred.jpg";
	string file_dst6 = "C:\\Users\\mon25\\Desktop\\BlindDeconvolution\\dst_deblurred2.jpg";
	//string file_dst = "C:\\Users\\Yuki Momma\\Desktop\\BlindDeconvolution\\dst.jpg";	// 出力画像のファイル名
	////string file_dst2 = "C:\\Users\\Yuki Momma\\Desktop\\BlindDeconvolution\\dst_hist.jpg";
	////string file_dst3 = "C:\\Users\\Yuki Momma\\Desktop\\BlindDeconvolution\\src_hist.jpg";
	//string file_dst4 = "C:\\Users\\Yuki Momma\\Desktop\\BlindDeconvolution\\dst_kernel.jpg";
	//string file_dst5 = "C:\\Users\\Yuki Momma\\Desktop\\BlindDeconvolution\\dst_deblurred.jpg";
	//string file_dst6 = "C:\\Users\\Yuki Momma\\Desktop\\BlindDeconvolution\\dst_deblurred2.jpg";

	/* ウィンドウ生成 */
	namedWindow(win_src, WINDOW_AUTOSIZE);
	namedWindow(win_dst, WINDOW_AUTOSIZE);
	namedWindow(win_dst2, WINDOW_AUTOSIZE);
	namedWindow(win_dst3, WINDOW_AUTOSIZE);
	namedWindow(win_dst4, WINDOW_AUTOSIZE);

	/* 画像の表示 & 保存 */
	imshow(win_src, Image_src);				// 入力画像を表示
	imshow(win_dst, Image_dst);				// 出力画像を表示
	imwrite(file_dst, Image_dst);			// 処理結果の保存
	//imwrite(file_dst2, Image_dst_hist);		// 出力ヒストグラム画像の保存
	//imwrite(file_dst3, Image_src_hist);		// 入力ヒストグラム画像の保存
	imshow(win_dst2, Image_kernel);			// 出力画像を表示
	imwrite(file_dst4, Image_kernel);		// 処理結果の保存
	imshow(win_dst3, Image_dst_deblurred);		// 出力画像を表示
	imwrite(file_dst5, Image_dst_deblurred);	// 処理結果の保存
	imshow(win_dst4, Image_dst_deblurred2);		// 出力画像を表示
	imwrite(file_dst6, Image_dst_deblurred2);	// 処理結果の保存

	waitKey(0); // キー入力待ち
}

// カーネル画像を用いたFFTによる畳み込み
void ConvolvedImage(Mat& InputImage, Mat& OutputImage, KERNEL& KernelImage) {
	int c;

	/* 前処理 */
	// 画像をCV_64Fに変換
	Mat TrueKernel;
	KernelImage.Kernel_normalized.copyTo(TrueKernel);
	Mat TrueImg;
	InputImage.convertTo(TrueImg, CV_64FC3);

	// DFT変換のサイズを計算
	int Mplus = TrueImg.rows + TrueKernel.rows;
	int Nplus = TrueImg.cols + TrueKernel.cols;
	int Msize = getOptimalDFTSize(Mplus);
	int Nsize = getOptimalDFTSize(Nplus);
	cout << " FFT Size  : (" << Mplus << "," << Nplus << ") => (" << Msize << "," << Nsize << ")" << endl;	// 確認

	// 3つのチャネルB, G, Rに分離
	Mat doubleTrueImg_sub[3] = { Mat::zeros(TrueImg.size(), CV_64F), Mat::zeros(TrueImg.size(), CV_64F), Mat::zeros(TrueImg.size(), CV_64F) };
	split(TrueImg, doubleTrueImg_sub);
	Mat doubleTrueImg[3];
	for (c = 0; c < 3; c++) {
		Mat planes[] = { Mat_<double>(doubleTrueImg_sub[c]), Mat::zeros(doubleTrueImg_sub[c].size(), CV_64F) };
		merge(planes, 2, doubleTrueImg[c]);
	}


	/* DFT */
	// カーネルのDFT
	Mat dft_TrueKernel = Mat::zeros(Msize, Nsize, CV_64FC2);
	Mat planes2[] = { Mat_<double>(TrueKernel), Mat::zeros(TrueKernel.size(), CV_64F) };
	merge(planes2, 2, TrueKernel);
	copyMakeBorder(TrueKernel, dft_TrueKernel, 0, Msize - TrueKernel.rows, 0, Nsize - TrueKernel.cols, BORDER_CONSTANT, (0.0, 0.0));
	dft(dft_TrueKernel, dft_TrueKernel, 0, dft_TrueKernel.rows);
	//visualbule_complex(dft_TrueKernel, Image_dst_deblurred2);	// 確認

	// 真の元画像のDFT
	Mat dft_doubleTrueImg[3] = { Mat::zeros(Msize, Nsize, CV_64FC2), Mat::zeros(Msize, Nsize, CV_64FC2), Mat::zeros(Msize, Nsize, CV_64FC2) };
	for (c = 0; c < 3; c++) {
		copyMakeBorder(doubleTrueImg[c], dft_doubleTrueImg[c], TrueKernel.rows / 2, Msize - Mplus + TrueKernel.rows / 2, TrueKernel.cols / 2, Nsize - Nplus + TrueKernel.cols / 2, BORDER_REPLICATE);
		dft(dft_doubleTrueImg[c], dft_doubleTrueImg[c], 0, dft_doubleTrueImg[c].rows);
	}
	//visualbule_complex(dft_doubleTrueImg_2[0], Image_dst_deblurred2);	// 確認


	/* ぼけ画像を求める */
	Mat dft_doubleBlurredImg[3];
	Mat dft_denomImg[3];
	for (c = 0; c < 3; c++) {
		mulSpectrums(dft_doubleTrueImg[c], dft_TrueKernel, dft_doubleBlurredImg[c], 0, false);
	}
	//visualbule_complex(dft_doubleBlurredImg[0], Image_dst_deblurred2);	// 確認


	/* 出力 */
	// inverseDFT
	Mat doubleBlurredImg[3];
	for (c = 0; c < 3; c++) {
		dft(dft_doubleBlurredImg[c], dft_doubleBlurredImg[c], cv::DFT_INVERSE + cv::DFT_SCALE | cv::DFT_REAL_OUTPUT);
		doubleBlurredImg[c] = dft_doubleBlurredImg[c](Rect(TrueKernel.cols, TrueKernel.rows, TrueImg.cols, TrueImg.rows));
	}
	Mat BlurredImage;
	merge(doubleBlurredImg, 3, BlurredImage);

	BlurredImage.convertTo(BlurredImage, CV_8UC3);
	BlurredImage.copyTo(OutputImage);
}

// カーネルを用いたFFTによる逆畳み込み (ウィーナ・フィルター)
void DeconvolvedImage(Mat& InputImage, Mat& OutputImage, KERNEL& KernelImage, Mat& FilterParameter) {
	int x, y, c;

	/* 前処理 */
	// 画像をCV_64Fに変換
	Mat TrueKernel;
	KernelImage.Kernel_normalized.copyTo(TrueKernel);
	Mat BlurredImg;
	InputImage.convertTo(BlurredImg, CV_64FC3);

	// DFT変換のサイズを計算
	int Mplus = BlurredImg.rows + TrueKernel.rows;
	int Nplus = BlurredImg.cols + TrueKernel.cols;
	int Msize = getOptimalDFTSize(Mplus);
	int Nsize = getOptimalDFTSize(Nplus);
	cout << " FFT Size  : (" << Mplus << "," << Nplus << ") => (" << Msize << "," << Nsize << ")" << endl;	// 確認

	// 3つのチャネルB, G, Rに分離
	Mat doubleBlurredImg_sub[3] = { Mat::zeros(BlurredImg.size(), CV_64F), Mat::zeros(BlurredImg.size(), CV_64F), Mat::zeros(BlurredImg.size(), CV_64F) };
	split(BlurredImg, doubleBlurredImg_sub);
	Mat doubleBlurredImg[3];
	for (c = 0; c < 3; c++) {
		Mat planes[] = { Mat_<double>(doubleBlurredImg_sub[c]), Mat::zeros(doubleBlurredImg_sub[c].size(), CV_64F) };
		merge(planes, 2, doubleBlurredImg[c]);
	}


	/* DFT */
	// カーネルのDFT
	Mat dft_TrueKernel = Mat::zeros(Msize, Nsize, CV_64FC2);
	Mat planes3[] = { Mat_<double>(TrueKernel), Mat::zeros(TrueKernel.size(), CV_64F) };
	merge(planes3, 2, TrueKernel);
	copyMakeBorder(TrueKernel, dft_TrueKernel, 0, Msize - TrueKernel.rows, 0, Nsize - TrueKernel.cols, BORDER_CONSTANT, (0.0, 0.0));
	dft(dft_TrueKernel, dft_TrueKernel, 0, dft_TrueKernel.rows);
	//visualbule_complex(dft_TrueKernel, Image_dst_deblurred);	// 確認

	// ぼけ画像のDFT
	Mat dft_doubleBlurredImg[3] = { Mat::zeros(Msize, Nsize, CV_64FC2), Mat::zeros(Msize, Nsize, CV_64FC2), Mat::zeros(Msize, Nsize, CV_64FC2) };
	for (c = 0; c < 3; c++) {
		copyMakeBorder(doubleBlurredImg[c], dft_doubleBlurredImg[c], TrueKernel.rows / 2, Msize - Mplus + TrueKernel.rows / 2, TrueKernel.cols / 2, Nsize - Nplus + TrueKernel.cols / 2, BORDER_REPLICATE);
		dft(dft_doubleBlurredImg[c], dft_doubleBlurredImg[c], 0, dft_doubleBlurredImg[c].rows);
	}
	//visualbule_complex(dft_doubleBlurredImg_2[0], Image_dst_deblurred);	// 確認


	/* 真の元画像を求める */
	Mat dft_doubleTrueImg[3];
	Mat dft_WienerKernel;
	wiener_filter(dft_TrueKernel, dft_WienerKernel, FilterParameter);		// 2次元ベクトルの逆数(ウィーナ・フィルター)
	//visualbule_complex(dft_WienerKernel, Image_dst_deblurred);	// 確認
	for (c = 0; c < 3; c++) {
		mulSpectrums(dft_doubleBlurredImg[c], dft_WienerKernel, dft_doubleTrueImg[c], 0, false);
	}
	//visualbule_complex(dft_doubleTrueImg[0], Image_dst_deblurred);	// 確認


	/* 出力 */
	// inverseDFT
	Mat doubleTrueImg[3];
	for (c = 0; c < 3; c++) {
		dft(dft_doubleTrueImg[c], dft_doubleTrueImg[c], cv::DFT_INVERSE + cv::DFT_SCALE | cv::DFT_REAL_OUTPUT);
		doubleTrueImg[c] = dft_doubleTrueImg[c](Rect(TrueKernel.cols / 2, TrueKernel.rows / 2, BlurredImg.cols, BlurredImg.rows));
		//doubleTrueImg[c] = dft_doubleTrueImg[c](Rect(TrueKernel.cols, TrueKernel.rows, BlurredImg.cols, BlurredImg.rows));
	}
	Mat TrueImage;
	merge(doubleTrueImg, 3, TrueImage);
	TrueImage.convertTo(TrueImage, CV_8U);
	TrueImage.copyTo(OutputImage);

	// カーネル
	reciprocal_complex(dft_WienerKernel, dft_WienerKernel);
	//visualbule_complex(dft_WienerKernel, Image_dst_deblurred);	// 確認
	dft(dft_WienerKernel, dft_WienerKernel, cv::DFT_INVERSE + cv::DFT_SCALE | cv::DFT_REAL_OUTPUT);
	copyMakeBorder(dft_WienerKernel, dft_WienerKernel, TrueKernel.rows / 2, TrueKernel.rows / 2, TrueKernel.cols / 2, TrueKernel.cols / 2, BORDER_WRAP);
	Mat KernelTmp = dft_WienerKernel(Rect(0, 0, TrueKernel.cols, TrueKernel.rows));
	//Mat KernelTmp = dft_WienerKernel(Rect(0, 0, Nsize, Msize));
	double double_num;
#pragma omp parallel for private(x)
	for (y = 0; y < KernelTmp.rows; y++) {
		for (x = 0; x < KernelTmp.cols; x++) {
			double_num = KernelTmp.at<double>(y, x);	// 負の値は0にする
			//cout << "  double_num = " << double_num << endl;	// 確認用
			if (double_num < 0) {
				double_num = 0.0;
				KernelTmp.at<double>(y, x) = double_num;
			}
		}
	}
	KernelTmp.copyTo(KernelImage.Kernel_normalized);
}
// カーネルを用いたFFTによる単純(理論的)逆畳み込み
void DeconvolvedImage_simple(Mat& InputImage, Mat& OutputImage1, Mat& OutputImage2, KERNEL& KernelImage) {
	int c;

	/* 前処理 */
	// 画像をCV_64Fに変換
	Mat TrueKernel;
	KernelImage.Kernel_normalized.copyTo(TrueKernel);
	Mat BlurredImg;
	InputImage.convertTo(BlurredImg, CV_64FC3);

	// DFT変換のサイズを計算
	int Mplus = BlurredImg.rows + TrueKernel.rows;
	int Nplus = BlurredImg.cols + TrueKernel.cols;
	int Msize = getOptimalDFTSize(Mplus);
	int Nsize = getOptimalDFTSize(Nplus);
	cout << " FFT Size  : (" << Mplus << "," << Nplus << ") => (" << Msize << "," << Nsize << ")" << endl;	// 確認

	// 3つのチャネルB, G, Rに分離
	Mat doubleBlurredImg_sub[3] = { Mat::zeros(BlurredImg.size(), CV_64F), Mat::zeros(BlurredImg.size(), CV_64F), Mat::zeros(BlurredImg.size(), CV_64F) };
	split(BlurredImg, doubleBlurredImg_sub);
	Mat doubleBlurredImg[3];
	for (c = 0; c < 3; c++) {
		Mat planes[] = { Mat_<double>(doubleBlurredImg_sub[c]), Mat::zeros(doubleBlurredImg_sub[c].size(), CV_64F) };
		merge(planes, 2, doubleBlurredImg[c]);
	}


	/* DFT */
	// カーネルのDFT
	Mat dft_TrueKernel = Mat::zeros(Msize, Nsize, CV_64FC2);
	Mat planes3[] = { Mat_<double>(TrueKernel), Mat::zeros(TrueKernel.size(), CV_64F) };
	merge(planes3, 2, TrueKernel);
	copyMakeBorder(TrueKernel, dft_TrueKernel, 0, Msize - TrueKernel.rows, 0, Nsize - TrueKernel.cols, BORDER_CONSTANT, (0.0, 0.0));
	dft(dft_TrueKernel, dft_TrueKernel, 0, dft_TrueKernel.rows);
	//visualbule_complex(dft_TrueKernel, Image_dst_deblurred2);	// 確認

	// ぼけ画像のDFT
	Mat dft_doubleBlurredImg[3] = { Mat::zeros(Msize, Nsize, CV_64FC2), Mat::zeros(Msize, Nsize, CV_64FC2), Mat::zeros(Msize, Nsize, CV_64FC2) };
	for (c = 0; c < 3; c++) {
		copyMakeBorder(doubleBlurredImg[c], dft_doubleBlurredImg[c], TrueKernel.rows / 2, Msize - Mplus + TrueKernel.rows / 2, TrueKernel.cols / 2, Nsize - Nplus + TrueKernel.cols / 2, BORDER_REPLICATE);
		dft(dft_doubleBlurredImg[c], dft_doubleBlurredImg[c], 0, dft_doubleBlurredImg[c].rows);
	}
	//visualbule_complex(dft_doubleBlurredImg_2[0], Image_dst_deblurred2);	// 確認


	/* 真の元画像を求める */
	Mat dft_doubleTrueImg[3];
	Mat dft_doubleTrueImg2[3];
	Mat dft_denomImg[3], dft_denomImg2[3];
	for (c = 0; c < 3; c++) {
		abs_pow_complex(dft_TrueKernel, dft_denomImg[c]);
		reciprocal_complex(dft_denomImg[c], dft_denomImg[c]);
		mulSpectrums(dft_doubleBlurredImg[c], dft_TrueKernel, dft_doubleTrueImg[c], 0, true);	// 複素共役
		mulSpectrums(dft_doubleTrueImg[c], dft_denomImg[c], dft_doubleTrueImg[c], 0, false);

		reciprocal_complex(dft_TrueKernel, dft_denomImg2[c]);
		//visualbule_complex(dft_denomImg2[0], Image_dst_deblurred);	// 確認
		mulSpectrums(dft_doubleBlurredImg[c], dft_denomImg2[c], dft_doubleTrueImg2[c], 0, false);
	}
	//visualbule_complex(dft_doubleTrueImg[0], Image_dst_deblurred2);	// 確認
	//visualbule_complex(dft_doubleTrueImg2[0], Image_dst_deblurred2);	// 確認


	/* 出力 */
	// inverseDFT
	Mat doubleTrueImg[3];
	Mat doubleTrueImg2[3];
	for (c = 0; c < 3; c++) {
		dft(dft_doubleTrueImg[c], dft_doubleTrueImg[c], cv::DFT_INVERSE + cv::DFT_SCALE | cv::DFT_REAL_OUTPUT);
		dft(dft_doubleTrueImg2[c], dft_doubleTrueImg2[c], cv::DFT_INVERSE + cv::DFT_SCALE | cv::DFT_REAL_OUTPUT);
		doubleTrueImg[c] = dft_doubleTrueImg[c](Rect(TrueKernel.cols, TrueKernel.rows, BlurredImg.cols, BlurredImg.rows));
		doubleTrueImg2[c] = dft_doubleTrueImg2[c](Rect(TrueKernel.cols / 2, TrueKernel.rows / 2, BlurredImg.cols, BlurredImg.rows));
	}
	Mat TrueImage;
	Mat TrueImage2;
	merge(doubleTrueImg, 3, TrueImage);
	merge(doubleTrueImg2, 3, TrueImage2);

	TrueImage.convertTo(TrueImage, CV_8U);
	TrueImage2.convertTo(TrueImage2, CV_8U);
	TrueImage.copyTo(OutputImage1);
	TrueImage2.copyTo(OutputImage2);
}

// FFTの誤差より真の係数を求める
void DeconvolvedImage_simple_true(Mat& TrueImage, Mat& InputImage, Mat& OutputParameter, KERNEL& KernelImage) {
	int x, y, c;

	/* 前処理 */
	// 画像をCV_64Fに変換
	Mat TrueKernel;
	KernelImage.Kernel_normalized.copyTo(TrueKernel);
	Mat TrueImg;
	TrueImage.convertTo(TrueImg, CV_64FC3);
	Mat BlurredImg;
	InputImage.convertTo(BlurredImg, CV_64FC3);

	// DFT変換のサイズを計算
	int Mplus = TrueImg.rows + TrueKernel.rows;
	int Nplus = TrueImg.cols + TrueKernel.cols;
	int Msize = getOptimalDFTSize(Mplus);
	int Nsize = getOptimalDFTSize(Nplus);
	//cout << " FFT Size  : (" << Mplus << "," << Nplus << ") => (" << Msize << "," << Nsize << ")" << endl;	// 確認

	// 3つのチャネルB, G, Rに分離
	Mat doubleTrueImg_sub[3] = { Mat::zeros(TrueImg.size(), CV_64F), Mat::zeros(TrueImg.size(), CV_64F), Mat::zeros(TrueImg.size(), CV_64F) };
	split(TrueImg, doubleTrueImg_sub);
	Mat doubleTrueImg[3];
	for (c = 0; c < 3; c++) {
		Mat planes[] = { Mat_<double>(doubleTrueImg_sub[c]), Mat::zeros(doubleTrueImg_sub[c].size(), CV_64F) };
		merge(planes, 2, doubleTrueImg[c]);
	}
	Mat doubleBlurredImg_sub[3] = { Mat::zeros(BlurredImg.size(), CV_64F), Mat::zeros(BlurredImg.size(), CV_64F), Mat::zeros(BlurredImg.size(), CV_64F) };
	split(BlurredImg, doubleBlurredImg_sub);
	Mat doubleBlurredImg[3];
	for (c = 0; c < 3; c++) {
		Mat planes[] = { Mat_<double>(doubleBlurredImg_sub[c]), Mat::zeros(doubleBlurredImg_sub[c].size(), CV_64F) };
		merge(planes, 2, doubleBlurredImg[c]);
	}


	/* DFT */
	// カーネルのDFT
	Mat dft_TrueKernel = Mat::zeros(Msize, Nsize, CV_64FC2);
	Mat planes2[] = { Mat_<double>(TrueKernel), Mat::zeros(TrueKernel.size(), CV_64F) };
	merge(planes2, 2, TrueKernel);
	copyMakeBorder(TrueKernel, dft_TrueKernel, 0, Msize - TrueKernel.rows, 0, Nsize - TrueKernel.cols, BORDER_CONSTANT, (0.0, 0.0));
	dft(dft_TrueKernel, dft_TrueKernel, 0, dft_TrueKernel.rows);
	//visualbule_complex(dft_TrueKernel, Image_dst_deblurred2);	// 確認

	// 真の元画像のDFT
	Mat dft_doubleTrueImg[3] = { Mat::zeros(Msize, Nsize, CV_64FC2), Mat::zeros(Msize, Nsize, CV_64FC2), Mat::zeros(Msize, Nsize, CV_64FC2) };
	for (c = 0; c < 3; c++) {
		copyMakeBorder(doubleTrueImg[c], dft_doubleTrueImg[c], TrueKernel.rows / 2, Msize - Mplus + TrueKernel.rows / 2, TrueKernel.cols / 2, Nsize - Nplus + TrueKernel.cols / 2, BORDER_REPLICATE);
		dft(dft_doubleTrueImg[c], dft_doubleTrueImg[c], 0, dft_doubleTrueImg[c].rows);
	}
	//visualbule_complex(dft_doubleTrueImg[0], Image_dst_deblurred);	// 確認

	// ぼけ画像のDFT
	Mat dft_doubleBlurredImg[3] = { Mat::zeros(Msize, Nsize, CV_64FC2), Mat::zeros(Msize, Nsize, CV_64FC2), Mat::zeros(Msize, Nsize, CV_64FC2) };
	for (c = 0; c < 3; c++) {
		copyMakeBorder(doubleBlurredImg[c], dft_doubleBlurredImg[c], TrueKernel.rows / 2, Msize - Mplus + TrueKernel.rows / 2, TrueKernel.cols / 2, Nsize - Nplus + TrueKernel.cols / 2, BORDER_REPLICATE);
		dft(dft_doubleBlurredImg[c], dft_doubleBlurredImg[c], 0, dft_doubleBlurredImg[c].rows);
	}
	//visualbule_complex(dft_doubleBlurredImg[0], Image_dst_deblurred2);	// 確認

	/* 真の係数を求める */
	Mat dft_doubleTrueImg2[3];
	Mat dft_denomImg[3], dft_denomImg2[3];
	for (c = 0; c < 3; c++) {
		abs_pow_complex(dft_TrueKernel, dft_denomImg[c]);
		reciprocal_complex(dft_denomImg[c], dft_denomImg2[c]);
		//visualbule_complex(dft_denomImg2[c], Image_dst_deblurred2);	// 確認
		mulSpectrums(dft_doubleBlurredImg[c], dft_TrueKernel, dft_doubleTrueImg2[c], 0, true);	// 複素共役
		//mulSpectrums(dft_doubleTrueImg2[c], dft_denomImg2[c], dft_doubleTrueImg2[c], 0, false);
	}
	//visualbule_complex(dft_denomImg2[0], Image_dst_deblurred2);	// 確認
	//visualbule_complex(dft_doubleTrueImg2[0], Image_dst_deblurred2);	// 確認

	// 真の係数のMat
	Mat Noise[3] = { Mat::zeros(Msize, Nsize, CV_64FC2), Mat::zeros(Msize, Nsize, CV_64FC2), Mat::zeros(Msize, Nsize, CV_64FC2) };
	Vec2d Error = { 0.0, 0.0 };
	Vec2d number, number1, number2, number3;
	Vec2d one = { 1.0, 0.0 };
	Vec2d c_number = { 3.0, 0.0 };
	double all_num = (double)Msize * (double)Nsize;
	Vec2d all_number = { all_num, 0.0 };
	Vec2d ave_number[3];
	for (c = 0; c < 3; c++) {
		ave_number[c] = { 0.0, 0.0 };
#pragma omp parallel for private(x)
		for (y = 0; y < Msize; y++) {
			for (x = 0; x < Nsize; x++) {
				number1 = dft_doubleTrueImg[c].at<Vec2d>(y, x);
				number2 = dft_doubleTrueImg2[c].at<Vec2d>(y, x);
				number3 = dft_denomImg[c].at<Vec2d>(y, x);
				//cout << "  number1 = " << number1 << " , number2 = " << number2 << endl;	// 確認用
				divi_complex_2(number, number2, number1);
				number = number - number3;
				Noise[c].at<Vec2d>(y, x) = number;
				ave_number[c] = ave_number[c] + number;
				//cout << "  ave = " << ave_number[c] << " , number = " << number << endl;	// 確認用
			}
		}
		divi_complex_2(ave_number[c], ave_number[c], all_number);
		//cout << "  " << c << " : " << ave_number[c] << endl;	// 確認用

		Error += ave_number[c];
	}
	divi_complex_2(Error, Error, c_number);
	//cout << " Error = " << Error << endl;	// 確認用
	//Noise[0] = Mat(Msize, Nsize, CV_64FC2, TrueParameter);
	//visualbule_complex(Noise[0], Image_dst_deblurred2);	// 確認

	Mat Gamma_Mat = Mat::zeros(Msize, Nsize, CV_64FC2);
#pragma omp parallel for private(x)
	for (y = 0; y < Msize; y++) {
		for (x = 0; x < Nsize; x++) {
			number1 = Noise[0].at<Vec2d>(y, x);
			number2 = Noise[1].at<Vec2d>(y, x);
			number3 = Noise[2].at<Vec2d>(y, x);
			number = number1 + number2 + number3;
			//cout << "  number1 = " << number1 << "  number2 = " << number2 << " , number3 = " << number3 << endl;	// 確認用
			divi_complex_2(number, number, c_number);
			//cout << "  number = " << number << endl;	// 確認用
			Gamma_Mat.at<Vec2d>(y, x) = number;
		}
	}
	//checkMat(GammaMat);	// 確認
	//visualbule_complex(GammaMat, Image_dst_deblurred2);	// 確認

	/* 出力 */
	Gamma_Mat.copyTo(OutputParameter);

	//// ウィーナ・フィルター確認 ※
	//Mat W_Filter[3];
	//Mat doubleTrueImg2[3];
	//for (c = 0; c < 3; c++) {
	//	double theredhold = 10;
	//	dft_denomImg2[c].copyTo(W_Filter[c]);
	//	//GaussianBlur(W_Filter[c], W_Filter[c], Size(11, 11), 1, 1);		// ガウスノイズ付加(I = I + n)
	//	wiener_filter(dft_TrueKernel, W_Filter[c], Noise[c]);			// 2次元ベクトルの逆数(ウィーナ・フィルター)
	//	visualbule_complex(W_Filter[c], Image_dst_deblurred);	// 確認

	//	//mulSpectrums(W_Filter[c], dft_TrueKernel, dft_doubleTrueImg2[c], 0, true);	// 推定PSF
	//	//reciprocal_complex(dft_doubleTrueImg2[c], dft_doubleTrueImg2[c]);

	//	mulSpectrums(dft_doubleTrueImg2[c], W_Filter[c], dft_doubleTrueImg2[c], 0, false);
	//	//mulSpectrums(dft_doubleTrueImg2[c], dft_denomImg2[c], dft_doubleTrueImg2[c], 0, false);
	//	//visualbule_complex(dft_doubleTrueImg2[c], Image_dst_deblurred2);	// 確認

	//	dft(dft_doubleTrueImg2[c], dft_doubleTrueImg2[c], cv::DFT_INVERSE + cv::DFT_SCALE | cv::DFT_REAL_OUTPUT);
	//	/*dft_doubleTrueImg2[c] = dft_doubleTrueImg2[c] + dft_doubleTrueImg2[c];
	//	dft_doubleTrueImg2[c] = dft_doubleTrueImg2[c] / 2.0;*/
	//	doubleTrueImg2[c] = dft_doubleTrueImg2[c](Rect(TrueKernel.cols / 2, TrueKernel.rows / 2, BlurredImg.cols, BlurredImg.rows));
	//}
	//Mat TrueImage2;
	//merge(doubleTrueImg2, 3, TrueImage2);
	//TrueImage2.convertTo(TrueImage2, CV_8U);
	////TrueImage2.copyTo(Image_dst_deblurred);	// 確認

	//// カーネルの確認 ※
	///*dft(dft_TrueKernel, dft_TrueKernel, cv::DFT_INVERSE + cv::DFT_SCALE | cv::DFT_REAL_OUTPUT);
	//Mat KernelTmp = dft_TrueKernel(Rect(0, 0, TrueKernel.cols, TrueKernel.rows));*/
	//Mat KernelTmp = dft_doubleTrueImg2[0](Rect(0, 0, TrueKernel.cols, TrueKernel.rows));
	////KernelImage.inputKernel_normalized(KernelTmp);
	//KernelTmp = KernelTmp * (double)KernelImage.sum;
	//KernelTmp.copyTo(Image_dst_deblurred2);
	//KernelImage_Visualization(Image_dst_deblurred2);
}
