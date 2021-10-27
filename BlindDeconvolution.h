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
double Myu = 1.0e-05/*0.4e-03*/;
double Rambda = 0.4e-03;
double Tau = 1.0e-03;

/* 関数 */
double CostCalculateLeast(int X, int Y, Mat& Quant_Img, Mat& Now_Img, Mat& contrast_Img);
void ConjugateGradientMethod(Mat& QuantImg, Mat& BlurrImg, Mat& Kernel, Mat& LagrngianMlutipliers, Vec2d& PenaltyParameter, Mat& NewKernel);
void ConjugateGradientMethod_Gray(Mat& QuantImg, Mat& BlurrImg, Mat& Kernel, Mat& LagrngianMlutipliers, Vec2d& PenaltyParameter, Mat& NewKernel);


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
	void UpdateQuantizedImage_kmeans(Mat&, QuantMatDouble&);
	void UpdateImage(Mat&, Mat&, KERNEL&, Mat&);
	void UpdateImage_check(Mat&, Mat&, KERNEL&, Mat&);
	void UpdateKarnel(KERNEL&, Mat&, Mat&);
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
	for (pyr = 0/*PYRAMID_NUM*/; pyr >= 0/*3*/; pyr--) {
		cout << "Deconvoluting in " << (int)pyr << endl;		// 実行確認用
		/*checkMat(Img[pyr]);
		checkMat(QuantImg[pyr].QMat);
		checkMat(BlurrImg[pyr]);
		checkMat(Kernel[pyr].Kernel);*/

		for (int i = 0; i < MAX_Iteration; i++) {
			/* Update x~ */
			cout << " Update QuantImg... " << endl;				// 実行確認用
			//UpdateQuantizedImage(Img[pyr], QuantImg[pyr]);
			UpdateQuantizedImage_kmeans(Img[pyr], QuantImg[pyr]);

			/* Update x */
			cout << " Update Img... " << endl;					// 実行確認用
			//UpdateImage(Img[pyr], QuantImg[pyr].QMat, Kernel[pyr], BlurrImg[pyr]);
			//UpdateImage_check(Img[pyr], QuantImg[pyr].QMat, Kernel[pyr], BlurrImg[pyr]);

			/*if (i == 1) {
				break;
			}*/

			/* Update k */
			Mat before_Kernel, after_Kernel;
			Kernel[pyr].Kernel_normalized.copyTo(before_Kernel);
			cout << " Update Karnel... " << endl;				// 実行確認用
			//UpdateKarnel(Kernel[pyr], QuantImg[pyr].QMat, BlurrImg[pyr]);
			Kernel[pyr].Kernel_normalized.copyTo(after_Kernel);

			//double diff_Kernel = (double)norm(before_Kernel, after_Kernel, NORM_L2);
			//diff_Kernel = (double)sqrt(diff_Kernel) / (double)Kernel[pyr].size;
			//cout << "  diff_Kernel = " << diff_Kernel << endl;		// 実行確認用
			//if (diff_Kernel < (double)1.0e-04) { break; }

			if (i == 0) {
				break;
			}
		}

		/* 出力 */
		BlurrImg[pyr].convertTo(Image_dst, CV_8UC3);		// 確認用
		TrueImg[pyr].convertTo(Img_true, CV_8UC3);
		Img[pyr].convertTo(Img_inoutput, CV_8UC3);
		QuantImg[pyr].QMat.convertTo(Image_dst_deblurred2, CV_8UC3);		// 確認用
		//TrueImg[pyr].convertTo(Image_dst_deblurred2, CV_8UC3);
		Kernel_inoutput.copy(Kernel[pyr]);
		//Mat resize_kernel_original;	// 確認用
		//for (int pyr_index = 0; pyr_index < pyr; pyr_index++) {
		//	if (pyr_index == 0) { resize(Image_kernel_original, resize_kernel_original, Size(), ResizeFactor, ResizeFactor); }
		//	else { resize(resize_kernel_original, resize_kernel_original, Size(), ResizeFactor, ResizeFactor); }
		//}
		//KernelMat_Normalization(resize_kernel_original);
		//normalize(resize_kernel_original, resize_kernel_original, 0, 100, NORM_MINMAX);
		////resize_kernel_original.copyTo(Image_kernel_original);
		//cout << "推定カーネル と 真カーネル (double)" << endl;		// 確認用
		//Evaluation_MSE_PSNR_SSIM(Kernel[pyr].Kernel, resize_kernel_original);
		////checkMat(Kernel[pyr].Kernel);
		////checkMat(resize_kernel_original);

		/* Upsample */
		if (pyr != 0) {
			cout << " Upsample..." << endl;						// 実行確認用
			Upsampling(pyr);
		}
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

	Mat Img_input_tmp;
	Img_input.copyTo(Img_input_tmp);
	Img.push_back(Img_input_tmp);
	BlurrImg.push_back(Img_input);
	TrueImg.push_back(Img_true);
	QuantMatDouble quantMat = QuantMatDouble(10, Img_input_tmp, 0);
	quantMat.quantedQMat();
	QuantImg.push_back(quantMat);
	//Kernel.push_back(Kernel_input);	// カーネルコピー
	KERNEL Kernel_tmp = KERNEL(3);	// カーネル適当な初期値
	Kernel.push_back(Kernel_tmp);
	K_X_SIZE.push_back(Kernel_input.Kernel.cols);
	K_Y_SIZE.push_back(Kernel_input.Kernel.rows);
	//Kernel[0].display_detail();	// 確認用
	cout << " [0] : " << Img[0].size() << endl;		// 実行確認用
	cout << "     : [" << Kernel[0].rows << " x " << Kernel[0].cols << "]" << endl;	// 実行確認用
	int pyr_next = 0;
	Mat Img_tmp;
	QuantMatDouble QuantImg_tmp;
	KERNEL Karnel_tmp;
	Mat TrueImg_tmp, BlurrImg_tmp;
	for (pyr = 0; pyr < PYRAMID_NUM; pyr++) {
		resize(Img[pyr], Img_tmp, Size(), ResizeFactor, ResizeFactor);
		resize(TrueImg[pyr], TrueImg_tmp, Size(), ResizeFactor, ResizeFactor);
		QuantImg_tmp = QuantMatDouble(10, Img_tmp);
		QuantImg_tmp.quantedQMat();
		Karnel_tmp = KERNEL();
		Karnel_tmp.resize_copy(ResizeFactor, Kernel[pyr]);
		Img.push_back(Img_tmp);
		Img_tmp.copyTo(BlurrImg_tmp);
		BlurrImg.push_back(BlurrImg_tmp);
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
	//contrust.convertTo(Image_dst_deblurred2, CV_8U);		// 確認用
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
void Blind_Deconvolution::UpdateQuantizedImage_kmeans(Mat& Img_Now, QuantMatDouble& QuantImg_Now) {
	int Iteration_Number = 1;
	double Error = 1.0e-04;

	//QuantImg_Now.quantedQMat();
	/* Optimizing using k-means */
	Mat NewQuantImg;
	Img_Now.convertTo(NewQuantImg, CV_32FC3/*, 1.0 / 255.0*/);

	/* k-means クラスタリング */
	Mat feature;
	NewQuantImg.copyTo(feature);
	//画像の画素を1列（3チャネル）に並べる
	feature = feature.reshape(3, feature.rows * feature.cols);
	//cout << "  " << NewQuantImg.cols * NewQuantImg.rows << endl;	// 確認用
	//checkMat(feature);	// 確認用
	//checkMat_detail(feature);	// 確認用

	Mat_<int> labels(feature.size(), CV_32SC1);
	Mat centers;

	//kmeans法による画像の分類（領域分割）
	const int MAX_CLUSTERS = 20;
	kmeans(feature, MAX_CLUSTERS, labels, TermCriteria(TermCriteria::COUNT, 10, 1.0), 1, KMEANS_RANDOM_CENTERS, centers);
	//centers.convertTo(Image_dst_deblurred2, CV_8UC3);		// 確認用
	//checkMat_detail(feature);	// 確認用

	// ラベリング結果の描画色を決定
	vector<Vec3b> colors(MAX_CLUSTERS+1);
	colors[0] = Vec3d(0.0, 0.0, 0.0);
	//cout << "  " << colors[0];	// 確認用
	for (int label = 1; label <= MAX_CLUSTERS; ++label)
	{
		Vec3d ave_colors = Vec3d(0.0, 0.0, 0.0);
		int ave_counter = 0;
		SelectAverageRGB CalcAveRGB = SelectAverageRGB();
		//クラスタの平均値を計算
		for (y = 0; y < NewQuantImg.rows; y++) {
			for (x = 0; x < NewQuantImg.cols; x++) {
				int pix = y * NewQuantImg.cols + x;
				//int pix = (y * NewQuantImg.cols + x) * 3;
				//cout << "  " << (int)labels.data[pix];	// 確認用
				if ((int)labels.data[pix] == label) {
					ave_colors[0] += (double)Img_Now.at<Vec3b>(y, x)[0];
					ave_colors[1] += (double)Img_Now.at<Vec3b>(y, x)[1];
					ave_colors[2] += (double)Img_Now.at<Vec3b>(y, x)[2];
					CalcAveRGB.put(ave_colors);
					//ave_colors += feature.at<Vec3d>(y, x);
					//ave_colors += NewQuantImg.at<Vec3d>(y, x);
					ave_counter++;
					/*if (pix % 3 == 0) { ave_colors[0] += (double)NewQuantImg.data[pix * 3 + 0]; }
					else if (pix % 3 == 1) { ave_colors[1] += (double)NewQuantImg.data[pix * 3 + 1]; }
					else {
						ave_colors[2] += (double)NewQuantImg.data[pix * 3 + 2];
						ave_counter++;
					}*/
					//cout << "  ave_colors = " << (Vec3d)ave_colors << " , color''= " << Img_Now.at<Vec3b>(y, x) << endl;	// 確認用
					//cout << "  ave_colors = " << (Vec3d)ave_colors << " , color' = " << feature.at<Vec3d>(y, x) << endl;	// 確認用
					//cout << "  ave_colors = " << (Vec3d)ave_colors << " , color = " << NewQuantImg.at<Vec3d>(y, x) << endl;	// 確認用
					//cout << "  ave_colors = " << (Vec3d)ave_colors << " , color = [" << (double)NewQuantImg.data[pix * 3 + 0] << ", " << (double)NewQuantImg.data[pix * 3 + 1] << ", " << (double)NewQuantImg.data[pix * 3 + 2] << "]" << endl;	// 確認用
				}
			}
		}
		cout << "  ave_colors_sum = " << (Vec3d)ave_colors << endl;	// 確認用
		cout << "  ave_counter = " << ave_counter << endl;	// 確認用
		if (ave_counter != 0) {
			ave_colors[0] /= (double)ave_counter;
			ave_colors[1] /= (double)ave_counter;
			ave_colors[2] /= (double)ave_counter;
		}
		CalcAveRGB.selectAverage(ave_colors);
		//cout << "  ave_colors = " << (Vec3d)ave_colors << endl;	// 確認用

		//クラスタ毎に色を描画
		//colors[label] = ave_colors;
		//colors[label] = Vec3b(ave_colors[0], ave_colors[1], ave_colors[2]);
		colors[label][0] = (uchar)ave_colors[0];
		colors[label][1] = (uchar)ave_colors[1];
		colors[label][2] = (uchar)ave_colors[2];
		//cout << " , " << colors[label];	// 確認用
		cout << "  ave_colors = " << (Vec3d)ave_colors << " => " << (Vec3b)colors[label] << endl;	// 確認用

		////ラベル番号に対して色をランダムに割り当てる
		//colors[label] = Vec3b((rand() & 255), (rand() & 255), (rand() & 255));
	}
	//cout << endl;	// 確認用

	//ラベリング結果画像の作成
	Mat dst(NewQuantImg.size(), CV_8UC3);
	//結果画像の各画素の色を決定していく
	MatIterator_<Vec3b> itd = dst.begin<Vec3b>(), itd_end = dst.end<Vec3b>();
	for (int i = 0; itd != itd_end; ++itd, ++i) {
		int label = labels(i); //ラベリング画像の各画素のラベル番号を抽出
		(*itd) = colors[label]; //ラベリング画像の各画素の色を割り当て
	}

	/*center = NewQuantImg.uint8(center)
	res = center[label.flatten()]
	res2 = res.reshape((img.shape))*/

	dst.convertTo(NewQuantImg, CV_64FC3/*, 255.0*/);
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
	//checkMat_detail(doubleKernel);	// 確認
	// ぼけ画像＆量子化画像
	Mat dft_doubleBlurredImg[3] = { Mat::zeros(Msize, Nsize, CV_64FC2), Mat::zeros(Msize, Nsize, CV_64FC2), Mat::zeros(Msize, Nsize, CV_64FC2) };
	Mat dft_doubleQuantImg[3] = { Mat::zeros(Msize, Nsize, CV_64FC2), Mat::zeros(Msize, Nsize, CV_64FC2), Mat::zeros(Msize, Nsize, CV_64FC2) };
	for (c = 0; c < 3; c++) {
		copyMakeBorder(doubleBlurredImg[c], dft_doubleBlurredImg[c], doubleKernel.rows / 2, Msize - Mplus + doubleKernel.rows / 2, doubleKernel.cols / 2, Nsize - Nplus + doubleKernel.cols / 2, BORDER_REPLICATE);
		dft(dft_doubleBlurredImg[c], dft_doubleBlurredImg[c]);
		copyMakeBorder(doubleQuantImg[c], dft_doubleQuantImg[c], doubleKernel.rows / 2, Msize - Mplus + doubleKernel.rows / 2, doubleKernel.cols / 2, Nsize - Nplus + doubleKernel.cols / 2, BORDER_REPLICATE);
		dft(dft_doubleQuantImg[c], dft_doubleQuantImg[c]);
	}
	//visualbule_complex(dft_doubleBlurredImg[0], Image_dst_deblurred2);	// 確認
	//visualbule_complex(dft_doubleQuantImg[0], Image_dst_deblurred2);	// 確認
	//copyMakeBorder(doubleBlurredImg_sub[0], Image_dst_deblurred2, doubleKernel.rows / 2, Msize - Mplus + doubleKernel.rows / 2, doubleKernel.cols / 2, Nsize - Nplus + doubleKernel.cols / 2, BORDER_REPLICATE);
	//Image_dst_deblurred2.convertTo(Image_dst_deblurred2, CV_8UC1);

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
	//dft_doubleNewImg[0].copyTo(Image_dst_deblurred2);	// 確認
	//Image_dst_deblurred2.convertTo(Image_dst_deblurred2, CV_8UC1);	// 確認
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
	//checkMat_detail(doubleKernel);	// 確認
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
	int Iterate_Num = 5000;
	double ERROR_END_NUM = 1.0e-04;

	//CG_method(doubleNewImg1[0], doubleNewImg1[1], doubleNewImg1[2], doubleBlurredImg_sub[0], doubleBlurredImg_sub[1], doubleBlurredImg_sub[2], doubleNewImg2[0], doubleNewImg2[1], doubleNewImg2[2]);
	//Mat NextX[3];
	//for (c = 0; c < 3; c++) {
	//	doubleBlurredImg_sub[c].copyTo(NextX[c]);
	//}

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
		//cout << "   color_mean = " << (double)color_mean << endl;	// 確認用

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
			if (energy[c] < ERROR_END_NUM || i_number == Iterate_Num - 1) {
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
		//cout << "   before_color_mean = " << (double)before_color_mean << endl;	// 確認用
		Mat tmp;
		normalize(NextX[c], tmp, 0, 255, NORM_MINMAX);
		double after_color_mean = (double)mean(tmp)[0];
		//NextX[c] = NextX[c] + (color_mean - after_color_mean);
		//NextX[c] = NextX[c] + (color_mean - before_color_mean);
		//normalize(NextX[c], NextX[c], 0, 255, NORM_MINMAX);
		NextX[c] = NextX[c] * (double)(color_mean / before_color_mean);
		//cout << "   diff_color_mean = " << (double)(color_mean - after_color_mean) << endl;	// 確認用
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
void Blind_Deconvolution::UpdateKarnel(KERNEL& Karnel_Now, Mat& QuantImg_Now, Mat& BlurrImg_Now) {
	/* Optimizing k */
	double PenaltyParameter = 1.0e+15/*1.0e+01*/;
	double Error = 1.0e-04;

	Mat doubleKernel;			// k
	Karnel_Now.Kernel_normalized.convertTo(doubleKernel, CV_64F);
	Mat doubleKernel2;			// k'
	Karnel_Now.Kernel_normalized.convertTo(doubleKernel2, CV_64F);
	//Mat doubleA = Mat::ones(Karnel_Now.Kernel_normalized.size(), CV_64F);	// correspond to a transformed vector of Lagrange multiplier
	Mat doubleA = Mat::zeros(Karnel_Now.Kernel_normalized.size(), CV_64F);

	double energy = 0.0;
	Mat Before, Before_const, After;
	doubleKernel.copyTo(Before_const);
	Vec2d PenaltyParameter_Vec2 = { PenaltyParameter / 2.0 , 0.0 };
	//Vec2d incr_Parameter = { 2.0 , 0.0 }, decr_Parameter = { 2.0 , 0.0 }, blanc_Parameter = { 10.0 , 0.0 };
	double incr_Parameter = 2.0, decr_Parameter = 2.0, blanc_Parameter = 10.0;
	for (int k_index = 0; k_index < 10; k_index++) {
		energy = 0.0;
		doubleKernel.copyTo(Before);
		//checkMat_detail(doubleKernel);	// 確認

		/* Calculate doubleKernel_ */
		// 画像をCV_64FC3に変換(前処理)
		Mat BlurredImg;
		BlurrImg_Now.convertTo(BlurredImg, CV_64FC3);
		Mat QuantImg;
		QuantImg_Now.convertTo(QuantImg, CV_64FC3);
		//PenaltyParameter_Vec2 = { PenaltyParameter / 2.0 , 0.0 };
		ConjugateGradientMethod(QuantImg, BlurredImg, doubleKernel, doubleA, PenaltyParameter_Vec2, doubleKernel2);
		//ConjugateGradientMethod_Gray(QuantImg, BlurredImg, doubleKernel, doubleA, PenaltyParameter_Vec2, doubleKernel2);
		KernelMat_Normalization(doubleKernel2);
		//checkMat_detail(doubleKernel2);	// 確認
		//doubleKernel2.copyTo(Image_dst_deblurred2);	// 確認
		//normalize(Image_dst_deblurred2, Image_dst_deblurred2, 0, 200, NORM_MINMAX);
		//Image_dst_deblurred2.convertTo(Image_dst_deblurred2, CV_8UC1);

		/* Calculate doubleKernel */
		Mat doubleKernel_sub;
		doubleKernel.copyTo(doubleKernel_sub);
		double sign_calc = 0.0;
		double threshold = (double)(Tau / PenaltyParameter_Vec2[0]);
		cout << "  threshold = " << threshold  << " , PenaltyParameter = " << PenaltyParameter_Vec2[0] << endl;
#pragma omp parallel for private(x)
		for (y = 0; y < Karnel_Now.rows; y++) {
			for (x = 0; x < Karnel_Now.cols; x++) {
				sign_calc = (double)abs((double)doubleKernel2.at<double>(y, x) - (double)doubleA.at<double>(y, x));
				//cout << "   " << sign_calc << endl;	// 確認用
				/*if (sign_calc >= threshold) {
					doubleKernel_sub.at<double>(y, x) = (double)sign_calc - (double)threshold;
				}
				else {
					doubleKernel_sub.at<double>(y, x) = 0.0;
				}*/
				if (sign_calc >= threshold) {
					doubleKernel_sub.at<double>(y, x) = (double)sign_calc - (double)threshold;
				}
				else if (sign_calc > -threshold) {
					doubleKernel_sub.at<double>(y, x) = 0.0;
				}
				else { doubleKernel_sub.at<double>(y, x) = (double)sign_calc - (double)threshold; }
				doubleKernel_sub.at<double>(y, x) = (double)sign_calc - (double)threshold;
				//cout << doubleKernel_sub.at<double>(y, x) << endl;	// 確認用
			}
		}
		//#pragma omp parallel for private(x)
		//		for (y = 0; y < doubleKernel_sub.rows; y++) {
		//			for (x = 0; x < doubleKernel_sub.cols; x++) {
		//				double kernel_num = doubleKernel_sub.at<double>(y, x);
		//				if (kernel_num < 0) { doubleKernel_sub.at<double>(y, x) = 0.0; }	// 負の値は0にする
		//				//else if (kernel_num > 1) { doubleKernel_sub.at<double>(y, x) = 1.0; }
		//			}
		//		}
		doubleKernel_sub.copyTo(doubleKernel);
		KernelMat_Normalization(doubleKernel);
		//checkMat_detail(doubleKernel);	// 確認


		/* Calculate doubleA */
		Mat doubleA_sub;
		double doubleKernel_diff;
		doubleA.copyTo(doubleA_sub);
#pragma omp parallel for private(x)
		for (y = 0; y < Karnel_Now.rows; y++) {
			for (x = 0; x < Karnel_Now.cols; x++) {
				doubleKernel_diff = (double)doubleKernel2.at<double>(y, x) - (double)doubleKernel.at<double>(y, x);
				doubleA_sub.at<double>(y, x) = (double)((double)doubleA.at<double>(y, x) - (double)doubleKernel_diff);
			}
		}
		doubleA_sub.copyTo(doubleA);
		//doubleA_sub.copyTo(Image_dst_deblurred2);	// 確認
		//normalize(Image_dst_deblurred2, Image_dst_deblurred2, 0, 200, NORM_MINMAX);
		//Image_dst_deblurred2.convertTo(Image_dst_deblurred2, CV_8UC1);

		/* 収束条件付加 */
		//doubleKernel.copyTo(After);
		//energy = norm(Before, After, NORM_L2) / (double)Karnel_Now.size;
		//cout << "  " << (int)k_index << " : energy = " << (double)energy << endl;	// 確認用
		//if (energy < Error) { break; }

		double main_diff = (double)norm(doubleKernel);
		double sub_diff = (double)norm(doubleA) * (double)PenaltyParameter_Vec2[0] * (double)blanc_Parameter;
		//cout << "   main_diff = " << (double)main_diff << " , sub_diff = " << (double)sub_diff << endl;	// 確認用
		if (main_diff > sub_diff) { PenaltyParameter_Vec2[0] *= incr_Parameter; }
		else if (main_diff < sub_diff) { PenaltyParameter_Vec2[0] /= decr_Parameter; }
	}

	//double Unorm = norm(Before_const, After, NORM_L2);
	//double e = (double)(Unorm) / (double)(Karnel_Now.size * 3.0);
	//cout << "  e = " << (double)e << "  ( before & after )" << endl;	// 確認用

	KernelMat_Normalization(doubleKernel);
	doubleKernel.copyTo(Karnel_Now.Kernel_normalized);
	//doubleKernel.copyTo(Image_dst_deblurred2);	// 確認
}
void Blind_Deconvolution::Upsampling(int before_pyrLEVEL) {
	int after_pyrLEVEL = before_pyrLEVEL - 1;
	Mat imgNEXT;
	//Kernel[after_pyrLEVEL] = KERNEL();
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


// 量子化画像決定時のコスト計算(比較部分のみ)
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
				if (Quant_Img.at<double>(Y, X) != Quant_Img.at<double>(Y, X - 1)) { cout_tmp += wight; }

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
				if (Quant_Img.at<double>(Y, X) != Quant_Img.at<double>(Y, X + 1)) { cout_tmp += wight; }

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
				if (Quant_Img.at<double>(Y, X) != Quant_Img.at<double>(Y - 1, X)) { cout_tmp += wight; }

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
				if (Quant_Img.at<double>(Y, X) != Quant_Img.at<double>(Y + 1, X)) { cout_tmp += wight; }

				diff = (double)abs(adj - now[c]);
				norm = (double)pow(diff, 2);
				Uorm += norm;
			}
		}
	}
	Uorm /= (double)Max_Pix;
	Cost = Myu * Uorm;
	cout_tmp /= (double)Max_Pix;
	//cout << "  Cost(p) = " << Cost << " , Cost(x) = " << cout_tmp << endl;	// 確認用
	Cost += cout_tmp;

	return Cost;
}

// Kernel Estimate by CG method in ADMM
void ConjugateGradientMethod(Mat& QuantImg, Mat& BlurrImg, Mat& Kernel, Mat& LagrngianMlutipliers, Vec2d& PenaltyParameter, Mat& NewKernel) {
	int Iterate_Number = 100;
	double ERROR_END = 1.0e-04;
	int x, y, c;
	//Vec2d incr_Parameter = { 2.0 , 0.0 }, decr_Parameter = { 2.0 , 0.0 }, blanc_Parameter = { 10.0 , 0.0 };

	/* Ax=bを解くとしてAxとbを求める */
	// 画像をCV_64FC1に変換(前処理)
	Mat doubleBlurredImg_sub[3] = { Mat::zeros(BlurrImg.size(), CV_64F), Mat::zeros(BlurrImg.size(), CV_64F), Mat::zeros(BlurrImg.size(), CV_64F) };
	split(BlurrImg, doubleBlurredImg_sub);
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
	// 入力画像をDFTに指定した大きさに広げる
	int Mplus = BlurrImg.rows + Kernel.rows;
	int Nplus = BlurrImg.cols + Kernel.cols;
	int Msize = getOptimalDFTSize(Mplus);
	int Nsize = getOptimalDFTSize(Nplus);
	// DFT
	Mat dft_doubleBlurredImg[3] = { Mat::zeros(Msize, Nsize, CV_64FC2), Mat::zeros(Msize, Nsize, CV_64FC2), Mat::zeros(Msize, Nsize, CV_64FC2) };
	Mat dft_doubleQuantImg[3] = { Mat::zeros(Msize, Nsize, CV_64FC2), Mat::zeros(Msize, Nsize, CV_64FC2), Mat::zeros(Msize, Nsize, CV_64FC2) };
	Mat dft_doubleBlurredImg_part[3], dft_doubleQuantImg_part[3];
	for (c = 0; c < 3; c++) {
		copyMakeBorder(doubleBlurredImg[c], dft_doubleBlurredImg[c], Kernel.rows / 2, Msize - Mplus + Kernel.rows / 2, Kernel.cols / 2, Nsize - Nplus + Kernel.cols / 2, BORDER_REPLICATE);
		dft(dft_doubleBlurredImg[c], dft_doubleBlurredImg[c]);
		copyMakeBorder(doubleQuantImg[c], dft_doubleQuantImg[c], Kernel.rows / 2, Msize - Mplus + Kernel.rows / 2, Kernel.cols / 2, Nsize - Nplus + Kernel.cols / 2, BORDER_REPLICATE);
		dft(dft_doubleQuantImg[c], dft_doubleQuantImg[c]);
	}
	// カーネル
	Mat dft_doubleK = Mat::zeros(Msize, Nsize, CV_64FC2);
	Mat planes2[] = { Mat_<double>(NewKernel), Mat::zeros(NewKernel.size(), CV_64F) };
	Mat doubleK_sub;
	merge(planes2, 2, doubleK_sub);
	copyMakeBorder(doubleK_sub, dft_doubleK, 0, Msize - NewKernel.rows, 0, Nsize - NewKernel.cols, BORDER_CONSTANT, (0.0, 0.0));
	dft(dft_doubleK, dft_doubleK, 0, dft_doubleK.rows);
	Mat dft_doubleK2 = Mat::zeros(Msize, Nsize, CV_64FC2);
	Mat planes2_2[] = { Mat_<double>(Kernel), Mat::zeros(Kernel.size(), CV_64F) };
	Mat doubleK_sub2;
	merge(planes2_2, 2, doubleK_sub2);
	copyMakeBorder(doubleK_sub2, dft_doubleK2, 0, Msize - Kernel.rows, 0, Nsize - Kernel.cols, BORDER_CONSTANT, (0.0, 0.0));
	dft(dft_doubleK2, dft_doubleK2, 0, dft_doubleK2.rows);
	Mat dft_A = Mat::zeros(Msize, Nsize, CV_64FC2);
	Mat planes3[] = { Mat_<double>(LagrngianMlutipliers), Mat::zeros(LagrngianMlutipliers.size(), CV_64F) };
	Mat A_sub;
	merge(planes3, 2, A_sub);
	copyMakeBorder(A_sub, dft_A, 0, Msize - LagrngianMlutipliers.rows, 0, Nsize - LagrngianMlutipliers.cols, BORDER_CONSTANT, (0.0, 0.0));
	dft(dft_A, dft_A, 0, dft_A.rows);
	//visualbule_complex(dft_doubleBlurredImg[0], Image_dst_deblurred2);	// 確認
	//visualbule_complex(dft_doubleQuantImg[0], Image_dst_deblurred2);	// 確認
	//visualbule_complex(dft_doubleK, Image_dst_deblurred2);	// 確認
	//visualbule_complex(dft_A, Image_dst_deblurred2);	// 確認

	// 計算
	Mat dft_doubleNewImg[3], dft_doubleNewImg1[3], dft_doubleNewImg2[3];
	Mat pow_dft_doubleQuantImg[3];
	Vec2d before, after, before2, after2, before3, after3;
	for (c = 0; c < 3; c++) {
		dft_doubleNewImg[c] = Mat::zeros(Msize, Nsize, CV_64FC2);
		dft_doubleNewImg1[c] = Mat::zeros(Msize, Nsize, CV_64FC2);
		dft_doubleNewImg2[c] = Mat::zeros(Msize, Nsize, CV_64FC2);

		abs_pow_complex(dft_doubleQuantImg[c], pow_dft_doubleQuantImg[c]);		// 2次元ベクトルの大きさの２乗
		mulSpectrums(dft_doubleBlurredImg[c], dft_doubleQuantImg[c], dft_doubleNewImg2[c], 0, true);	// 複素共役
#pragma omp parallel for private(x)
		for (y = 0; y < Msize; y++) {
			for (x = 0; x < Nsize; x++) {
				before = pow_dft_doubleQuantImg[c].at<Vec2d>(y, x);
				after = before + PenaltyParameter;
				//cout << "  before = " << before << " , after = " << after << endl;	// 確認用
				dft_doubleNewImg[c].at<Vec2d>(y, x) = after;
				//cout << "   " << dft_doubleNewImg[c].at<Vec2d>(y, x) << " <- " << after << endl;	// 確認用

				before2 = dft_doubleK2.at<Vec2d>(y, x);
				before3 = dft_A.at<Vec2d>(y, x);
				after2 = before2 + before3;
				multi_complex_2(after2, after2, PenaltyParameter);
				after3 = dft_doubleNewImg2[c].at<Vec2d>(y, x);
				after3 += after2;
				dft_doubleNewImg2[c].at<Vec2d>(y, x) = after3;
			}
		}
		mulSpectrums(dft_doubleNewImg[c], dft_doubleK, dft_doubleNewImg1[c], 0, false);
	}
	//visualbule_complex(dft_doubleNewImg[0], Image_dst_deblurred2);	// 確認
	//visualbule_complex(dft_doubleNewImg1[0], Image_dst_deblurred2);	// 確認
	//visualbule_complex(dft_doubleNewImg2[0], Image_dst_deblurred2);	// 確認
	//checkMat_detail(Image_dst_deblurred2);	// 確認

	//inverseDFT
	Mat doubleNewImg[3], doubleNewImg1[3], doubleNewImg2[3];
	for (c = 0; c < 3; c++) {
		/*dft(dft_doubleNewImg[c], dft_doubleNewImg[c], cv::DFT_INVERSE + cv::DFT_SCALE | cv::DFT_REAL_OUTPUT);
		copyMakeBorder(dft_doubleNewImg[c], dft_doubleNewImg[c], NewKernel.rows / 2, NewKernel.rows / 2, NewKernel.cols / 2, NewKernel.cols / 2, BORDER_WRAP);
		doubleNewImg[c] = dft_doubleNewImg[c](Rect(0, 0, NewKernel.cols, NewKernel.rows));*/
		dft(dft_doubleNewImg1[c], dft_doubleNewImg1[c], cv::DFT_INVERSE + cv::DFT_SCALE | cv::DFT_REAL_OUTPUT);
		//copyMakeBorder(dft_doubleNewImg1[c], dft_doubleNewImg1[c], NewKernel.rows / 2, NewKernel.rows / 2, NewKernel.cols / 2, NewKernel.cols / 2, BORDER_WRAP);
		doubleNewImg1[c] = dft_doubleNewImg1[c](Rect(0, 0, NewKernel.cols, NewKernel.rows));
		dft(dft_doubleNewImg2[c], dft_doubleNewImg2[c], cv::DFT_INVERSE + cv::DFT_SCALE | cv::DFT_REAL_OUTPUT);
		//copyMakeBorder(dft_doubleNewImg2[c], dft_doubleNewImg2[c], NewKernel.rows / 2, NewKernel.rows / 2, NewKernel.cols / 2, NewKernel.cols / 2, BORDER_WRAP);
		doubleNewImg2[c] = dft_doubleNewImg2[c](Rect(0, 0, NewKernel.cols, NewKernel.rows));
	}
	//checkMat_detail(dft_doubleNewImg[0]);	// 確認
	//KernelMat_Normalization(dft_doubleNewImg[0]);
	//normalize(dft_doubleNewImg[0], dft_doubleNewImg[0], 0, 255, NORM_MINMAX);
	//dft_doubleNewImg[0].convertTo(Image_dst_deblurred2, CV_8UC1);	// 確認
	//checkMat_detail(Image_dst_deblurred2);	// 確認
	//doubleNewImg[0].convertTo(Image_dst_deblurred2, CV_8UC1);	// 確認
	//doubleNewImg1[0].convertTo(Image_dst_deblurred2, CV_8UC1);	// 確認
	//doubleNewImg2[0].convertTo(Image_dst_deblurred2, CV_8UC1);	// 確認


	/* 実行列 A について */
	//Mat DST_Img;
	//make_matrix_A2(doubleBlurredImg_sub[0], NewKernel.cols, PenaltyParameter[0], DST_Img);
	//DST_Img.convertTo(Image_dst_deblurred2, CV_8UC1);	// 確認
	//checkMat(Image_dst_deblurred2);	// 確認


	/*Mat NextKernel;
	NewKernel.copyTo(NextKernel);
	Mat Ax_ave = (doubleNewImg1[0] + doubleNewImg1[1] + doubleNewImg1[2]) / 3.0;
	Mat b_ave = (doubleNewImg2[0] + doubleNewImg2[1] + doubleNewImg2[2]) / 3.0;
	CG_method(Ax_ave, NextKernel, b_ave, PenaltyParameter[0]);*/

	/* 初期値設定 */
	Mat NextKernel, LastKernel;		// 初期値Kernel
	NewKernel.copyTo(LastKernel);
	NewKernel.copyTo(NextKernel);
	Mat Residual = Mat::zeros(NewKernel.size(), CV_64F);
	Mat P_base = Mat::zeros(NewKernel.size(), CV_64F);
	Mat Mat_tmp_ave = Mat::zeros(NewKernel.size(), CV_64F);
	Mat Mat_tmp[3];
	for (c = 0; c < 3; c++) {
		Mat_tmp[c] = doubleNewImg2[c] - doubleNewImg1[c];
		Mat_tmp_ave += Mat_tmp[c];
	}
	Mat_tmp_ave /= (double)3.0;
	Mat_tmp_ave.copyTo(Residual);
	Mat_tmp_ave.copyTo(P_base);
	//checkMat_detail(doubleNewImg2[0]);	// 確認
	//checkMat_detail(doubleNewImg1[0]);	// 確認
	//checkMat_detail(P_base);	// 確認

	//Mat Alpha[3] = { Mat::zeros(NewKernel.size(), CV_64F), Mat::zeros(NewKernel.size(), CV_64F), Mat::zeros(NewKernel.size(), CV_64F) };
	//Mat Beta[3] = { Mat::zeros(NewKernel.size(), CV_64F), Mat::zeros(NewKernel.size(), CV_64F), Mat::zeros(NewKernel.size(), CV_64F) };
	double ALPHA = 0.0, BETA = 0.0;
	double energy;
	for (int i_number = 0; i_number < Iterate_Number; i_number++) {
		// Calculate ALPHA
		ALPHA = 0.0;
		double Numerator, Denominator;
		for (c = 0; c < 3; c++) {
			Numerator = multi_vector(P_base, Residual);		// ベクトルの内積
			multi_matrix_vector(P_base, dft_doubleNewImg[c], doubleNewImg[c]);
			//checkMat_detail(doubleNewImg[0]);	// 確認
			//doubleNewImg[0].convertTo(Image_dst_deblurred2, CV_8UC1);	// 確認
			Denominator = multi_vector(P_base, doubleNewImg[c]);
			ALPHA += (double)(Numerator / Denominator);
		}
		ALPHA /= 3.0;
		//cout << "  ALPHA = " << (double)ALPHA << endl;	// 確認用

		// Calculate Kernel
		NextKernel = LastKernel + ALPHA * P_base;
#pragma omp parallel for private(x)
		for (y = 0; y < NextKernel.rows; y++) {
			for (x = 0; x < NextKernel.cols; x++) {
				double kernel_num = NextKernel.at<double>(y, x);
				//cout << "   " << (double)kernel_num << endl;	// 確認用
				if (kernel_num < 0) { NextKernel.at<double>(y, x) = 0.0; }	// 負の値は0にする
				//else if (kernel_num > 1) { NextKernel.at<double>(y, x) = 1.0; }
				//cout << "   " << (double)NextKernel.at<double>(y, x) << endl;	// 確認用
			}
		}
		//cout << endl;
		//KernelMat_Normalization(NextKernel);
		//checkMat_detail(LastKernel);	// 確認
		//checkMat_detail(NextKernel);	// 確認

		Mat Residual_before;
		Mat Residual_tmp = Mat::zeros(NewKernel.size(), CV_64F);
		// Calculate Residual
		for (c = 0; c < 3; c++) {
			Residual.copyTo(Residual_before);
			Residual_tmp += Residual_before - ALPHA * doubleNewImg[c];
		}
		Residual_tmp /= (double)3.0;
		Residual_tmp.copyTo(Residual);

		energy = (double)norm(Residual);
		//energy = (double)norm(Residual[0]) + (double)norm(Residual[1]) + (double)norm(Residual[2]);
		//energy = (double)mean(Residual[0])[0] + (double)mean(Residual[1])[0] + (double)mean(Residual[2])[0];
		//energy /= (double)((double)Residual[0].cols * (double)Residual[0].rows * 3.0);
		//cout << "  " << (int)i_number << " : energy = " << (double)energy << endl;	// 確認用
		if (energy < ERROR_END) {
			cout << "   " << (int)i_number << " : energy = " << (double)energy << endl;	// 確認用
			break;
		}

		// Calculate BETA
		BETA = 0.0;
		double Numerator2, Denominator2;
		Numerator2 = multi_vector(Residual_before, Residual_before);
		Denominator2 = multi_vector(Residual, Residual);
		BETA += (double)(Numerator2 / Denominator2);
		//BETA /= 3.0;	//※
		//cout << "  BETA = " << (double)BETA << endl;	// 確認用

		// Calculate P_base
		P_base = Residual + BETA * P_base;

		NextKernel.copyTo(LastKernel);
		//checkMat_detail(NextKernel);	// 確認用
	}
	cout << "   energy = " << (double)energy << endl;	// 確認用

//#pragma omp parallel for private(x)
//	for (y = 0; y < NextKernel.rows; y++) {
//		for (x = 0; x < NextKernel.cols; x++) {
//			double kernel_num = NextKernel.at<double>(y, x);
//			if (kernel_num < 0) { NextKernel.at<double>(y, x) = 0.0; }	// 負の値は0にする
//			//else if (kernel_num > 1) { NextKernel.at<double>(y, x) = 1.0; }
//		}
//	}
	KernelMat_Normalization(NextKernel);

	NextKernel.copyTo(NewKernel);
}

void ConjugateGradientMethod_Gray(Mat& QuantImg, Mat& BlurrImg, Mat& Kernel, Mat& LagrngianMlutipliers, Vec2d& PenaltyParameter, Mat& NewKernel) {
	int Iterate_Number = 100;
	double ERROR_END = 1.0e-04;
	int x, y, c;

	/* Ax=bを解くとしてAxとbを求める */
	// 画像をCV_64FC1に変換(前処理)
	Mat BlurredImg_Color, QuantImg_Color;
	BlurrImg.convertTo(BlurredImg_Color, CV_8UC3);
	QuantImg.convertTo(QuantImg_Color, CV_8UC3);
	Mat BlurredImg_Gray, QuantImg_Gray;
	cvtColor(BlurredImg_Color, BlurredImg_Gray, COLOR_RGB2GRAY);
	cvtColor(QuantImg_Color, QuantImg_Gray, COLOR_RGB2GRAY);
	Mat doubleBlurredImg_Gray, doubleQuantImg_Gray;
	BlurrImg.convertTo(doubleBlurredImg_Gray, CV_64FC1);
	QuantImg.convertTo(doubleQuantImg_Gray, CV_64FC1);

	Mat doubleBlurredImg_sub = Mat::zeros(doubleBlurredImg_Gray.size(), CV_64F);
	Mat doubleQuantImg_sub = Mat::zeros(doubleQuantImg_Gray.size(), CV_64F);
	Mat doubleBlurredImg, doubleQuantImg;
	Mat planes_BI[] = { Mat_<double>(doubleBlurredImg_sub), Mat::zeros(doubleBlurredImg_sub.size(), CV_64F) };
	merge(planes_BI, 2, doubleBlurredImg);
	Mat planes_QI[] = { Mat_<double>(doubleQuantImg_sub), Mat::zeros(doubleQuantImg_sub.size(), CV_64F) };
	merge(planes_QI, 2, doubleQuantImg);
	
	// 入力画像をDFTに指定した大きさに広げる
	int Mplus = doubleBlurredImg_Gray.rows + Kernel.rows;
	int Nplus = doubleBlurredImg_Gray.cols + Kernel.cols;
	int Msize = getOptimalDFTSize(Mplus);
	int Nsize = getOptimalDFTSize(Nplus);
	// DFT
	Mat dft_doubleBlurredImg = Mat::zeros(Msize, Nsize, CV_64FC2);
	Mat dft_doubleQuantImg = Mat::zeros(Msize, Nsize, CV_64FC2);
	Mat dft_doubleBlurredImg_part, dft_doubleQuantImg_part;
	copyMakeBorder(doubleBlurredImg, dft_doubleBlurredImg, Kernel.rows / 2, Msize - Mplus + Kernel.rows / 2, Kernel.cols / 2, Nsize - Nplus + Kernel.cols / 2, BORDER_REPLICATE);
	dft(dft_doubleBlurredImg, dft_doubleBlurredImg);
	copyMakeBorder(doubleQuantImg, dft_doubleQuantImg, Kernel.rows / 2, Msize - Mplus + Kernel.rows / 2, Kernel.cols / 2, Nsize - Nplus + Kernel.cols / 2, BORDER_REPLICATE);
	dft(dft_doubleQuantImg, dft_doubleQuantImg);
	
	// カーネル
	Mat dft_doubleK = Mat::zeros(Msize, Nsize, CV_64FC2);
	Mat planes2[] = { Mat_<double>(NewKernel), Mat::zeros(NewKernel.size(), CV_64F) };
	Mat doubleK_sub;
	merge(planes2, 2, doubleK_sub);
	copyMakeBorder(doubleK_sub, dft_doubleK, 0, Msize - NewKernel.rows, 0, Nsize - NewKernel.cols, BORDER_CONSTANT, (0.0, 0.0));
	dft(dft_doubleK, dft_doubleK, 0, dft_doubleK.rows);
	Mat dft_doubleK2 = Mat::zeros(Msize, Nsize, CV_64FC2);
	Mat planes2_2[] = { Mat_<double>(Kernel), Mat::zeros(Kernel.size(), CV_64F) };
	Mat doubleK_sub2;
	merge(planes2_2, 2, doubleK_sub2);
	copyMakeBorder(doubleK_sub2, dft_doubleK2, 0, Msize - Kernel.rows, 0, Nsize - Kernel.cols, BORDER_CONSTANT, (0.0, 0.0));
	dft(dft_doubleK2, dft_doubleK2, 0, dft_doubleK2.rows);
	Mat dft_A = Mat::zeros(Msize, Nsize, CV_64FC2);
	Mat planes3[] = { Mat_<double>(LagrngianMlutipliers), Mat::zeros(LagrngianMlutipliers.size(), CV_64F) };
	Mat A_sub;
	merge(planes3, 2, A_sub);
	copyMakeBorder(A_sub, dft_A, 0, Msize - LagrngianMlutipliers.rows, 0, Nsize - LagrngianMlutipliers.cols, BORDER_CONSTANT, (0.0, 0.0));
	dft(dft_A, dft_A, 0, dft_A.rows);
	//visualbule_complex(dft_doubleBlurredImg, Image_dst_deblurred2);	// 確認
	//visualbule_complex(dft_doubleQuantImg, Image_dst_deblurred2);	// 確認
	//visualbule_complex(dft_doubleK, Image_dst_deblurred2);	// 確認
	//visualbule_complex(dft_A, Image_dst_deblurred2);	// 確認

	// 計算
	Mat dft_doubleNewImg, dft_doubleNewImg1, dft_doubleNewImg2;
	Mat pow_dft_doubleQuantImg;
	Vec2d before, after, before2, after2, before3, after3;
	dft_doubleNewImg = Mat::zeros(Msize, Nsize, CV_64FC2);
	dft_doubleNewImg1 = Mat::zeros(Msize, Nsize, CV_64FC2);
	dft_doubleNewImg2 = Mat::zeros(Msize, Nsize, CV_64FC2);

	abs_pow_complex(dft_doubleQuantImg, pow_dft_doubleQuantImg);		// 2次元ベクトルの大きさの２乗
	mulSpectrums(dft_doubleBlurredImg, dft_doubleQuantImg, dft_doubleNewImg2, 0, true);	// 複素共役
#pragma omp parallel for private(x)
	for (y = 0; y < Msize; y++) {
		for (x = 0; x < Nsize; x++) {
			before = pow_dft_doubleQuantImg.at<Vec2d>(y, x);
			after = before + PenaltyParameter;
			//cout << "  before = " << before << " , after = " << after << endl;	// 確認用
			dft_doubleNewImg.at<Vec2d>(y, x) = after;
			//cout << "   " << dft_doubleNewImg[c].at<Vec2d>(y, x) << " <- " << after << endl;	// 確認用

			before2 = dft_doubleK2.at<Vec2d>(y, x);
			before3 = dft_A.at<Vec2d>(y, x);
			after2 = before2 + before3;
			multi_complex_2(after2, after2, PenaltyParameter);
			after3 = dft_doubleNewImg2.at<Vec2d>(y, x);
			after3 += after2;
			dft_doubleNewImg2.at<Vec2d>(y, x) = after3;
		}
	}
	mulSpectrums(dft_doubleNewImg, dft_doubleK, dft_doubleNewImg1, 0, false);
	//visualbule_complex(dft_doubleNewImg, Image_dst_deblurred2);	// 確認
	//visualbule_complex(dft_doubleNewImg1, Image_dst_deblurred2);	// 確認
	//visualbule_complex(dft_doubleNewImg2, Image_dst_deblurred2);	// 確認
	//checkMat_detail(Image_dst_deblurred2);	// 確認

	//inverseDFT
	Mat doubleNewImg, doubleNewImg1, doubleNewImg2;
	dft(dft_doubleNewImg1, dft_doubleNewImg1, cv::DFT_INVERSE + cv::DFT_SCALE | cv::DFT_REAL_OUTPUT);
	doubleNewImg1 = dft_doubleNewImg1(Rect(0, 0, NewKernel.cols, NewKernel.rows));
	dft(dft_doubleNewImg2, dft_doubleNewImg2, cv::DFT_INVERSE + cv::DFT_SCALE | cv::DFT_REAL_OUTPUT);
	doubleNewImg2 = dft_doubleNewImg2(Rect(0, 0, NewKernel.cols, NewKernel.rows));
	
	//checkMat_detail(dft_doubleNewImg[0]);	// 確認
	//KernelMat_Normalization(dft_doubleNewImg[0]);
	//normalize(dft_doubleNewImg[0], dft_doubleNewImg[0], 0, 255, NORM_MINMAX);
	//dft_doubleNewImg[0].convertTo(Image_dst_deblurred2, CV_8UC1);	// 確認
	//checkMat_detail(Image_dst_deblurred2);	// 確認
	//doubleNewImg[0].convertTo(Image_dst_deblurred2, CV_8UC1);	// 確認
	//doubleNewImg1[0].convertTo(Image_dst_deblurred2, CV_8UC1);	// 確認
	//doubleNewImg2[0].convertTo(Image_dst_deblurred2, CV_8UC1);	// 確認


	/* 初期値設定 */
	Mat NextKernel, LastKernel;		// 初期値Kernel
	NewKernel.copyTo(LastKernel);
	NewKernel.copyTo(NextKernel);
	Mat Residual = Mat::zeros(NewKernel.size(), CV_64F);
	Mat P_base = Mat::zeros(NewKernel.size(), CV_64F);
	Mat Mat_tmp_ave = Mat::zeros(NewKernel.size(), CV_64F);
	Mat_tmp_ave = doubleNewImg2 - doubleNewImg1;
	Mat_tmp_ave.copyTo(Residual);
	Mat_tmp_ave.copyTo(P_base);
	//checkMat_detail(Mat_tmp_ave);	// 確認

	double ALPHA = 0.0, BETA = 0.0;
	double energy;
	for (int i_number = 0; i_number < Iterate_Number; i_number++) {
		// Calculate ALPHA
		ALPHA = 0.0;
		double Numerator, Denominator;
		Numerator = multi_vector(P_base, Residual);		// ベクトルの内積
		multi_matrix_vector(P_base, dft_doubleNewImg, doubleNewImg);
		//checkMat_detail(doubleNewImg[0]);	// 確認
		//doubleNewImg[0].convertTo(Image_dst_deblurred2, CV_8UC1);	// 確認
		Denominator = multi_vector(P_base, doubleNewImg);
		ALPHA += (double)(Numerator / Denominator);
		//cout << "  ALPHA = " << (double)ALPHA << endl;	// 確認用

		// Calculate Kernel
		NextKernel = LastKernel + ALPHA * P_base;
#pragma omp parallel for private(x)
		for (y = 0; y < NextKernel.rows; y++) {
			for (x = 0; x < NextKernel.cols; x++) {
				double kernel_num = NextKernel.at<double>(y, x);
				//cout << "   " << (double)kernel_num << endl;	// 確認用
				if (kernel_num < 0) { NextKernel.at<double>(y, x) = 0.0; }	// 負の値は0にする
				//else if (kernel_num > 1) { NextKernel.at<double>(y, x) = 1.0; }
				//cout << "   " << (double)NextKernel.at<double>(y, x) << endl;	// 確認用
			}
		}
		//cout << endl;
		//KernelMat_Normalization(NextKernel);
		//checkMat_detail(LastKernel);	// 確認
		//checkMat_detail(NextKernel);	// 確認

		Mat Residual_before;
		Mat Residual_tmp = Mat::zeros(NewKernel.size(), CV_64F);
		// Calculate Residual
		Residual.copyTo(Residual_before);
		Residual_tmp += Residual_before - ALPHA * doubleNewImg;
		Residual_tmp.copyTo(Residual);

		energy = (double)norm(Residual);
		//energy = (double)norm(Residual[0]) + (double)norm(Residual[1]) + (double)norm(Residual[2]);
		//energy = (double)mean(Residual[0])[0] + (double)mean(Residual[1])[0] + (double)mean(Residual[2])[0];
		//energy /= (double)((double)Residual[0].cols * (double)Residual[0].rows * 3.0);
		//cout << "  " << (int)i_number << " : energy = " << (double)energy << endl;	// 確認用
		if (energy < ERROR_END) {
			cout << "   " << (int)i_number << " : energy = " << (double)energy << endl;	// 確認用
			break;
		}

		// Calculate BETA
		BETA = 0.0;
		double Numerator2, Denominator2;
		Numerator2 = multi_vector(Residual_before, Residual_before);
		Denominator2 = multi_vector(Residual, Residual);
		BETA += (double)(Numerator2 / Denominator2);
		//cout << "  BETA = " << (double)BETA << endl;	// 確認用

		// Calculate P_base
		P_base = Residual + BETA * P_base;

		NextKernel.copyTo(LastKernel);
		//checkMat_detail(NextKernel);	// 確認用
	}
	cout << "   energy = " << (double)energy << endl;	// 確認用

//#pragma omp parallel for private(x)
//	for (y = 0; y < NextKernel.rows; y++) {
//		for (x = 0; x < NextKernel.cols; x++) {
//			double kernel_num = NextKernel.at<double>(y, x);
//			if (kernel_num < 0) { NextKernel.at<double>(y, x) = 0.0; }	// 負の値は0にする
//			//else if (kernel_num > 1) { NextKernel.at<double>(y, x) = 1.0; }
//		}
//	}
	KernelMat_Normalization(NextKernel);

	NextKernel.copyTo(NewKernel);
}


#endif