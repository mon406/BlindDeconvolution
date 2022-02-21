#include "main.h"
#include "Image_Evaluation.h"		// 画像の評価(MSE, PSNR, SSIM)
//#include "Image_Histgram.h"		// ヒストグラム取得
#include "Fourier_Transform_Mat.h"
#include "Convolution.h"
#include "Blind_Deconvolution.h"


int main() {
	/* 画像の入力 */
	Input_Image();
	Image_src.copyTo(Image_dst);
	Image_src.copyTo(Image_dst_deblurred);
	Image_src.copyTo(Image_dst_deblurred2);

	clock_t start, end;	// 処理時間表示用
	start = clock();
	//--- 画像処理 -------------------------------------------------------------------------------
	/* カーネル生成 */
	cout << "カーネル生成..." << endl;
	KERNEL kernel_original = KERNEL(2);

	/* ぼけ画像生成 */
	cout << "ぼけ画像生成..." << endl;
	Convolution MakeBlurredImage = Convolution();
	MakeBlurredImage.convolved(Image_src, Image_dst, kernel_original, 2);	// I = x ** k
	//MakeBlurredImage.Evaluatuion();
	GaussianBlur(Image_dst, Image_dst, Size(5, 5), 1, 1);					// I = I + n (ガウスノイズ付加)
	Image_dst.copyTo(Image_dst_deblurred);
	Image_dst.copyTo(Image_dst_deblurred2);

	/* ぼけ除去 */
	cout << "ぼけ除去処理..." << endl;			// 実行確認用
	//Deconvolution DeblurredImage_InverseFilter = Deconvolution();
	//DeblurredImage_InverseFilter.deconvolved(Image_dst, Image_dst_deblurred, kernel_original);		// inverse filter
	//Deconvolution_WF DeblurredImage_WieneFilter = Deconvolution_WF();
	//DeblurredImage_WieneFilter.calcWienerFilterConstant(MakeBlurredImage, Image_dst);
	//DeblurredImage_InverseFilter.Evaluatuion(Image_src);
	//DeblurredImage_WieneFilter.deconvolved_WF(Image_dst, Image_dst_deblurred2, kernel_original);	// wiener filter
	//DeblurredImage_WieneFilter.Evaluatuion(Image_src);

	KERNEL kernel = KERNEL(1);
	/*BlindDeconvolution DeblurredImage_MRF = BlindDeconvolution();
	DeblurredImage_MRF.initialization(Image_src, Image_dst, kernel_original, 1);
	DeblurredImage_MRF.deblurring(Image_dst_deblurred, kernel);
	DeblurredImage_MRF.Evaluatuion(7);*/

	/* 確認用 */
	//MakeBlurredImage = Convolution();
	//MakeBlurredImage.convolved(Image_dst_deblurred2, Image_dst_deblurred2, kernel_original, 2);	// I = x ** k
	//GaussianBlur(Image_dst_deblurred2, Image_dst_deblurred2, Size(5, 5), 1, 1);					// I = I + n (ガウスノイズ付加)
	//--------------------------------------------------------------------------------------------
	end = clock();
	double time_difference = (double)end - (double)start;
	const double time = time_difference / CLOCKS_PER_SEC * 1000.0;
	cout << "time : " << time << " [ms]" << endl;
	cout << endl;

	/* カーネル可視化 */
	kernel_original.visualization(Image_kernel_original);
	//kernel.normalization();
	kernel.visualization(Image_kernel);

	/* 画像の評価 */
	//cout << "ぼけ画像 と 真画像" << endl;
	//Evaluation_MSE_PSNR_SSIM(Image_src, Image_dst);
	//cout << "ぼけ除去画像 と 真画像" << endl;
	//Evaluation_MSE_PSNR_SSIM(Image_src, Image_dst_deblurred);
	cout << "ぼけ除去画像2 と 真画像" << endl;
	Evaluation_MSE_PSNR_SSIM(Image_src, Image_dst_deblurred2);
	//cout << "推定カーネル と 真のカーネル" << endl;
	//Evaluation_MSE_PSNR_SSIM(Image_kernel, Image_kernel_original);

	/* 画像の出力 */
	Output_Image();

	return 0;
}

// 画像の入力
void Input_Image() {
	//string file_src = "img\\src.jpg";	// 入力画像のファイル名
	string file_src = "img\\src.png";
	Image_src = imread(file_src, 1);	// 入力画像（カラー）の読み込み
	Image_src_gray = imread(file_src, 0);	// 入力画像（グレースケール）の読み込み

	/* パラメータ定義 */
	WIDTH = Image_src.cols;
	HEIGHT = Image_src.rows;
	MAX_DATA = WIDTH * HEIGHT;
	cout << "INPUT : WIDTH = " << WIDTH << " , HEIGHT = " << HEIGHT << endl;
	cout << endl;

	Image_dst = Mat(Size(WIDTH, HEIGHT), CV_8UC3);	// 出力画像（カラー）の初期化
}
// 画像の出力
void Output_Image() {
	//string file_dst = "img\\dst.jpg";	// 出力画像のファイル名
	string file_dst = "img\\dst.png";
	string file_dst2 = "img\\dst_kernel.jpg";
	string file_dst3 = "img\\dst_deblurred.jpg";
	string file_dst4 = "img\\dst_deblurred2.jpg";

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
	imshow(win_dst2, Image_kernel);
	imwrite(file_dst2, Image_kernel);
	imshow(win_dst3, Image_dst_deblurred);
	imwrite(file_dst3, Image_dst_deblurred);
	imshow(win_dst4, Image_dst_deblurred2);
	imwrite(file_dst4, Image_dst_deblurred2);

	imwrite(file_dst, Image_src_gray);

	waitKey(0); // キー入力待ち
}
