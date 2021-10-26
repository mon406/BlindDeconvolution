#include "main.h"
#include "MakeKernel.h"
#include "CalculateComplexNumber.h"
#include "DiscreteFourierTransform.h"
#include "BlindDeconvolution.h"

/* �֐�(���`) */
void ConvolvedImage(Mat& InputImage, Mat& OutputImage, KERNEL& KernelImage);								// �J�[�l���摜��p������ݍ���
void DeconvolvedImage(Mat& InputImage, Mat& OutputImage, KERNEL& KernelImage, Mat& FilterParameter);		// �J�[�l����p�����t��ݍ���(wiener filter)
void DeconvolvedImage_simple(Mat& InputImage, Mat& OutputImage1, Mat& OutputImage2, KERNEL& KernelImage);	// �J�[�l����p�����t��ݍ���(inverse filter)
void DeconvolvedImage_simple_true(Mat& TrueImage, Mat& InputImage, Mat& OutputParameter, KERNEL& KernelImage);	// FFT�̌덷���^�̌W�������߂�


int main() {
	/* �摜�̓��� */
	Input_Image();
	Image_src.copyTo(Image_dst);
	Image_dst.copyTo(Image_dst_deblurred);
	Image_dst.copyTo(Image_dst_deblurred2);

	clock_t start, end;	// �������ԕ\���p
	start = clock();
	//--- �摜���� -------------------------------------------------------------------------------
	/* �J�[�l������ */
	cout << "�J�[�l������..." << endl;			// ���s�m�F�p
	KERNEL kernel = KERNEL(0);
	kernel.Kernel.copyTo(Image_kernel_original);
	//kernel.display_detail();		// �m�F�p

	/* �ڂ��摜���� */
	cout << "�ڂ��摜����..." << endl;			// ���s�m�F�p
	ConvolvedImage(Image_src, Image_dst, kernel);				// I = x ** k
	// �m�C�Y���l�����邽�߂̌W��������
	Mat GammaMat;
	DeconvolvedImage_simple_true(Image_src, Image_dst, GammaMat, kernel);
	// �K�E�X�m�C�Y�t��
	GaussianBlur(Image_dst, Image_dst, Size(11, 11), 1, 1);		// I = I + n
	Image_dst.copyTo(Image_dst_deblurred);
	Image_dst.copyTo(Image_dst_deblurred2);

	/* �ʎq���摜�`�F�b�N */
	//QuantMatDouble QuantizedImageD = QuantMatDouble(10, Image_src);		// Quantize x
	//QuantizedImageD.quantedQMat();
	//QuantizedImageD.QMat.convertTo(Image_dst_deblurred2, CV_8UC3);
	//ConvolvedImage(Image_dst_deblurred2, Image_dst_deblurred2, kernel);				// I = x ** k
	//GaussianBlur(Image_dst_deblurred2, Image_dst_deblurred2, Size(11, 11), 1, 1);	// I = I + n


	/* �ڂ����� */
	cout << "�ڂ���������..." << endl;			// ���s�m�F�p
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

	/* �J�[�l������ */
	KernelImage_Visualization(Image_kernel_original);
	//KernelImage_Visualization_double(Image_kernel_original);
	kernel.inverse_normalization();
	kernel.visualization(Image_kernel);
	//kernel.display_detail();	// �m�F�p

	/* �摜�̕]�� */
	cout << "�ڂ��摜 �� �^�摜" << endl;				// ���s�m�F�p
	Evaluation_MSE_PSNR_SSIM(Image_src, Image_dst);
	cout << "�ڂ������摜 �� �^�摜" << endl;			// ���s�m�F�p
	Evaluation_MSE_PSNR_SSIM(Image_src, Image_dst_deblurred);
	cout << "�ڂ������摜2 �� �^�摜" << endl;			// ���s�m�F�p
	Evaluation_MSE_PSNR_SSIM(Image_src, Image_dst_deblurred2);
	cout << "����J�[�l�� �� �^�̃J�[�l��" << endl;		// ���s�m�F�p
	Evaluation_MSE_PSNR_SSIM(Image_kernel, Image_kernel_original);

	//Image_kernel.convertTo(Image_kernel, CV_8UC1);
	//Image_kernel_original.convertTo(Image_kernel_original, CV_8UC1);

	/* �q�X�g�O�����쐬 */
	//DrawHist(Image_src, Image_src_hist);
	//DrawHist(Image_dst, Image_dst_hist);

	/* �摜�̏o�� */
	Output_Image();

	return 0;
}

// �摜�̓���
void Input_Image() {
	string file_src = "C:\\Users\\mon25\\Desktop\\BlindDeconvolution\\src.jpg";		// ���͉摜�̃t�@�C����
	//string file_src = "C:\\Users\\Yuki Momma\\Desktop\\BlindDeconvolution\\src.jpg";	// ���͉摜�̃t�@�C����
	Image_src = imread(file_src, 1);		// ���͉摜�i�J���[�j�̓ǂݍ���
	Image_src_gray = imread(file_src, 0);	// ���͉摜�i�O���[�X�P�[���j�̓ǂݍ���

	/* �p�����[�^��` */
	WIDTH = Image_src.cols;
	HEIGHT = Image_src.rows;
	MAX_DATA = WIDTH * HEIGHT;
	cout << "INPUT : WIDTH = " << WIDTH << " , HEIGHT = " << HEIGHT << endl;
	cout << endl;

	Image_dst = Mat(Size(WIDTH, HEIGHT), CV_8UC3);	// �o�͉摜�i�J���[�j�̏�����
	Image_dst_deblurred = Mat(Size(WIDTH, HEIGHT), CV_8UC3);
	Image_dst_deblurred2 = Mat(Size(WIDTH, HEIGHT), CV_8UC3);
}
// �摜�̏o��
void Output_Image() {
	string file_dst = "C:\\Users\\mon25\\Desktop\\BlindDeconvolution\\dst.jpg";		// �o�͉摜�̃t�@�C����
	//string file_dst2 = "C:\\Users\\mon25\\Desktop\\BlindDeconvolution\\dst_hist.jpg";
	//string file_dst3 = "C:\\Users\\mon25\\Desktop\\BlindDeconvolution\\src_hist.jpg";
	string file_dst4 = "C:\\Users\\mon25\\Desktop\\BlindDeconvolution\\dst_kernel.jpg";
	string file_dst5 = "C:\\Users\\mon25\\Desktop\\BlindDeconvolution\\dst_deblurred.jpg";
	string file_dst6 = "C:\\Users\\mon25\\Desktop\\BlindDeconvolution\\dst_deblurred2.jpg";
	//string file_dst = "C:\\Users\\Yuki Momma\\Desktop\\BlindDeconvolution\\dst.jpg";	// �o�͉摜�̃t�@�C����
	////string file_dst2 = "C:\\Users\\Yuki Momma\\Desktop\\BlindDeconvolution\\dst_hist.jpg";
	////string file_dst3 = "C:\\Users\\Yuki Momma\\Desktop\\BlindDeconvolution\\src_hist.jpg";
	//string file_dst4 = "C:\\Users\\Yuki Momma\\Desktop\\BlindDeconvolution\\dst_kernel.jpg";
	//string file_dst5 = "C:\\Users\\Yuki Momma\\Desktop\\BlindDeconvolution\\dst_deblurred.jpg";
	//string file_dst6 = "C:\\Users\\Yuki Momma\\Desktop\\BlindDeconvolution\\dst_deblurred2.jpg";

	/* �E�B���h�E���� */
	namedWindow(win_src, WINDOW_AUTOSIZE);
	namedWindow(win_dst, WINDOW_AUTOSIZE);
	namedWindow(win_dst2, WINDOW_AUTOSIZE);
	namedWindow(win_dst3, WINDOW_AUTOSIZE);
	namedWindow(win_dst4, WINDOW_AUTOSIZE);

	/* �摜�̕\�� & �ۑ� */
	imshow(win_src, Image_src);				// ���͉摜��\��
	imshow(win_dst, Image_dst);				// �o�͉摜��\��
	imwrite(file_dst, Image_dst);			// �������ʂ̕ۑ�
	//imwrite(file_dst2, Image_dst_hist);		// �o�̓q�X�g�O�����摜�̕ۑ�
	//imwrite(file_dst3, Image_src_hist);		// ���̓q�X�g�O�����摜�̕ۑ�
	imshow(win_dst2, Image_kernel);			// �o�͉摜��\��
	imwrite(file_dst4, Image_kernel);		// �������ʂ̕ۑ�
	imshow(win_dst3, Image_dst_deblurred);		// �o�͉摜��\��
	imwrite(file_dst5, Image_dst_deblurred);	// �������ʂ̕ۑ�
	imshow(win_dst4, Image_dst_deblurred2);		// �o�͉摜��\��
	imwrite(file_dst6, Image_dst_deblurred2);	// �������ʂ̕ۑ�

	waitKey(0); // �L�[���͑҂�
}

// �J�[�l���摜��p����FFT�ɂ���ݍ���
void ConvolvedImage(Mat& InputImage, Mat& OutputImage, KERNEL& KernelImage) {
	int c;

	/* �O���� */
	// �摜��CV_64F�ɕϊ�
	Mat TrueKernel;
	KernelImage.Kernel_normalized.copyTo(TrueKernel);
	Mat TrueImg;
	InputImage.convertTo(TrueImg, CV_64FC3);

	// DFT�ϊ��̃T�C�Y���v�Z
	int Mplus = TrueImg.rows + TrueKernel.rows;
	int Nplus = TrueImg.cols + TrueKernel.cols;
	int Msize = getOptimalDFTSize(Mplus);
	int Nsize = getOptimalDFTSize(Nplus);
	cout << " FFT Size  : (" << Mplus << "," << Nplus << ") => (" << Msize << "," << Nsize << ")" << endl;	// �m�F

	// 3�̃`���l��B, G, R�ɕ���
	Mat doubleTrueImg_sub[3] = { Mat::zeros(TrueImg.size(), CV_64F), Mat::zeros(TrueImg.size(), CV_64F), Mat::zeros(TrueImg.size(), CV_64F) };
	split(TrueImg, doubleTrueImg_sub);
	Mat doubleTrueImg[3];
	for (c = 0; c < 3; c++) {
		Mat planes[] = { Mat_<double>(doubleTrueImg_sub[c]), Mat::zeros(doubleTrueImg_sub[c].size(), CV_64F) };
		merge(planes, 2, doubleTrueImg[c]);
	}


	/* DFT */
	// �J�[�l����DFT
	Mat dft_TrueKernel = Mat::zeros(Msize, Nsize, CV_64FC2);
	Mat planes2[] = { Mat_<double>(TrueKernel), Mat::zeros(TrueKernel.size(), CV_64F) };
	merge(planes2, 2, TrueKernel);
	copyMakeBorder(TrueKernel, dft_TrueKernel, 0, Msize - TrueKernel.rows, 0, Nsize - TrueKernel.cols, BORDER_CONSTANT, (0.0, 0.0));
	dft(dft_TrueKernel, dft_TrueKernel, 0, dft_TrueKernel.rows);
	//visualbule_complex(dft_TrueKernel, Image_dst_deblurred2);	// �m�F

	// �^�̌��摜��DFT
	Mat dft_doubleTrueImg[3] = { Mat::zeros(Msize, Nsize, CV_64FC2), Mat::zeros(Msize, Nsize, CV_64FC2), Mat::zeros(Msize, Nsize, CV_64FC2) };
	for (c = 0; c < 3; c++) {
		copyMakeBorder(doubleTrueImg[c], dft_doubleTrueImg[c], TrueKernel.rows / 2, Msize - Mplus + TrueKernel.rows / 2, TrueKernel.cols / 2, Nsize - Nplus + TrueKernel.cols / 2, BORDER_REPLICATE);
		dft(dft_doubleTrueImg[c], dft_doubleTrueImg[c], 0, dft_doubleTrueImg[c].rows);
	}
	//visualbule_complex(dft_doubleTrueImg_2[0], Image_dst_deblurred2);	// �m�F


	/* �ڂ��摜�����߂� */
	Mat dft_doubleBlurredImg[3];
	Mat dft_denomImg[3];
	for (c = 0; c < 3; c++) {
		mulSpectrums(dft_doubleTrueImg[c], dft_TrueKernel, dft_doubleBlurredImg[c], 0, false);
	}
	//visualbule_complex(dft_doubleBlurredImg[0], Image_dst_deblurred2);	// �m�F


	/* �o�� */
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

// �J�[�l����p����FFT�ɂ��t��ݍ��� (�E�B�[�i�E�t�B���^�[)
void DeconvolvedImage(Mat& InputImage, Mat& OutputImage, KERNEL& KernelImage, Mat& FilterParameter) {
	int x, y, c;

	/* �O���� */
	// �摜��CV_64F�ɕϊ�
	Mat TrueKernel;
	KernelImage.Kernel_normalized.copyTo(TrueKernel);
	Mat BlurredImg;
	InputImage.convertTo(BlurredImg, CV_64FC3);

	// DFT�ϊ��̃T�C�Y���v�Z
	int Mplus = BlurredImg.rows + TrueKernel.rows;
	int Nplus = BlurredImg.cols + TrueKernel.cols;
	int Msize = getOptimalDFTSize(Mplus);
	int Nsize = getOptimalDFTSize(Nplus);
	cout << " FFT Size  : (" << Mplus << "," << Nplus << ") => (" << Msize << "," << Nsize << ")" << endl;	// �m�F

	// 3�̃`���l��B, G, R�ɕ���
	Mat doubleBlurredImg_sub[3] = { Mat::zeros(BlurredImg.size(), CV_64F), Mat::zeros(BlurredImg.size(), CV_64F), Mat::zeros(BlurredImg.size(), CV_64F) };
	split(BlurredImg, doubleBlurredImg_sub);
	Mat doubleBlurredImg[3];
	for (c = 0; c < 3; c++) {
		Mat planes[] = { Mat_<double>(doubleBlurredImg_sub[c]), Mat::zeros(doubleBlurredImg_sub[c].size(), CV_64F) };
		merge(planes, 2, doubleBlurredImg[c]);
	}


	/* DFT */
	// �J�[�l����DFT
	Mat dft_TrueKernel = Mat::zeros(Msize, Nsize, CV_64FC2);
	Mat planes3[] = { Mat_<double>(TrueKernel), Mat::zeros(TrueKernel.size(), CV_64F) };
	merge(planes3, 2, TrueKernel);
	copyMakeBorder(TrueKernel, dft_TrueKernel, 0, Msize - TrueKernel.rows, 0, Nsize - TrueKernel.cols, BORDER_CONSTANT, (0.0, 0.0));
	dft(dft_TrueKernel, dft_TrueKernel, 0, dft_TrueKernel.rows);
	//visualbule_complex(dft_TrueKernel, Image_dst_deblurred);	// �m�F

	// �ڂ��摜��DFT
	Mat dft_doubleBlurredImg[3] = { Mat::zeros(Msize, Nsize, CV_64FC2), Mat::zeros(Msize, Nsize, CV_64FC2), Mat::zeros(Msize, Nsize, CV_64FC2) };
	for (c = 0; c < 3; c++) {
		copyMakeBorder(doubleBlurredImg[c], dft_doubleBlurredImg[c], TrueKernel.rows / 2, Msize - Mplus + TrueKernel.rows / 2, TrueKernel.cols / 2, Nsize - Nplus + TrueKernel.cols / 2, BORDER_REPLICATE);
		dft(dft_doubleBlurredImg[c], dft_doubleBlurredImg[c], 0, dft_doubleBlurredImg[c].rows);
	}
	//visualbule_complex(dft_doubleBlurredImg_2[0], Image_dst_deblurred);	// �m�F


	/* �^�̌��摜�����߂� */
	Mat dft_doubleTrueImg[3];
	Mat dft_WienerKernel;
	wiener_filter(dft_TrueKernel, dft_WienerKernel, FilterParameter);		// 2�����x�N�g���̋t��(�E�B�[�i�E�t�B���^�[)
	//visualbule_complex(dft_WienerKernel, Image_dst_deblurred);	// �m�F
	for (c = 0; c < 3; c++) {
		mulSpectrums(dft_doubleBlurredImg[c], dft_WienerKernel, dft_doubleTrueImg[c], 0, false);
	}
	//visualbule_complex(dft_doubleTrueImg[0], Image_dst_deblurred);	// �m�F


	/* �o�� */
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

	// �J�[�l��
	reciprocal_complex(dft_WienerKernel, dft_WienerKernel);
	//visualbule_complex(dft_WienerKernel, Image_dst_deblurred);	// �m�F
	dft(dft_WienerKernel, dft_WienerKernel, cv::DFT_INVERSE + cv::DFT_SCALE | cv::DFT_REAL_OUTPUT);
	copyMakeBorder(dft_WienerKernel, dft_WienerKernel, TrueKernel.rows / 2, TrueKernel.rows / 2, TrueKernel.cols / 2, TrueKernel.cols / 2, BORDER_WRAP);
	Mat KernelTmp = dft_WienerKernel(Rect(0, 0, TrueKernel.cols, TrueKernel.rows));
	//Mat KernelTmp = dft_WienerKernel(Rect(0, 0, Nsize, Msize));
	double double_num;
#pragma omp parallel for private(x)
	for (y = 0; y < KernelTmp.rows; y++) {
		for (x = 0; x < KernelTmp.cols; x++) {
			double_num = KernelTmp.at<double>(y, x);	// ���̒l��0�ɂ���
			//cout << "  double_num = " << double_num << endl;	// �m�F�p
			if (double_num < 0) {
				double_num = 0.0;
				KernelTmp.at<double>(y, x) = double_num;
			}
		}
	}
	KernelTmp.copyTo(KernelImage.Kernel_normalized);
}
// �J�[�l����p����FFT�ɂ��P��(���_�I)�t��ݍ���
void DeconvolvedImage_simple(Mat& InputImage, Mat& OutputImage1, Mat& OutputImage2, KERNEL& KernelImage) {
	int c;

	/* �O���� */
	// �摜��CV_64F�ɕϊ�
	Mat TrueKernel;
	KernelImage.Kernel_normalized.copyTo(TrueKernel);
	Mat BlurredImg;
	InputImage.convertTo(BlurredImg, CV_64FC3);

	// DFT�ϊ��̃T�C�Y���v�Z
	int Mplus = BlurredImg.rows + TrueKernel.rows;
	int Nplus = BlurredImg.cols + TrueKernel.cols;
	int Msize = getOptimalDFTSize(Mplus);
	int Nsize = getOptimalDFTSize(Nplus);
	cout << " FFT Size  : (" << Mplus << "," << Nplus << ") => (" << Msize << "," << Nsize << ")" << endl;	// �m�F

	// 3�̃`���l��B, G, R�ɕ���
	Mat doubleBlurredImg_sub[3] = { Mat::zeros(BlurredImg.size(), CV_64F), Mat::zeros(BlurredImg.size(), CV_64F), Mat::zeros(BlurredImg.size(), CV_64F) };
	split(BlurredImg, doubleBlurredImg_sub);
	Mat doubleBlurredImg[3];
	for (c = 0; c < 3; c++) {
		Mat planes[] = { Mat_<double>(doubleBlurredImg_sub[c]), Mat::zeros(doubleBlurredImg_sub[c].size(), CV_64F) };
		merge(planes, 2, doubleBlurredImg[c]);
	}


	/* DFT */
	// �J�[�l����DFT
	Mat dft_TrueKernel = Mat::zeros(Msize, Nsize, CV_64FC2);
	Mat planes3[] = { Mat_<double>(TrueKernel), Mat::zeros(TrueKernel.size(), CV_64F) };
	merge(planes3, 2, TrueKernel);
	copyMakeBorder(TrueKernel, dft_TrueKernel, 0, Msize - TrueKernel.rows, 0, Nsize - TrueKernel.cols, BORDER_CONSTANT, (0.0, 0.0));
	dft(dft_TrueKernel, dft_TrueKernel, 0, dft_TrueKernel.rows);
	//visualbule_complex(dft_TrueKernel, Image_dst_deblurred2);	// �m�F

	// �ڂ��摜��DFT
	Mat dft_doubleBlurredImg[3] = { Mat::zeros(Msize, Nsize, CV_64FC2), Mat::zeros(Msize, Nsize, CV_64FC2), Mat::zeros(Msize, Nsize, CV_64FC2) };
	for (c = 0; c < 3; c++) {
		copyMakeBorder(doubleBlurredImg[c], dft_doubleBlurredImg[c], TrueKernel.rows / 2, Msize - Mplus + TrueKernel.rows / 2, TrueKernel.cols / 2, Nsize - Nplus + TrueKernel.cols / 2, BORDER_REPLICATE);
		dft(dft_doubleBlurredImg[c], dft_doubleBlurredImg[c], 0, dft_doubleBlurredImg[c].rows);
	}
	//visualbule_complex(dft_doubleBlurredImg_2[0], Image_dst_deblurred2);	// �m�F


	/* �^�̌��摜�����߂� */
	Mat dft_doubleTrueImg[3];
	Mat dft_doubleTrueImg2[3];
	Mat dft_denomImg[3], dft_denomImg2[3];
	for (c = 0; c < 3; c++) {
		abs_pow_complex(dft_TrueKernel, dft_denomImg[c]);
		reciprocal_complex(dft_denomImg[c], dft_denomImg[c]);
		mulSpectrums(dft_doubleBlurredImg[c], dft_TrueKernel, dft_doubleTrueImg[c], 0, true);	// ���f����
		mulSpectrums(dft_doubleTrueImg[c], dft_denomImg[c], dft_doubleTrueImg[c], 0, false);

		reciprocal_complex(dft_TrueKernel, dft_denomImg2[c]);
		//visualbule_complex(dft_denomImg2[0], Image_dst_deblurred);	// �m�F
		mulSpectrums(dft_doubleBlurredImg[c], dft_denomImg2[c], dft_doubleTrueImg2[c], 0, false);
	}
	//visualbule_complex(dft_doubleTrueImg[0], Image_dst_deblurred2);	// �m�F
	//visualbule_complex(dft_doubleTrueImg2[0], Image_dst_deblurred2);	// �m�F


	/* �o�� */
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

// FFT�̌덷���^�̌W�������߂�
void DeconvolvedImage_simple_true(Mat& TrueImage, Mat& InputImage, Mat& OutputParameter, KERNEL& KernelImage) {
	int x, y, c;

	/* �O���� */
	// �摜��CV_64F�ɕϊ�
	Mat TrueKernel;
	KernelImage.Kernel_normalized.copyTo(TrueKernel);
	Mat TrueImg;
	TrueImage.convertTo(TrueImg, CV_64FC3);
	Mat BlurredImg;
	InputImage.convertTo(BlurredImg, CV_64FC3);

	// DFT�ϊ��̃T�C�Y���v�Z
	int Mplus = TrueImg.rows + TrueKernel.rows;
	int Nplus = TrueImg.cols + TrueKernel.cols;
	int Msize = getOptimalDFTSize(Mplus);
	int Nsize = getOptimalDFTSize(Nplus);
	//cout << " FFT Size  : (" << Mplus << "," << Nplus << ") => (" << Msize << "," << Nsize << ")" << endl;	// �m�F

	// 3�̃`���l��B, G, R�ɕ���
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
	// �J�[�l����DFT
	Mat dft_TrueKernel = Mat::zeros(Msize, Nsize, CV_64FC2);
	Mat planes2[] = { Mat_<double>(TrueKernel), Mat::zeros(TrueKernel.size(), CV_64F) };
	merge(planes2, 2, TrueKernel);
	copyMakeBorder(TrueKernel, dft_TrueKernel, 0, Msize - TrueKernel.rows, 0, Nsize - TrueKernel.cols, BORDER_CONSTANT, (0.0, 0.0));
	dft(dft_TrueKernel, dft_TrueKernel, 0, dft_TrueKernel.rows);
	//visualbule_complex(dft_TrueKernel, Image_dst_deblurred2);	// �m�F

	// �^�̌��摜��DFT
	Mat dft_doubleTrueImg[3] = { Mat::zeros(Msize, Nsize, CV_64FC2), Mat::zeros(Msize, Nsize, CV_64FC2), Mat::zeros(Msize, Nsize, CV_64FC2) };
	for (c = 0; c < 3; c++) {
		copyMakeBorder(doubleTrueImg[c], dft_doubleTrueImg[c], TrueKernel.rows / 2, Msize - Mplus + TrueKernel.rows / 2, TrueKernel.cols / 2, Nsize - Nplus + TrueKernel.cols / 2, BORDER_REPLICATE);
		dft(dft_doubleTrueImg[c], dft_doubleTrueImg[c], 0, dft_doubleTrueImg[c].rows);
	}
	//visualbule_complex(dft_doubleTrueImg[0], Image_dst_deblurred);	// �m�F

	// �ڂ��摜��DFT
	Mat dft_doubleBlurredImg[3] = { Mat::zeros(Msize, Nsize, CV_64FC2), Mat::zeros(Msize, Nsize, CV_64FC2), Mat::zeros(Msize, Nsize, CV_64FC2) };
	for (c = 0; c < 3; c++) {
		copyMakeBorder(doubleBlurredImg[c], dft_doubleBlurredImg[c], TrueKernel.rows / 2, Msize - Mplus + TrueKernel.rows / 2, TrueKernel.cols / 2, Nsize - Nplus + TrueKernel.cols / 2, BORDER_REPLICATE);
		dft(dft_doubleBlurredImg[c], dft_doubleBlurredImg[c], 0, dft_doubleBlurredImg[c].rows);
	}
	//visualbule_complex(dft_doubleBlurredImg[0], Image_dst_deblurred2);	// �m�F

	/* �^�̌W�������߂� */
	Mat dft_doubleTrueImg2[3];
	Mat dft_denomImg[3], dft_denomImg2[3];
	for (c = 0; c < 3; c++) {
		abs_pow_complex(dft_TrueKernel, dft_denomImg[c]);
		reciprocal_complex(dft_denomImg[c], dft_denomImg2[c]);
		//visualbule_complex(dft_denomImg2[c], Image_dst_deblurred2);	// �m�F
		mulSpectrums(dft_doubleBlurredImg[c], dft_TrueKernel, dft_doubleTrueImg2[c], 0, true);	// ���f����
		//mulSpectrums(dft_doubleTrueImg2[c], dft_denomImg2[c], dft_doubleTrueImg2[c], 0, false);
	}
	//visualbule_complex(dft_denomImg2[0], Image_dst_deblurred2);	// �m�F
	//visualbule_complex(dft_doubleTrueImg2[0], Image_dst_deblurred2);	// �m�F

	// �^�̌W����Mat
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
				//cout << "  number1 = " << number1 << " , number2 = " << number2 << endl;	// �m�F�p
				divi_complex_2(number, number2, number1);
				number = number - number3;
				Noise[c].at<Vec2d>(y, x) = number;
				ave_number[c] = ave_number[c] + number;
				//cout << "  ave = " << ave_number[c] << " , number = " << number << endl;	// �m�F�p
			}
		}
		divi_complex_2(ave_number[c], ave_number[c], all_number);
		//cout << "  " << c << " : " << ave_number[c] << endl;	// �m�F�p

		Error += ave_number[c];
	}
	divi_complex_2(Error, Error, c_number);
	//cout << " Error = " << Error << endl;	// �m�F�p
	//Noise[0] = Mat(Msize, Nsize, CV_64FC2, TrueParameter);
	//visualbule_complex(Noise[0], Image_dst_deblurred2);	// �m�F

	Mat Gamma_Mat = Mat::zeros(Msize, Nsize, CV_64FC2);
#pragma omp parallel for private(x)
	for (y = 0; y < Msize; y++) {
		for (x = 0; x < Nsize; x++) {
			number1 = Noise[0].at<Vec2d>(y, x);
			number2 = Noise[1].at<Vec2d>(y, x);
			number3 = Noise[2].at<Vec2d>(y, x);
			number = number1 + number2 + number3;
			//cout << "  number1 = " << number1 << "  number2 = " << number2 << " , number3 = " << number3 << endl;	// �m�F�p
			divi_complex_2(number, number, c_number);
			//cout << "  number = " << number << endl;	// �m�F�p
			Gamma_Mat.at<Vec2d>(y, x) = number;
		}
	}
	//checkMat(GammaMat);	// �m�F
	//visualbule_complex(GammaMat, Image_dst_deblurred2);	// �m�F

	/* �o�� */
	Gamma_Mat.copyTo(OutputParameter);

	//// �E�B�[�i�E�t�B���^�[�m�F ��
	//Mat W_Filter[3];
	//Mat doubleTrueImg2[3];
	//for (c = 0; c < 3; c++) {
	//	double theredhold = 10;
	//	dft_denomImg2[c].copyTo(W_Filter[c]);
	//	//GaussianBlur(W_Filter[c], W_Filter[c], Size(11, 11), 1, 1);		// �K�E�X�m�C�Y�t��(I = I + n)
	//	wiener_filter(dft_TrueKernel, W_Filter[c], Noise[c]);			// 2�����x�N�g���̋t��(�E�B�[�i�E�t�B���^�[)
	//	visualbule_complex(W_Filter[c], Image_dst_deblurred);	// �m�F

	//	//mulSpectrums(W_Filter[c], dft_TrueKernel, dft_doubleTrueImg2[c], 0, true);	// ����PSF
	//	//reciprocal_complex(dft_doubleTrueImg2[c], dft_doubleTrueImg2[c]);

	//	mulSpectrums(dft_doubleTrueImg2[c], W_Filter[c], dft_doubleTrueImg2[c], 0, false);
	//	//mulSpectrums(dft_doubleTrueImg2[c], dft_denomImg2[c], dft_doubleTrueImg2[c], 0, false);
	//	//visualbule_complex(dft_doubleTrueImg2[c], Image_dst_deblurred2);	// �m�F

	//	dft(dft_doubleTrueImg2[c], dft_doubleTrueImg2[c], cv::DFT_INVERSE + cv::DFT_SCALE | cv::DFT_REAL_OUTPUT);
	//	/*dft_doubleTrueImg2[c] = dft_doubleTrueImg2[c] + dft_doubleTrueImg2[c];
	//	dft_doubleTrueImg2[c] = dft_doubleTrueImg2[c] / 2.0;*/
	//	doubleTrueImg2[c] = dft_doubleTrueImg2[c](Rect(TrueKernel.cols / 2, TrueKernel.rows / 2, BlurredImg.cols, BlurredImg.rows));
	//}
	//Mat TrueImage2;
	//merge(doubleTrueImg2, 3, TrueImage2);
	//TrueImage2.convertTo(TrueImage2, CV_8U);
	////TrueImage2.copyTo(Image_dst_deblurred);	// �m�F

	//// �J�[�l���̊m�F ��
	///*dft(dft_TrueKernel, dft_TrueKernel, cv::DFT_INVERSE + cv::DFT_SCALE | cv::DFT_REAL_OUTPUT);
	//Mat KernelTmp = dft_TrueKernel(Rect(0, 0, TrueKernel.cols, TrueKernel.rows));*/
	//Mat KernelTmp = dft_doubleTrueImg2[0](Rect(0, 0, TrueKernel.cols, TrueKernel.rows));
	////KernelImage.inputKernel_normalized(KernelTmp);
	//KernelTmp = KernelTmp * (double)KernelImage.sum;
	//KernelTmp.copyTo(Image_dst_deblurred2);
	//KernelImage_Visualization(Image_dst_deblurred2);
}
