#include "main.h"
#include "Image_Evaluation.h"		// �摜�̕]��(MSE, PSNR, SSIM)
//#include "Image_Histgram.h"		// �q�X�g�O�����擾
#include "Fourier_Transform_Mat.h"
#include "Convolution.h"
#include "Blind_Deconvolution.h"


int main() {
	/* �摜�̓��� */
	Input_Image();
	Image_src.copyTo(Image_dst);
	Image_src.copyTo(Image_dst_deblurred);
	Image_src.copyTo(Image_dst_deblurred2);

	clock_t start, end;	// �������ԕ\���p
	start = clock();
	//--- �摜���� -------------------------------------------------------------------------------
	/* �J�[�l������ */
	cout << "�J�[�l������..." << endl;
	KERNEL kernel_original = KERNEL(2);

	/* �ڂ��摜���� */
	cout << "�ڂ��摜����..." << endl;
	Convolution MakeBlurredImage = Convolution();
	MakeBlurredImage.convolved(Image_src, Image_dst, kernel_original, 2);	// I = x ** k
	//MakeBlurredImage.Evaluatuion();
	GaussianBlur(Image_dst, Image_dst, Size(5, 5), 1, 1);					// I = I + n (�K�E�X�m�C�Y�t��)
	Image_dst.copyTo(Image_dst_deblurred);
	Image_dst.copyTo(Image_dst_deblurred2);

	/* �ڂ����� */
	cout << "�ڂ���������..." << endl;			// ���s�m�F�p
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

	/* �m�F�p */
	//MakeBlurredImage = Convolution();
	//MakeBlurredImage.convolved(Image_dst_deblurred2, Image_dst_deblurred2, kernel_original, 2);	// I = x ** k
	//GaussianBlur(Image_dst_deblurred2, Image_dst_deblurred2, Size(5, 5), 1, 1);					// I = I + n (�K�E�X�m�C�Y�t��)
	//--------------------------------------------------------------------------------------------
	end = clock();
	double time_difference = (double)end - (double)start;
	const double time = time_difference / CLOCKS_PER_SEC * 1000.0;
	cout << "time : " << time << " [ms]" << endl;
	cout << endl;

	/* �J�[�l������ */
	kernel_original.visualization(Image_kernel_original);
	//kernel.normalization();
	kernel.visualization(Image_kernel);

	/* �摜�̕]�� */
	//cout << "�ڂ��摜 �� �^�摜" << endl;
	//Evaluation_MSE_PSNR_SSIM(Image_src, Image_dst);
	//cout << "�ڂ������摜 �� �^�摜" << endl;
	//Evaluation_MSE_PSNR_SSIM(Image_src, Image_dst_deblurred);
	cout << "�ڂ������摜2 �� �^�摜" << endl;
	Evaluation_MSE_PSNR_SSIM(Image_src, Image_dst_deblurred2);
	//cout << "����J�[�l�� �� �^�̃J�[�l��" << endl;
	//Evaluation_MSE_PSNR_SSIM(Image_kernel, Image_kernel_original);

	/* �摜�̏o�� */
	Output_Image();

	return 0;
}

// �摜�̓���
void Input_Image() {
	//string file_src = "img\\src.jpg";	// ���͉摜�̃t�@�C����
	string file_src = "img\\src.png";
	Image_src = imread(file_src, 1);	// ���͉摜�i�J���[�j�̓ǂݍ���
	Image_src_gray = imread(file_src, 0);	// ���͉摜�i�O���[�X�P�[���j�̓ǂݍ���

	/* �p�����[�^��` */
	WIDTH = Image_src.cols;
	HEIGHT = Image_src.rows;
	MAX_DATA = WIDTH * HEIGHT;
	cout << "INPUT : WIDTH = " << WIDTH << " , HEIGHT = " << HEIGHT << endl;
	cout << endl;

	Image_dst = Mat(Size(WIDTH, HEIGHT), CV_8UC3);	// �o�͉摜�i�J���[�j�̏�����
}
// �摜�̏o��
void Output_Image() {
	//string file_dst = "img\\dst.jpg";	// �o�͉摜�̃t�@�C����
	string file_dst = "img\\dst.png";
	string file_dst2 = "img\\dst_kernel.jpg";
	string file_dst3 = "img\\dst_deblurred.jpg";
	string file_dst4 = "img\\dst_deblurred2.jpg";

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
	imshow(win_dst2, Image_kernel);
	imwrite(file_dst2, Image_kernel);
	imshow(win_dst3, Image_dst_deblurred);
	imwrite(file_dst3, Image_dst_deblurred);
	imshow(win_dst4, Image_dst_deblurred2);
	imwrite(file_dst4, Image_dst_deblurred2);

	imwrite(file_dst, Image_src_gray);

	waitKey(0); // �L�[���͑҂�
}
