#ifndef __INCLUDED_H_Blind_Deconvolution__
#define __INCLUDED_H_Blind_Deconvolution__

#include "main.h"
#include "Image_Evaluation.h"
#include "Fourier_Transform_Mat.h"
#include "Calculate_Complex_Number.h"
#include "Conjugate_Gradient.h"


/* �萔 */
const int MAX_Iteration = 10;		// �ő唽����
const int MAX_Iteration_ADMM = 10;

/* �p�����[�^ */
const double Myu = 0.4e-03;
const double Rambda = 0.4e-03;
const double Tau = 1.0e-03;


/*--- BlindDeconvolution�N���X ----------------------------------------
	�J�[�l����������t��ݍ��݂��s���N���X (FTMat�N���X���g�p)
	TrueImage:			�^�摜 (���͗p)
	ConvImage:			��ݍ��݉摜 (�o�͗p)
	KernelFilter:		�J�[�l�� (�o�͗p)
	KernelFastOrder:	�J�[�l���w��l (���͗p)
-----------------------------------------------------------------------*/
class BlindDeconvolution {
private:
	int x = 0, y = 0, c = 0;
	int pyr = 0, index = 0;
public:
	vector<FTMat3D> DeconvImg;	// �t��ݍ���(�ڂ�����)�摜
	vector<FTMat3D> BlurrImg;	// �ڂ��摜
	vector<FTMat3D> QuantImg;	// �ʎq���摜
	vector<KERNEL> Kernel;		// ����J�[�l��
	vector<FTMat3D> TrueImg;	// �^�摜
	vector<KERNEL> TrueKernel;	// �^�J�[�l��

	vector<int> X_SIZE;
	vector<int> Y_SIZE;
	vector<int> MAX_PIX;
	vector<int> K_X_SIZE;
	vector<int> K_Y_SIZE;
	const int PYRAMID_NUM = 8;														// �s���~�b�h�K�w��
	const double ResizeFactor[8] = { 0.1, 0.2, 0.25, 0.4, 0.5, 0.6, 0.75, 1.0 };	// �s���~�b�h�̏k���v�f

	BlindDeconvolution();							// ������
	void initialization(Mat& TrueIMG, Mat& ConvIMG, KERNEL& InputTrueKernel, int KernelFastOrder);
	void deblurring(Mat& OutputIMG, KERNEL& OutputKernel);
	void UpdateQuantizedImage(int NowPyrNum);
	void UpdateQuantizedImage_wighted(int NowPyrNum);
	void UpdateDeconvImage(int NowPyrNum);
	void UpdateDeconvImage_FFT(int NowPyrNum);
	void UpdateKarnel(int NowPyrNum);
	void Upsampling(int NextPyramidNumber);
	void Evaluatuion(int ComperePyramidNumber);		// �摜��r�]��
};
BlindDeconvolution::BlindDeconvolution() {
	DeconvImg.clear();
	BlurrImg.clear();
	QuantImg.clear();
	Kernel.clear();
	TrueImg.clear();
	TrueKernel.clear();

	X_SIZE.clear();
	Y_SIZE.clear();
	MAX_PIX.clear();
	K_X_SIZE.clear();
	K_Y_SIZE.clear();
}
void BlindDeconvolution::initialization(Mat& TrueIMG, Mat& ConvIMG, KERNEL& InputTrueKernel, int KernelFastOrder) {
	/* �����l�ݒ� */
	cout << "Initialization..." << endl;		// ���s�m�F�p
	KERNEL SetKernel = KERNEL(KernelFastOrder);
	Mat doubleTrueIMG, doubleConvIMG, TrueKernelIMG, KernelIMG;
	TrueIMG.convertTo(doubleTrueIMG, CV_64FC3);
	ConvIMG.convertTo(doubleConvIMG, CV_64FC3);
	InputTrueKernel.ImgMat.copyTo(TrueKernelIMG);
	SetKernel.ImgMat.copyTo(KernelIMG);

	Mat NowdoubleTrueIMG, NowdoubleConvIMG, NowTrueKernelIMG, NowKernelIMG;
	FTMat3D Now_TrueIMG, Now_BlurrIMG, Now_ConvIMG;
	KERNEL Now_TrueKernelIMG, Now_KernelIMG;
	double MAX_PIX_tmp;
	for (pyr = 0; pyr < PYRAMID_NUM; pyr++) {
		// �摜�̃��T�C�Y
		resize(doubleTrueIMG, NowdoubleTrueIMG, Size(), ResizeFactor[pyr], ResizeFactor[pyr]);
		resize(doubleConvIMG, NowdoubleConvIMG, Size(), ResizeFactor[pyr], ResizeFactor[pyr]);
		X_SIZE.push_back(NowdoubleTrueIMG.cols);
		Y_SIZE.push_back(NowdoubleTrueIMG.rows);
		MAX_PIX_tmp = (double)NowdoubleTrueIMG.cols * (double)NowdoubleTrueIMG.rows;
		MAX_PIX.push_back(MAX_PIX_tmp);

		Now_TrueIMG = FTMat3D(NowdoubleTrueIMG);
		TrueImg.push_back(Now_TrueIMG);
		Now_BlurrIMG = FTMat3D(NowdoubleConvIMG);
		Now_ConvIMG = FTMat3D(NowdoubleConvIMG);
		DeconvImg.push_back(Now_ConvIMG);
		BlurrImg.push_back(Now_BlurrIMG);
		QuantImg.push_back(Now_ConvIMG);

		// �J�[�l���̃��T�C�Y
		resize(TrueKernelIMG, NowTrueKernelIMG, Size(), ResizeFactor[pyr], ResizeFactor[pyr]);
		resize(KernelIMG, NowKernelIMG, Size(), ResizeFactor[pyr], ResizeFactor[pyr]);
		K_X_SIZE.push_back(NowTrueKernelIMG.cols);
		K_Y_SIZE.push_back(NowTrueKernelIMG.rows);

		Now_TrueKernelIMG = KERNEL(NowTrueKernelIMG);
		TrueKernel.push_back(Now_TrueKernelIMG);
		Now_KernelIMG = KERNEL(NowKernelIMG);
		Kernel.push_back(Now_KernelIMG);
	}
	cout << endl;
}
void BlindDeconvolution::deblurring(Mat& OutputIMG, KERNEL& OutputKernel) {
	if (TrueImg.size() == 0) { cout << "WARNING! BlindDeconvolution : Initialization didn't finish." << endl; }
	else {
		/* �ڂ����� */
		for (pyr = 0/*PYRAMID_NUM - 1*/; pyr < PYRAMID_NUM; pyr++) {
			cout << "Deconvoluting in " << (int)pyr << endl;		// ���s�m�F�p

			for (int i = 0; i < MAX_Iteration; i++) {
				/* Update x~ */
				//UpdateQuantizedImage(pyr);
				UpdateQuantizedImage_wighted(pyr);

				/* Update x */
				//UpdateDeconvImage(pyr);
				UpdateDeconvImage_FFT(pyr);
				//if (i == 0) { break; }

				/* Update k */
				Mat before_Kernel, after_Kernel;
				Kernel[pyr].ImgMat.copyTo(before_Kernel);
				UpdateKarnel(pyr);
				Kernel[pyr].ImgMat.copyTo(after_Kernel);

				double diff_Kernel = (double)norm(before_Kernel, after_Kernel, NORM_L2);
				diff_Kernel = (double)diff_Kernel / (double)Kernel[pyr].size;
				cout << "  diff_Kernel = " << diff_Kernel << endl;		// ���s�m�F�p
				if (diff_Kernel < (double)1.0e-08) { break; }

				//if (i == 0) { break; }
			}

			/* �摜�̕]�� */
			//Evaluatuion(pyr);		// �m�F�p

			/* �o�� (���o�]��) */
			DeconvImg[pyr].output(OutputIMG);
			BlurrImg[pyr].output(Image_dst);
			QuantImg[pyr].output(Image_dst_deblurred2);
			TrueImg[pyr].output(Image_src);
			Kernel[pyr].copyTo(OutputKernel);

			/* Upsample */
			Upsampling(pyr + 1);
		}

		/* �o�� */
		//int InputImage_PYRAMID_NUM = PYRAMID_NUM - 1;
		//DeconvImg[InputImage_PYRAMID_NUM].output(OutputIMG);
		//QuantImg[InputImage_PYRAMID_NUM].output(Image_dst_deblurred2);	// �m�F�p
		//Kernel[InputImage_PYRAMID_NUM].copyTo(OutputKernel);
	}
}
void BlindDeconvolution::UpdateQuantizedImage(int NowPyrNum) {
	/* Update x~ */
	cout << " Update QuantImg... " << endl;			// ���s�m�F�p
	/* Optimizing using k-means */
	Mat NewQuantImg;
	QuantImg[NowPyrNum].output(NewQuantImg);
	NewQuantImg.convertTo(NewQuantImg, CV_32FC3);

	/* k-means �N���X�^�����O */
	Mat feature;
	NewQuantImg.copyTo(feature);
	//�摜�̉�f��1��i3�`���l���j�ɕ��ׂ�
	feature = feature.reshape(3, feature.rows * feature.cols);

	Mat_<int> labels(feature.size(), CV_32SC1);
	Mat centers;

	//kmeans�@�ɂ��摜�̕��ށi�̈敪���j
	const int MAX_CLUSTERS = 15/*10*/;	// �ő�F�� n
	kmeans(feature, MAX_CLUSTERS, labels, TermCriteria(TermCriteria::COUNT, 10, 1.0), 1, KMEANS_RANDOM_CENTERS, centers);

	// ���x�����O���ʂ̕`��F������
	vector<Vec3b> colors(MAX_CLUSTERS + 1);
	Vec3d center_colors;
	for (int label = 0; label <= MAX_CLUSTERS; ++label)
	{
		center_colors = Vec3d(0.0, 0.0, 0.0);
		//�N���X�^�̒��S���W
		for (c = 0; c < 3; c++) {
			center_colors[c] = (double)centers.at<float>(label, c);
		}

		//�N���X�^���ɐF��`��
		for (c = 0; c < 3; c++) {
			colors[label][c] = (uchar)center_colors[c];
		}
		//cout << "  ave_colors = " << (Vec3d)ave_colors << " => " << (Vec3b)colors[label] << endl;	// �m�F�p

		////���x���ԍ��ɑ΂��ĐF�������_���Ɋ��蓖�Ă�
		//colors[label] = Vec3b((rand() & 255), (rand() & 255), (rand() & 255));
	}

	//���x�����O���ʉ摜�̍쐬
	Mat dst(NewQuantImg.size(), CV_8UC3);
	//���ʉ摜�̊e��f�̐F�����肵�Ă���
	MatIterator_<Vec3b> itd = dst.begin<Vec3b>(), itd_end = dst.end<Vec3b>();
	for (int i = 0; itd != itd_end; ++itd, ++i) {
		int label = labels(i);	//���x�����O�摜�̊e��f�̃��x���ԍ��𒊏o
		(*itd) = colors[label];	//���x�����O�摜�̊e��f�̐F�����蓖��
	}

	dst.convertTo(NewQuantImg, CV_64FC3);
	QuantImg[NowPyrNum] = FTMat3D(NewQuantImg);
}
void BlindDeconvolution::UpdateQuantizedImage_wighted(int NowPyrNum) {
	/* Update x~ */
	cout << " Update QuantImg... " << endl;			// ���s�m�F�p
	/* Optimizing using k-means */
	Mat NewQuantImg;
	QuantImg[NowPyrNum].output(NewQuantImg);
	NewQuantImg.convertTo(NewQuantImg, CV_32FC3);

	/* k-means �N���X�^�����O */
	Mat feature;
	NewQuantImg.copyTo(feature);
	//�摜�̉�f��1��i3�`���l���j�ɕ��ׂ�
	feature = feature.reshape(3, feature.rows * feature.cols);

	Mat_<int> labels(feature.size(), CV_32SC1);
	Mat centers;
	//kmeans�@�ɂ��摜�̕��ށi�̈敪���j
	const int MAX_CLUSTERS = 15/*10*/;	// �ő�F�� n
	kmeans(feature, MAX_CLUSTERS, labels, TermCriteria(TermCriteria::COUNT, 10, 1.0), 1, KMEANS_RANDOM_CENTERS, centers);

	// ���x�����O���ʂ̕`��F������
	vector<Vec3b> colors(MAX_CLUSTERS + 1);
	Vec3d center_colors;
	for (int label = 0; label <= MAX_CLUSTERS; ++label)
	{
		center_colors = Vec3d(0.0, 0.0, 0.0);
		//�N���X�^�̒��S���W
		for (c = 0; c < 3; c++) {
			center_colors[c] = (double)centers.at<float>(label, c);
		}

		//�N���X�^���ɐF��`��
		for (c = 0; c < 3; c++) {
			colors[label][c] = (uchar)center_colors[c];
		}
		//cout << "  ave_colors = " << (Vec3d)ave_colors << " => " << (Vec3b)colors[label] << endl;	// �m�F�p

		////���x���ԍ��ɑ΂��ĐF�������_���Ɋ��蓖�Ă�
		//colors[label] = Vec3b((rand() & 255), (rand() & 255), (rand() & 255));
	}

	//���x�����O���ʉ摜�̍쐬
	Mat dst(NewQuantImg.size(), CV_8UC3);
	//���ʉ摜�̊e��f�̐F�����肵�Ă���
	MatIterator_<Vec3b> itd = dst.begin<Vec3b>(), itd_end = dst.end<Vec3b>();
	for (int i = 0; itd != itd_end; ++itd, ++i) {
		int label = labels(i);	//���x�����O�摜�̊e��f�̃��x���ԍ��𒊏o
		(*itd) = colors[label];	//���x�����O�摜�̊e��f�̐F�����蓖��
	}


	/* �G�b�W���ŏd�݂Â�����MRF�œK�� */
	// ����摜�̃G�b�W��񂩂�d�݂�����
	Mat color_Img, gray_Img, contrust, Deconv;
	DeconvImg[NowPyrNum].output(color_Img);
	cvtColor(color_Img, gray_Img, COLOR_BGR2GRAY);
	Laplacian(gray_Img, contrust, CV_64F, 3);
	//contrust.convertTo(Image_dst_deblurred2, CV_8U);		// �m�F�p
	//normalize(Image_dst_deblurred2, Image_dst_deblurred2, 0, 150, NORM_MINMAX);	// �m�F�p
	convertScaleAbs(contrust, contrust, 1, 0);
	contrust.convertTo(contrust, CV_64FC1);
	color_Img.convertTo(Deconv, CV_64FC3);

	// �ʎq���摜�̍œK��
	Vec3d kmeans_color, img_color, quant_color;
	Vec3d min_color;
	double min_Energy, now_Energy;
	double diff, diff_pow, wight, egde;
	Mat dst_tmp = Mat::zeros(dst.size(), CV_64FC3);
	dst.convertTo(dst_tmp, CV_64FC3);
	/* �m�F�p */
	Mat NowTrueImg;
	TrueImg[NowPyrNum].output(NowTrueImg);
	cout << "�ʎq���摜 �� �^�摜" << endl;
	Evaluation_MSE_PSNR_SSIM(dst_tmp, NewQuantImg);

	for (int MRF_itrate = 0; MRF_itrate < MAX_Iteration; MRF_itrate++) {
#pragma omp parallel for private(x)
		for (y = 0; y < dst_tmp.rows; y++) {
			for (x = 0; x < dst_tmp.cols; x++) {
				kmeans_color = dst_tmp.at<Vec3d>(y, x);
				img_color = Deconv.at<Vec3d>(y, x);
				min_color = kmeans_color;

				min_Energy = DBL_MAX; // �ő�l����
				for (int label = 0; label <= MAX_CLUSTERS; ++label) {
					quant_color = (Vec3d)colors[label];
					// ||x~-x||^2
					diff = (double)abs(quant_color[0] - img_color[0]) + (double)abs(quant_color[1] - img_color[1]) + (double)abs(quant_color[2] - img_color[2]);
					diff /= 3.0;
					diff_pow = (double)pow(diff, 2);
					// Sigma w_pq * delta
					egde = 0.0;
					if (x > 0) {
						wight = (double)contrust.at<double>(y, x - 1);
						if (wight != 0) { wight = 1.0 - (1.0 / (double)wight); }
						else { wight = 0.0; }
						if (dst_tmp.at<double>(y, x) != dst_tmp.at<double>(y, x - 1)) { egde += wight; }
					}
					if (x < dst_tmp.cols - 1) {
						wight = (double)contrust.at<double>(y, x + 1);
						if (wight != 0) { wight = 1.0 - (1.0 / (double)wight); }
						else { wight = 0.0; }
						if (dst_tmp.at<double>(y, x) != dst_tmp.at<double>(y, x + 1)) { egde += wight; }
					}if (y > 0) {
						wight = (double)contrust.at<double>(y - 1, x);
						if (wight != 0) { wight = 1.0 - (1.0 / (double)wight); }
						else { wight = 0.0; }
						if (dst_tmp.at<double>(y, x) != dst_tmp.at<double>(y - 1, x)) { egde += wight; }
					}
					if (y < dst_tmp.rows - 1) {
						wight = (double)contrust.at<double>(y + 1, x);
						if (wight != 0) { wight = 1.0 - (1.0 / (double)wight); }
						else { wight = 0.0; }
						if (dst_tmp.at<double>(y, x) != dst_tmp.at<double>(y + 1, x)) { egde += wight; }
					}

					now_Energy = Myu * diff_pow + egde;
					if (now_Energy < min_Energy) {
						min_Energy = now_Energy;
						min_color = quant_color;
					}
				}

				dst_tmp.at<Vec3d>(y, x) = (Vec3d)min_color;
				//cout << "  kmeans_color = " << kmeans_color << " , dst_tmp = " << min_color << " = " << dst_tmp.at<Vec3d>(y, x) << endl;
			}
		}
	}

	//dst.convertTo(NewQuantImg, CV_64FC3);
	dst_tmp.convertTo(NewQuantImg, CV_64FC3);
	QuantImg[NowPyrNum] = FTMat3D(NewQuantImg);

	/* �m�F�p */
	cout << "MRF-based�ʎq���摜 �� �^�摜" << endl;
	Evaluation_MSE_PSNR_SSIM(NowTrueImg, NewQuantImg);
}
void BlindDeconvolution::UpdateDeconvImage(int NowPyrNum) {
	/* Update x */
	cout << " Update Img... " << endl;				// ���s�m�F�p
	// DFT�ϊ��̃T�C�Y���v�Z
	int Mplus = BlurrImg[NowPyrNum].FT_Mat[0].ImgMat.rows + Kernel[NowPyrNum].ImgMat.rows;
	int Nplus = BlurrImg[NowPyrNum].FT_Mat[0].ImgMat.cols + Kernel[NowPyrNum].ImgMat.cols;
	int Msize = getOptimalDFTSize(Mplus);
	int Nsize = getOptimalDFTSize(Nplus);
	//cout << " FFT Size  : (" << Nplus << "," << Mplus << ") => (" << Nsize << "," << Msize << ")" << endl;	// �m�F

	/* �O���� */
	// ���z�t�B���^��FTMat�N���X�ɕϊ�
	Mat grad_h = (Mat_<double>(3, 3)	// 3*3
		<< -1, 0, 1,
		-2, 0, 2,
		-1, 0, 1);
	Mat grad_v = (Mat_<double>(3, 3)	// 3*3
		<< -1, -2, -1,
		0, 0, 0,
		1, 2, 1);
	FTMat Grad_H = FTMat(grad_h, 0);
	FTMat Grad_V = FTMat(grad_v, 0);

	// �C���f�b�N�X���w�肵��1�����x�N�g���ɕϊ�
	BlurrImg[NowPyrNum].toVector(1, 0, 1, Nsize, Msize);
	QuantImg[NowPyrNum].toVector(1, 0, 1, Nsize, Msize);
	DeconvImg[NowPyrNum].toVector(1, 0, 1, Nsize, Msize);
	Kernel[NowPyrNum].toVector(1, 1, 0, Nsize, Msize);
	Grad_H.toVector(1, 1, 0, Nsize, Msize);
	Grad_V.toVector(1, 1, 0, Nsize, Msize);
	// DFT�ϊ�
	BlurrImg[NowPyrNum].DFT();
	QuantImg[NowPyrNum].DFT();
	DeconvImg[NowPyrNum].DFT();
	Kernel[NowPyrNum].DFT();
	Grad_H.DFT();
	Grad_V.DFT();

	/* �ڂ������摜(�^�̌��摜)�����߂� */
	Mat dft_NewImg_Ax[3], dft_NewImg_b[3];				// CG�@Ax=b�ł�A��DEF,Ax��b�����߂�
	for (c = 0; c < 3; c++) {
		mulSpectrums(BlurrImg[NowPyrNum].FT_Mat[c].dft_ImgVec, Kernel[NowPyrNum].dft_ImgVec, dft_NewImg_b[c], 0, true);	// ���f����
	}
	Mat abs_pow_Kernel, abs_pow_Grad_H, abs_pow_Grad_V;
	abs_pow_complex_Mat(Kernel[NowPyrNum].dft_ImgVec, abs_pow_Kernel);	// 2�����x�N�g���̑傫���̂Q��
	abs_pow_complex_Mat(Grad_H.dft_ImgVec, abs_pow_Grad_H);
	abs_pow_complex_Mat(Grad_V.dft_ImgVec, abs_pow_Grad_V);
	// FTMat�Ɉ�x�ϊ����ăp�����[�^�𑫂�(A���v�Z)
	FTMat NewImg_K, NewImg_H, NewImg_V;
	NewImg_K = FTMat(abs_pow_Kernel, 2);
	NewImg_H = FTMat(abs_pow_Grad_H, 2);
	NewImg_V = FTMat(abs_pow_Grad_V, 2);
	NewImg_K.settingB(1, 1, 0, Nsize, Msize);
	NewImg_H.settingB(1, 1, 0, Nsize, Msize);
	NewImg_V.settingB(1, 1, 0, Nsize, Msize);
	// inverseDFT�ϊ�
	NewImg_K.iDFT();
	NewImg_H.iDFT();
	NewImg_V.iDFT();
	Mat NewImg_A_tmp = Mat::zeros(1, Msize * Nsize, CV_64F);
	double denom = 0, num_K = 0, num_H_V = 0;
#pragma omp parallel
	for (x = 0; x < Nsize * Msize; x++) {
		num_K = NewImg_K.ImgVec.at<double>(0, x);
		num_H_V = NewImg_H.ImgVec.at<double>(0, x) + NewImg_V.ImgVec.at<double>(0, x);
		denom = num_K + Rambda * num_H_V + Myu;
		NewImg_A_tmp.at<double>(0, x) = denom;
	}
	FTMat NewImg_A = FTMat(NewImg_A_tmp, 1);
	NewImg_A.settingA(1, 1, BlurrImg[NowPyrNum].FT_Mat[0].ImgMat.cols, BlurrImg[NowPyrNum].FT_Mat[0].ImgMat.rows);
	NewImg_A.settingB(1, 1, 0, Nsize, Msize);
	// DFT�ϊ�
	NewImg_A.DFT();		// 1:A��DEF
	for (c = 0; c < 3; c++) {
		mulSpectrums(DeconvImg[NowPyrNum].FT_Mat[c].dft_ImgVec, NewImg_A.dft_ImgVec, dft_NewImg_Ax[c], 0, false);
	}
	FTMat3D NewImg_Ax = FTMat3D(dft_NewImg_Ax[0], dft_NewImg_Ax[1], dft_NewImg_Ax[2]);
	NewImg_Ax.settingB(1, 0, 1, Nsize, Msize);
	NewImg_Ax.settingAverageColor(BlurrImg[NowPyrNum]);
	FTMat3D NewImg_b = FTMat3D(dft_NewImg_b[0], dft_NewImg_b[1], dft_NewImg_b[2]);
	NewImg_b.settingB(1, 0, 1, Nsize, Msize);
	NewImg_b.settingAverageColor(BlurrImg[NowPyrNum]);
	// inverseDFT�ϊ�
	NewImg_Ax.iDFT();
	NewImg_b.iDFT();
	double b_tmp = 0;
#pragma omp parallel for private(x)
	for (c = 0; c < 3; c++) {
		for (x = 0; x < Nsize * Msize; x++) {
			b_tmp = NewImg_b.FT_Mat[c].ImgVec.at<double>(0, x);
			b_tmp += Myu * QuantImg[NowPyrNum].FT_Mat[c].ImgVec.at<double>(0, x);
			NewImg_b.FT_Mat[c].ImgVec.at<double>(0, x) = b_tmp;
		}
	}
	// 2�����x�N�g���ɕϊ�
	NewImg_Ax.toMatrix(1, 0, BlurrImg[NowPyrNum].FT_Mat[0].ImgMat.cols, BlurrImg[NowPyrNum].FT_Mat[0].ImgMat.rows);	// 2:Ax
	NewImg_b.toMatrix(1, 0, BlurrImg[NowPyrNum].FT_Mat[0].ImgMat.cols, BlurrImg[NowPyrNum].FT_Mat[0].ImgMat.rows);	// 3:b
	//checkMat_detail(NewImg_A.dft_ImgVec);			// �m�F�p
	//checkMat_detail(NewImg_Ax.FT_Mat[0].ImgMat);	// �m�F�p
	//checkMat_detail(NewImg_b.FT_Mat[0].ImgMat);		// �m�F�p

	/* CG method */
	CG_method_x(DeconvImg[NowPyrNum], NewImg_A, NewImg_Ax, NewImg_b, Nsize, Msize);

	// �������̉��
	Grad_H = FTMat();
	Grad_V = FTMat();
	NewImg_K = FTMat();
	NewImg_H = FTMat();
	NewImg_V = FTMat();
}
void BlindDeconvolution::UpdateDeconvImage_FFT(int NowPyrNum) {
	/* Update x */
	cout << " Update Img (FFT)... " << endl;				// ���s�m�F�p
	// DFT�ϊ��̃T�C�Y���v�Z
	int Mplus = BlurrImg[NowPyrNum].FT_Mat[0].ImgMat.rows + Kernel[NowPyrNum].ImgMat.rows;
	int Nplus = BlurrImg[NowPyrNum].FT_Mat[0].ImgMat.cols + Kernel[NowPyrNum].ImgMat.cols;
	int Msize = getOptimalDFTSize(Mplus);
	int Nsize = getOptimalDFTSize(Nplus);
	//cout << " FFT Size  : (" << Nplus << "," << Mplus << ") => (" << Nsize << "," << Msize << ")" << endl;	// �m�F

	/* �O���� */
	// ���z�t�B���^��FTMat�N���X�ɕϊ�
	Mat grad_h = (Mat_<double>(3, 3)	// 3*3
		<< -1, 0, 1,
		-2, 0, 2,
		-1, 0, 1);
	Mat grad_v = (Mat_<double>(3, 3)	// 3*3
		<< -1, -2, -1,
		0, 0, 0,
		1, 2, 1);
	FTMat Grad_H = FTMat(grad_h, 0);
	FTMat Grad_V = FTMat(grad_v, 0);

	// �C���f�b�N�X���w�肵��1�����x�N�g���ɕϊ�
	BlurrImg[NowPyrNum].toVector(1, 0, 1, Nsize, Msize);
	QuantImg[NowPyrNum].toVector(1, 0, 1, Nsize, Msize);
	DeconvImg[NowPyrNum].toVector(1, 0, 1, Nsize, Msize);
	Kernel[NowPyrNum].toVector(1, 1, 0, Nsize, Msize);
	Grad_H.toVector(1, 1, 0, Nsize, Msize);
	Grad_V.toVector(1, 1, 0, Nsize, Msize);
	// DFT�ϊ�
	BlurrImg[NowPyrNum].DFT();
	QuantImg[NowPyrNum].DFT();
	DeconvImg[NowPyrNum].DFT();
	Kernel[NowPyrNum].DFT();
	Grad_H.DFT();
	Grad_V.DFT();

	/* �ڂ������摜(�^�̌��摜)�����߂� */
	Mat dft_NewImg_x[3], dft_NewImg_b[3];				// Ax=b�ł�A��DEF��b��DEF�����߂�
	for (c = 0; c < 3; c++) {
		mulSpectrums(BlurrImg[NowPyrNum].FT_Mat[c].dft_ImgVec, Kernel[NowPyrNum].dft_ImgVec, dft_NewImg_b[c], 0, true);	// ���f����
	}
	FTMat3D NewImg_b = FTMat3D(dft_NewImg_b[0], dft_NewImg_b[1], dft_NewImg_b[2]);
	NewImg_b.settingB(1, 0, 1, Nsize, Msize);
	NewImg_b.settingAverageColor(BlurrImg[NowPyrNum]);
	// inverseDFT�ϊ�
	NewImg_b.iDFT();
	double b_tmp = 0;
#pragma omp parallel for private(x)
	for (c = 0; c < 3; c++) {
		for (x = 0; x < Nsize * Msize; x++) {
			b_tmp = NewImg_b.FT_Mat[c].ImgVec.at<double>(0, x);
			b_tmp += Myu * QuantImg[NowPyrNum].FT_Mat[c].ImgVec.at<double>(0, x);
			NewImg_b.FT_Mat[c].ImgVec.at<double>(0, x) = b_tmp;
		}
	}
	// DFT�ϊ�
	NewImg_b.DFT();	// b��DEF

	Mat abs_pow_Kernel, abs_pow_Grad_H, abs_pow_Grad_V;
	abs_pow_complex_Mat(Kernel[NowPyrNum].dft_ImgVec, abs_pow_Kernel);	// 2�����x�N�g���̑傫���̂Q��
	abs_pow_complex_Mat(Grad_H.dft_ImgVec, abs_pow_Grad_H);
	abs_pow_complex_Mat(Grad_V.dft_ImgVec, abs_pow_Grad_V);
	// FTMat�Ɉ�x�ϊ����ăp�����[�^�𑫂�(A���v�Z)
	FTMat NewImg_K, NewImg_H, NewImg_V;
	NewImg_K = FTMat(abs_pow_Kernel, 2);
	NewImg_H = FTMat(abs_pow_Grad_H, 2);
	NewImg_V = FTMat(abs_pow_Grad_V, 2);
	NewImg_K.settingB(1, 1, 0, Nsize, Msize);
	NewImg_H.settingB(1, 1, 0, Nsize, Msize);
	NewImg_V.settingB(1, 1, 0, Nsize, Msize);
	// inverseDFT�ϊ�
	NewImg_K.iDFT();
	NewImg_H.iDFT();
	NewImg_V.iDFT();
	Mat NewImg_A_tmp = Mat::zeros(1, Msize * Nsize, CV_64F);
	double denom = 0, num_K = 0, num_H_V = 0;
#pragma omp parallel
	for (x = 0; x < Nsize * Msize; x++) {
		num_K = NewImg_K.ImgVec.at<double>(0, x);
		num_H_V = NewImg_H.ImgVec.at<double>(0, x) + NewImg_V.ImgVec.at<double>(0, x);
		denom = num_K + Rambda * num_H_V + Myu;
		NewImg_A_tmp.at<double>(0, x) = denom;
	}
	FTMat NewImg_A = FTMat(NewImg_A_tmp, 1);
	NewImg_A.settingA(1, 1, BlurrImg[NowPyrNum].FT_Mat[0].ImgMat.cols, BlurrImg[NowPyrNum].FT_Mat[0].ImgMat.rows);
	NewImg_A.settingB(1, 1, 0, Nsize, Msize);
	// DFT�ϊ�
	NewImg_A.DFT();		// A��DEF

	Mat denom_NewImg_A;
	reciprocal_complex_Mat(NewImg_A.dft_ImgVec, denom_NewImg_A);			// 2�����x�N�g���̋t��
	for (c = 0; c < 3; c++) {
		mulSpectrums(NewImg_b.FT_Mat[c].dft_ImgVec, denom_NewImg_A, dft_NewImg_x[c], 0, false);
	}
	DeconvImg[NowPyrNum] = FTMat3D(dft_NewImg_x[0], dft_NewImg_x[1], dft_NewImg_x[2]);
	DeconvImg[NowPyrNum].settingB(1, 0, 1, Nsize, Msize);
	DeconvImg[NowPyrNum].settingAverageColor(BlurrImg[NowPyrNum]);
	// inverseDFT�ϊ�
	DeconvImg[NowPyrNum].iDFT();
	// 2�����x�N�g���ɕϊ�
	DeconvImg[NowPyrNum].toMatrix(1, 0, BlurrImg[NowPyrNum].FT_Mat[0].ImgMat.cols, BlurrImg[NowPyrNum].FT_Mat[0].ImgMat.rows);

	// �������̉��
	Grad_H = FTMat();
	Grad_V = FTMat();
	NewImg_K = FTMat();
	NewImg_H = FTMat();
	NewImg_V = FTMat();
}
void BlindDeconvolution::UpdateKarnel(int NowPyrNum) {
	/* Update k */
	cout << " Update Karnel... " << endl;			// ���s�m�F�p
	// DFT�ϊ��̃T�C�Y���v�Z
	int Mplus = BlurrImg[NowPyrNum].FT_Mat[0].ImgMat.rows + Kernel[NowPyrNum].ImgMat.rows;
	int Nplus = BlurrImg[NowPyrNum].FT_Mat[0].ImgMat.cols + Kernel[NowPyrNum].ImgMat.cols;
	int Msize = getOptimalDFTSize(Mplus);
	int Nsize = getOptimalDFTSize(Nplus);
	//cout << " FFT Size  : (" << Nplus << "," << Mplus << ") => (" << Nsize << "," << Msize << ")" << endl;	// �m�F

	/* �O���� */
	// ���z�t�B���^��FTMat�N���X�ɕϊ�
	Mat Trans_Vector = Mat::zeros(Kernel[NowPyrNum].ImgMat.size(), CV_64F);
	FTMat TransVector = FTMat(Trans_Vector, 0);
	KERNEL Kernel_sub = KERNEL();
	Kernel[NowPyrNum].copyTo(Kernel_sub);
	//checkMat_detail(Kernel[NowPyrNum].ImgMat);	// �m�F�p
	//checkMat_detail(Kernel_sub.ImgMat);	// �m�F�p

	/* �J�[�l�������߂� */
	double PenaltyParameter = 1.0e+03;
	double threshold = (double)(Tau / PenaltyParameter);
	cout << "  threshold = " << threshold << " , PenaltyParameter = " << PenaltyParameter << endl;
	double incr_Parameter = 2.0, decr_Parameter = 2.0, blanc_Parameter = 10.0;
	for (int k_index = 0; k_index < MAX_Iteration_ADMM; k_index++) {
		/* Calculate Kernel_sub by CG method */
		CG_method_k(Kernel_sub, Kernel[NowPyrNum], TransVector, BlurrImg[NowPyrNum], QuantImg[NowPyrNum], PenaltyParameter, Nsize, Msize);

		/* Calculate Kernel */
		double sign_calc = 0.0;
#pragma omp parallel for private(x)
		for (y = 0; y < Kernel_sub.ImgMat.rows; y++) {
			for (x = 0; x < Kernel_sub.ImgMat.cols; x++) {
				sign_calc = (double)abs((double)Kernel_sub.ImgMat.at<double>(y, x) - (double)Kernel[NowPyrNum].ImgMat.at<double>(y, x));
				//sign_calc = (double)Kernel_sub.ImgMat.at<double>(y, x);
				//cout << "   " << Kernel[NowPyrNum].ImgMat.at<double>(y, x) << " => " << (double)Kernel_sub.ImgMat.at<double>(y, x) << endl;	// �m�F�p
				//cout << "   " << sign_calc << endl;	// �m�F�p
				if (sign_calc > threshold) {
					Kernel[NowPyrNum].ImgMat.at<double>(y, x) = (double)sign_calc - (double)threshold;
				}
				else if (sign_calc < -threshold) {
					Kernel[NowPyrNum].ImgMat.at<double>(y, x) = -((double)sign_calc + (double)threshold);
				}
				else { Kernel[NowPyrNum].ImgMat.at<double>(y, x) = 0.0; }
				/*if (sign_calc >= threshold) {
					Kernel[NowPyrNum].ImgMat.at<double>(y, x) = (double)sign_calc - (double)threshold;
				}
				else { Kernel[NowPyrNum].ImgMat.at<double>(y, x) = 0.0; }*/
				//cout << "   -> " << Kernel[NowPyrNum].ImgMat.at<double>(y, x) << endl;	// �m�F�p
			}
		}
		Kernel[NowPyrNum].normalization();

		/* Calculate TransVector */
		TransVector.ImgMat = TransVector.ImgMat - Kernel_sub.ImgMat + Kernel[NowPyrNum].ImgMat;

		/* ��PenaltyParameter */
		/*double main_diff = (double)norm(Kernel[NowPyrNum].ImgMat);
		double sub_diff = (double)norm(TransVector.ImgMat) * PenaltyParameter;
		if (main_diff > sub_diff * (double)blanc_Parameter) { PenaltyParameter *= incr_Parameter; }
		else if (main_diff < sub_diff * (double)blanc_Parameter) { PenaltyParameter /= decr_Parameter; }*/
	}

}
void BlindDeconvolution::Upsampling(int NextPyramidNumber) {
	/* Upsample */
	int before_pyrLEVEL = NextPyramidNumber - 1;
	int after_pyrLEVEL = NextPyramidNumber;

	if (before_pyrLEVEL >= 0 && after_pyrLEVEL < PYRAMID_NUM) {
		cout << " Upsample..." << endl;				// ���s�m�F�p
		double ReResizeFactor = ResizeFactor[after_pyrLEVEL] / ResizeFactor[before_pyrLEVEL];
		DeconvImg[after_pyrLEVEL].resizeTo(DeconvImg[before_pyrLEVEL], ReResizeFactor);
		QuantImg[after_pyrLEVEL].resizeTo(QuantImg[before_pyrLEVEL], ReResizeFactor);
		Kernel[after_pyrLEVEL].resizeTo(Kernel[before_pyrLEVEL], ReResizeFactor);
	}
	else {
		cout << "Finished Blind Deconvolution." << endl;				// ���s�m�F�p
	}
	cout << endl;
}
void BlindDeconvolution::Evaluatuion(int ComperePyramidNumber) {
	if (ComperePyramidNumber >= 0 && ComperePyramidNumber < PYRAMID_NUM) {
		Mat NowTrueImg, NowBlurrImg, NowQuantImg, NowDeconvImg, NowKernel, NowTrueKernel;
		TrueImg[ComperePyramidNumber].output(NowTrueImg);
		BlurrImg[ComperePyramidNumber].output(NowBlurrImg);
		QuantImg[ComperePyramidNumber].output(NowQuantImg);
		DeconvImg[ComperePyramidNumber].output(NowDeconvImg);
		TrueKernel[ComperePyramidNumber].visualization(NowTrueKernel);
		Kernel[ComperePyramidNumber].visualization(NowKernel);

		cout << "�ڂ��摜 �� �^�摜" << endl;
		Evaluation_MSE_PSNR_SSIM(NowTrueImg, NowBlurrImg);
		cout << "�ʎq���摜 �� �^�摜" << endl;
		Evaluation_MSE_PSNR_SSIM(NowTrueImg, NowQuantImg);
		cout << "�ڂ������摜 �� �^�摜" << endl;
		Evaluation_MSE_PSNR_SSIM(NowTrueImg, NowDeconvImg);
		cout << "����J�[�l�� �� �^�J�[�l��" << endl;
		Evaluation_MSE_PSNR_SSIM(NowTrueKernel, NowKernel);
	}
	else { cout << "ERROR! BlindDeconvolution::Evaluatuion(() : ComperePyramidNumber is wrong." << endl; }
}


#endif