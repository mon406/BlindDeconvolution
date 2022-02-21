#ifndef __INCLUDED_H_Conjugate_Gradient__
#define __INCLUDED_H_Conjugate_Gradient__

#include "main.h"
#include "Fourier_Transform_Mat.h"
#include "Calculate_Complex_Number.h"

/* �萔 */
const int MAX_Iteration_CG = 10;			// �ő唽����
const int MAX_Iteration_CG_2 = 10;
const double ERROR_END_NUM_CG = 1.0e-08;	// ��������l

/* �֐��錾 (Conjugate Gradient Method) */
void CG_method_x(FTMat3D& OutputImg, FTMat& dft_NewImg_A, FTMat3D& NewImg_Ax, FTMat3D& NewImg_b, int FFT_Xsize, int FFT_Ysize);
void CG_method_k(KERNEL& OutputKernel, KERNEL& InputKernel, FTMat& TransVector, FTMat3D& BlurrImg, FTMat3D& QuantImg, double PenaltyParameter, int FFT_Xsize, int FFT_Ysize);

void multi_Matrix_Vector_3D(int fft_x, int fft_y, FTMat3D& InputImg, FTMat& dft_InputTransVector, FTMat3D& OutputImg);		// 2�����x�N�g����1�����x�N�g���̏�Z
void multi_Matrix_Vector(int fft_x, int fft_y, FTMat& InputImg, FTMat& dft_InputTransVector, FTMat& OutputImg);
double multi_vector(Mat& srcImg, Mat& srcImg2);							// 1�����x�N�g���̏�Z(����)

/* �֐� */
/*--- CG_method_x �N���X ----------------------------------------------
	CG�@��Ax=b�������ċt��ݍ��݉摜�����߂�N���X
	OutputImg:		�t��ݍ��݉摜 x�i���o�͗p�j
	dft_NewImg_A:	Ax=b��A�̃t�[���G�X�y�N�g���i���͗p�j
	NewImg_Ax:		Ax=b��Ax�i���͗p�j
	NewImg_b:		Ax=b��b�i���͗p�j
	FFT_Xsize:		�t�[���G�ϊ��� x�����T�C�Y�i���͗p�j
	FFT_Ysize:		�t�[���G�ϊ��� y�����T�C�Y�i���͗p�j
-----------------------------------------------------------------------*/
void CG_method_x(FTMat3D& OutputImg, FTMat& dft_NewImg_A, FTMat3D& NewImg_Ax, FTMat3D& NewImg_b, int FFT_Xsize, int FFT_Ysize) {
	int c;
	int Msize = FFT_Ysize;
	int Nsize = FFT_Xsize;
	int Row_size = OutputImg.FT_Mat[0].ImgMat.rows;
	int Col_size = OutputImg.FT_Mat[0].ImgMat.cols;

	/* �����l�ݒ� */
	FTMat3D NextX, LastX;
	NextX = FTMat3D(OutputImg);
	LastX = FTMat3D(OutputImg);
	FTMat3D Residual, P_base;
	Mat Residual_tmp[3], Residual_tmp3D;
	for (c = 0; c < 3; c++) {
		Residual_tmp[c] = NewImg_b.FT_Mat[c].ImgMat - NewImg_Ax.FT_Mat[c].ImgMat;
	}
	merge(Residual_tmp, 3, Residual_tmp3D);
	Residual = FTMat3D(Residual_tmp3D);
	P_base = FTMat3D(Residual_tmp3D);

	FTMat3D AP = FTMat3D();
	/* �����@�Ōv�Z */
	double ALPHA = 0.0, BETA = 0.0;
	double energy;
	for (int i_number = 0; i_number < MAX_Iteration_CG; i_number++) {
		// Calculate AP
		multi_Matrix_Vector_3D(Nsize, Msize, Residual, dft_NewImg_A, AP);

		// Calculate ALPHA
		ALPHA = 0.0;
		double Numerator, Denominator;
		for (c = 0; c < 3; c++) {
			//Numerator = multi_vector(P_base.FT_Mat[c].ImgMat, Residual.FT_Mat[c].ImgMat);		// �x�N�g���̓���
			Numerator = multi_vector(Residual.FT_Mat[c].ImgMat, Residual.FT_Mat[c].ImgMat);
			Denominator = multi_vector(P_base.FT_Mat[c].ImgMat, AP.FT_Mat[c].ImgMat);
			ALPHA += (double)(Numerator / Denominator);
		}
		ALPHA /= 3.0;

		// Calculate x
		for (c = 0; c < 3; c++) {
			NextX.FT_Mat[c].ImgMat = LastX.FT_Mat[c].ImgMat + ALPHA * P_base.FT_Mat[c].ImgMat;
			/*NextX.FT_Mat[c].ImgMat += NextX.AverageColor[c];
			normalize(NextX[c], NextX[c], 0, 1, NORM_MINMAX);
			NextX.FT_Mat[c].ImgMat -= NextX.AverageColor[c];*/
		}

		// Calculate Residual
		Mat Residual_before[3];
		for (c = 0; c < 3; c++) {
			Residual.FT_Mat[c].ImgMat.copyTo(Residual_before[c]);
			Residual.FT_Mat[c].ImgMat = Residual.FT_Mat[c].ImgMat - ALPHA * AP.FT_Mat[c].ImgMat;
		}

		energy = 0.0;
		for (c = 0; c < 3; c++) {
			energy += (double)norm(Residual.FT_Mat[c].ImgMat);
		}
		energy /= (double)((double)Residual.FT_Mat[c].ImgMat.cols * (double)Residual.FT_Mat[c].ImgMat.rows * 3.0);
		if (energy < ERROR_END_NUM_CG || i_number == MAX_Iteration_CG - 1) {
			cout << "  " << (int)i_number << " : energy = " << (double)energy << endl;	// �m�F�p
			break;
		}

		// Calculate BETA
		BETA = 0.0;
		double Numerator2, Denominator2;
		for (c = 0; c < 3; c++) {
			Numerator2 = multi_vector(Residual.FT_Mat[c].ImgMat, Residual.FT_Mat[c].ImgMat);
			Denominator2 = multi_vector(Residual_before[c], Residual_before[c]);
			BETA += (double)(Numerator2 / Denominator2);
		}
		BETA /= 3.0;

		// Calculate P_base
		for (c = 0; c < 3; c++) {
			P_base.FT_Mat[c].ImgMat = Residual.FT_Mat[c].ImgMat + BETA * P_base.FT_Mat[c].ImgMat;
		}

		NextX.copyTo(LastX);
	}
	NextX.copyTo(OutputImg);

	// �������̉��
	NextX = FTMat3D();
	LastX = FTMat3D();
	Residual = FTMat3D();
	P_base = FTMat3D();
	AP = FTMat3D();
}

/*--- CG_method_k �N���X ----------------------------------------------
	CG�@��Ax=b�������ăJ�[�l���𐄒肷��N���X
	OutputKernel:		�J�[�l�� k'�i���o�͗p�j
	InputKernel:		�J�[�l�� k�i���͗p�j
	TransVector:		ADMM�@�ł̃x�N�g���i���͗p�j
	BlurrImg:			�ڂ��摜�i���͗p�j
	QuantImg:			�ʎq���摜�i���͗p�j
	PenaltyParameter:	ADMM�@�ł̃y�i���e�B�E�p�����[�^
	FFT_Xsize:			�t�[���G�ϊ��� x�����T�C�Y�i���͗p�j
	FFT_Ysize:			�t�[���G�ϊ��� y�����T�C�Y�i���͗p�j
-----------------------------------------------------------------------*/
void CG_method_k(KERNEL& OutputKernel, KERNEL& InputKernel, FTMat& TransVector, FTMat3D& BlurrImg, FTMat3D& QuantImg, double PenaltyParameter, int FFT_Xsize, int FFT_Ysize) {
	int x, c;
	int Msize = FFT_Ysize;
	int Nsize = FFT_Xsize;
	int Row_size = OutputKernel.rows;
	int Col_size = OutputKernel.cols;

	// �C���f�b�N�X���w�肵��1�����x�N�g���ɕϊ�
	BlurrImg.toVector(1, 1, 0, Nsize, Msize);
	QuantImg.toVector(1, 1, 0, Nsize, Msize);
	OutputKernel.toVector(1, 0, 1, Nsize, Msize);
	InputKernel.toVector(1, 0, 1, Nsize, Msize);
	TransVector.toVector(1, 0, 1, Nsize, Msize);
	// DFT�ϊ�
	BlurrImg.DFT();
	QuantImg.DFT();
	OutputKernel.DFT();

	/* CG�@Ax=b�ł�A��DEF,Ax��b�����߂� */
	Mat dft_NewImg_A[3], dft_NewImg_Ax[3], dft_NewImg_b[3];				// CG�@Ax=b�ł�A��DEF,Ax��b�����߂�
	for (c = 0; c < 3; c++) {
		mulSpectrums(BlurrImg.FT_Mat[c].dft_ImgVec, QuantImg.FT_Mat[c].dft_ImgVec, dft_NewImg_b[c], 0, true);	// ���f����
		abs_pow_complex_Mat(OutputKernel.dft_ImgVec, dft_NewImg_A[c]);	// 2�����x�N�g���̑傫���̂Q��
	}

	FTMat3D NewImg_A;
	NewImg_A = FTMat3D(dft_NewImg_A[0], dft_NewImg_A[1], dft_NewImg_A[2]);
	NewImg_A.settingB(1, 1, 1, Nsize, Msize);
	NewImg_A.settingAverageColor(BlurrImg);
	FTMat3D NewImg_b;
	NewImg_b = FTMat3D(dft_NewImg_b[0], dft_NewImg_b[1], dft_NewImg_b[2]);
	NewImg_b.settingB(1, 0, 1, Nsize, Msize);
	NewImg_b.settingAverageColor(BlurrImg);
	// inverseDFT�ϊ�
	NewImg_A.iDFT();
	NewImg_b.iDFT();
	double denom = 0, b_tmp1 = 0, b_tmp2 = 0;
	double PenaltyParameter_half = (double)(PenaltyParameter / 2.0);
#pragma omp parallel for private(x)
	for (c = 0; c < 3; c++) {
		for (x = 0; x < Nsize * Msize; x++) {
			denom = NewImg_A.FT_Mat[c].ImgVec.at<double>(0, x);
			denom += PenaltyParameter_half;
			NewImg_A.FT_Mat[c].ImgVec.at<double>(0, x) = denom;

			b_tmp1 = NewImg_b.FT_Mat[c].ImgVec.at<double>(0, x);
			b_tmp2 = InputKernel.ImgVec.at<double>(0, x) + TransVector.ImgVec.at<double>(0, x);
			NewImg_b.FT_Mat[c].ImgVec.at<double>(0, x) = b_tmp1 + b_tmp2 * PenaltyParameter_half;
		}
	}
	NewImg_A.settingA(1, 1, OutputKernel.ImgMat.cols, OutputKernel.ImgMat.rows);
	NewImg_A.settingB(1, 1, 0, Nsize, Msize);
	// DFT�ϊ�
	NewImg_A.DFT();		// 1:A��DEF
	for (c = 0; c < 3; c++) {
		mulSpectrums(OutputKernel.dft_ImgVec, NewImg_A.FT_Mat[c].dft_ImgVec, dft_NewImg_Ax[c], 0, false);
	}
	FTMat3D NewImg_Ax = FTMat3D(dft_NewImg_Ax[0], dft_NewImg_Ax[1], dft_NewImg_Ax[2]);
	NewImg_Ax.settingB(1, 0, 1, Nsize, Msize);
	NewImg_Ax.settingAverageColor(BlurrImg);
	// inverseDFT�ϊ�
	NewImg_Ax.iDFT();
	// 2�����x�N�g���ɕϊ�
	NewImg_Ax.toMatrix(1, 0, OutputKernel.ImgMat.cols, OutputKernel.ImgMat.rows);	// 2:Ax
	NewImg_b.toMatrix(1, 0, OutputKernel.ImgMat.cols, OutputKernel.ImgMat.rows);	// 3:b

	// ���ς��Ƃ���1������
	Vec2d number_V2d, denom_V2d = { 3.0, 0.0 }, input_V2d;
	Mat NewImg_A_ave = Mat::zeros(1, Msize * Nsize, CV_64FC2);
	Mat NewImg_Ax_ave = Mat::zeros(OutputKernel.ImgMat.rows, OutputKernel.ImgMat.cols, CV_64F);
	Mat NewImg_b_ave = Mat::zeros(OutputKernel.ImgMat.rows, OutputKernel.ImgMat.cols, CV_64F);
#pragma omp parallel for private(x)
	for (c = 0; c < 3; c++) {
		for (x = 0; x < Nsize * Msize; x++) {
			number_V2d = NewImg_A.FT_Mat[c].dft_ImgVec.at<Vec2d>(0, x);
			divi_Vec2d(input_V2d, number_V2d, denom_V2d);
			NewImg_A_ave.at<Vec2d>(0, x) += input_V2d;
		}

		NewImg_Ax_ave = NewImg_Ax_ave + NewImg_Ax.FT_Mat[c].ImgMat;
		NewImg_b_ave = NewImg_b_ave + NewImg_b.FT_Mat[c].ImgMat;
	}
	NewImg_Ax_ave /= 3.0;
	NewImg_b_ave /= 3.0;
	FTMat NewImg_A_Ave, NewImg_Ax_Ave, NewImg_b_Ave;
	NewImg_A_Ave = FTMat(NewImg_A_ave, 2);
	NewImg_A_Ave.settingA(1, 0, OutputKernel.ImgMat.cols, OutputKernel.ImgMat.rows);
	NewImg_A_Ave.settingB(1, 0, 1, Nsize, Msize);
	NewImg_Ax_Ave = FTMat(NewImg_Ax_ave, 0);
	NewImg_b_Ave = FTMat(NewImg_b_ave, 0);
	//checkMat(NewImg_A_Ave.dft_ImgVec);	// �m�F�p
	//checkMat(NewImg_Ax_Ave.ImgMat);		// �m�F�p
	//checkMat(NewImg_b_Ave.ImgMat);		// �m�F�p


	/* �����l�ݒ� */
	KERNEL NextK, LastK;
	NextK = KERNEL(OutputKernel);
	LastK = KERNEL(OutputKernel);
	FTMat Residual, P_base;
	Mat Residual_tmp = NewImg_b_Ave.ImgMat - NewImg_Ax_Ave.ImgMat;
	normalize(Residual_tmp, Residual_tmp, 0, 1, NORM_MINMAX);
	Residual = FTMat(Residual_tmp, 0);
	P_base = FTMat(Residual_tmp, 0);
	//checkMat_detail(Residual_tmp);	// �m�F�p

	FTMat AP = FTMat();
	/* �����@�Ōv�Z */
	double ALPHA = 0.0, BETA = 0.0;
	double energy;
	for (int i_number = 0; i_number < MAX_Iteration_CG_2; i_number++) {
		// Calculate AP
		multi_Matrix_Vector(Nsize, Msize, Residual, NewImg_A_Ave, AP);

		// Calculate ALPHA
		ALPHA = 0.0;
		//double Numerator = multi_vector(P_base.ImgMat, Residual.ImgMat);		// �x�N�g���̓���
		double Numerator = multi_vector(Residual.ImgMat, Residual.ImgMat);
		double Denominator = multi_vector(P_base.ImgMat, AP.ImgMat);
		ALPHA = (double)(Numerator / Denominator);
		//cout << "  ALPHA = " << (double)ALPHA << endl;	// �m�F�p

		// Calculate x
		NextK.ImgMat = LastK.ImgMat + ALPHA * P_base.ImgMat;
		NextK.normalization();

		// Calculate Residual
		Mat Residual_before;
		Residual.ImgMat.copyTo(Residual_before);
		Residual.ImgMat = Residual.ImgMat - ALPHA * AP.ImgMat;
		normalize(Residual.ImgMat, Residual.ImgMat, 0, 1, NORM_MINMAX);

		energy = 0.0;
		energy += (double)norm(Residual.ImgMat);
		energy /= (double)((double)Residual.ImgMat.cols * (double)Residual.ImgMat.rows);
		//cout << "  " << (int)i_number << " : energy = " << (double)energy << endl;	// �m�F�p
		if (energy < ERROR_END_NUM_CG || i_number == MAX_Iteration_CG - 1) {
			cout << "  " << (int)i_number << " : energy = " << (double)energy << endl;	// �m�F�p
			break;
		}

		// Calculate BETA
		BETA = 0.0;
		double Numerator2 = multi_vector(Residual.ImgMat, Residual.ImgMat);
		double Denominator2 = multi_vector(Residual_before, Residual_before);
		BETA = (double)(Numerator2 / Denominator2);
		//cout << "  BETA = " << (double)BETA << endl;	// �m�F�p

		// Calculate P_base
		P_base.ImgMat = Residual.ImgMat + BETA * P_base.ImgMat;
		normalize(P_base.ImgMat, P_base.ImgMat, 0, 1, NORM_MINMAX);

		NextK.copyTo(LastK);
	}

	NextK.copyTo(OutputKernel);
	OutputKernel.normalization();
}


/*--- multi_Matrix_Vector_3D() ----------------------------------------
	2�����x�N�g����1�����x�N�g���̏�Z���v�Z����N���X
	fft_x:		�t�[���G�ϊ��� x�����T�C�Y�i���͗p�j
	fft_y:		�t�[���G�ϊ��� y�����T�C�Y�i���͗p�j
	InputImg:				1�����x�N�g���i���͗p�j
	dft_InputTransVector:	2�����x�N�g����DFT�i���͗p�j
	OutputImg:				1�����x�N�g���i�o�͗p�j
------------------------------------------------------------------------*/
void multi_Matrix_Vector_3D(int fft_x, int fft_y, FTMat3D& InputImg, FTMat& dft_InputTransVector, FTMat3D& OutputImg) {
	int c;
	int Msize = fft_y;
	int Nsize = fft_x;
	OutputImg = FTMat3D();

	// �C���f�b�N�X���w�肵��1�����x�N�g���ɕϊ�
	InputImg.toVector(1, 0, 1, Nsize, Msize);
	//dft_InputTransVector.toVector(1, 1, 0, Nsize, Msize);
	// DFT�ϊ�
	InputImg.DFT();
	//dft_InputTransVector.DFT();
	// 2�����x�N�g����1�����x�N�g���̏�Z�����߂�
	Mat dft_NewImage[3];
	for (c = 0; c < 3; c++) {
		mulSpectrums(InputImg.FT_Mat[c].dft_ImgVec, dft_InputTransVector.dft_ImgVec, dft_NewImage[c], 0, false);
	}

	OutputImg = FTMat3D(dft_NewImage[0], dft_NewImage[1], dft_NewImage[2]);
	OutputImg.settingB(1, 0, 1, Nsize, Msize);
	OutputImg.settingAverageColor(InputImg);
	// inverseDFT�ϊ�
	OutputImg.iDFT();
	// 2�����x�N�g���ɕϊ�
	OutputImg.toMatrix(1, 0, InputImg.FT_Mat[0].ImgMat.cols, InputImg.FT_Mat[0].ImgMat.rows);
}
/*--- multi_Matrix_Vector() -------------------------------------------
	2�����x�N�g����1�����x�N�g���̏�Z���v�Z����N���X
	fft_x:		�t�[���G�ϊ��� x�����T�C�Y�i���͗p�j
	fft_y:		�t�[���G�ϊ��� y�����T�C�Y�i���͗p�j
	InputImg:				1�����x�N�g���i���͗p�j
	dft_InputTransVector:	2�����x�N�g����DFT�i���͗p�j
	OutputImg:				1�����x�N�g���i�o�͗p�j
------------------------------------------------------------------------*/
void multi_Matrix_Vector(int fft_x, int fft_y, FTMat& InputImg, FTMat& dft_InputTransVector, FTMat& OutputImg) {
	int c;
	int Msize = fft_y;
	int Nsize = fft_x;
	OutputImg = FTMat();

	// �C���f�b�N�X���w�肵��1�����x�N�g���ɕϊ�
	InputImg.toVector(1, 0, 1, Nsize, Msize);
	//dft_InputTransVector.toVector(1, 1, 0, Nsize, Msize);
	// DFT�ϊ�
	InputImg.DFT();
	//dft_InputTransVector.DFT();
	// 2�����x�N�g����1�����x�N�g���̏�Z�����߂�
	Mat dft_NewImage;
	mulSpectrums(InputImg.dft_ImgVec, dft_InputTransVector.dft_ImgVec, dft_NewImage, 0, false);

	OutputImg = FTMat(dft_NewImage, 2);
	OutputImg.settingB(1, 0, 1, Nsize, Msize);
	// inverseDFT�ϊ�
	OutputImg.iDFT();
	// 2�����x�N�g���ɕϊ�
	OutputImg.toMatrix(1, 0, InputImg.ImgMat.cols, InputImg.ImgMat.rows);
}

/*--- multi_vector() --------------------------------------------------
	1�����x�N�g���̏�Z(����)���v�Z����N���X

	srcImg:		1�����x�N�g��A�i���͗p�j
	srcImg2:	1�����x�N�g��B�i���͗p�j
------------------------------------------------------------------------*/
double multi_vector(Mat& srcImg, Mat& srcImg2) {
	double Result = 0.0;
	int x, y;
	double in, in2, out;

	if (srcImg.cols != srcImg2.cols || srcImg.rows != srcImg2.rows) { cout << "ERROR! multi_vector() : Can't translate because of wrong sizes." << endl; }
	if (srcImg.channels() == 1) {
#pragma omp parallel for private(x)
		for (y = 0; y < srcImg.rows; y++) {
			for (x = 0; x < srcImg.cols; x++) {
				in = srcImg.at<double>(y, x);
				in2 = srcImg2.at<double>(y, x);
				out = in * in2;
				Result += out;
			}
		}
	}
	else { cout << "ERROR! multi_vector() : Can't translate because of wrong channels." << endl; }

	return Result;
}


#endif