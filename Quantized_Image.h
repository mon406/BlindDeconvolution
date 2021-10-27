#ifndef __INCLUDED_Quantized_Image__
#define __INCLUDED_Quantized_Image__

/* クラス */
class QuantMatDouble {
private:
	int Qx, Qy, Qc;
	int Qindex;
public:
	int cols;		// 画像の幅
	int rows;		// 画像の高さ
	int size;		// 画像の総ピクセル数
	int color_number;	// 総画素値数
	vector<double> QuantNumber;	// 全ての取り得る画素値
	Mat QMat;

	QuantMatDouble();
	QuantMatDouble(Mat&);
	QuantMatDouble(int, Mat&);
	QuantMatDouble(int, Mat&, int);
	//void setColorNumber(int);
	void quantedQMat();
	void searchUpDown(double&, double&, double&); // 対称の値の上下値を取得
};
QuantMatDouble::QuantMatDouble() {
	cols = WIDTH;
	rows = HEIGHT;
	size = cols * rows;
	color_number = 0;
	QuantNumber.clear();
	QMat = Mat(Size(WIDTH, HEIGHT), CV_64FC3);
}
QuantMatDouble::QuantMatDouble(Mat& inputIMG) {
	cols = inputIMG.cols;
	rows = inputIMG.rows;
	size = cols * rows;
	inputIMG.convertTo(QMat, CV_64FC3);

	double color_simpler;
#pragma omp parallel for private(Qx, Qc)
	for (Qy = 0; Qy < rows; Qy++) {
		for (Qx = 0; Qx < cols; Qx++) {
			for (Qc = 0; Qc < 3; Qc++) {
				Qindex = (Qy * cols + Qx) * 3 + Qc;
				color_simpler = (double)inputIMG.data[Qindex];
				/* 色がRGB全てで15以上になるように色を減らす */
				if (color_simpler < 32) { color_simpler = 0.0; }
				else if (color_simpler < 95) { color_simpler = 63.0; }
				else if (color_simpler < 159) { color_simpler = 127.0; }
				else if (color_simpler < 223) { color_simpler = 191.0; }
				else { color_simpler = (double)MAX_INTENSE; }

				QMat.data[Qindex] = (double)color_simpler;
			}
		}
	}
	color_number = 5;
	QuantNumber = { 0.0, 63.0, 127.0, 191.0, (double)MAX_INTENSE };

	/* 確認用 */
	cout << " QuantNumber(Default) : ";
	for (int i = 0; i < QuantNumber.size(); i++) {
		cout << " " << QuantNumber[i];
	}
	cout << endl;
}
QuantMatDouble::QuantMatDouble(int quantized, Mat& inputIMG) {
	cols = inputIMG.cols;
	rows = inputIMG.rows;
	size = cols * rows;
	inputIMG.convertTo(QMat, CV_64FC3);

	if (quantized <= 0) { cout << " ERROR! QuantMat : Can't make quantized image because of the small number." << endl; }
	else if (quantized > 255) { cout << " ERROR! QuantMat : Can't make quantized because of the big number." << endl; }
	else {
		double quantized2 = (double)quantized * 2.0;
		double doubleQuantized = (double)quantized2;
		QuantNumber.clear();
		QuantNumber.push_back(0.0);
		while (doubleQuantized <= MAX_INTENSE) {
			QuantNumber.push_back(doubleQuantized);
			doubleQuantized = doubleQuantized + (double)quantized2;
		}
		doubleQuantized = doubleQuantized - (double)quantized2;
		if (doubleQuantized != MAX_INTENSE) { QuantNumber.push_back((double)MAX_INTENSE); }
		color_number = (int)QuantNumber.size();
	}

	/* 確認用 */
	/*cout << " QuantNumber : ";
	for (int i = 0; i < QuantNumber.size(); i++) {
		cout << " " << QuantNumber[i];
	}
	cout << endl;*/
}
QuantMatDouble::QuantMatDouble(int quantized, Mat& inputIMG, int display_checker) {
	cols = inputIMG.cols;
	rows = inputIMG.rows;
	size = cols * rows;
	inputIMG.convertTo(QMat, CV_64FC3);

	if (quantized <= 0) { cout << " ERROR! QuantMat : Can't make quantized image because of the small number." << endl; }
	else if (quantized > 255) { cout << " ERROR! QuantMat : Can't make quantized because of the big number." << endl; }
	else {
		double quantized2 = (double)quantized * 2.0;
		double doubleQuantized = (double)quantized2;
		QuantNumber.clear();
		QuantNumber.push_back(0.0);
		while (doubleQuantized <= MAX_INTENSE) {
			QuantNumber.push_back(doubleQuantized);
			doubleQuantized = doubleQuantized + (double)quantized2;
		}
		doubleQuantized = doubleQuantized - (double)quantized2;
		if (doubleQuantized != MAX_INTENSE) { QuantNumber.push_back((double)MAX_INTENSE); }
		color_number = (int)QuantNumber.size();
	}

	/* 確認用 */
	if (display_checker == 0) {
		cout << " QuantNumber : ";
		for (int i = 0; i < QuantNumber.size(); i++) {
			cout << " " << QuantNumber[i];
		}
		cout << endl;
	}
}
//void QuantMatDouble::setColorNumber(int LeastColorNumber) {
//	color_number = 0;
//	QuantNumber.clear();
//
//	int LeastColor = (int)(255 / LeastColorNumber);
//	Qindex = 0;
//	double NOW_Color = 0.0;
//	while (NOW_Color < 255) {
//		QuantNumber.push_back(NOW_Color);
//		NOW_Color += (double)LeastColor;
//		Qindex++;
//	}
//	if(QuantNumber[Qindex - 1] != 255){ QuantNumber.push_back((double)255); }
//	color_number = QuantNumber.size();
//
//	/* 確認用 */
//	cout << " QuantNumber : ";
//	for (int i = 0; i < QuantNumber.size(); i++) {
//		cout << " " << QuantNumber[i];
//	}
//	cout << endl;
//}
void QuantMatDouble::quantedQMat() {
	double now_color, quanted_color;
	Vec3d now_color_3, quanted_color_3;
	int center = (int)(QuantNumber.size() / 2);
	double centerNUM;
	if (center > 0) { centerNUM = (double)QuantNumber[center - 1]; }
	else { centerNUM = (double)QuantNumber[0]; }

#pragma omp parallel for private(Qx, Qc)
	for (Qy = 0; Qy < rows; Qy++) {
		for (Qx = 0; Qx < cols; Qx++) {
			now_color_3 = QMat.at<Vec3d>(Qy, Qx);

			for (Qc = 0; Qc < 3; Qc++) {
				Qindex = (Qy * cols + Qx) * 3 + Qc;
				now_color = (double)now_color_3[Qc];
				/* 色の量子化 */
				quanted_color = 0;
				double diff1, diff2, beforeNUM = (double)QuantNumber[0];
				int ind;
				if (now_color < centerNUM) {
					for (ind = 0; ind <= center - 1; ind++) {
						if (now_color <= (double)QuantNumber[ind]) {
							quanted_color = (double)QuantNumber[ind];
							if (ind != 0) { beforeNUM = (double)QuantNumber[(int)(ind - 1)]; }
							break;
						}
					}
					if (now_color != quanted_color && ind != 0) {
						diff1 = abs(now_color - beforeNUM);
						diff2 = abs(quanted_color - now_color);
						if (diff1 < diff2) { quanted_color = beforeNUM; }
					}
				}
				else {
					for (ind = center - 1; ind < QuantNumber.size(); ind++) {
						if (now_color <= (double)QuantNumber[ind]) {
							quanted_color = (double)QuantNumber[ind];
							beforeNUM = (double)QuantNumber[(int)(ind - 1)];
							break;
						}
					}
					if (ind == QuantNumber.size()) {
						quanted_color = (double)QuantNumber[(int)(ind - 1)];
					}
					else if (now_color != quanted_color) {
						diff1 = abs(now_color - beforeNUM);
						diff2 = abs(quanted_color - now_color);
						if (diff1 <= diff2) { quanted_color = beforeNUM; }
					}
				}

				//cout << "  " << (double)now_color << "->" << (double)quanted_color << endl;	// 確認用
				quanted_color_3[Qc] = (double)quanted_color;
			}

			QMat.at<Vec3d>(Qy, Qx) = quanted_color_3;
			//cout << "  " << quanted_color_3 << endl;	// 確認用
		}
	}
}
void QuantMatDouble::searchUpDown(double& current, double& down, double& up) {
	down = current;
	up = current;

	int current_index = -1;
	for (int i = 0; i < QuantNumber.size(); i++) {
		if (current == QuantNumber[i]) { current_index = i; }
		else if (current > QuantNumber[i]) { current_index = i; /*cout << "WARNING!" << endl;*/ }
	}
	if (current_index < 0) { cout << "WARNING! QuantMatDouble::searchUpDown() : Input number is wrong. " << current << endl; }
	else {
		// down number
		if (current_index == 0) { down = 0.0; }
		else { down = QuantNumber[current_index - 1]; }
		// up number
		if (current_index == (QuantNumber.size() - 1)) { up = 255.0; }
		else { up = QuantNumber[current_index + 1]; }
	}
}


// すべての取りうるカラー値から平均値を計算
class SelectAverageRGB {
private:
	int index, index2;
public:
	int size;		// 総比較カラー数
	double most_color[3];	// 最頻カラー値
	vector<Vec3d> RGB;	// 全ての取り得る画素値

	SelectAverageRGB();
	void put(Vec3d&);
	void selectAverage(Vec3d&);
};
SelectAverageRGB::SelectAverageRGB() {
	size = 0;
	most_color[0] = 0.0;
	most_color[1] = 0.0;
	most_color[2] = 0.0;
	RGB.clear();
}
void SelectAverageRGB::put(Vec3d& input) {
	size++;
	RGB.push_back(input);
}
void SelectAverageRGB::selectAverage(Vec3d& output) {
	Vec3d now;
	vector<Vec3d> AllRGB;
	vector<int> counter;
#pragma omp parallel for private(index2)
	for (index = 0; index < size; index++) {
		now = RGB[index];
		for (index2 = 0; index2 < AllRGB.size(); index2++) {
			if (AllRGB[index2] == now) {
				counter[index2]++;
			}
			else {
				AllRGB.push_back(now);
				counter.push_back(1);
			}
		}
	}

	int most_color_counter = 0;
	vector<double> most_colorB, most_colorG, most_colorR;
	for (index2 = 0; index2 < AllRGB.size(); index2++) {
		if (counter[index2] > most_color_counter) {
			most_color_counter = counter[index2];
			most_colorB.clear();
			most_colorG.clear();
			most_colorR.clear();
			most_colorB.push_back(AllRGB[index2][0]);
			most_colorG.push_back(AllRGB[index2][1]);
			most_colorR.push_back(AllRGB[index2][2]);
		}
		else if(counter[index2] == most_color_counter) {
			most_colorB.push_back(AllRGB[index2][0]);
			most_colorG.push_back(AllRGB[index2][1]);
			most_colorR.push_back(AllRGB[index2][2]);
		}
	}
	if (most_color_counter == 1) {
		most_color[0] = most_colorB[0];
		most_color[1] = most_colorG[0];
		most_color[2] = most_colorR[0];
	}
	else {
		double ave_colorB = 0.0, ave_colorG = 0.0, ave_colorR = 0.0;
		for (int i = 0; i < most_colorB.size(); i++) {
			ave_colorB += most_colorB[i];
			ave_colorG += most_colorG[i];
			ave_colorR += most_colorR[i];
		}
		ave_colorB /= (double)most_colorB.size();
		ave_colorG /= (double)most_colorG.size();
		ave_colorR /= (double)most_colorR.size();
		most_color[0] = ave_colorB;
		most_color[1] = ave_colorG;
		most_color[2] = ave_colorR;
	}

	output = Vec3d( most_color[0], most_color[1], most_color[2] );
	AllRGB.clear();
	counter.clear();
	most_colorB.clear();
	most_colorG.clear();
	most_colorR.clear();
}


#endif