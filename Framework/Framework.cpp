#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>
#include <vector>

#define PI 3.14159
#define MIN(x1, x2, x3) std::min(std::min(x1, x2), x3)
#define MAX(x1, x2, x3) std::max(std::max(x1, x2), x3)

double m(cv::Mat& I, int p, int q) {
	int sum = 0;
	CV_Assert(I.depth() != sizeof(uchar));
	switch (I.channels())  {
	case 1:
		for(int i = 0; i < I.rows; ++i) {
			for(int j = 0; j < I.cols; ++j) {
				if (I.at<uchar>(i, j) == 255) {
					sum += pow(j, p) * pow(i, q);
				}
			}
		}
		break;
	case 3:
		cv::Mat_<cv::Vec3b> _I = I;
		for (int i = 0; i < I.rows; ++i){
			for (int j = 0; j < I.cols; ++j){
				if (_I(i, j)[1] == 0) {
					sum += pow(j, p) * pow(i, q);
				}
			}
		}
		break;
	}
	return sum;
}

double i(cv::Mat& image) {
	return m(image, 1, 0) / m(image, 0, 0);
}

double j(cv::Mat& image) {
	return m(image, 0, 1) / m(image, 0, 0);
}

double M11(cv::Mat& image) {
	return m(image, 1, 1) - m(image, 1, 0) * m(image, 0, 1) / m(image, 0, 0);
}

double M20(cv::Mat& image) {
	return m(image, 2, 0) - pow(m(image, 1, 0), 2) / m(image, 0, 0);
}

double M02(cv::Mat& image) {
	return m(image, 0, 2) - pow(m(image, 0, 1), 2) / m(image, 0, 0);
}

double M21(cv::Mat& image) {
	return m(image, 2, 1) - 2 * m(image, 1, 1) * i(image) - m(image, 2, 0) * j(image) + 2 * m(image, 0, 1) * pow(i(image), 2);
}

double M12(cv::Mat& image) {
	return m(image, 1, 2) - 2 * m(image, 1, 1) * j(image) - m(image, 0, 2) * i(image) + 2 * m(image, 1, 0) * pow(j(image), 2);
}

double M30(cv::Mat& image) {
	return m(image, 3, 0) - 3 * m(image, 2, 0) * i(image) + 2 * m(image, 1, 0) * pow(i(image), 2);
}

double M03(cv::Mat& image) {
	return m(image, 0, 3) - 3 * m(image, 0, 2) * j(image) + 2 * m(image, 0, 1) * pow(j(image), 2);
}

double computeM1(cv::Mat& image) {
	return (M20(image) + M02(image)) / pow(m(image, 0, 0), 2);
}

double computeM2(cv::Mat& image) {
	return (pow(M20(image) - M02(image), 2) + 4 * pow(m(image, 1, 1), 2)) / pow(m(image, 0, 0), 4);
}

double computeM3(cv::Mat& image) {
	return (pow(M30(image) - 3 * M12(image), 2) + pow(3 * M21(image) - M03(image), 2)) / pow(m(image, 0, 0), 5);
}

double computeM4(cv::Mat& image) {
	return (pow(M30(image) + M12(image), 2) + pow(M21(image) + M03(image), 2)) / pow(m(image, 0, 0), 5);
}

double computeM5(cv::Mat& image) {
	return (M30(image) - 3 * M12(image) * (M30(image) + M12(image)) * (pow(M30(image) + M12(image), 2) - 3 * pow(M21(image) + M03(image), 2)) + 
		(3 * M21(image) - M03(image)) * (M21(image) + M03(image)) * (3 * pow(M30(image) + M12(image), 2) - pow(M21(image) + M03(image), 2))) / pow(m(image, 0, 0), 10);
}

double computeM7(cv::Mat& image) {
	return (M20(image) * M02(image) - pow(M11(image), 2)) / pow(m(image, 0, 0), 4);
}

/*

void drawBindingBox(cv::Mat& image, int index) {
	cv::Rect rect = findBoundingBox(image, index);
	cv::rectangle(image, rect, cv::Scalar(0, 0, 0));
	int size = computeS(image, index);

	double m10 = m(image(rect), 1, 0, index);
	double m01 = m(image(rect), 0, 1, index);

	int r = m01 / size;
	int c = m10 / size;

	cv::Point boundingBoxCenter(rect.x + rect.width / 2, rect.y + rect.height / 2);
	cv::Point weightCenter(rect.x + r, rect.y + c);

	cv::circle(image, boundingBoxCenter, 2, cv::Scalar(0, 0, 0), CV_FILLED);
	cv::circle(image, weightCenter, 3, cv::Scalar(0, 0, 0), CV_FILLED);

	cv::Point direction = weightCenter - boundingBoxCenter;

	double cos = direction.x / sqrt(direction.x * direction.x + direction.y * direction.y);
	int angle = acos(cos) * 180 / PI ;
	
	std::cout << index << ":" << std::endl;
	std::cout << "\tS: " << size << std::endl;
	std::cout << "\tangle: " << angle << std::endl;
}
*/

void RGBtoHSV(cv::Mat& I) { // correct (for H=0 and S=0 work unproperly)
	int sum = 0;
	CV_Assert(I.depth() != sizeof(uchar));
	cv::Mat  res(I.rows, I.cols, CV_8UC3);
	cv::Mat_<cv::Vec3b> _I = I;
	for (int i = 0; i < I.rows; ++i) {
		for (int j = 0; j < I.cols; ++j) {
			float min, max, delta;
			int r = _I(i, j)[2], g = _I(i, j)[1], b = _I(i, j)[0];
			min = std::min(r, std::min(g, b));
			max = std::max(r, std::max(g, b));
			_I(i, j)[2] = max;				// v

			delta = max - min;	

			if(max != 0) {
				_I(i, j)[1] = delta * 255 / max;		// s
			} else {
				// r = g = b = 0		// s = 0, v is undefined
				_I(i, j)[1] = 0;
				_I(i, j)[0] = -1;
				continue;
			}

			if(r == max) {
				_I(i, j)[0] = 30 * (g - b) / delta;		// between yellow & magenta
			} else if(g == max) {
				_I(i, j)[0] = 60 + 30 * (b - r) / delta;	// between cyan & yellow
			} else {
				_I(i, j)[0] = 120 + 30 * (r - g) / delta;	// between magenta & cyan
			}

			if(_I(i, j)[0] < 0) {
				_I(i, j)[0] += 180;
			}
		}
	}
}

cv::Mat segmentation(cv::Mat& I, int minvalue, int maxvalue) {
	int sum = 0;
	CV_Assert(I.depth() != sizeof(uchar));
	cv::Mat res(I.rows, I.cols, CV_8UC3);
	cv::Mat_<cv::Vec3b> _I = I;
	for (int i = 0; i < I.rows; ++i) {
		for (int j = 0; j < I.cols; ++j) {
			if (minvalue < maxvalue) {
				if (_I(i, j)[0] >= minvalue && _I(i, j)[0] <= maxvalue && _I(i, j)[1] > 50 && _I(i, j)[2] > 75) {
					_I(i, j)[0] = 0;
					_I(i, j)[1] = 0;
					_I(i, j)[2] = 0;
				} else {
					_I(i, j)[0] = 255;
					_I(i, j)[1] = 255;
					_I(i, j)[2] = 255;
				}
			} else {
				if ((_I(i, j)[0] >= minvalue || _I(i, j)[0] <= maxvalue) && _I(i, j)[1] > 50 && _I(i, j)[2] > 75) {
					_I(i, j)[0] = 0;
					_I(i, j)[1] = 0;
					_I(i, j)[2] = 0;
				} else {
					_I(i, j)[0] = 255;
					_I(i, j)[1] = 255;
					_I(i, j)[2] = 255;
				}
			}
		}
	}
	return _I;
}

bool** getBooleanImage(const cv::Mat& image) {
	bool** mask = new bool*[image.cols];
	for (int i = 0; i < image.cols; ++i) {
		mask[i] = new bool[image.rows];
	}

	cv::Mat_<cv::Vec3b> _I = image;
	for (int i = 0; i < image.cols; ++i) {
		for (int j = 0; j < image.rows; ++j) {
			if (_I(j, i)[0] == 0) {
				mask[i][j] = true;
			} else {
				mask[i][j] = false;
			}
		}
	}
	return mask;
}

struct Slice {
public:
	Slice(int x1_, int x2_, int y1_, int y2_, int width_, int heigth_) : x1(x1_), x2(x2_), y1(y1_), y2(y2_) { 
		mask = cv::Mat(heigth_, width_, CV_8UC1, cv::Scalar(255)); 
	}

	int x1;
	int x2;
	int y1;
	int y2;
	cv::Mat mask;

	void add(const cv::Point& point) {
		x1 = std::min(x1, point.x);
		x2 = std::max(x2, point.x);
		y1 = std::min(y1, point.y);
		y2 = std::max(y2, point.y);
		
		mask.at<uchar>(point) = 0;
	}

	cv::Mat getMask() {
		return mask(cv::Rect(x1, y1, x2 - x1, y2 - y1));
	}

	cv::Rect getRect() {
		return cv::Rect(x1, y1, x2 - x1, y2 - y1);
	}

	int getWidth() {
		return x2 - x1;
	}

	int getHeigth() {
		return y2 - y1;
	}
};

std::vector<Slice*> getSlices(cv::Mat& image, bool** mask, int minSliceArea) {
	std::vector<Slice*> slices;

	for (int i = 0; i < image.cols; ++i) {
		for (int j = 0; j < image.rows; ++j) {
			if (mask[i][j] == true) {
				std::vector<cv::Point> stack;
				stack.push_back(cv::Point(i, j));
				int count = 0;
				
				Slice* slice = new Slice(i, i, j, j, image.cols, image.rows);
				while (!stack.empty()) {
					++count;
					cv::Point point = stack.back();
					stack.pop_back();

					if (mask[point.x][point.y]) {
						mask[point.x][point.y] = false;
						slice->add(point);

						if (point.x > 0 && mask[point.x - 1][point.y]) {
							stack.push_back(cv::Point(point.x - 1, point.y));
						}
						if (point.y > 0 && mask[point.x][point.y - 1]) {
							stack.push_back(cv::Point(point.x, point.y - 1));
						}
						if (point.x + 1 < image.cols && mask[point.x + 1][point.y]) {
							stack.push_back(cv::Point(point.x + 1, point.y));
						}
						if (point.y + 1 < image.rows && mask[point.x][point.y + 1]) {
							stack.push_back(cv::Point(point.x, point.y + 1));
						}
					}
				}
				if (count > minSliceArea) {
					slices.push_back(slice);
				}
			}
		}
	}
	return slices;
}

std::vector<Slice*> detectBlueCircle(cv::Mat& image) {
	segmentation(image, 200 / 2,  225 / 2);
	bool** mask = getBooleanImage(image);

	std::vector<Slice*> slices = getSlices(image, mask, 250);
	std::vector<Slice*> outSlices;

	for (std::vector<Slice*>::iterator iter = slices.begin(); iter != slices.end(); ++iter) {
		double m1 = computeM1((*iter)->getMask());
		double m7 = computeM7((*iter)->getMask());
		
		std::cout << "M1: " << m1 << std::endl;
		std::cout << "M7: " << m7;
		if ((m1 > 0.8 && m1 < 2.6) && (m7 > 0.16 && m7 < 1.7))
		{
			std::cout << " <<< passed";
			outSlices.push_back(*iter);
		}
		std::cout << std::endl;
	}

	return outSlices;
}

std::vector<Slice*> detectPart(cv::Mat& image, int minHueValue, int maxHueValue) {
	segmentation(image, minHueValue, maxHueValue);
	bool** mask = getBooleanImage(image);

	std::vector<Slice*> slices = getSlices(image, mask, 350);
	std::vector<Slice*> outSlices;

	for (std::vector<Slice*>::iterator iter = slices.begin(); iter != slices.end(); ++iter) {
		double m1 = computeM1((*iter)->getMask());
		double m7 = computeM7((*iter)->getMask());

		std::cout << "M1: " << m1 << std::endl;
		std::cout << "M7: " << m7;
		if ((m1 > 0.26 && m1 < 0.7) && (m7 > 0.014 && m7 < 0.057))
		{
			std::cout << " <<< passed";
			outSlices.push_back(*iter);
		}
		std::cout << std::endl;
	}

	return outSlices;
}

void drawRects(cv::Mat image, std::vector<Slice*> slices, const cv::Scalar& color) {
	for (std::vector<Slice*>::iterator iter = slices.begin(); iter != slices.end(); ++iter) {
		cv::Rect rect((*iter)->x1, (*iter)->y1, (*iter)->x2 - (*iter)->x1 + 1, (*iter)->y2 - (*iter)->y1 + 1);
		cv::rectangle(image, rect, color, 1);
	}
}

int findDistance(const cv::Point& point1, const cv::Point& point2) {
	cv::Point difference = point2 - point1;
	return sqrt(difference.x * difference.x + difference.y * difference.y);
}

Slice* getNearestSlice(Slice* blueSlice, const std::vector<Slice*>& slices) {
	int minDistance = blueSlice->getWidth() + blueSlice->getHeigth();
	Slice* nearestSlice = NULL;
	cv::Point blueWeightCenter(blueSlice->x1 + i(blueSlice->getMask()), blueSlice->y1 + j(blueSlice->getMask()));

	for (std::vector<Slice*>::const_iterator iter = slices.begin(); iter != slices.end(); ++iter) {	
		cv::Point weightCenter((*iter)->x1 + i((*iter)->getMask()), (*iter)->y1 + j((*iter)->getMask()));
		
		int distance = findDistance(blueWeightCenter, weightCenter);
		if (distance < minDistance) {
			minDistance = distance;
			nearestSlice = (*iter);
		}
	}
	return nearestSlice;
}

cv::Point getWeightCenter(Slice* slice) {
	int r = i(slice->getMask());
	int c = j(slice->getMask());
	return cv::Point(slice->x1 + r, slice->y1 + c);
}


void image(std::string name) {
	std::cout << std::endl << "Image: " << name << std::endl;
	cv::Mat image = cv::imread(name);
	cv::Mat imageRed, imageGreen, imageYellow, imageBlue;
	
	image.copyTo(imageBlue);
	image.copyTo(imageGreen);
	image.copyTo(imageRed);
	image.copyTo(imageYellow);
	
	RGBtoHSV(imageBlue);
	RGBtoHSV(imageGreen);
	RGBtoHSV(imageRed);
	RGBtoHSV(imageYellow);

	std::cout << "Blue: " << std::endl;
	std::vector<Slice*> blueSlices = detectBlueCircle(imageBlue);
	std::cout << "Green: " << std::endl;
	std::vector<Slice*> greenSlices = detectPart(imageGreen, 95 / 2,  160 / 2);
	std::cout << "Red: " << std::endl;
	std::vector<Slice*> redSlices = detectPart(imageRed, 330 / 2, 15 / 2);
	std::cout << "Yellow: " << std::endl;
	std::vector<Slice*> yellowSlices = detectPart(imageYellow, 35 / 2, 65 / 2);

	drawRects(image, blueSlices, cv::Scalar(255, 0, 0));
	drawRects(image, greenSlices, cv::Scalar(0, 255, 0));
	drawRects(image, redSlices, cv::Scalar(0, 0, 255));
	drawRects(image, yellowSlices, cv::Scalar(0, 255, 255));

	for (std::vector<Slice*>::iterator blueIter = blueSlices.begin(); blueIter != blueSlices.end(); ++blueIter) {

		cv::Point blueWeightCenter((*blueIter)->x1 + i((*blueIter)->getMask()), (*blueIter)->y1 + j((*blueIter)->getMask()));
		cv::circle(image, blueWeightCenter, 3, cv::Scalar(255, 255, 255), CV_FILLED);
		
		Slice* greenSlice = getNearestSlice(*blueIter, greenSlices);
		Slice* redSlice = getNearestSlice(*blueIter, redSlices);
		Slice* yellowSlice = getNearestSlice(*blueIter, yellowSlices);
		
		if (greenSlice == NULL || redSlice == NULL || yellowSlice == NULL) {
			continue;
		}

		cv::Point greenWeightCenter = getWeightCenter(greenSlice);
		cv::Point redWeightCenter = getWeightCenter(redSlice);
		cv::Point yellowWeightCenter = getWeightCenter(yellowSlice);

		int distance = findDistance(greenWeightCenter, blueWeightCenter);

		int maxDefect = 20 * std::max((*blueIter)->getWidth(), (*blueIter)->getHeigth()) / std::min((*blueIter)->getWidth(), (*blueIter)->getHeigth());
		if (findDistance(blueWeightCenter, greenWeightCenter) - findDistance(blueWeightCenter, redWeightCenter) < maxDefect &&
			findDistance(blueWeightCenter, redWeightCenter) - findDistance(blueWeightCenter, yellowWeightCenter) < maxDefect &&
			findDistance(blueWeightCenter, yellowWeightCenter) - findDistance(blueWeightCenter, greenWeightCenter) < maxDefect) {
			cv::Rect rect;
			rect.x = MIN(greenSlice->x1, redSlice->x1, yellowSlice->x1);
			rect.y = MIN(greenSlice->y1, redSlice->y1, yellowSlice->y1);
			rect.width = MAX(greenSlice->x2, redSlice->x2, yellowSlice->x2) - rect.x;
			rect.height = MAX(greenSlice->y2, redSlice->y2, yellowSlice->y2) - rect.y;
			
			cv::rectangle(image, rect, cv::Scalar(255, 255, 255), 1);
			cv::circle(image, greenWeightCenter, 3, cv::Scalar(255, 255, 255), CV_FILLED);
			cv::circle(image, redWeightCenter, 3, cv::Scalar(255, 255, 255), CV_FILLED);
			cv::circle(image, yellowWeightCenter, 3, cv::Scalar(255, 255, 255), CV_FILLED);
		}

	}

	/*
	*/
	int scale = 1;
	cv::resize(image, image, cv::Size(scale * image.cols, scale * image.rows));
	cv::imshow(name, image);
}

int main(int, char *[]) {
    std::cout << "Start ..." << std::endl;
	
	image("chrome1.jpg");
	image("chrome2.jpg");
	image("chrome3.jpg");
	image("chrome4.jpg");
	image("chrome5.jpg");
	image("chrome6.jpg");
	image("chrome7.jpg");
	image("chrome8.jpg");
	image("chrome9.jpg");
	image("chrome10.jpg");
	image("chrome11.jpg");
	image("chrome_3D.jpg");

    cv::waitKey(-1);
    return 0;
}

/*
double m1 = computeM1((*iter)->getMask());
	double m2 = computeM2((*iter)->getMask());
	double m3 = computeM3((*iter)->getMask());
	double m4 = computeM4((*iter)->getMask());
	//double m5 = computeM5((*iter)->getMask());
	double m7 = computeM7((*iter)->getMask());
		
	std::cout << std::endl << "M1: " << m1;
	std::cout << std::endl << "M7: " << m7;
	//for blue - if ((m1 > 1 && m1 < 1.6)&& (m7 > 0.2 && m7 < 0.65))
	if ((m1 > 0.27 && m1 < 0.355) && (m7 > 0.025 && m7 < 0.03))
	{
		std::cout << "passed" << std::endl;
			
			
		std::cout << std::endl << "M2: " << m2;
		std::cout << std::endl << "M3: " << m3;
		std::cout << std::endl << "M4: " << m4;
		//std::cout << std::endl << "M5: " << m5;
			
	}

//S, L, W1, W3, M1, M7

//W3 (prost) ~ 0.294
//M7 (kolo, elipsa) ~ 0.00633
//W3 (prost,...) ~ 1.6

cv::Mat& computeL(cv::Mat& I, int& L) {
	L = 0;
  CV_Assert(I.depth() != sizeof(uchar));
  switch(I.channels())  {
  case 1:
    for( int i = 0; i < I.rows; ++i)
        for( int j = 0; j < I.cols; ++j )
            I.at<uchar>(i,j) = (I.at<uchar>(i,j)/32)*32;
    break;
  case 3:
	  cv::Mat_<cv::Vec3b> _I = I;
	
	for (int i = 1; i < I.rows - 1; ++i){
		for (int j = 1; j < I.cols - 1; ++j){

			if ((_I(i, j)[0] == 255) &&
				(							     (_I(i, j - 1)[0] != 255) || 
				 (_I(i - 1, j    )[0] != 255) ||							    (_I(i + 1, j    )[0] != 255)
											  || (_I(i, j + 1)[0] != 255)   							   )) {
				++L;
			}
			else {

			}
		}
	}
    I = _I;
    break;
  }
  return I;
}

cv::Rect findBoundingBox(cv::Mat& I, int index){
	cv::Rect rect;
	int min_x = I.cols;
	int min_y = I.rows;
	int max_x = 0;
	int max_y = 0;
    CV_Assert(I.depth() != sizeof(uchar));
    cv::Mat res(I.rows,I.cols, CV_8UC3);
    switch(I.channels())  {
    case 3:
        cv::Mat_<cv::Vec3b> _I = I;
		
		for (int i = 0; i < I.rows; ++i){
			for (int j = 0; j < I.cols; ++j){
				if (_I(i, j)[1] == index * 45) {
					if (j < min_x) {
						min_x = j;
					}
					if (j > max_x) {
						max_x = j;
					}
					if (i < min_y) {
						min_y = i;
					}
					if (i > max_y) {
						max_y = i;
					}
				}
			}
		}
		I = _I;
    }
	
	return cv::Rect(min_x, min_y, max_x - min_x, max_y - min_y);
}
*/