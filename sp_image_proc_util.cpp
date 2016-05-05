/*
 * sp_image_proc_util.cpp
 *
 *  Created on: 24 באפר 2016
 *      Author: Maayan Sivroni
 */
#include <opencv2/imgproc.hpp>//calcHist
#include <opencv2/core.hpp>//Mat
#include <opencv2/highgui.hpp>
#include <vector>
#include <iostream>
#include <opencv2/highgui.hpp> //imshow, drawKeypoints, waitKey
#include <opencv2/xfeatures2d.hpp>//SiftDescriptorExtractor
#include <opencv2/features2d.hpp>
#include <stdio.h>
#include <math.h>
#include <stdlib.h>

//this structure will be used in spBestSIFTL2SquaredDistance function
struct Feature {
	int index; //this will store the index of the image this feature belongs to
	double distance; // this will store the distance of the feature from featureA

};
//headers
double ** matToArray(cv::Mat mat);
int** spGetRGBHist(char* str, int nBins);
double getArrayMax(Feature array [], int size);
int getArrayMaxIndex (Feature array [], double max, int size);
int * featureArrToIntArr(Feature array [], int size, int * targetArray);
int compare (const void * a, const void * b);
using namespace cv;


/*
 * Calculates the RGB channels histogram. The histogram will be stored in a
 * two dimensional array of dimensions 3XnBins . The first row is the
 * red channel histogram, the second row is the green channel histogram
 * and the third row is the blue channel histogram.
 *
 * @param str - The path of the image for which the histogram will be calculated
 * @param nBins - The number of subdivision for the intensity histogram
 * @return NULL if str is NULL or nBins <= 0 or allocation error occurred,
 *  otherwise a two dimensional array representing the histogram.
 */
/*
int main(){

	int** temp = spGetRGBHist("adi.jpg", 500);
	return 1;
}

*/
int** spGetRGBHist(char* str, int nBins){ // red, green, blue
	Mat src;

	if ( (nBins<=0) ||( (str == NULL) && (str[0] =='\0') ) )
		return NULL;

	src = imread(str, CV_LOAD_IMAGE_COLOR); // load image to src

	/// Separate the image in 3 places ( B, G and R )
	std::vector<Mat> bgr_planes;
	split(src, bgr_planes);

	/// Set the ranges ( for B,G,R) )
	float range[] = { 0, 256 };
	const float* histRange = { range };

	/// Set the other parameters:
	int nImages = 1;

	// Temp matrices (will be united to one mat)
	Mat b_hist, g_hist, r_hist;

	/// Compute the histograms:
	/// The results will be store in b_hist,g_hist,r_hist.
	/// The output type of the matrices is CV_32F (float)
	calcHist(&bgr_planes[0], nImages, 0, Mat(), b_hist, 1, &nBins, &histRange);
	calcHist(&bgr_planes[1], nImages, 0, Mat(), g_hist, 1, &nBins, &histRange);
	calcHist(&bgr_planes[2], nImages, 0, Mat(), r_hist, 1, &nBins, &histRange);

	// Creates output array - c
	int **c = (int **) malloc(3 * sizeof( int* ));
	if ( c == NULL )
		return NULL; // allocation error

	for ( int i = 0; i < 3; i++ ) {
		c[i] = (int *) malloc(nBins * sizeof(int));
		if ( c[i]== NULL )
			return NULL; // allocation error
	}

	for (int i=0; i<nBins ; i++){ // RED
		c[0][i] = (int) b_hist.at<float>(i);
	}
	for (int i=0; i<nBins ; i++){ // GREEN
			c[1][i] = (int) g_hist.at<float>(i);
	}
	for (int i=0; i<nBins ; i++){ // BLUE
			c[2][i] = (int) b_hist.at<float>(i);
	}


	//*****************************************************************
	// This is not relevant for the Assignment - You can read it for fun
	//*****************************************************************
	// Draw the histograms for B, G and R
	int hist_w = 512;
	int hist_h = 400;
	int bin_w = cvRound((double) hist_w / nBins);

	Mat histImage(hist_h, hist_w, CV_8UC3, Scalar(0, 0, 0));

	/// Normalize the result to [ 0, histImage.rows ]
	normalize(b_hist, b_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat());
	normalize(g_hist, g_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat());
	normalize(r_hist, r_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat());

	/// Draw for each channel
	for (int i = 1; i < nBins; i++) {
		line(histImage,
				Point(bin_w * (i - 1),
						hist_h - cvRound(b_hist.at<float>(i - 1))),
				Point(bin_w * (i), hist_h - cvRound(b_hist.at<float>(i))),
				Scalar(255, 0, 0), 2, 8, 0);
		line(histImage,
				Point(bin_w * (i - 1),
						hist_h - cvRound(g_hist.at<float>(i - 1))),
				Point(bin_w * (i), hist_h - cvRound(g_hist.at<float>(i))),
				Scalar(0, 255, 0), 2, 8, 0);
		line(histImage,
				Point(bin_w * (i - 1),
						hist_h - cvRound(r_hist.at<float>(i - 1))),
				Point(bin_w * (i), hist_h - cvRound(r_hist.at<float>(i))),
				Scalar(0, 0, 255), 2, 8, 0);
	}

	/// Display
	namedWindow("calcHist Demo", CV_WINDOW_AUTOSIZE);
	imshow("calcHist Demo", histImage);

	waitKey(0);




	return c;


}

/**
 * Returns the average L2-squared distance between histA and histB. The
 * histA and histB are histograms of the RGB channels. Both histograms
 * must be in the same dimension (3 X nBins).
 * @param histA - RGB histogram of image A
 * @param histB - RGB histogram of image B
 * @return -1 if nBins <= 0 or histA/histB is null, otherwise the average L@-squared distance.
 */
double spRGBHistL2Distance(int** histA, int** histB, int nBins){
	double sum0 = 0;
	double sum1 = 0;
	double sum2 = 0; // sum of each row
	double res;

	if ((nBins<=0) ||  (histA == NULL) || (histB == NULL))
		return -1;

	for (int i=0; i<nBins; i++){
		sum0 += (histA[0][i] - histB[0][i]) * (histA[0][i] - histB[0][i]);
	}
	for (int i=0; i<nBins; i++){
		sum1 += (histA[1][i] - histB[1][i]) * (histA[1][i] - histB[1][i]);
	}
	for (int i=0; i<nBins; i++){
		sum2 += (histA[2][i] - histB[2][i]) * (histA[2][i] - histB[2][i]);
	}
	res = 0.33*sum0 + 0.33*sum1 + 0.33*sum2;
	return res;
}

double** spGetSiftDescriptors(char* str, int maxNFeautres, int *nFeatures){

	cv::Mat image;

	image = cv::imread(str, CV_LOAD_IMAGE_GRAYSCALE); //load image to matrix
/*
	if (image.empty())
		return NULL;
*/

	int maxFeatures = maxNFeautres ; //stores the max num of features the user wants

	// keypoints will be stored in this vector
	std::vector<cv::KeyPoint> keypoints;
	cv::Mat descriptors; //feature values will be stored in dsMat

	//Creating  a Sift Descriptor extractor
	cv::Ptr<cv::xfeatures2d::SiftDescriptorExtractor> detect =
			cv::xfeatures2d::SIFT::create(maxFeatures);
	detect->detect(image, keypoints, cv::Mat());
	// this func returns an
	detect->compute(image, keypoints, descriptors);

	//convert descriptors Mat to 2d array
	double** result = matToArray(descriptors);

	* nFeatures = descriptors.rows;

	return result;


}


double spL2SquaredDistance(double* featureA, double* featureB){

	double sum = 0;

	for(int i = 0; i < 128; i++){
		double a = *(featureA + i);
		double b = *(featureB + i);
	//	double result = pow(a-b,2);
		double result = (a-b) * (a-b);
		sum += result;
	}
	return sum;

}

int* spBestSIFTL2SquaredDistance(int bestNFeatures, double* featureA,
		double*** databaseFeatures, int numberOfImages,
		int* nFeaturesPerImage){

	struct Feature bestFeatures[bestNFeatures];

	int cnt = 0;

		for(int i = 0; i < numberOfImages; i++){
				for(int j = 0; j < nFeaturesPerImage[i]; j++){
					if(cnt < bestNFeatures){
						struct Feature feature;
						feature.index = i;
						feature.distance = spL2SquaredDistance(featureA, databaseFeatures[i][j]);
						bestFeatures[cnt] = feature;
						cnt++;
					}
					else{
						j = nFeaturesPerImage[i];
						i = numberOfImages;
						break;
					}
				}
		}


	// loop over the images, and in each image loop over its features
	for(int i = 0; i < numberOfImages; i++){
		for(int j = 0; j < nFeaturesPerImage[i]; j++ ){

			struct Feature f;
			f.index = i;
			f.distance = spL2SquaredDistance(featureA, databaseFeatures[i][j]);
			double maxDistance = getArrayMax(bestFeatures,bestNFeatures );
			int maxIndex = getArrayMaxIndex(bestFeatures, maxDistance, bestNFeatures);
			if(f.distance < maxDistance){ // if we found a feature closer to A than the max, switch them
				bestFeatures[maxIndex] = f;
			}
		}

	}
	//use quick sort to sort the array by distances
	qsort(bestFeatures, bestNFeatures, sizeof(Feature), compare);

	/*now we checked all the features of each image, we have the top ones stored in an array of Features
	  and we need to return only the images index's */
	int targetArray[bestNFeatures];
	int * result; // this is a pointer to the first elem of "result" array
	result = (int *)malloc(bestNFeatures * sizeof(int)); //allocate enough memory space
	//TODO remember to free this space!
	result = featureArrToIntArr(bestFeatures, bestNFeatures, targetArray);
	return result;

}


double ** matToArray(cv::Mat mat ){

	// get the matrix's dims
	int cols = mat.cols;
	int rows = mat.rows;

	// allocate memory space
	double ** result;
	result = (double**)malloc(rows*sizeof(double*));
	for (int i = 0; i < cols; i++)
		result[i] = (double*)malloc(cols*sizeof(double));

	//TODO remember to do free to both arrays!

	//insert values to our array 'result'
	for(int i=0; i<rows; i++){
		for(int j=0; j<cols; j++){
			result[i][j] = mat.at<double>(i,j);
		}
	}

	return result;
}

// this function recieves an array of Features and returns the max element (by distance)
double getArrayMax(Feature array [], int size){
	double max = array[0].distance;

	for(int i = 1; i < size; i++){
		if(array[i].distance > max){
			max = array[i].distance;
		}
	}
	return max;
}

//this function returns the index of the max element
int getArrayMaxIndex (Feature array [], double max, int size){

	for(int i = 0; i < size; i++){
		if(array[i].distance == max )
			return i;
	}
	return -1;

}

int * featureArrToIntArr(Feature array [], int size, int * targetArray){


	for(int i = 0; i < size; i++){
		targetArray[i] = array[i].index;
	}

	return targetArray;
}

int compare (const void * a, const void * b){

	Feature * f1 = (Feature *) a;
	Feature * f2 = (Feature *) b;

	if(f1->distance == f2->distance){ // if distances are equal - choose first the smaller index
		return (f1->index - f2->index);
	}
	else
		return (f1->distance - f2->distance);
}

Feature * sortByIndex(Feature * array, int bestNFeatures){

	for(int i = 0; i < bestNFeatures-1; i++){

		if(array[i].distance == array[i+1].distance){
			if(array[i].index > array[i+1].index){ //we need to switch them
				swap(array[i], array[i+1]);
			}
		}

	}

	return array;

}

void swap (Feature * a, Feature * b){
	   Feature temp = *a;
	   *a = *b;
	   *b = temp;

}
