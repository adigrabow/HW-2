/*
 * main_aux.cpp
 *
 *  Created on: May 5, 2016
 *      Author: adigrabow
 */

#include <opencv2/imgproc.hpp>//calcHist
#include <opencv2/core.hpp>//Mat
#include <opencv2/highgui.hpp>
#include <vector>
#include <iostream>

struct Hist {
	int index; //this will store the index of the image this feature belongs to
	double distance; // this will store the distance of the feature from featureA

};
int* bestHistMatch(int ** histA, int numberOfImages){ // we compare hist A to all other histograms

	struct Hist bestHist[5];

	int cnt = 0;

		for(int i = 0; i < numberOfImages; i++){ // for each photo
			//	for(int j = 0; j < nFeaturesPerImage[i]; j++){
					if(cnt < 5){
						struct Hist hist;
						hist.index = i;
						hist.distance = spRGBHistL2Distance(featureA, databaseFeatures[i][j]);
						bestFeatures[cnt] = feature;
						cnt++;
					}
					else{
						j = nFeaturesPerImage[i];
						i = numberOfImages;
						break;
					}
				//}
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

