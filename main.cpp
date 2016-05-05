#include <opencv2/imgproc.hpp>//calcHist
#include <opencv2/core.hpp>//Mat
#include <opencv2/highgui.hpp>
#include <vector>
#include <iostream>

#define DIRECTORY "Enter images directory path:"
#define PREFIX "Enter images prefix:"
#define IMG_NUM "Enter number of images:\n"
#define ERROR "An error occurred - invalid number of images\n"
#define SUFFIX "Enter images suffix:\n"
#define BINS "Enter number of bins:\n"
#define BINS_ERRORS "An error occurred - invalid number of bins\n"
#define FEAT_NUM "Enter number of features:\n"
#define FEAT_NUM_ERROR "An error occurred - invalid number of features\n"
#define ENTER_QUERY "Enter a query image or # to terminate:\n"
#define ENTER_Q_TERMINATE "Enter a query image or # to terminate:\n"
#define EXIT "Exiting...\n"
#define MAX_PATH_LEN 1024


using namespace cv;
int** spGetRGBHist(char* str, int nBins);
double** spGetSiftDescriptors(char* str, int maxNFeautres, int *nFeatures);



int main() {

	char dir[MAX_PATH_LEN], prefix[MAX_PATH_LEN], suffix[MAX_PATH_LEN], query[MAX_PATH_LEN];
	int num, bins, featNum;

	setvbuf(stdout, NULL, _IONBF, 0); // from main in assignment 1
	puts(DIRECTORY);
	fflush(NULL);
	scanf("%s", dir); // for example ./images/

	puts(PREFIX);
//	puts("Enter images directory path:\n"); I guess this is a mistake right?
	fflush(NULL);
	scanf("%s", prefix); // for example img

	puts(IMG_NUM); // numbered 0 to n-1
	fflush(NULL);
	scanf("%d", &num);
	if (num < 1){
		puts(ERROR);
		fflush(NULL);
		// func free + end program,
		// we need to free the mem we allocated using malloc...
	}

	puts(SUFFIX);
	fflush(NULL);
	scanf("%s", suffix); // for example .jpg

	puts(BINS);
	fflush(NULL);
	scanf("%d", &bins);
	if (bins < 1){
			puts(BINS_ERRORS);
			fflush(NULL);
			// func free + end program
	}

	puts(FEAT_NUM);
	fflush(NULL);
	scanf("%d", &featNum);
	if (featNum < 1){
			puts(FEAT_NUM_ERROR);
			fflush(NULL);
			// func free + end program
	}

	//section 7
	int *** rgbHistDB; //this 3d array will store all the RGBHist of all images in the directory
	double *** siftDB; //this 3d array will store all the SIFT descriptors of all images

	for(int i = 0; i < num; i++){ //loop over all the images in the directory given
		int nFeaturesNum = 0;
		char img [MAX_PATH_LEN * 3]; //make sure we have enough memory to store the full directory
		strcpy(img,""); // initialize img in the beginning of each loop

		char index [MAX_PATH_LEN];
		sprintf(index, "%d", i); //convert i to a string and store it in 'index'

		strcat(img, dir);
		strcat(img, prefix);
		strcat(img, index);
		strcat(img, suffix); //now we have the full image path in 'img' and we can use it

		// Now the program will calculate the RGB histogram for each image in the images
		rgbHistDB[i] = spGetRGBHist(img, bins);

		//calculate the SIFT descriptors
		siftDB[i] = spGetSiftDescriptors(img, featNum, &nFeaturesNum);

	}

	puts(ENTER_QUERY); // first input
	fflush(NULL);
	scanf("%s", query);

	while(strcmp(query,"#/0") != 0 ){ // main loop
		int n = 0;


			// calculate the query's hist and SIFT
			int ** queryHist = spGetRGBHist(query, bins);
			double ** querySiftDescriptors = spGetSiftDescriptors(query, featNum, &n);

			//Search using Global Features
			for(int i = 0; i < num; i++){

			}

			puts(ENTER_Q_TERMINATE);
			fflush(NULL);
			scanf("%s", query);

	}
	puts(EXIT);
	fflush(NULL);
	// func free + end program
	return 0;
}
