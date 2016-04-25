// this is my new function
/*
char* str - is the img path
int maxNFeatures is the max num of features i want to find
int * nFeatures - is a pointer to the number of features extracted
*/
cv::Mat spGetSiftDescriptors(char* str, int maxNFeautres, int *nFeatures){
	cv::Mat image;
	image = cv::imread(str, CV_LOAD_IMAGE_GRAYSCALE); //load image to matrix
	if (image.empty()){
		return -1;
	}
	int maxFeatures = maxNFeatures ; //stores the max num of features the user wants
	
	// keypoints will be stored in this vector
	std::vector<cv::KeyPoint> keypoints; 
	cv::Mat descriptors; //feature values will be stored in dsMat
	
	//Creating  a Sift Descriptor extractor
	cv::Ptr<cv::xfeatures2d::SiftDescriptorExtractor> detect =
			cv::xfeatures2d::SIFT::create(maxFeatures);
	detect->detect(image, keypoints, cv::(Mat));
	// this func returns an 
	detect->compute(image, keypoints, descriptors); 
	
	
	return descriptors;
	

}


double spL2SquaredDistance(double* featureA, double* featureB){
	
	double sum;
	
	for(int i = 0; i < 128; i++){
		double a = *(featureA + i); 
		double b = *(featureB + i);
	//	double result = pow(a-b,2);
		double result = (a-b) * (a-b);
		sum += result;
		
	
	}
	return sum;

}