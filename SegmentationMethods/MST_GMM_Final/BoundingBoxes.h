#include <opencv2/opencv.hpp>
#include <eigen3/Eigen/Dense>
#include <opencv2/core/eigen.hpp>

using namespace cv;

//Gets all non-zero pixels in an image
template<typename T> void  getNonZeroPix(Mat mask, Mat im, Mat& nonZeroSubset);

//Fins contours from a binary image and writes bounding box results fo file
void findContoursAndWriteResults(Mat& obstacleMap, Mat& image, std::ofstream& scoreFile, std::string outputName);

//Thresholding method that uses otsu thresholding until a certain amount of the image remains
void customOtsuThreshold(Mat& im);
