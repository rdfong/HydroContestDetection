#include <opencv2/opencv.hpp>
#include <eigen3/Eigen/Dense>
#include <opencv2/core/eigen.hpp>

using namespace cv;

//Finding final contours from binary map
template<typename T> void  getNonZeroPix(Mat mask, Mat im, Mat& nonZeroSubset);

void findContoursAndWriteResults(Mat& obstacleMap, Mat& image, std::ofstream& scoreFile, std::string outputName, bool display);

void customOtsuThreshold(Mat& im);
