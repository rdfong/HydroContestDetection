#include <opencv2/opencv.hpp>
#include <eigen3/Eigen/Dense>
#include <opencv2/core/eigen.hpp>
#include <vector>
using namespace cv;

#define USE_WEAK_PRIOR 0

void parseHorizonInfo(Mat& image, std::string horizonFileName);
void initializeKernelInfo(Mat& image);
void initializePriorsAndPosteriorStructures(Mat& image);
void initializeLabelPriors(Mat& image, bool usePrevious);
void initializeGaussianModels(Mat& image);
void setDataFromFrame(Mat& image);
void updatePriorsAndPosteriors(Mat& image);
void updateGaussianParameters(Mat& image);
void findHorizonLine(std::vector<int>& shoreLine, cv::Size horizonSize);
void findShoreLine(Mat& coloredImage, std::vector<int>& shoreLine, bool display);
void findSeedNodes(Mat& image, Mat& seedNodes, bool display);
void findObstacles(std::vector<int>& shoreLine, Mat& obstacles, Mat& obstaclesInWater, bool display);
int runEM(Mat& image);
