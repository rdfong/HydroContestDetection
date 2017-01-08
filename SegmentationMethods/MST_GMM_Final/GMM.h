#include <opencv2/opencv.hpp>
#include <eigen3/Eigen/Dense>
#include <opencv2/core/eigen.hpp>
#include <vector>
using namespace cv;

//Best not to use a weak prior, doesn't change results much and is not adaptabile at all
#define USE_WEAK_PRIOR 0

//Get horizon line information
void parseHorizonInfo(Mat& image, std::string horizonFileName);

//Initialize kernel for Markov Random Field
void initializeKernelInfo(Mat& image);

//Initialize priors and posterior storage
void initializePriorsAndPosteriorStructures(Mat& image);

//Initialize values for priors
void initializeLabelPriors(Mat& image, bool usePrevious);

//Initialize values for gaussian parameters
void initializeGaussianModels(Mat& image);

//Set color and position data from a frame
void setDataFromFrame(Mat& image);

//Update priors and posteriors based on gaussian parameters
void updatePriorsAndPosteriors(Mat& image);

//Update gaussian parameters based on priors and posteriors
void updateGaussianParameters(Mat& image);

//Find pixel by pixel horizon line coordinates
void findHorizonLine(std::vector<int>& shoreLine, cv::Size horizonSize);

//Find pixel by pixel shore line coordinates based on the water mapping found via the GMM
void findShoreLine(Mat& coloredImage, std::vector<int>& shoreLine, bool display);

//Determine which pixels correspond to background seed nodes to be used by the MST method
void findSeedNodes(Mat& image, Mat& dis_image, Mat& seedNodes, bool display);

//Create a binary map of all obstacles in the scene taking the horizon/shoreline into account
void findObstaclesInWater(std::vector<int>& shoreLine, Mat& obstacles, Mat& obstaclesInWater, bool display);

//Run expectation maximization until posterior assignment converges
int runEM(Mat& image);
