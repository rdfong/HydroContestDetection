#include <opencv2/opencv.hpp>
#include <eigen3/Eigen/Dense>
#include <opencv2/core/eigen.hpp>
using namespace cv;

#define USE_WEAK_PRIOR 0
// Disable weak prior, not flexible enough for purposes and actually made some results worse
#if USE_WEAK_PRIOR == 1
Mat positionMeanPriors[3];
Mat positionCovPriors[3];
Mat ipositionCovPriors[3];
#endif


//M Step update
Mat means[3];
Mat covars[3];
Mat icovars[3];

//E step update
Mat imagePriors[3]; //We don't actually need a prior for the uniform component
Mat oldPriors[3];
Mat posteriorQ[3];
Mat posteriorP[4];

//Y-data
Mat imageFeatures;

//Step 1
Mat kern2d;
Mat lambda0;
Mat lambda1;

double uniformComponent;

//SETUP FOR BOX SELECTION
int imgCount = 1;
int dims = 3;
const int sizes[] = {64,64,64};
const int channels[] = {0,1,2};
float rRange[] = {0,256};
float gRange[] = {0,256};
float bRange[] = {0,256};
const float *ranges[] = {rRange,gRange,bRange};
Mat mask = Mat();
std::vector<std::vector<Point> > contours;
std::vector<Vec4i> hierarchy;
std::vector<Rect> boundRects;
std::vector<Rect> originalRects;
Mat hist1, temp1, hist2, temp2, nonZeroSubset;
Mat bgr[3];
Rect curRect, otherRect, originalRect, intersection, rectUnion;
std::vector<std::vector<Rect> > intersectionGroups;
std::vector<std::pair<Point2i, Point2i> > finalBoxBounds;
std::vector<Mat> input(3);
std::vector<int> groupsToMerge;


template<typename T> void  getNonZeroPix(Mat mask, Mat im, Mat& nonZeroSubset);
void initializePriorsAndPosteriorStructures(Mat image);
void initializeLabelPriors(Mat image, bool usePrevious);
void initializeGaussianModels(Mat image);
void setDataFromFrame(Mat image);
void updatePriorsAndPosteriors(Mat image);
void updateGaussianParameters(Mat image);
void findShoreLine(Mat coloredImage, std::map<int, int>& shoreLine, bool useHorizon, bool display);
void drawMapping(Mat image, Mat& zoneMapping, Mat& obstacleMap, bool display);
void findObstacles(std::map<int, int>& shoreLine, Mat& obstacles, Mat& obstaclesInWater, bool display);
