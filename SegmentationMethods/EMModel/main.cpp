#include <iostream>
#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <fstream>
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

//SETUP FOR BOX SELECTION AND MERGING
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

//File setup
std::ofstream scoreFile;
std::string name;
std::string output;
std::string waitString;
std::string imageFolder;

int leftIntercept, rightIntercept;

//-------------------Bounding box code---------------------//
/**
 * @brief getNonZeroPix     Gets all values of pixels specified by the mask
 * @param mask              Mask of which pixels to get values from im
 * @param im                The image to retrieve values from
 * @param nonZeroSubset     The output of the desired values
 */
 template<typename T> void  getNonZeroPix(Mat mask, Mat im, Mat& nonZeroSubset) {
     Mat imValues = im.reshape(0, im.rows*im.cols);
     Mat flattenedMask = mask.reshape(0, im.rows*im.cols);
     Mat idx;
     findNonZero(flattenedMask, idx);
     nonZeroSubset = Mat::zeros(idx.rows, 1, CV_8U);
     auto im_it = nonZeroSubset.begin<T>();
     for (int i = 0; i < idx.rows; i++) {
         *im_it= imValues.at<T>(idx.at<int>(i, 1), 0);
         ++im_it;
     }
 }


/**
 * @brief findContoursAndWriteResults   Finds all contours and surrounding bounding boxes, writes to file and displays
 * @param obstacleMap                   The binary map of obstacles
 * @param image                         The original image on top of which we'll write rectangles
 * @param scoreFile                     The file handle to write to
 * @param outputName                    The file name to write the bounding box coordinates and dimensions to
 */
void findContoursAndWriteResults(Mat& obstacleMap, Mat& image, bool display) {
    hierarchy.clear();
    contours.clear();
    boundRects.clear();
    originalRects.clear();
    findContours( obstacleMap.clone(), contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE, Point(0, 0) );
    //get bounding rects from contours
    int expand = 1;
    for (int i =0; i < contours.size(); i++) {
        curRect = boundingRect(contours[i]);
        //don't add a box if it is too long or too tall in aspect ratio
        if ((double)curRect.width/curRect.height < 0.05|| (double)curRect.height/curRect.width < 0.05)
            continue;
        Point2i newTL(max(curRect.tl().x-expand, 0), max(curRect.tl().y-expand,0));
        Point2i newBR(min(curRect.br().x+expand, obstacleMap.cols-1), min(curRect.br().y+expand,obstacleMap.rows-1));
        boundRects.push_back(Rect(newTL, newBR));
        originalRects.push_back(curRect);
    }

    //intersection groups mirrors finalboxbounds in size, but final box bounds contains information on originalRects, intersection groups is for the expanded rects
    intersectionGroups.clear();
    finalBoxBounds.clear();

    for (int k = 0; k < boundRects.size(); k++) {
          curRect = boundRects[k];
          originalRect = originalRects[k];
          bool intersectionFound = false;
          groupsToMerge.clear();
          //check for intersections
         for (int i = 0; i < intersectionGroups.size(); i++) {
              for (int j = 0; j < intersectionGroups[i].size(); j++) {
                  otherRect = intersectionGroups[i][j];
                  intersection = curRect & otherRect;
                  //one is contained by the other
                  if (intersection.area() == curRect.area() || intersection.area() == otherRect.area()) {
                      intersectionGroups[i].push_back(curRect);
                      finalBoxBounds[i].first = Point2i(min(finalBoxBounds[i].first.x, originalRect.tl().x), min(finalBoxBounds[i].first.y, originalRect.tl().y));
                      finalBoxBounds[i].second = Point2i(max(finalBoxBounds[i].second.x, originalRect.br().x), max(finalBoxBounds[i].second.y, originalRect.br().y));
                      //multiple intersecting groups may be found, need to find out what to merge
                      intersectionFound = true;
                      groupsToMerge.push_back(i);
                      break;
                  } else if (intersection.area() > 0) {
                      //COLOR SIMILARITY MEASURE
                      Mat mask1, mask2;
                      obstacleMap(curRect).copyTo(mask1);
                      temp1 = image(curRect);
                      split(temp1, bgr);
                      getNonZeroPix<unsigned char>(mask1, bgr[0], bgr[0]);
                      getNonZeroPix<unsigned char>(mask1, bgr[1], bgr[1]);
                      getNonZeroPix<unsigned char>(mask1, bgr[2], bgr[2]);
                      input[2] = bgr[2];
                      input[1] = bgr[1];
                      input[0] = bgr[0];
                      cv::merge(input, nonZeroSubset);
                      calcHist(&nonZeroSubset, imgCount, channels, Mat(), hist1, dims, sizes, ranges);
                      normalize( hist1, hist1);
                      int numPix1 = nonZeroSubset.rows;

                      obstacleMap(otherRect).copyTo(mask2);
                      temp2 = image(otherRect);
                      split(temp2, bgr);
                      getNonZeroPix<unsigned char>(mask2, bgr[0], bgr[0]);
                      getNonZeroPix<unsigned char>(mask2, bgr[1], bgr[1]);
                      getNonZeroPix<unsigned char>(mask2, bgr[2], bgr[2]);
                      input[2] = bgr[2];
                      input[1] = bgr[1];
                      input[0] = bgr[0];
                      cv::merge(input, nonZeroSubset);
                      calcHist(&nonZeroSubset, imgCount, channels, Mat(), hist2, dims, sizes, ranges);
                      normalize( hist2, hist2);
                      int numPix2 = nonZeroSubset.rows;
                      double colorSim = compareHist(hist1, hist2, CV_COMP_INTERSECT);
                      //std::cout << colorSim << std::endl;

                      // SIZE SIMILARITY MEASURE - current is used if it improves the average fill of the separated boxes
                      rectUnion = curRect | otherRect;
                      double sizeSim = 1.0 - ((double)rectUnion.area() - numPix1 - numPix2)/(rectUnion.area());
                     // std::cout << sizeSim << std::endl;

                      if (colorSim < 2.0 || sizeSim > 0.5) {
                         //merge curRect with otherRect
                          intersectionGroups[i].push_back(curRect);
                          finalBoxBounds[i].first = Point2i(min(finalBoxBounds[i].first.x, originalRect.tl().x), min(finalBoxBounds[i].first.y, originalRect.tl().y));
                          finalBoxBounds[i].second = Point2i(max(finalBoxBounds[i].second.x, originalRect.br().x), max(finalBoxBounds[i].second.y, originalRect.br().y));

                          intersectionFound = true;
                          groupsToMerge.push_back(i);
                          break;
                      }
                  }
              }
          }
          if (groupsToMerge.size() > 1) {
              //merge groups
              for (int i = groupsToMerge.size()-1; i > 0; i--) {
                  int mergeTo = groupsToMerge[0];
                  int mergeFrom = groupsToMerge[i];
                  intersectionGroups[mergeTo].insert(intersectionGroups[mergeTo].begin(), intersectionGroups[mergeFrom].begin(), intersectionGroups[mergeFrom].end());
                  finalBoxBounds[mergeTo].first = Point2i(min(finalBoxBounds[mergeTo].first.x, finalBoxBounds[mergeFrom].first.x),
                                                          min(finalBoxBounds[mergeTo].first.y, finalBoxBounds[mergeFrom].first.y));

                  finalBoxBounds[mergeTo].second = Point2i(max(finalBoxBounds[mergeTo].second.x, finalBoxBounds[mergeFrom].second.x),
                                                           max(finalBoxBounds[mergeTo].second.y, finalBoxBounds[mergeFrom].second.y));

                  intersectionGroups.erase(intersectionGroups.begin()+mergeFrom);
                  finalBoxBounds.erase(finalBoxBounds.begin()+mergeFrom);
              }
          }

          //no intersections found
          if (!intersectionFound) {
              int curSize = intersectionGroups.size();
              intersectionGroups.resize(curSize+1);
              intersectionGroups[curSize].push_back(curRect);
              finalBoxBounds.push_back(std::pair<Point2i, Point2i>(originalRect.tl(), originalRect.br()));
          }
    }
    for (int i = 0; i < finalBoxBounds.size(); i++) {
        curRect = Rect(finalBoxBounds[i].first, finalBoxBounds[i].second);
        if (curRect.area() > 25) {
          rectangle(image, curRect, Scalar(0, 255,0), 2);
          scoreFile << "other\n" << curRect.tl().x << " " << curRect.tl().y << " "
                              << curRect.width << " " << curRect.height <<std::endl;
        }
    }

    if(display)
        imshow("boxOutput", image);
    imwrite(output+name, image);
}

//-----------------------------------------------//

/**
 * @brief initializePriorsAndPosteriorStructures    Initialize structures containing prior and posterior information
 * @param image
 */
void initializePriorsAndPosteriorStructures(Mat image) {
    for (int i = 0; i < 4; i++) {
        if (i < 3) {
            imagePriors[i] = Mat::zeros(image.rows, image.cols, CV_64F);
            oldPriors[i] = Mat::zeros(image.rows, image.cols, CV_64F);
            posteriorQ[i] = Mat::zeros(image.rows, image.cols, CV_64F);
        }
        posteriorP[i] = Mat::zeros(image.rows, image.cols, CV_64F);
    }
    uniformComponent = 1.0/(image.rows*image.cols*10988544.0);
#if USE_WEAK_PRIOR == 1
    //From the last 25 images in the Test Media set
    positionMeanPriors[0] = (Mat_<double>(1,5) <<
                             64.9531377612962,
                             12.7114319140013,
                             131.306726742091,
                             37.6804563745202,
                             209.603955587930);
    positionCovPriors[0] = (Mat_<double>(5,5) <<
                            1302.50738403309,	5.16331221476810,	-46.7216623195916,	-110.858320587398,	211.576800090794,
                            5.16331221476810,	81.6189693481908,	-26.2254714610841,	-95.9970782044438,	41.4804757273633,
                            -46.7216623195916,	-26.2254714610841,	2255.05475103781,	217.614570462177,	235.615320627296,
                            -110.858320587398,	-95.9970782044438,	217.614570462177,	937.364180598562,	-226.841694186760,
                            211.576800090794,	41.4804757273633,	235.615320627296,	-226.841694186760,	1418.65820871393);

    positionMeanPriors[1]  = (Mat_<double>(1,5) <<
                              61.0766578884367,
                              19.7486952358152,
                              124.392423935630,
                              56.3788256363787,
                              163.641273641369);
    positionCovPriors[1] = (Mat_<double>(5,5) <<
                            1294.26820228977,	5.91109861033112,	-111.673449250060,	-46.9881375544757,	111.149025396833,
                            5.91109861033112,	128.957433835868,	27.5042450642121,	-27.3340299527174,	-38.9960389424345,
                            -111.673449250060,	27.5042450642121,	2275.59398350067,	35.7121829533379,	874.677879980121,
                            -46.9881375544757,	-27.3340299527174,	35.7121829533379,	986.066547997161,	-588.816020422674,
                            111.149025396833,	-38.9960389424345,	874.677879980121,	-588.816020422674,	2708.35928186555);

    positionMeanPriors[2] = (Mat_<double>(1,5) <<
                             62.9513882257448,
                             52.4619580786860,
                             139.700614100230,
                             58.7715504338722,
                             141.914856283422);
    positionCovPriors[2] = (Mat_<double>(5,5) <<
                            1301.85757803354,	2.43698126869371,	-11.6478726137114,	-34.5677475344341,	100.146853820201,
                            2.43698126869371,	305.797563254135,	29.2006044450966,	157.289335323213,	-274.892037087511,
                            -11.6478726137114,	29.2006044450966,	1297.75359197849,	264.350519819200,	-71.4327776469228,
                            -34.5677475344341,	157.289335323213,	264.350519819200,	2123.53137783777,	-899.579866784132,
                            100.146853820201,	-274.892037087511,	-71.4327776469228,	-899.579866784132,	2447.66602013560);

    invert(positionCovPriors[0], ipositionCovPriors[0]);
    invert(positionCovPriors[1], ipositionCovPriors[1]);
    invert(positionCovPriors[2], ipositionCovPriors[2]);
#endif
}

/**
 * @brief initializeLabelPriors Initialize priors either with an even distribution or information from the previous frame
 * @param image                 The current frame
 * @param usePrevious           Whether or not to use previous frame information
 */
void initializeLabelPriors(Mat image, bool usePrevious) {
    if (usePrevious) {
        double nonUniformSum = 1.0-uniformComponent;
        std::cout << nonUniformSum << std::endl;
        Mat QSum = posteriorQ[0] + posteriorQ[1] + posteriorQ[2];
        for (int i = 0; i < 3; i++) {
            //normalize posteriorQ, blur, and set prior equal to
            Mat normalizedQ = (posteriorQ[i]/QSum)*nonUniformSum;
            //Blur and set to imagePriors
            GaussianBlur(normalizedQ,imagePriors[i],Size(3,3), 3.0, 3.0, BORDER_REPLICATE);
        }
    } else {
        double gaussianComponent = (1.0-uniformComponent)/3.0;
        for (int row = 0; row < image.rows; row++) {
            double *imPriRow0 = imagePriors[0].ptr<double>(row);
            double *imPriRow1 = imagePriors[1].ptr<double>(row);
            double *imPriRow2 = imagePriors[2].ptr<double>(row);
            for (int col = 0; col < image.cols; col++) {
                imPriRow0[col] = gaussianComponent;
                imPriRow1[col] = gaussianComponent;
                imPriRow2[col] = gaussianComponent;
            }
        }
    }
}

/**
 * @brief initializeGaussianModels  Initialize gaussian model parameters based on the horizon line
 * @param image                     The current frame
 */
void initializeGaussianModels(Mat image) {
    int skyCount = 0;
    int landCount = 0;
    int waterCount = 0;

    std::map<int,std::pair<int, int> > indexToRegion;
    int i;

    double hMidPercent = (leftIntercept + rightIntercept)/(2.0*image.rows);
    for (i = 0; i < imageFeatures.rows; i++) {
        double curX = imageFeatures.at<double>(i, 0);
        double curY = imageFeatures.at<double>(i, 1);
        int curHPoint = (rightIntercept-leftIntercept)*(curX/image.cols)+leftIntercept;
        if (hMidPercent <= .01) {
            //just do a 33/33/33 flat split
            if (imageFeatures.at<double>(i, 1) < .33*image.rows) {
                indexToRegion[i] = std::pair<int, int>(0, skyCount);
                skyCount++;
            } else if (imageFeatures.at<double>(i, 1)  < .66*image.rows) {
                indexToRegion[i] = std::pair<int, int>(1, landCount);
                landCount++;
            } else {
                indexToRegion[i] = std::pair<int, int>(2, waterCount);
                waterCount++;
            }
        } else if (hMidPercent <= .05) {
            //Set everything above horizon to sky
            if (curHPoint > curY) {
                indexToRegion[i] = std::pair<int, int>(0, skyCount);
                skyCount++;
            } else {
                //Set everything below to water and land, curY is too high?
                if (curY-curHPoint < (1.0-hMidPercent)/2.0*image.rows) {
                    indexToRegion[i] = std::pair<int, int>(1, landCount);
                    landCount++;
                } else {
                    indexToRegion[i] = std::pair<int, int>(2, waterCount);
                    waterCount++;
                }
            }
        } else {
            //Set everything below horizon to water
            if (curHPoint < curY) {
                indexToRegion[i] = std::pair<int, int>(2, waterCount);
                waterCount++;
            } else {
                //Set everything above to sky and land.
                if (curHPoint-curY < (hMidPercent)/2.0*image.rows) {
                    indexToRegion[i] = std::pair<int, int>(1, landCount);
                    landCount++;
                } else {
                    indexToRegion[i] = std::pair<int, int>(0, skyCount);
                    skyCount++;
                }
            }
        }
    }
    Mat regionMats[3];

    regionMats[0] = Mat::zeros(skyCount, 5, CV_64F);
    regionMats[1] = Mat::zeros(landCount, 5, CV_64F);
    regionMats[2] = Mat::zeros(waterCount, 5, CV_64F);

    for (i = 0; i < imageFeatures.rows; i++) {
        imageFeatures.row(i).copyTo(regionMats[indexToRegion[i].first].row(indexToRegion[i].second));
    }

    //Calculate gaussian parameters
    cv::calcCovarMatrix(regionMats[0], covars[0], means[0], CV_COVAR_NORMAL | CV_COVAR_ROWS | CV_COVAR_SCALE);
    cv::calcCovarMatrix(regionMats[1], covars[1], means[1], CV_COVAR_NORMAL | CV_COVAR_ROWS | CV_COVAR_SCALE);
    cv::calcCovarMatrix(regionMats[2], covars[2], means[2], CV_COVAR_NORMAL | CV_COVAR_ROWS | CV_COVAR_SCALE);
}

/**
 * @brief setDataFromFrame  Set 5 feature vector info from frame
 * @param image             The current frame
 */
void setDataFromFrame(Mat image) {
    imageFeatures = Mat::zeros(image.rows*image.cols, 5, CV_64F);
    int index = 0;
    for (int row = 0; row < image.rows; row++) {
        for (int col = 0; col < image.cols; col++) {
            cv::Vec3b color = image.at<cv::Vec3b>(row,col);
            double *rowPtr = imageFeatures.ptr<double>(index);
            rowPtr[0] = col;
            rowPtr[1] = row;
            rowPtr[2] = color[0];
            rowPtr[3] = color[1];
            rowPtr[4] = color[2];
            index++;
        }
    }
}

/**
 * @brief updatePriorsAndPosteriors The expectation step of the the EM algorithm
 * @param image                     The current frame
 */
void updatePriorsAndPosteriors(Mat image) {
    //Make sure matrices are well conditioned
    int result0 = invert(covars[0]+Mat::eye(5,5,CV_64F).mul(covars[0])*1e-10, icovars[0]);
    int result1 = invert(covars[1]+Mat::eye(5,5,CV_64F).mul(covars[1])*1e-10, icovars[1]);
    int result2 = invert(covars[2]+Mat::eye(5,5,CV_64F).mul(covars[2])*1e-10, icovars[2]);
    assert(result0 && result1 && result2);

    //Using equation 12 from paper
    Mat priMultMat[4];
    Mat priSum = Mat::zeros(image.rows, image.cols, CV_64F);
    Mat mahalanobis;
    for (int i = 0; i < 3; i++) {
        Mat meanRepeated = cv::repeat(means[i], image.rows*image.cols, 1);
        Mat meanDiff = imageFeatures-meanRepeated;

        mahalanobis = (meanDiff*icovars[i]).mul(meanDiff);
        cv::reduce(mahalanobis,mahalanobis,1,CV_REDUCE_SUM, CV_64F);
        cv::sqrt(mahalanobis, mahalanobis);
        mahalanobis = mahalanobis.reshape(0, image.rows);

        cv::exp(-.5*(mahalanobis.mul(mahalanobis)), priMultMat[i]);
        priMultMat[i] = imagePriors[i].mul(priMultMat[i]/sqrt(cv::determinant(2*M_PI*covars[i])));
        add(priSum, priMultMat[i], priSum);
    }

    priMultMat[3] = Mat::ones(image.rows, image.cols, CV_64F)*uniformComponent;
    add(priSum, priMultMat[3], priSum);

    posteriorP[0] = priMultMat[0]/priSum;
    posteriorP[1] = priMultMat[1]/priSum;
    posteriorP[2] = priMultMat[2]/priSum;
    posteriorP[3] = priMultMat[3]/priSum;

    //Update prior using S and Q
    Mat SMat[3];
    Mat normalizingMatS = Mat::zeros(image.rows, image.cols, CV_64F);
    Mat normalizingMatQ= Mat::zeros(image.rows, image.cols, CV_64F);
    for (int i = 0; i < 3; i++) {
        SMat[i] = Mat::zeros(image.rows, image.cols, CV_64F);
        cv::filter2D(imagePriors[i], SMat[i], -1, lambda0.clone(),Point(-1, -1), 0, BORDER_REPLICATE);
        SMat[i] = imagePriors[i].mul(SMat[i]);
        add(normalizingMatS, SMat[i], normalizingMatS);
        cv::filter2D(posteriorP[i], posteriorQ[i], -1, lambda0.clone(), Point(-1, -1), 0, BORDER_REPLICATE);
        posteriorQ[i] = posteriorP[i].mul(posteriorQ[i]);
        add(normalizingMatQ, posteriorQ[i], normalizingMatQ);
    }
    for (int i = 0; i < 3; i++) {
        SMat[i] = SMat[i].mul(1.0/normalizingMatS);
        cv::filter2D(SMat[i], SMat[i], -1, lambda1.clone(), Point(-1, -1), 0, BORDER_REPLICATE);
        posteriorQ[i] = posteriorQ[i].mul(1.0/normalizingMatQ);
        cv::filter2D(posteriorQ[i], posteriorQ[i], -1, lambda1.clone(), Point(-1, -1), 0, BORDER_REPLICATE);
        imagePriors[i] = (SMat[i] + posteriorQ[i])/4.0;
    }
}

/**
 * @brief updateGaussianParameters  The maximization step of the EM algortihm
 * @param image                     The current frame
 */
void updateGaussianParameters(Mat image) {
    Mat lambda, meanDiff, meanDiffT, featureSum;
    Mat meanOpt, covarOpt;
    //Use equations 10 and 11 from paper
    for (int i = 0; i < 3; i++) {
        //Used by both updates
       double Bk = cv::sum(posteriorQ[i])[0];
       //Update Covariance
       covarOpt = Mat::zeros(5,5, CV_64F);
       featureSum = Mat::zeros(1,5, CV_64F);
       Mat meanRepeated = cv::repeat(means[i], image.rows*image.cols,1);
       Mat posteriorQReshaped = posteriorQ[i].clone();
       posteriorQReshaped = posteriorQReshaped.reshape(0, image.rows*image.cols);
       posteriorQReshaped  = cv::repeat(posteriorQReshaped, 1, 5);
       meanDiff = imageFeatures-meanRepeated;

       //EIGEN NOT SO HELPFUL, TIME TO CONVERT IS TOO LONG
       //Eigen::MatrixXd X = Eigen::MatrixXd(meanDiff.rows,meanDiff.cols);
       //Eigen::MatrixXd Q = Eigen::MatrixXd(posteriorQReshaped.rows,posteriorQReshaped.cols);
       //cv::cv2eigen(meanDiff, X);
       //cv::cv2eigen(posteriorQReshaped, Q);
       //Eigen::MatrixXd result = (1.0/Bk)*(X.array()*Q.array()).matrix().transpose()*X;
       //cv::eigen2cv(result, covarOpt);

       transpose(meanDiff.mul(posteriorQReshaped), meanDiffT);
       covarOpt = (1.0/Bk)*meanDiffT*meanDiff;

       cv::reduce(imageFeatures.mul(posteriorQReshaped), featureSum, 0, CV_REDUCE_SUM, CV_64F);
       meanOpt = (1.0/Bk)*featureSum;

       assert(covarOpt.rows == 5 && covarOpt.cols == 5);
       assert(meanOpt.rows == 1 && meanOpt.cols == 5);

#if USE_WEAK_PRIOR == 1
       invert(icovars[i] + ipositionCovPriors[i], lambda);
       Mat intermediateTerm = meanOpt*icovars[i]+positionMeanPriors[i]*ipositionCovPriors[i];
       transpose(intermediateTerm, intermediateTerm);
       meanOpt = lambda*intermediateTerm;
       transpose(meanOpt, meanOpt);
#endif
        covars[i] = covarOpt.clone();
        means[i] = meanOpt.clone();
    }
}

/**
 * @brief findZonesAndObstacles   Draws the water/land/sky zone mapping. Also turns zones to water if enough of it lies beneath the shore line
 *                              Also finds all potential obstacles.
 * @param image                 The current frame
 * @param zoneMapping           Map that will contain the output zones (sky, land, water)
 * @param obstacleMap           Map that will contain the detected potential obstacles
 * @param display               If true, displays obstacles
 */
void findZonesAndObstacles(Mat image, Mat& zoneMapping, Mat& obstacleMap, bool display) {
    zoneMapping = image.clone();
    for (int i = 0; i < 3; i++) {
        GaussianBlur(posteriorP[i],posteriorP[i],Size(3,3), 3.0, 3.0, BORDER_REPLICATE);
    }
    obstacleMap = Mat::zeros(zoneMapping.rows, zoneMapping.cols, CV_8U);
    int totalRed = 0;
    int redUnder = 0;
    int totalGreen = 0;
    int greenUnder = 0;
    double* posteriorRows[4];
    for (int row = 0; row < image.rows; row++) {
        posteriorRows[0] = posteriorP[0].ptr<double>(row);
        posteriorRows[1] = posteriorP[1].ptr<double>(row);
        posteriorRows[2] = posteriorP[2].ptr<double>(row);
        posteriorRows[3] = posteriorP[3].ptr<double>(row);
        unsigned char* obstacleMapRow = obstacleMap.ptr<uint8_t>(row);
        for (int col = 0; col < image.cols; col++) {
            Vec3b color;
            double probability = 0.0;
            int maxIndex = -1;
            for (int i = 0; i < 3; i++) {
                double curProb = posteriorRows[i][col];
                if (curProb > probability) {
                    probability = curProb;
                    maxIndex = i;
                }
            }
            if (posteriorRows[3][col] > probability)
                obstacleMapRow[col] = 255;

            int curHPoint = (rightIntercept-leftIntercept)*((double)col/image.cols)+leftIntercept;
            switch(maxIndex) {
            case 0:
                color = Vec3b(0, 0, 255);
                totalRed++;
                if (curHPoint < row)
                    redUnder++;
                break;
            case 1:
                color = Vec3b(0, 255, 0);
                totalGreen++;
                if (curHPoint < row)
                    greenUnder++;
                break;
            case 2:
                //if (probability > .995) (choose top 10 in every 100 x 100 ush, with a lower bound
                 color = Vec3b(255, 0, 0);
                break;
            }
            zoneMapping.at<Vec3b>(row,col) = color;
        }
    }
    //go through zonemappings, if green or red lie mostly below the horizon line, then turn them blue instead
    //This only matters for two reasons
    // a) If we decide to use the zones to determine shoreline instead of just defaulting to the horizon line
    // b) For detecting green/red colored areas that are islands in the blue areas
    //      If enough of the green/red area is under neath the horizon line we can assume that it is modeling water
    //      Thus we should not model any green/red islands as obstacles since they are likely still just water
    double redUnderRatio = (double)redUnder/totalRed;
    double greenUnderRatio = (double)greenUnder/totalGreen;
    for (int row = 0; row < zoneMapping.rows; row++) {
        unsigned char *obstaclePtr = obstacleMap.ptr<uint8_t>(row);
        for (int col = 0; col < zoneMapping.cols; col++) {
            Vec3b& color = zoneMapping.at<Vec3b>(row,col);
            if (color[1] == 255) {
                if (greenUnderRatio > 0.5) {
                    color[1] = 0;
                    color[0] = 255;
                } else if (obstaclePtr[col] != 255){
                    obstaclePtr[col] = 128;
                }
            } else if (color[2] == 255) {
                if (redUnderRatio > 0.5) {
                    color[2] = 0;
                    color[0] = 255;
                } else if (obstaclePtr[col] != 255){
                    obstaclePtr[col] = 128;
                }
            }
        }
    }

    if (display) {
        imshow("Obstacle Map", obstacleMap);
    }
}

/**
 * @brief findShoreLine     Finds the vector of row positions (indexed by column positions) to be considered as the boundary with water
 * @param coloredImage      The current frame
 * @param shoreLine         The vector that stores the water line row positions
 * @param useHorizon        If true we use the shoreline defined by the model, otherwise just use horizon line information
 * @param display           If true we display the resulting line on the zone mapping
 */
void findShoreLine(Mat coloredImage, std::map<int, int>& shoreLine, bool useHorizon, bool display) {
    if (useHorizon) {
        for (int col = 0; col < coloredImage.cols; col++) {
            int curHPoint = (rightIntercept-leftIntercept)*((double)col/coloredImage.cols)+leftIntercept;
            coloredImage.at<Vec3b>(curHPoint,col) = Vec3b(0,0,0);
            shoreLine[col] = curHPoint;
        }
    } else {
        float areaRatioLimit = 0.1;
        cv::Scalar lowerb = cv::Scalar(0,0,0);
        cv::Scalar upperb = cv::Scalar(255,0,0);
        Mat mask;
        cv::inRange(coloredImage, lowerb, upperb, mask);
        Mat labels;
        Mat stats;
        Mat centroids;
        int connectedCount = cv::connectedComponentsWithStats(mask, labels, stats, centroids);
        Mat waterBinary = Mat::zeros(mask.rows, mask.cols, CV_8U);
        for (int label = 0; label < connectedCount; label++) {
            Mat temp;
            int area = stats.at<int>(label, CC_STAT_AREA);
            if ((float)area/(mask.rows * mask.cols) > areaRatioLimit) {
                cv::inRange(labels, Scalar(label), Scalar(label), temp);
                add(waterBinary, temp, waterBinary,mask);
            }
        }
        //Anything white component that is above whose lowest pixel starts at the black line or below is considered as an obstacle
        for (int col = 0; col < waterBinary.cols; col++) {
            int row;
            for (row = 0; row < waterBinary.rows; row++) {
                if (waterBinary.at<uint8_t>(row, col) == 255) {
                    break;
                }
            }
            row = std::max(--row,0);
            coloredImage.at<Vec3b>(row,col) = Vec3b(0,0,0);
            shoreLine[col] = row;
        }
    }

    if (display)
        imshow("Water Zone", coloredImage);
}


/**
 * @brief findObstaclesInWater  Find all obstacles that are under the water line
 * @param shoreLine             The water line, calculated from findShoreLine
 * @param obstacles             The full obstacle map
 * @param obstaclesInWater      The resulting filtered obstacle map
 * @param display               If true, display binary map of obstacles in water
 */
void findObstaclesInWater(std::map<int, int>& shoreLine, Mat& obstacles, Mat& obstaclesInWater, bool display) {
    obstaclesInWater = Mat::zeros(obstacles.rows, obstacles.cols, CV_8U);
    Mat uniformObstacles = Mat::zeros(obstacles.rows, obstacles.cols, CV_8U);
    Mat nonUniformObstacles = Mat::zeros(obstacles.rows, obstacles.cols, CV_8U);
    uniformObstacles.setTo(255, obstacles==255);
    nonUniformObstacles.setTo(255, obstacles == 128);
    Mat labels;
    Mat stats;
    Mat centroids;
    Mat temp;
    int connectedCount = cv::connectedComponentsWithStats(uniformObstacles, labels, stats, centroids);
    //Find obstacles defined by the uniform component where the bottom edge of the blob in question must be underneath the shore line
    for (int label = 0; label < connectedCount; label++) {
         int* statsForLabel = stats.ptr<int>(label);
         int bottomRow = statsForLabel[CC_STAT_TOP]+statsForLabel[CC_STAT_HEIGHT]-1;
         uint8_t* obsRowBot = uniformObstacles.ptr<uint8_t>(bottomRow);
         bool obstacleDetected = false;
         for (int col = statsForLabel[CC_STAT_LEFT]; col < statsForLabel[CC_STAT_LEFT]+statsForLabel[CC_STAT_WIDTH]; col++) {
            if (obsRowBot[col] == 255 && bottomRow > shoreLine[col]) {
                obstacleDetected = true;
                break;
            }
         }
         if (obstacleDetected) {
             cv::inRange(labels, Scalar(label), Scalar(label), temp);
             add(obstaclesInWater, temp, obstaclesInWater, uniformObstacles);
         }
    }
    //Find obstacles defined by regions of non water and non uniform areas contained in the water area (must be fully contained)
    connectedCount = cv::connectedComponentsWithStats(nonUniformObstacles, labels, stats, centroids);
    for (int label = 0; label < connectedCount; label++) {
         int* statsForLabel = stats.ptr<int>(label);
         uint8_t* obsRowTop = nonUniformObstacles.ptr<uint8_t>(statsForLabel[CC_STAT_TOP]);
         bool obstacleDetected = true;
         for (int col = statsForLabel[CC_STAT_LEFT]; col < statsForLabel[CC_STAT_LEFT]+statsForLabel[CC_STAT_WIDTH]; col++) {
            if (obsRowTop[col] == 255 && statsForLabel[CC_STAT_TOP] < shoreLine[col]) {
                obstacleDetected = false;
                break;
            }
         }
         if (obstacleDetected) {
             cv::inRange(labels, Scalar(label), Scalar(label), temp);
             add(obstaclesInWater, temp, obstaclesInWater, nonUniformObstacles);
         }
    }
    if (display)
        imshow("Obstacles in Water", obstaclesInWater);
}

int main(int argc, char *argv[])
{
    std::cout << "Path name: " << argv[1] <<std::endl;
    std::cout << "Image name: " << argv[2] <<std::endl;
    std::cout << "Output folder: " << argv[3] <<std::endl;
    std::cout << "Wait string: " << argv[4] << std::endl;
    imageFolder = std::string(argv[1]);
    name = std::string(argv[2]);
    output = std::string(argv[3]);
    waitString = std::string(argv[4]);
    bool wait = (waitString == "WAIT");

    std::ifstream horizonFile;

    Mat originalImage;

    originalImage = imread(imageFolder+name, CV_LOAD_IMAGE_COLOR);

    scoreFile.open(output+name+std::string(".txt"));

    if (!originalImage.data)
    {
        printf("No image data \n");
        return -1;
    }

    //Get horizon information
    horizonFile.open(imageFolder+name+std::string("_horizon.txt"));
    std::string line;
    std::getline(horizonFile, line);
    std::istringstream iss(line);
    int hLeftIntercept, hRightIntercept, hWidth, hHeight;
    iss >> hLeftIntercept >> hRightIntercept >> hWidth >> hHeight;

    horizonFile.close();
    float scale = .25;
    Size size(scale*originalImage.cols, scale*originalImage.rows);
    Mat image;
    resize(originalImage, image, size);

    leftIntercept = hLeftIntercept*image.rows/(double)hHeight;
    rightIntercept = hRightIntercept*image.rows/(double)hHeight;

    //Initialize kernel info once
    int kernelWidth = (2*((int)(.1*image.rows)))+1;
    Mat kern = getGaussianKernel(kernelWidth, kernelWidth/1.5);
    Mat kernT;
    transpose(kern, kernT);
    kern2d = kern*kernT;
    lambda0 = kern2d.clone();
    lambda0.at<double>(kernelWidth/2, kernelWidth/2) = 0.0;
    double zeroSum = 1.0/cv::sum(lambda0)[0];
    lambda0 = lambda0.clone()*zeroSum;
    lambda1 = lambda0.clone();
    lambda1.at<double>(kernelWidth/2,kernelWidth/2) = 1.0;

    Mat zones;
    Mat obstacles;
    std::map<int,int> shoreLine;
    bool useHorizon = true;
    Mat obstaclesInWater;
    Mat totalDiff;
    Mat sqrtOldP, sqrtNewP;

    initializePriorsAndPosteriorStructures(image);

    int64 t1 = getTickCount();
    //Initialize model
    cvtColor(image, image, CV_BGR2HSV);
    setDataFromFrame(image);
    initializeLabelPriors(image, false);
    initializeGaussianModels(image);

    //Run EM iterations until convergence
    int iter = 0;
    while (iter < 5) {
        oldPriors[0] = imagePriors[0].clone();
        oldPriors[1] = imagePriors[1].clone();
        oldPriors[2] = imagePriors[2].clone();

        updatePriorsAndPosteriors(image);
        //Now check for convergence
        totalDiff = Mat::zeros(image.rows, image.cols, CV_64F);
        for (int i = 0; i < 3; i++) {
            cv::sqrt(oldPriors[i], sqrtOldP);
            cv::sqrt(imagePriors[i], sqrtNewP);
            totalDiff = totalDiff + sqrtOldP-sqrtNewP;
        }
        //sort totalDiff in ascending order and take mean of second half
        totalDiff = totalDiff.reshape(0,1);
        cv::sort(totalDiff, totalDiff, CV_SORT_DESCENDING);
        double meanDiff = cv::sum(totalDiff(Range(0,1), Range(0, totalDiff.cols/2)))[0]/(totalDiff.cols/2);
        if (meanDiff <= 0.01) {
            break;
        }
        updateGaussianParameters(image);
        iter++;
    }
    //Find the different regions and obstacles
    findZonesAndObstacles(image, zones, obstacles, false);
    findShoreLine(zones, shoreLine, useHorizon, false);
    findObstaclesInWater(shoreLine, obstacles, obstaclesInWater, false);
    int64 t2 = getTickCount();
    std::cout << (t2-t1)/getTickFrequency() << std::endl;

    resize(obstaclesInWater, obstaclesInWater, Size(originalImage.cols, originalImage.rows),0,0,INTER_NEAREST);
    findContoursAndWriteResults(obstaclesInWater, originalImage, true);
    if (wait)
        waitKey(0);
    scoreFile.close();
    return 0;
}
