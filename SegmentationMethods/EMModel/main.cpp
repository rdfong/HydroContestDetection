#include <iostream>
#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <fstream>

using namespace cv;

#define VIDEO 0

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

 //File setup
 std::ofstream scoreFile;
 std::string name;
 std::string output;
 std::string waitString;
 std::string imageFolder;

int hLeftIntercept, hRightIntercept, hWidth, hHeight;

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


void findContoursAndWriteResults(Mat& obstacleMap, Mat& image, bool display) {
    hierarchy.clear();
    contours.clear();
    boundRects.clear();
    findContours( obstacleMap.clone(), contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE, Point(0, 0) );
    //get bounding rects from contours
    int expand = 3;
    for (int i =0; i < contours.size(); i++) {
        curRect = boundingRect(contours[i]);
        //don't add a box if it is too long or too tall in aspect ratio
        if ((double)curRect.width/curRect.height < 0.1 || (double)curRect.height/curRect.width < 0.1)
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

void initializeLabelPriors(Mat image) {
    //from their implementation for ycrcb
    double gaussianComponent = (1.0-uniformComponent)/3.0;
    for (int row = 0; row < image.rows; row++) {
        for (int col = 0; col < image.cols; col++) {
            imagePriors[0].at<double>(row,col) = gaussianComponent;
            imagePriors[1].at<double>(row,col) = gaussianComponent;
            imagePriors[2].at<double>(row,col) = gaussianComponent;
        }
    }
}

void initializeGaussianModels(Mat image) {
    //If we have confident horizon line estimatse we can have good estimates
    //If do not however we assume the height spread for the gaussians, 0.0-0.2, 0.2-0.4,0.6-1.0
    //It is assumed here that the horizon line lies between 0.4 and 0.6
    
    //We start with this implementing under this assumption, if a better horizon line detection comes up, we'll use it
    int skyCount = 0;
    int landCount = 0;
    int waterCount = 0;

    std::map<int,std::pair<int, int> > indexToRegion;
    int i;

    int leftIntercept = hLeftIntercept*image.rows/(double)hHeight;
    int rightIntercept = hRightIntercept*image.rows/(double)hHeight;
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
        } else if (hMidPercent <= .1) {
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

    cv::calcCovarMatrix(regionMats[0], covars[0], means[0], CV_COVAR_NORMAL | CV_COVAR_ROWS | CV_COVAR_SCALE);
    cv::calcCovarMatrix(regionMats[1], covars[1], means[1], CV_COVAR_NORMAL | CV_COVAR_ROWS | CV_COVAR_SCALE);
    cv::calcCovarMatrix(regionMats[2], covars[2], means[2], CV_COVAR_NORMAL | CV_COVAR_ROWS | CV_COVAR_SCALE);
}

void setDataFromFrame(Mat image) {
    imageFeatures = Mat::zeros(image.rows*image.cols, 5, CV_64F);
    int index = 0;
    for (int row = 0; row < image.rows; row++) {
        for (int col = 0; col < image.cols; col++) {
            cv::Vec3b color = image.at<cv::Vec3b>(row,col);
            imageFeatures.at<double>(index, 0) = col;
            imageFeatures.at<double>(index, 1) = row;
            imageFeatures.at<double>(index, 2) = color[0];
            imageFeatures.at<double>(index, 3) = color[1];
            imageFeatures.at<double>(index, 4) = color[2];
            index++;
        }
    }
}


void updatePriorsAndPosteriors(Mat image) {
    //Calculate Posterior
    double div0 = sqrt(cv::determinant(2*M_PI*covars[0]));
    double div1 = sqrt(cv::determinant(2*M_PI*covars[1]));
    double div2 = sqrt(cv::determinant(2*M_PI*covars[2]));
    int index = 0;
    //Make sure matrices are well conditioned
    int result0 = invert(covars[0]+Mat::eye(5,5,CV_64F).mul(covars[0])*1e-10, icovars[0]);
    int result1 = invert(covars[1]+Mat::eye(5,5,CV_64F).mul(covars[1])*1e-10, icovars[1]);
    int result2 = invert(covars[2]+Mat::eye(5,5,CV_64F).mul(covars[2])*1e-10, icovars[2]);
    assert(result0 && result1 && result2);
    for (int row = 0; row < image.rows; row++) {
        for (int col = 0; col < image.cols; col++) {
            double mah0 = cv::Mahalanobis(imageFeatures.row(index),means[0], icovars[0]);
            double mah1 = cv::Mahalanobis(imageFeatures.row(index),means[1], icovars[1]);
            double mah2 = cv::Mahalanobis(imageFeatures.row(index),means[2], icovars[2]);

            double pri0 = imagePriors[0].at<double>(row,col) * exp(-.5*mah0*mah0)/div0;
            double pri1 = imagePriors[1].at<double>(row,col) * exp(-.5*mah1*mah1)/div1;
            double pri2 = imagePriors[2].at<double>(row,col) * exp(-.5*mah2*mah2)/div2;
            double pri3 = uniformComponent;
            double priSum = pri0 + pri1 + pri2 + pri3;


            posteriorP[0].at<double>(row,col) = pri0/priSum;
            posteriorP[1].at<double>(row,col) = pri1/priSum;
            posteriorP[2].at<double>(row,col) = pri2/priSum;
            posteriorP[3].at<double>(row,col) = pri3/priSum;

            assert(posteriorP[0].at<double>(row,col) >= 0 &&
                    posteriorP[1].at<double>(row,col) >= 0 &&
                    posteriorP[2].at<double>(row,col) >= 0 &&
                    posteriorP[3].at<double>(row,col) >= 0);

            index++;
        }
    }

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

void updateGaussianParameters(Mat image) {
    Mat lambda, meanDiff, meanDiffT, featureSum;
    Mat meanOpt, covarOpt;
    //Use equations 10 and 11
    for (int i = 0; i < 3; i++) {
        //Used by both updates
       double Bk = cv::sum(posteriorQ[i])[0];
       //Update Covariance
       int index = 0;
       covarOpt = Mat::zeros(5,5, CV_64F);
       featureSum = Mat::zeros(1,5, CV_64F); //Used for mean update, may as well calculate in this loop

       for (int row = 0; row < image.rows; row++) {
           double *curRowQ = posteriorQ[i].ptr<double>(row);
           for (int col = 0; col < image.cols; col++) {
               meanDiff = imageFeatures.row(index)-means[i];
               transpose(meanDiff, meanDiffT);
               covarOpt = covarOpt+curRowQ[col]*meanDiffT*meanDiff;
               featureSum = featureSum + curRowQ[col]*imageFeatures.row(index);
               index++;
           }
       }
       meanDiff = imageFeatures.row(0)-means[i];
       covarOpt = (1.0/Bk)*covarOpt;
       assert(covarOpt.rows == 5 && covarOpt.cols == 5);
       assert(featureSum.rows == 1 && featureSum.cols == 5);

        //Update mean
       meanOpt = (1.0/Bk)*featureSum;
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

void findShoreLine(Mat coloredImage, std::map<int, int>& shoreLine, bool useHorizon, bool display) {
    if (useHorizon) {
        int leftIntercept = hLeftIntercept*coloredImage.rows/(double)hHeight;
        int rightIntercept = hRightIntercept*coloredImage.rows/(double)hHeight;
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

void drawMapping(Mat image, Mat& zoneMapping, Mat& obstacleMap, bool display) {
    zoneMapping = image.clone();
    for (int i = 0; i < 3; i++) {
        GaussianBlur(posteriorP[i],posteriorP[i],Size(3,3), 3.0, 3.0, BORDER_REPLICATE);
    }
    obstacleMap = Mat::zeros(zoneMapping.rows, zoneMapping.cols, CV_8U);
    int leftIntercept = hLeftIntercept*image.rows/(double)hHeight;
    int rightIntercept = hRightIntercept*image.rows/(double)hHeight;
    int totalRed = 0;
    int redUnder = 0;
    int totalGreen = 0;
    int greenUnder = 0;
    for (int row = 0; row < image.rows; row++) {
        for (int col = 0; col < image.cols; col++) {
            Vec3b color;
            double probability = 0.0;
            int maxIndex = -1;
            for (int i = 0; i < 3; i++) {
                double curProb = posteriorP[i].at<double>(row,col);
                if (curProb > probability) {
                    probability = curProb;
                    maxIndex = i;
                }
            }
            if (posteriorP[3].at<double>(row,col) > probability)
                obstacleMap.at<unsigned char>(row,col) = 255;

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
        for (int col = 0; col < zoneMapping.cols; col++) {
            Vec3b& color = zoneMapping.at<Vec3b>(row,col);
            if (color[1] == 255) {
                if (greenUnderRatio > 0.5) {
                    color[1] = 0;
                    color[0] = 255;
                } else if (obstacleMap.at<unsigned char>(row,col) != 255){
                    obstacleMap.at<unsigned char>(row,col) = 128;
                }
            } else if (color[2] == 255) {
                if (redUnderRatio > 0.5) {
                    color[2] = 0;
                    color[0] = 255;
                } else if (obstacleMap.at<unsigned char>(row,col) != 255){
                    obstacleMap.at<unsigned char>(row,col) = 128;
                }
            }
        }
    }

    if (display) {
        imshow("Obstacle Map", obstacleMap);
    }
}

void findObstacles(std::map<int, int>& shoreLine, Mat& obstacles, Mat& obstaclesInWater, bool display) {
    //simply return any white connected white blobs that are under the zone shift
    //do it by scanning up, once the line has been passed, switch on a flag such that it tends once the current run ends
    obstaclesInWater = Mat::zeros(obstacles.rows, obstacles.cols, CV_8U);
    Mat uniformObstacles = Mat::zeros(obstacles.rows, obstacles.cols, CV_8U);
    Mat nonUniformObstacles = Mat::zeros(obstacles.rows, obstacles.cols, CV_8U);
    uniformObstacles.setTo(255, obstacles==255);
    nonUniformObstacles.setTo(255, obstacles == 128);
    Mat labels;
    Mat stats;
    Mat centroids;
    int connectedCount = cv::connectedComponentsWithStats(uniformObstacles, labels, stats, centroids);
    for (int label = 0; label < connectedCount; label++) {
         int* statsForLabel = stats.ptr<int>(label);
         int top = statsForLabel[CC_STAT_TOP];
         int left = statsForLabel[CC_STAT_LEFT];
         int width = statsForLabel[CC_STAT_WIDTH];
         int height = statsForLabel[CC_STAT_HEIGHT];
         int bottomRow = top+height-1;
         uint8_t* obsRowBot = uniformObstacles.ptr<uint8_t>(bottomRow);
         bool obstacleDetected = false;
         for (int col = left; col < left+width; col++) {
            if (obsRowBot[col] == 255 && bottomRow > shoreLine[col]) {
                obstacleDetected = true;
                break;
            }
         }
         if (obstacleDetected) {
             Mat temp;
             cv::inRange(labels, Scalar(label), Scalar(label), temp);
             add(obstaclesInWater, temp, obstaclesInWater, uniformObstacles);
         }
    }
    //non uniform
    connectedCount = cv::connectedComponentsWithStats(nonUniformObstacles, labels, stats, centroids);
    for (int label = 0; label < connectedCount; label++) {
         int* statsForLabel = stats.ptr<int>(label);
         int top = statsForLabel[CC_STAT_TOP];
         int left = statsForLabel[CC_STAT_LEFT];
         int width = statsForLabel[CC_STAT_WIDTH];
         uint8_t* obsRowTop = nonUniformObstacles.ptr<uint8_t>(top);
         bool obstacleDetected = true;
         for (int col = left; col < left+width; col++) {
            if (obsRowTop[col] == 255 && top < shoreLine[col]) {
                obstacleDetected = false;
                break;
            }
         }
         if (obstacleDetected) {
             Mat temp;
             cv::inRange(labels, Scalar(label), Scalar(label), temp);
             add(obstaclesInWater, temp, obstaclesInWater, nonUniformObstacles);
         }
    }
    if (display)
        imshow("Obstacles in Water", obstaclesInWater);

}

int main(int argc, char *argv[])
{
#if VIDEO == 0
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
    iss >> hLeftIntercept >> hRightIntercept >> hWidth >> hHeight;

    horizonFile.close();
    float scale = .25;
    Size size(scale*originalImage.cols, scale*originalImage.rows);
    Mat image;
    resize(originalImage, image, size);

    //Initialize kernel info once
    int kernelWidth = (2*((int)(.08*image.rows)))+1;
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

    initializePriorsAndPosteriorStructures(image);

    int64 t1 = getTickCount();
    //Initialize model
    imshow("Orig", image);
    cvtColor(image, image, CV_BGR2HSV);
    setDataFromFrame(image);
    initializeLabelPriors(image);
    initializeGaussianModels(image);
    int iter = 0;
    while (iter < 10) {
        oldPriors[0] = imagePriors[0].clone();
        oldPriors[1] = imagePriors[1].clone();
        oldPriors[2] = imagePriors[2].clone();

        updatePriorsAndPosteriors(image);
        //Now check for convergence
        Mat totalDiff = Mat::zeros(image.rows, image.cols, CV_64F);
        for (int i = 0; i < 3; i++) {
            Mat sqrtOldP, sqrtNewP;
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

    //TODO: optimization of cov update loop, more optimization, eigen to mkl
    Mat zones;
    Mat obstacles;
    drawMapping(image, zones, obstacles, true);
    std::map<int,int> shoreLine;
    bool useHorizon = true;
    findShoreLine(zones, shoreLine, useHorizon, true);
    Mat obstaclesInWater;
    findObstacles(shoreLine, obstacles, obstaclesInWater, true);

    int64 t2 = getTickCount();
    std::cout << (t2-t1)/getTickFrequency() << std::endl;

    resize(obstaclesInWater, obstaclesInWater, Size(originalImage.cols, originalImage.rows),0,0,INTER_NEAREST);
    imshow("large obs", obstaclesInWater);
    findContoursAndWriteResults(obstaclesInWater, originalImage, true);
    if (wait)
        waitKey(0);
    scoreFile.close();
#elif VIDEO == 1

    VideoCapture cap("../../TestMedia/videos/boatm30.mp4"); // open the default camera
    if(!cap.isOpened()) {  // check if we succeeded
        std::cout << "no vid" << std::endl;
        return -1;
    }

   Mat image;
   for(;;)
   {
       cap >> image; // get a new frame fro m camera
       if (image.rows == 0 || image.cols == 0)
           continue;

         waitKey(1);
   }

#endif
    return 0;
}
