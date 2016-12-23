#include <iostream>
#include <stdio.h>
#include <opencv2/opencv.hpp>

using namespace cv;

#define VIDEO 0

//Weak prior
Mat positionMeanPriors[3];
Mat positionCovPriors[3];
Mat ipositionCovPriors[3];

//M Step update
Mat means[3];
Mat covars[3];
Mat icovars[3];

//E step update
Mat imagePriors[3]; //We don't actually need a prior for the uniform component
Mat oldPosteriorP[4];
Mat posteriorP[4];
Mat posteriorQ[4];

//Y-data
Mat imageFeatures;

//Step 1
Mat kern2d;
Mat lambda0;
Mat lambda1;

double uniformComponent;

void initializePriorsAndPosteriorStructures(Mat image) {
    for (int i = 0; i < 4; i++) {
        if (i < 3)
            imagePriors[i] = Mat::zeros(image.rows, image.cols, CV_64F);
        posteriorP[i] = Mat::zeros(image.rows, image.cols, CV_64F);
        oldPosteriorP[i] = Mat::zeros(image.rows, image.cols, CV_64F);
        posteriorQ[i] = Mat::zeros(image.rows, image.cols, CV_64F);
    }

    //From the last 25 images in the Test Media set
    positionMeanPriors[0] = (Mat_<double>(1,5) <<
                             63.4862123463779,
                             12.4609487415375,
                             183.211685891539,
                             136.869671879791,
                             121.029556766447);
    positionCovPriors[0] = (Mat_<double>(5,5) <<
                            1306.45919492160,	-0.671543580406008,	288.809061389026,	-22.0480438947505,	7.91896527481572,
                            -0.671543580406008,	79.6475655935463,	65.0266429041669,	-20.1604608936357,	16.3323842052004,
                            288.809061389026,	65.0266429041669,	787.448913257439,	-57.0199904569081,	38.4255226830039,
                            -22.0480438947505,	-20.1604608936357,	-57.0199904569081,	77.8934453938443,	-55.1246154667244,
                            7.91896527481572,	16.3323842052004,	38.4255226830039,	-55.1246154667244,	42.3979988057096);

    positionMeanPriors[1]  = (Mat_<double>(1,5) <<
                              62.0588558982002,
                              19.4825633383010,
                              141.063372692881,
                              133.800940043563,
                              121.761068439757);
    positionCovPriors[1] = (Mat_<double>(5,5) <<
                            1296.85582420574,	-1.90516210596181,	138.250110567647,	-25.2637229580370,	15.1450070462499,
                            -1.90516210596181,	128.451471761792,	6.90423246968720,	5.26878100846356,	-5.78264335881975,
                            138.250110567647,	6.90423246968720,	1840.63834635385,	88.0459990305772,	-50.8091345248656,
                            -25.2637229580370,	5.26878100846356,	88.0459990305772,	109.258122654611,	-63.6579401982398,
                            15.1450070462499,	-5.78264335881975,	-50.8091345248656,	-63.6579401982398,	46.6681591739053);

    positionMeanPriors[2] = (Mat_<double>(1,5) <<
                             63.1211977439997,
                             52.2657027464609,
                             124.217469175706,
                             134.258633471121,
                             121.632972950533);
    positionCovPriors[2] = (Mat_<double>(5,5) <<
                            1301.92942071277,	-2.45330752316907,	74.3889535982663,	0.461727038858542,	6.66619754224339,
                            -2.45330752316907,	310.466989998298,	-229.579625029713,	4.96049427828225,	-4.63799671519768,
                            74.3889535982663,	-229.579625029713,	1830.39435535653,	-26.6999266425045,	-2.92121438726508,
                            0.461727038858542,	4.96049427828225,	-26.6999266425045,	47.5768906528705,	-49.1988005312233,
                            6.66619754224339,	-4.63799671519768,	-2.92121438726508,	-49.1988005312233,	88.6729064678893);

    invert(positionCovPriors[0], ipositionCovPriors[0]);
    invert(positionCovPriors[1], ipositionCovPriors[1]);
    invert(positionCovPriors[2], ipositionCovPriors[2]);
    uniformComponent = 1.0/(image.rows*image.cols*10988544.0)*0.001;
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
    for (i = 0; i < imageFeatures.rows; i++) {
        if (imageFeatures.at<double>(i, 1) < .2*image.rows) {
            indexToRegion[i] = std::pair<int, int>(0, skyCount);
            skyCount++;
        } else if (imageFeatures.at<double>(i, 1)  < .4*image.rows) {
            indexToRegion[i] = std::pair<int, int>(1, landCount);
            landCount++;
        } else if (imageFeatures.at<double>(i, 1)  > .6*image.rows) {
            indexToRegion[i] = std::pair<int, int>(2, waterCount);
            waterCount++;
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
    Mat SMat[4];
    for (int i = 0; i < 3; i++) {
        cv::filter2D(imagePriors[i], SMat[i], -1, lambda0,Point(-1, -1), 0, BORDER_REPLICATE);
        SMat[i] = imagePriors[i].mul(SMat[i]);
        SMat[i] = SMat[i]*(1.0/cv::sum(SMat[i])[0]);
        cv::filter2D(SMat[i], SMat[i], -1, lambda1, Point(-1, -1), 0, BORDER_REPLICATE);

        cv::filter2D(posteriorP[i], posteriorQ[i], -1, lambda0, Point(-1, -1), 0, BORDER_REPLICATE);
        posteriorQ[i] = posteriorP[i].mul(posteriorQ[i]);
        posteriorQ[i] = posteriorQ[i]*(1.0/cv::sum(posteriorQ[i])[0]);
        cv::filter2D(posteriorQ[i], posteriorQ[i], -1, lambda1, Point(-1, -1), 0, BORDER_REPLICATE);

        imagePriors[i] = (SMat[i] + posteriorQ[i])/4.0;
    }
}

void updateGaussianParameters(Mat image) {
    Mat iMeanCovar, iCovar, lambda, meanDiff, meanDiffT, featureSum;
    Mat meanOpt, covarOpt;
    //Use equations 10 and 11
    for (int i = 0; i < 3; i++) {
        //Used by both updates
       double Bk = 1.0/cv::sum(posteriorQ[i])[0];

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

       covarOpt = (1.0/Bk)*covarOpt;

       assert(covarOpt.rows == 5 && covarOpt.cols == 5);
       assert(featureSum.rows == 1 && featureSum.cols == 5);

        //Update mean
        invert(icovars[i] + ipositionCovPriors[i], lambda);
        Mat intermediateTerm = featureSum*icovars[i]+positionMeanPriors[i]*ipositionCovPriors[i];
        transpose(intermediateTerm, intermediateTerm);
        meanOpt = (1.0/Bk)*lambda*intermediateTerm;
        transpose(meanOpt, meanOpt);
        covars[i] = covarOpt;
        means[i] = meanOpt;
    }
}

void findShoreLine(Mat coloredImage, std::map<int, int>& shoreLine, bool display) {
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
    //TODO: Now draw the line (just for visualization, this should be combined in one loop with object detection)
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

    if (display)
        imshow("shoreLine", waterBinary);
}

void drawMapping(Mat image, Mat& zoneMapping, Mat& obstacleMap, bool display) {
    zoneMapping = image.clone();
    for (int i = 0; i < 4; i++) {
        filter2D(posteriorP[i], posteriorP[i], -1, kern2d, Point(-1, -1), 0, BORDER_REPLICATE);
    }
    obstacleMap = Mat::zeros(zoneMapping.rows, zoneMapping.cols, CV_8U);
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

            switch(maxIndex) {
            case 0:
                color = Vec3b(0, 0, 255);
                break;
            case 1:
                color = Vec3b(0, 255, 0);
                break;
            case 2:
                //if (probability > .995) (choose top 10 in every 100 x 100 ush, with a lower bound
                 color = Vec3b(255, 0, 0);
                break;
            }
            zoneMapping.at<Vec3b>(row,col) = color;
        }
    }

    if (display) {
        imshow("small with obs", obstacleMap);
        waitKey(1);
        imshow("small no obs", zoneMapping);
        waitKey(1);
    }
}

void findObstacles(std::map<int, int>& shoreLine, Mat& obstacles, Mat& obstaclesInWater, bool display) {
    //simply return any white connected white blobs that are under the zone shift
    //do it by scanning up, once the line has been passed, switch on a flag such that it tends once the current run ends
    obstaclesInWater = Mat::zeros(obstacles.rows, obstacles.cols, CV_8U);

    Mat labels;
    Mat stats;
    Mat centroids;
    int connectedCount = cv::connectedComponentsWithStats(obstacles, labels, stats, centroids);
    for (int label = 0; label < connectedCount; label++) {
         int* statsForLabel = stats.ptr<int>(label);
         int top = statsForLabel[CC_STAT_TOP];
         int left = statsForLabel[CC_STAT_LEFT];
         int width = statsForLabel[CC_STAT_WIDTH];
         int height = statsForLabel[CC_STAT_HEIGHT];
         int bottomRow = top+height-1;
         uint8_t* obsRow = obstacles.ptr<uint8_t>(bottomRow);
         for (int col = left; col < left+width; col++) {
             //If any part of the blob is below the shoreLine, consider it a water obstacle
            if (obsRow[col] == 255 && bottomRow > shoreLine[col]) {
                Mat temp;
                cv::inRange(labels, Scalar(label), Scalar(label), temp);
                add(obstaclesInWater, temp, obstaclesInWater, obstacles);
                break;
            }
         }
    }
    if (display)
        imshow("obs in water", obstaclesInWater);

}

int main(int argc, char *argv[])
{
#if VIDEO == 0
    Mat image;
    image = imread("../../TestMedia/images/21.jpg", CV_LOAD_IMAGE_COLOR);
    if (!image.data)
    {
        printf("No image data \n");
        return -1;
    }


    float scale = .25;
    Size size(scale*image.cols, scale*image.rows);
    resize(image, image, size);

    //Initialize kernel info once
    int kernelWidth = (2*((int)(.02*image.rows)))+1;
    Mat kern = getGaussianKernel(kernelWidth, 1.0);
    Mat kernT;
    transpose(kern, kernT);
    kern2d = kern*kernT;
    lambda0 = kern2d.clone();
    lambda0.at<double>(kernelWidth, kernelWidth) = 0.0;
    double zeroSum = (1.0/cv::sum(lambda0)[0]);
    lambda0 = lambda0.clone()*zeroSum;

    lambda1 = lambda0.clone();
    lambda1.at<double>(kernelWidth,kernelWidth) = 1.0;

    initializePriorsAndPosteriorStructures(image);

    int64 t1 = getTickCount();
    //Initialize model
    imshow("Orig", image);
    cvtColor(image, image, CV_BGR2YCrCb);
    setDataFromFrame(image);
    initializeLabelPriors(image);
    initializeGaussianModels(image);

    int iter = 0;
    while (iter < 10) {
        //Swap pointers to old and new posterior predictions
        cv::swap(oldPosteriorP[0], posteriorP[0]);
        cv::swap(oldPosteriorP[1], posteriorP[1]);
        cv::swap(oldPosteriorP[2], posteriorP[2]);
        cv::swap(oldPosteriorP[3], posteriorP[3]);
        updatePriorsAndPosteriors(image);

        //drawMapping(image);
        //Now check for convergence
        Mat totalDiff = Mat::zeros(image.rows, image.cols, CV_64F);
        for (int i = 0; i < 4; i++) {
            Mat sqrtOldP, sqrtNewP;
            cv::sqrt(oldPosteriorP[i], sqrtOldP);
            cv::sqrt(posteriorP[i], sqrtNewP);
            totalDiff = totalDiff + cv::abs(sqrtOldP-sqrtNewP);
        }
        //sort totalDiff in ascending order and take mean of second half
        totalDiff = totalDiff.reshape(0,1);
        cv::sort(totalDiff, totalDiff, CV_SORT_DESCENDING);
        double meanDiff = cv::sum(totalDiff(Range(0,1), Range(0, totalDiff.cols/2)))[0]/(totalDiff.cols/2);
        if (meanDiff <= 0.01)
            break;
        updateGaussianParameters(image);
        iter++;
    }

    int64 t2 = getTickCount();
    std::cout << (t2-t1)/getTickFrequency() << std::endl;
    //TODO: optimization of cov update loop, more optimization, eigen to mkl
    Mat zones;
    Mat obstacles;
    drawMapping(image, zones, obstacles, true);
    std::map<int,int> shoreLine;
    findShoreLine(zones, shoreLine, true);
    Mat obstaclesInWater;
    findObstacles(shoreLine, obstacles, obstaclesInWater, true);

    //Mat bigZones;
    //resize(zones, bigZones, Size(image.cols*4, image.rows*4));
    //imshow("final", bigZones);
    waitKey(0);
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
