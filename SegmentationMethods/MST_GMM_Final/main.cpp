#include <iostream>
#include <stdio.h>
#include <fstream>

#include "MST.h"
#include "GMM.h"
#include "BoundingBoxes.h"

using namespace cv;

#define VIDEO 0

//File setup
std::ofstream scoreFile;
std::string name;
std::string output;
std::string waitString;
std::string imageFolder;
int leftIntercept, rightIntercept;

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
    int hLeftIntercept, hRightIntercept, hWidth, hHeight;
    iss >> hLeftIntercept >> hRightIntercept >> hWidth >> hHeight;
    horizonFile.close();

    /**********GMM CODE**********/
    float scale = .25;
    Size size(scale*originalImage.cols, scale*originalImage.rows);
    Mat image;
    resize(originalImage, image, size);

    leftIntercept = hLeftIntercept*image.rows/(double)hHeight;
    rightIntercept = hRightIntercept*image.rows/(double)hHeight;

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

    //For use by method
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
    drawMapping(image, zones, obstacles, false);
    findShoreLine(zones, shoreLine, useHorizon, false);
    findObstacles(shoreLine, obstacles, obstaclesInWater, false);
    int64 t2 = getTickCount();
    std::cout << (t2-t1)/getTickFrequency() << std::endl;

    resize(obstaclesInWater, obstaclesInWater, Size(originalImage.cols, originalImage.rows),0,0,INTER_NEAREST);
    findContoursAndWriteResults(obstaclesInWater, originalImage, true);

    /***********MST CODE***********/
    Mat scaledImage, gray_image, lab, mbd_image, dis_image, new_dis_image, combined, rawCombined, intermediate;

    resize(image, scaledImage, size);

    vNodes.resize(scaledImage.rows*scaledImage.cols);//should only ever call thisonce
    createVertexGrid(scaledImage.rows, scaledImage.cols);
    initializeDiffBins();

    cvtColor(scaledImage, gray_image, CV_BGR2GRAY );

    //TODO: this is a trade off, smaller farther away objects get fucked, maybe the solution is to not use this and have better background seeds
    GaussianBlur(gray_image, gray_image, Size(5, 5), 3);
    //more blur deals with open water better...
    // GaussianBlur(gray_image, gray_image, Size(7, 7), 5);
   // GaussianBlur(gray_image, gray_image, Size(7, 7), 5);
   // GaussianBlur(gray_image, gray_image, Size(7, 7), 5);
    //This messes with smaller and more difficult to distinguish objects. would rather not remove information, maybe blur just the edges
    updateVertexGridWeights(gray_image);
    createMST(gray_image);
    passUp();
    passDown();

    cvtColor(scaledImage, lab, CV_BGR2Lab);
    int boundary_size = 20;
    int num_boundary_pixels = (boundary_size*2*(gray_image.cols+gray_image.rows)-4*boundary_size*boundary_size);
    std::vector<cv::Point3f> boundaryPixels(num_boundary_pixels);
    mbd_image = Mat::zeros(gray_image.rows, gray_image.cols, CV_32FC1);
    getMBDImageAndBoundaryPix(lab, mbd_image, boundaryPixels, boundary_size);

    dis_image = Mat::zeros(lab.rows, lab.cols, CV_32FC1);
    getDissimiliarityImage(boundaryPixels, lab, dis_image);
    treeFilter(dis_image, mbd_image, 5, 0.5);
    new_dis_image = Mat::zeros(lab.rows, lab.cols, CV_32FC1);
    bilateralFilter(dis_image, new_dis_image, 5, 0.5, 0.5);

    //combine images
    rawCombined = mbd_image + new_dis_image;

    double minVal;
    double maxVal;
    cv::minMaxLoc(rawCombined, &minVal, &maxVal);
    rawCombined /= maxVal;


    // POST PROCESSING FROM PAPER
    combined = rawCombined*255;
    combined.convertTo(combined, CV_8U);

    double tau = threshold(combined, intermediate, 0, 255, THRESH_OTSU);
    int gamma = 20;
    cv::exp(-gamma*(rawCombined-tau/255.0), intermediate);
    combined = 1.0/(1.0+intermediate);

     combined*=255;
     combined.convertTo(combined, CV_8U);

   customOtsuThreshold(combined);

   if (maxVal-minVal > 0.75) {
         contours.clear();
         boundRects.clear();
         findContours( combined.clone(), contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE, Point(0, 0) );
         //get bounding rects from contours
         int expand = 3;
         for (int i =0; i < contours.size(); i++) {
             curRect = boundingRect(contours[i]);
             meanStdDev(rawCombined(curRect), mean, std);
             if ((std.at<double>(0,0) < 0.1 && curRect.area() >= (.25*combined.rows*combined.cols)) ||
                     (double)curRect.width/curRect.height < 0.1 ||
                     (double)curRect.height/curRect.width < 0.1)
                 continue;
             Point2i newTL(max(curRect.tl().x-expand, 0), max(curRect.tl().y-expand,0));
             Point2i newBR(min(curRect.br().x+expand, combined.cols-1), min(curRect.br().y+expand,combined.rows-1));
             boundRects.push_back(Rect(newTL, newBR));
             originalRects.push_back(curRect);
         }

         //intersection groups mirros finalboxbounds in size, but final box bounds contains information on originalRects, intersection groups is for the expanded rects
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
                           combined(curRect).copyTo(mask1);
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

                           combined(otherRect).copyTo(mask2);
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
               rectangle(scaledImage, Rect(finalBoxBounds[i].first, finalBoxBounds[i].second), Scalar(0, 255,0), 2);
               scoreFile << "other\n" << curRect.tl().x << " " << curRect.tl().y << " "
                                   << curRect.width << " " << curRect.height <<std::endl;
             }
         }

         imwrite(output+name, scaledImage);
   }

    int64 t2 = getTickCount();
   imshow("mbd", mbd_image);
   imshow("dis_post", new_dis_image);
  // imshow("frei", frei_image);
   imshow("combined", combined);
   imshow("final result", scaledImage);
   std::cout << "PER FRAME TIME: " << (t2 - t1)/getTickFrequency() << std::endl;

    //**************END MAIN CODE SECTION*************//
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
