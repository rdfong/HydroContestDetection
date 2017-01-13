#include <iostream>
#include <stdio.h>
#include <fstream>

#include "MST.h"
#include "GMM.h"
#include "BoundingBoxes.h"

using namespace cv;

//File setup variables
std::ofstream scoreFile;
std::string name;
std::string output;
std::string waitString;
std::string imageFolder;

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

    Mat originalImage;
    originalImage = imread(imageFolder+name, CV_LOAD_IMAGE_COLOR);
    scoreFile.open(output+name+std::string(".txt"));

    if (!originalImage.data)
    {
        printf("No image data \n");
        return -1;
    }


    imshow("Original", originalImage);
    int cols = originalImage.cols;
    int rows = originalImage.rows;

    //Initialize things for both GMM and MST
    //GMM Initialization
    Mat seedNodes, obstacles, GMMimage;
    float scale = .25;
    Size size(scale*cols, scale*rows);
    resize(originalImage, GMMimage, size);
    initializeKernelInfo(GMMimage);
    initializePriorsAndPosteriorStructures(GMMimage);
    std::vector<int> shoreLine;
    shoreLine.resize(GMMimage.cols);

    //MST Initialization
    Mat gray_image, hsv, mst_image, dis_image, new_dis_image, combined;
    mst_image = Mat::zeros(rows, cols, CV_32FC1);
    dis_image = Mat::zeros(rows, cols, CV_32FC1);
    new_dis_image = Mat::zeros(rows, cols, CV_32FC1);
    std::vector<int> horizonLine;
    horizonLine.resize(originalImage.cols);
    int boundary_size = 20;
    int num_boundary_pixels = (boundary_size*2*(rows+cols)-4*boundary_size*boundary_size);
    std::vector<cv::Point3f> boundaryPixels(num_boundary_pixels);
    createVertexGrid(rows, cols);
    initializeDiffBins();

    int t1 = getTickCount();
    //PER IMAGE GMM/MST Code
    /**********GMM CODE**********/
    //Get boundary dissimiliarity image here for use by GMM Seed node finding
    cvtColor(originalImage, hsv, CV_BGR2HSV);
    getBoundaryPix(hsv, boundaryPixels, boundary_size);
    getDissimiliarityImage(boundaryPixels, hsv, dis_image);
    treeFilter(dis_image, mst_image, 4, 0.5);
    bilateralFilter(dis_image, new_dis_image, 4, 0.5, 0.5);

    //Initialize model
    cvtColor(GMMimage, GMMimage, CV_BGR2HSV);
    setDataFromFrame(GMMimage);
    parseHorizonInfo(GMMimage, imageFolder+name+std::string("_horizon.txt"));
    initializeLabelPriors(GMMimage, false);
    initializeGaussianModels(GMMimage);
    runEM(GMMimage);
    Mat disImageClone = new_dis_image.clone();
    findSeedNodes(GMMimage, disImageClone, seedNodes, true);
    resize(seedNodes, seedNodes, Size(originalImage.cols, originalImage.rows), 0, 0, INTER_NEAREST);
    imshow("sNodes", seedNodes);

    /***********MST CODE***********/
    //Create MST representation
    cvtColor(originalImage, gray_image, CV_BGR2GRAY );
    GaussianBlur(gray_image, gray_image, Size(3, 3), 2);
    updateVertexGridWeights(gray_image);
    setSeedNodes(seedNodes);
    createMST();
    passUp();
    passDown();
    //Get boundary dissimiliary and tree distance maps
    getMSTDistanceImage(mst_image);

    //combine images and normalize
    obstacles.convertTo(obstacles, CV_32F);
    obstacles = obstacles/255;
    combined = mst_image + new_dis_image;
    double minVal, maxVal;
    cv::minMaxLoc(combined, &minVal, &maxVal);
    combined /= maxVal;

    Mat obstaclesInWaterMST;
    //Post process and write results
    postProcessing(combined);
    customOtsuThreshold(combined);
    findHorizonLine(horizonLine, cv::Size(combined.cols, combined.rows));
    findObstaclesInWater(horizonLine, combined, obstaclesInWaterMST, true);
    findContoursAndWriteResults(obstaclesInWaterMST, originalImage, scoreFile, output+name);

    int t2 = getTickCount();
    std::cout << "Processing Time/Image: " << (t2-t1)/getTickFrequency() << std::endl;

   //Display results
   imshow("Tree Distance Image", mst_image);
   imshow("Boundary Dissimiliarty", new_dis_image);
   imshow("Combined MST result", combined);
   imshow("Bounding Boxes MST", originalImage);


    //**************END MAIN CODE SECTION*************//
   scoreFile.close();

   if (wait)
        waitKey(0);
    return 0;
}
