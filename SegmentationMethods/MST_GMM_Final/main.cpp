#include <iostream>
#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <fstream>
#include <eigen3/Eigen/Dense>
#include <opencv2/core/eigen.hpp>

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
