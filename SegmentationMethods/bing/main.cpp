#include <iostream>
#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <opencv2/saliency.hpp>
#include <ctime>

using namespace cv;
using namespace std;
#define VIDEO 0
int main(int argc, char *argv[])
{
#if VIDEO == 0
    Mat image;
    image = imread("../smallmedia/boat1.png", CV_LOAD_IMAGE_COLOR);
    if (!image.data)
    {
        printf("No image data \n");
        return -1;
    }

    //approximate size, 900 by 600

    float scale = 1.0;
    Size size(scale*image.cols, scale*image.rows);
    Mat scaledImage;
    resize(image, scaledImage, size);
/*
    Mat gray_image;
    cvtColor( scaledImage, gray_image, CV_BGR2GRAY );

    imshow("gray", gray_image);
*/
    //Start BINGness here
    cv::saliency::ObjectnessBING objProposal;

    string training_path = "../bing/ObjectnessTrainedModel";

        vector<Vec4i> saliencyMap;
        objProposal.setTrainingPath(training_path);

        // display some information about BING
        std:cout << "getBase() " << objProposal.getBase() << endl;
        cout << "getNSS() " << objProposal.getNSS() << endl;
        cout << "getW() " << objProposal.getW() << endl;
        int64 t1 = getTickCount();
        // do computation.
        objProposal.computeSaliency(scaledImage, saliencyMap);

        int64 t2 = getTickCount();
        std::cout << (t2 - t1)/getTickFrequency() << std::endl;

        for (int i = 0; i < std::min<int>(saliencyMap.size(), 100); i++)
        {
            Mat si = scaledImage.clone();
            rectangle(si, Point(saliencyMap[i][0], saliencyMap[i][1]), Point(saliencyMap[i][2], saliencyMap[i][3]), Scalar(255, 0, 0));
            imshow("object proposals", si);
            waitKey(0);
        }


       waitKey(0);
#elif VIDEO == 1
       VideoCapture cap("../media/boatm10.mp4"); // open the default camera
       if(!cap.isOpened()) {  // check if we succeeded
           std::cout << "no vid" << std::endl;
           return -1;
       }

       int framecounter = 0;
       /* cv::saliency::ObjectnessBING objProposal;
       string training_path = "../bing/ObjectnessTrainedModel";

        objProposal.setTrainingPath(training_path);

        // display some information about BING
        std:cout << "getBase() " << objProposal.getBase() << endl;
        cout << "getNSS() " << objProposal.getNSS() << endl;
        cout << "getW() " << objProposal.getW() << endl;*/

       for(;;)
       {
           Mat frame;
           cap >> frame; // get a new frame from camera
           if (frame.rows == 0 || frame.cols == 0)
               continue;
           int64 t2 = getTickCount();
           /*float scale = 1.0;
           Size size(scale*image.cols, scale*image.rows);
           Mat scaledImage;
           resize(image, scaledImage, size);*/
          //  vector<Vec4i> saliencyMap;
           objProposal.computeSaliency(scaledImage, saliencyMap);

           //std::cout << (t2 - t1)/getTickFrequency() << std::endl;

          /* for (int i = 0; i < std::min<int>(saliencyMap.size(), 10); i++)
           {
               rectangle(scaledImage, Point(saliencyMap[i][0], saliencyMap[i][1]), Point(saliencyMap[i][2], saliencyMap[i][3]), Scalar(255, 0, 0));
           }*/

           imshow("bing", frame);
           framecounter++;
       }

#endif
    return 0;
}
