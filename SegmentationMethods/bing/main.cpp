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
    Mat image;
    image = imread("../../TestMedia/images/japtest.JPG", CV_LOAD_IMAGE_COLOR);
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

    //Start BINGness here
    cv::saliency::ObjectnessBING objProposal;

    string training_path = "../../SegmentationMethods/bing/ObjectnessTrainedModel";

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

        Mat si = scaledImage.clone();
        for (int i = 0; i < std::min<int>(saliencyMap.size(), 10); i++)
        {
            rectangle(si, Point(saliencyMap[i][0], saliencyMap[i][1]), Point(saliencyMap[i][2], saliencyMap[i][3]), Scalar(255, 0, 0));
        }

       imshow("object proposals", si);
       waitKey(0);

    return 0;
}
