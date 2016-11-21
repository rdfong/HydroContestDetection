#include <iostream>
#include <stdio.h>
#include <opencv2/opencv.hpp>

using namespace cv;

std::vector<Rect> boundingBoxes;

void selectBoundingBoxes(Mat grayImage) {
    //Convert to binary image mat
    Mat binMat = grayImage;///255;
    imshow("test", binMat);
}

void drawBoundingBoxes(Mat displayImage) {
    for( size_t i = 0; i< boundingBoxes.size(); i++ )
    {
      Scalar color = Scalar(255,255,255);
      rectangle( displayImage, boundingBoxes[i].tl(), boundingBoxes[i].br(), color, 2, 8, 0 );
    }
    imshow("bounding", displayImage);
}

int main(int argc, char *argv[])
{
    Mat gray_image;
    gray_image = imread("../../TestMedia/binaryImages/boat9.png", CV_LOAD_IMAGE_GRAYSCALE);
    if (!gray_image.data)
    {
        printf("No image data \n");
        return -1;
    }

    selectBoundingBoxes(gray_image);
    drawBoundingBoxes(gray_image);

    waitKey(0);
    return 0;
}
