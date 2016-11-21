#include <iostream>
#include <stdio.h>
#include <opencv2/opencv.hpp>

using namespace cv;

std::vector<Rect> boundingBoxes;
std::queue<Rect> quadtreeBoxes;
const float divCrit = .75;
const int ignoreCrit = 0;
const int areaLimit = 25;

void selectBoundingBoxes(Mat grayImage) {
    //Convert to binary image mat
    Mat binMat = grayImage/255;
    quadtreeBoxes.push(Rect(Point2i(0,0), Point2i(grayImage.cols-1, grayImage.rows-1)));
    //rectangle(binMat, quadtreeBoxes[0], Scalar(255,255,255));
    //Start quadtree implementation
    while (!quadtreeBoxes.empty()) {
        Rect& curRect = quadtreeBoxes.front();
        int numWhite = sum(binMat(cv::Range(curRect.tl().y, curRect.br().y), cv::Range(curRect.tl().x, curRect.br().x))).val[0];
        float percentCoverage = (float)numWhite/curRect.area();
        if (percentCoverage >= divCrit) {
            boundingBoxes.push_back(curRect);
        } else if (numWhite > ignoreCrit && curRect.area() > areaLimit){
            //divide into 4 and push all in to the queue
            Point2i mid = (curRect.tl()+curRect.br())/2;
            quadtreeBoxes.push(Rect(curRect.tl(), mid));
            quadtreeBoxes.push(Rect(Point2i(mid.x, curRect.tl().y), Point2i(curRect.br().x, mid.y)));
            quadtreeBoxes.push(Rect(Point2i(curRect.tl().x, mid.y), Point2i(mid.x, curRect.br().y)));
            quadtreeBoxes.push(Rect(mid,curRect.br()));
        }
        quadtreeBoxes.pop();
    }
}

void drawBoundingBoxes(Mat& displayImage) {
    Scalar color = Scalar(128,128,128);
    for( size_t i = 0; i< boundingBoxes.size(); i++ )
    {
      rectangle( displayImage, boundingBoxes[i].tl(), boundingBoxes[i].br(), color, 1, 8, 0 );
    }
    imshow("bounding", displayImage);
}

int main(int argc, char *argv[])
{
    //TODO: step 1: combine boxes to form super boxes (remove small joints)
    //      step 2: split boxes as necessary (via differences)
    //      step 3: reshape and extend boxes as necessary (via canny style thresholding)
    //      step 4: blob filtering (taking size and height in image into consideration?)
    Mat gray_image;
    gray_image = imread("../../TestMedia/binaryImages/boat3.png", CV_LOAD_IMAGE_GRAYSCALE);
    if (!gray_image.data)
    {
        printf("No image data \n");
        return -1;
    }
    boundingBoxes.clear();

    int t1 = getTickCount();
    selectBoundingBoxes(gray_image);
    int t2 = getTickCount();
    std::cout << (t2-t1)/getTickFrequency() <<std::endl;

    drawBoundingBoxes(gray_image);
    waitKey(0);
    return 0;
}
