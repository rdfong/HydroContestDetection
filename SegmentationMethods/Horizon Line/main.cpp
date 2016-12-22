#include <iostream>
#include <stdio.h>
#include <opencv2/opencv.hpp>

using namespace cv;

#define VIDEO 1

int main(int argc, char *argv[])
{
#if VIDEO == 0
    Mat image;
    image = imread("../media/boat9.jpg", CV_LOAD_IMAGE_COLOR);
    if (!image.data)
    {
        printf("No image data \n");
        return -1;
    }

    //approximate size, 900 by 600
    int64 t1 = getTickCount();
    float scale = 0.2;
    Size size(scale*image.cols, scale*image.rows);
    Mat scaledImage;
    resize(image, scaledImage, size);

    Mat gray_image;
    cvtColor( scaledImage, gray_image, CV_BGR2GRAY );

    int erosion_size = 5;
    Mat element = getStructuringElement( MORPH_RECT,
                                         Size( 2*erosion_size + 1, 2*erosion_size+1 ),
                                         Point( erosion_size, erosion_size ) );


    Mat erosion_dst;
    erode( gray_image, erosion_dst, element );

    equalizeHist(erosion_dst, erosion_dst);
    imshow( "Erosion", erosion_dst );

    GaussianBlur(erosion_dst, erosion_dst, Size(5, 5), 3);
    Rect myROI(10, 10, erosion_dst.cols-10, erosion_dst.rows-10);
    erosion_dst = erosion_dst(myROI);
     Mat edgeImage;
     Canny( erosion_dst, edgeImage, 64, 128, 3);
     imshow( "Canny", edgeImage );


     std::vector<Vec4i> lines;
     int minVotes = scaledImage.cols*0.33;
       HoughLinesP(edgeImage, lines, 3, CV_PI/180, minVotes, 200, 10000 );

       if (lines.size()) {
           Vec4i l = lines[0];
           Point bestPointA(l[0]+10, l[1]-erosion_size+10);
           Point bestPointB(l[2]+10, l[3]-erosion_size+10);
           float slope = (bestPointA.y-bestPointB.y)/(bestPointA.x-bestPointB.x);
           Point leftPoint(0, bestPointA.y-bestPointA.x*slope);
           int maxX = scaledImage.cols-1;
           Point rightPoint(maxX, bestPointB.y+(maxX-bestPointB.x)*slope);
           line( scaledImage, leftPoint, rightPoint, Scalar(0,0,255), 2, CV_AA);
       }
       imshow( "Hough", scaledImage );
       int64 t2 = getTickCount();
       std::cout << (t2 - t1)/getTickFrequency() << std::endl;
       waitKey(0);
#elif VIDEO == 1
       VideoCapture cap("../../TestMedia/videos/boatm30.mp4"); // open the default camera
       if(!cap.isOpened()) {  // check if we succeeded
           std::cout << "no vid" << std::endl;
           return -1;
       }

       int framecounter = 0;
       for(;;)
       {
           Mat image;
           cap >> image; // get a new frame from camera
           int64 t2 = getTickCount();
           //float scale = 1.0;
           //Size size(scale*image.cols, scale*image.rows);
           //Mat scaledImage;
           //resize(image, scaledImage, size);

           Mat gray_image;
           cvtColor( image, gray_image, CV_BGR2GRAY );

        imshow("gray", gray_image);
           int erosion_size = 5;
           Mat element = getStructuringElement( MORPH_RECT,
                                                Size( 2*erosion_size + 1, 2*erosion_size+1 ),
                                                Point( erosion_size, erosion_size ) );

           Mat erosion_dst;
           erode( gray_image, erosion_dst, element );

           equalizeHist(erosion_dst, erosion_dst);

           GaussianBlur(erosion_dst, erosion_dst, Size(5, 5), 3);

           Rect myROI(10, 10, erosion_dst.cols-10, erosion_dst.rows-10);
           erosion_dst = erosion_dst(myROI);

           imshow( "Erosion", erosion_dst );
            Mat edgeImage;
            Canny( erosion_dst, edgeImage, 32, 128, 3);

            imshow( "Canny", edgeImage );

            std::vector<Vec4i> lines;
            int minVotes = image.cols*0.33;
              HoughLinesP(edgeImage, lines, 3, CV_PI/180, minVotes, 200, 10000 );
              if (lines.size()) {
                  Vec4i l = lines[0];
                  Point bestPointA(l[0]+10, l[1]+10-erosion_size);
                  Point bestPointB(l[2]+10, l[3]+10-erosion_size);
                  float slope = (bestPointA.y-bestPointB.y)/(bestPointA.x-bestPointB.x);
                  Point leftPoint(0, bestPointA.y-bestPointA.x*slope);
                  int maxX = image.cols-1;
                  Point rightPoint(maxX, bestPointB.y+(maxX-bestPointB.x)*slope);
                  line( image, leftPoint, rightPoint, Scalar(0,0,255), 2, CV_AA);
              }
           framecounter++;
          // if (framecounter%5 == 0) {
                imshow( "Hough", image );
                waitKey(1);
           //}
       }

#endif
    return 0;
}
