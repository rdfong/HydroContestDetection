#include <iostream>
#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <ctime>

using namespace cv;

#define VIDEO 1
//Doesn't work so well when it's too wavey, waves are detected as interest areas
//Doesn't work so well when boat are too close, we only see edges, so the side of the boat isn't seen
//and tracking methods may segment the boat.
//also doesn't handle reflections well

//mean shift clustering won't work well when the the radius can be any size

//requires a well posed problem

int main(int argc, char *argv[])
{
#if VIDEO == 0
    Mat image;
    image = imread("../media/boat1.jpg", CV_LOAD_IMAGE_COLOR);
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

    imshow("gray", gray_image);
    Mat gray_image2;
    GaussianBlur(gray_image, gray_image2, Size(5, 5), 3);
    GaussianBlur(gray_image2, gray_image, Size(5, 5), 3);

    float sigma1 = 3.0;
    int k1 = 5;

    float sigma2 = 1.0;
    float k2 = 1;

    Mat blur1;
    Mat blur2;
    GaussianBlur(gray_image, blur1, Size(k1, k1), sigma1);
    GaussianBlur(gray_image, blur2, Size(k2, k2), sigma2);
    Mat blurDiff1;
    addWeighted(blur1, 1.0, blur2, -1.0, 1.0, blurDiff1);
    //imshow("blur1", blurDiff1);

    Mat anomalies1;
    Mat meanM;
    Mat stdevM;
    meanStdDev(blurDiff1, meanM, stdevM);
    double mean = meanM.at<double>(0,0);
    double stdev = stdevM.at<double>(0,0);
    anomalies1 = (blurDiff1-mean)/stdev;
    anomalies1 = anomalies1.mul(anomalies1);


    int L = 10;
    float alpha = 0;
    int kernelSize = 2*L+1;
    Mat blurKernel;
    blurKernel.create(kernelSize,kernelSize,CV_64F);

    //todo: should be blurred in multiple directions, maybe rotated (look for cv functions)
    for (int row = -L; row <= L; row++) {
        for (int col = -L; col <= L; col++) {
            if (row == (int)L*sin(alpha) && abs(col) <= (int)L*cos(alpha))
                blurKernel.at<double>(row+L, col+L) = 1.0f/kernelSize;
            else
                blurKernel.at<double>(row+L, col+L) = 0.0f;
        }
    }

   // imshow("test", blurKernel);
    Mat linearBlur;
    filter2D(gray_image, linearBlur, -1 , blurKernel);
    imshow("Linear Motion BLur", linearBlur);

    GaussianBlur(linearBlur, blur1, Size(k1, k1), sigma1);
    GaussianBlur(linearBlur, blur2, Size(k2, k2), sigma2);
    Mat blurDiff2;
    addWeighted(blur1, 1.0, blur2, -1.0, 1.0, blurDiff2);
   // imshow("blurdiff2", blurDiff2);

    Mat anomalies2;
    meanStdDev(blurDiff2, meanM, stdevM);
    mean = meanM.at<double>(0,0);
    stdev = stdevM.at<double>(0,0);
    anomalies2 = (blurDiff2-mean)/stdev;
    anomalies2 = anomalies2.mul(anomalies2);

    Mat result = abs(anomalies1 - anomalies2);
    threshold(result, result, 8, 255, THRESH_BINARY);

    imshow("anomThresh", result);
    int morph_size = 5;
    Mat element = getStructuringElement( MORPH_ELLIPSE, Size( 2*morph_size + 1, 2*morph_size+1 ), Point( morph_size, morph_size ) );
    Mat final;
    /// Apply the specified morphology operation
     morphologyEx( result, final, MORPH_DILATE, element );
      imshow("final", final);
      // std::cout << (t2 - t1)/getTickFrequency() << std::endl;
       waitKey(0);
#elif VIDEO == 1
       VideoCapture cap("../media/test.mp4"); // open the default camera
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
           float scale = 0.5;
           Size size(scale*image.cols, scale*image.rows);
           Mat scaledImage;
           resize(image, scaledImage, size);

           Mat gray_image;
           cvtColor( scaledImage, gray_image, CV_BGR2GRAY );

           imshow("gray", gray_image);
           Mat gray_image2;
           GaussianBlur(gray_image, gray_image2, Size(5, 5), 3);
           GaussianBlur(gray_image2, gray_image, Size(5, 3), 3);

           float sigma1 = 3.0;
           int k1 = 5;

           float sigma2 = 1.0;
           float k2 = 1;

           Mat blur1;
           Mat blur2;
           GaussianBlur(gray_image, blur1, Size(k1, k1), sigma1);
           GaussianBlur(gray_image, blur2, Size(k2, k2), sigma2);
           Mat blurDiff1;
           addWeighted(blur1, 1.0, blur2, -1.0, 1.0, blurDiff1);
           //imshow("blur1", blurDiff1);

           Mat anomalies1;
           Mat meanM;
           Mat stdevM;
           meanStdDev(blurDiff1, meanM, stdevM);
           double mean = meanM.at<double>(0,0);
           double stdev = stdevM.at<double>(0,0);
           anomalies1 = (blurDiff1-mean)/stdev;
           anomalies1 = anomalies1.mul(anomalies1);


           int L = 10;
           float alpha = 0;
           int kernelSize = 2*L+1;
           Mat blurKernel;
           blurKernel.create(kernelSize,kernelSize,CV_64F);

           //todo: should be blurred in multiple directions, maybe rotated (look for cv functions)
           for (int row = -L; row <= L; row++) {
               for (int col = -L; col <= L; col++) {
                   if (row == (int)L*sin(alpha) && abs(col) <= (int)L*cos(alpha))
                       blurKernel.at<double>(row+L, col+L) = 1.0f/kernelSize;
                   else
                       blurKernel.at<double>(row+L, col+L) = 0.0f;
               }
           }

          // imshow("test", blurKernel);
           Mat linearBlur;
           filter2D(gray_image, linearBlur, -1 , blurKernel);
          // imshow("Linear Motion BLur", linearBlur);

           GaussianBlur(linearBlur, blur1, Size(k1, k1), sigma1);
           GaussianBlur(linearBlur, blur2, Size(k2, k2), sigma2);
           Mat blurDiff2;
           addWeighted(blur1, 1.0, blur2, -1.0, 1.0, blurDiff2);
          // imshow("blurdiff2", blurDiff2);

           Mat anomalies2;
           meanStdDev(blurDiff2, meanM, stdevM);
           mean = meanM.at<double>(0,0);
           stdev = stdevM.at<double>(0,0);
           anomalies2 = (blurDiff2-mean)/stdev;
           anomalies2 = anomalies2.mul(anomalies2);

           Mat result = abs(anomalies1 - anomalies2);
           threshold(result, result, 16, 255, THRESH_BINARY);
           imshow("anomThresh", result);
           waitKey(1);

           int morph_size = 5;
           Mat element = getStructuringElement( MORPH_ELLIPSE, Size( 2*morph_size + 1, 2*morph_size+1 ), Point( morph_size, morph_size ) );
           Mat final;
           /// Apply the specified morphology operation
            morphologyEx( result, final, MORPH_DILATE, element );
             imshow("final", final);
           //framecounter++;
       }

#endif
    return 0;
}
