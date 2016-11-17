#include <iostream>
#include <stdio.h>
#include <opencv2/opencv.hpp>

using namespace cv;

//if most weight closer to the outside of the tight bounding box, probably an object
int main(int argc, char *argv[])
{
   VideoCapture cap("../../TestMedia/videos/boatm30.mp4"); // open the default camera
   if(!cap.isOpened()) {  // check if we succeeded
       std::cout << "no vid" << std::endl;
       return -1;
   }

   float factor = 1.0/(2*sqrt(2));
   float factor2 = 1.0/6;
   float factor3 = 1.0/3;
   float data[81] = {factor, 0.5, factor, 0, 0, 0, -factor, -0.5, -factor,
                     factor, 0, -factor, 0.5, 0, -0.5, factor, 0, -factor,
                     0, -factor, 0.5, factor, 0, -factor, -0.5, factor, 0,
                     0.5, -factor, 0, -factor, 0, factor, 0, factor, -0.5,
                     0, 0.5, 0, -0.5, 0, -0.5, 0, 0.5, 0,
                     -0.5, 0, 0.5, 0, 0, 0, 0.5, 0, -0.5,
                     factor2, -2*factor2, factor2, -2*factor2, 4*factor2, -2*factor2, factor2, -2*factor2, factor2,
                     -2*factor2, factor2, -2*factor2, factor2, 4*factor2, factor2, -2*factor2, factor2, -2*factor2,
                     factor3, factor3, factor3, factor3, factor3, factor3};

   std::vector<Mat> fBank;
   float *pData = data;
   for (int i = 0; i < 9; i++) {
       fBank.push_back(Mat(3,3, CV_32F, pData));
       pData += 9;
   }

   Mat image, scaledImage;
   for(;;)
   {
       cap >> image; // get a new frame fro m camera
       if (image.rows == 0 || image.cols == 0)
           continue;

       int width = image.cols/2;
       int height = image.rows/2;
       Mat gray_image;
      // int64 t1 = getTickCount();

      // resize(image, scaledImage, Size(image.cols, image.rows));
       scaledImage = image;
       Mat frei_image = Mat::zeros(scaledImage.rows, scaledImage.cols, CV_32F);
       Mat m_term = Mat::zeros(scaledImage.rows, scaledImage.cols, CV_32F);
       Mat s_term = Mat::zeros(scaledImage.rows, scaledImage.cols, CV_32F);

       cvtColor( scaledImage, gray_image, CV_BGR2GRAY );
        GaussianBlur(gray_image, gray_image, Size(7, 7), 5);
        GaussianBlur(gray_image, gray_image, Size(7, 7), 5);

       gray_image.convertTo(gray_image, CV_32F);
         for (int f = 0; f < 9; f++) {
           filter2D(gray_image, frei_image, -1 , fBank[f]);
           s_term += frei_image.mul(frei_image);
           if (f == 1) {
               m_term = s_term.clone();
           }
       }
       cv::sqrt(m_term/s_term, frei_image);
       cv::threshold(frei_image, frei_image, 0.05, 1.0, THRESH_BINARY);

      // int64 t2 = getTickCount();
    //   std::cout << (t2 - t1)/getTickFrequency() << std::endl;

       imshow("threshold", frei_image);

       resize(image, image, Size(width, height));
       imshow("orig", image);

         waitKey(1);
   }

    return 0;
}



