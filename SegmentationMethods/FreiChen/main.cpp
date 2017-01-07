#include <iostream>
#include <stdio.h>
#include <opencv2/opencv.hpp>

using namespace cv;

//SETUP FOR BOX SELECTION
int imgCount = 1;
int dims = 3;
const int sizes[] = {64,64,64};
const int channels[] = {0,1,2};
float rRange[] = {0,256};
float gRange[] = {0,256};
float bRange[] = {0,256};
const float *ranges[] = {rRange,gRange,bRange};
Mat mask = Mat();
std::vector<std::vector<Point> > contours;
std::vector<Vec4i> hierarchy;
std::vector<Rect> boundRects;
std::vector<Rect> originalRects;
Mat hist1, temp1, hist2, temp2, nonZeroSubset;
Mat bgr[3];
Rect curRect, otherRect, originalRect, intersection, rectUnion;
std::vector<std::vector<Rect> > intersectionGroups;
std::vector<std::pair<Point2i, Point2i> > finalBoxBounds;
std::vector<Mat> input(3);
std::vector<int> groupsToMerge;

template<typename T> void  getNonZeroPix(Mat mask, Mat im, Mat& nonZeroSubset) {
    Mat imValues = im.reshape(0, im.rows*im.cols);
    Mat flattenedMask = mask.reshape(0, im.rows*im.cols);
    Mat idx;
    findNonZero(flattenedMask, idx);
    nonZeroSubset = Mat::zeros(idx.rows, 1, CV_8U);
    auto im_it = nonZeroSubset.begin<T>();
    for (int i = 0; i < idx.rows; i++) {
        *im_it= imValues.at<T>(idx.at<int>(i, 1), 0);
        ++im_it;
    }
}

void findContoursAndWriteResults(Mat& obstacleMap, Mat& image) {
   hierarchy.clear();
   contours.clear();
   boundRects.clear();
   originalRects.clear();
   findContours( obstacleMap.clone(), contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE, Point(0, 0) );
   //get bounding rects from contours
   int expand = 1;
   for (int i =0; i < contours.size(); i++) {
       curRect = boundingRect(contours[i]);
       if ((double)curRect.width/curRect.height < 0.1 || (double)curRect.height/curRect.width < 0.1)
           continue;
       originalRects.push_back(curRect);
       Point2i newTL(max(curRect.tl().x-expand, 0), max(curRect.tl().y-expand,0));
       Point2i newBR(min(curRect.br().x+expand, obstacleMap.cols-1), min(curRect.br().y+expand,obstacleMap.rows-1));
       boundRects.push_back(Rect(newTL, newBR));
   }

   //intersection groups mirrors finalboxbounds in size, but final box bounds contains information on originalRects, intersection groups is for the expanded rects
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
                 }  else if (intersection.area() > 0) {
                     //COLOR SIMILARITY MEASURE
                     Mat mask1, mask2;
                     obstacleMap(curRect).copyTo(mask1);
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

                     obstacleMap(otherRect).copyTo(mask2);
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
         rectangle(image, curRect, Scalar(0, 255,0), 2);
       }
   }
}

int main(int argc, char *argv[])
{
   VideoCapture cap("../../TestMedia/videos/boatm10.mp4"); // open the default camera
   if(!cap.isOpened()) {  // check if we succeeded
       std::cout << "no vid" << std::endl;
       return -1;
   }

   //Set up filter banks
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

        resize(image, scaledImage, Size(width, height));
       Mat frei_image = Mat::zeros(scaledImage.rows, scaledImage.cols, CV_32F);
       Mat m_term = Mat::zeros(scaledImage.rows, scaledImage.cols, CV_32F);
       Mat s_term = Mat::zeros(scaledImage.rows, scaledImage.cols, CV_32F);

       //blur image
       cvtColor( scaledImage, gray_image, CV_BGR2GRAY );
        GaussianBlur(gray_image, gray_image, Size(7, 7), 5);
        GaussianBlur(gray_image, gray_image, Size(7, 7), 5);

       gray_image.convertTo(gray_image, CV_32F);

       //run filters
         for (int f = 0; f < 9; f++) {
           filter2D(gray_image, frei_image, -1 , fBank[f]);
           s_term += frei_image.mul(frei_image);
           if (f == 1) {
               m_term = s_term.clone();
           }
       }
        cv::sqrt(m_term/s_term, frei_image);
        cv::threshold(frei_image, frei_image, 0.05, 1.0, THRESH_BINARY);


        imshow("threshold", frei_image);

        frei_image = frei_image*255;
        frei_image.convertTo(frei_image, CV_8U);

        findContoursAndWriteResults(frei_image, scaledImage);

        imshow("orig", scaledImage);

        waitKey(1);
   }

    return 0;
}



