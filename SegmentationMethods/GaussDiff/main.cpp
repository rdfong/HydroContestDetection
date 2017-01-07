#include <iostream>
#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <ctime>

using namespace cv;

#define VIDEO 1
//Issues with this approach:
//Doesn't work so well when it's too wavey, waves are detected as interest areas
//Doesn't work so well when boat are too close, we only see edges, so the side of the boat isn't seen and tracking methods may segment the boat.
//Doesn't handle reflections well
//Basically requires a well posed problem

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
       if (curRect.area() > 50) {
         rectangle(image, curRect, Scalar(0, 255,0), 2);
       }
   }
}

int main(int argc, char *argv[])
{
    //Create linear blur kernel
    int L = 10;
    float alpha = 0;
    int kernelSize = 2*L+1;
    Mat blurKernel;
    blurKernel.create(kernelSize,kernelSize,CV_64F);

    for (int row = -L; row <= L; row++) {
        for (int col = -L; col <= L; col++) {
            if (row == (int)L*sin(alpha) && abs(col) <= (int)L*cos(alpha))
                blurKernel.at<double>(row+L, col+L) = 1.0f/kernelSize;
            else
                blurKernel.at<double>(row+L, col+L) = 0.0f;
        }
    }
#if VIDEO == 0
    Mat image;
    image = imread("../../TestMedia/images/boat3.jpg", CV_LOAD_IMAGE_COLOR);
    if (!image.data)
    {
        printf("No image data \n");
        return -1;
    }

    float scale = 1.0;
    Size size(scale*image.cols, scale*image.rows);
    Mat scaledImage;
    resize(image, scaledImage, size);
#elif VIDEO == 1
    VideoCapture cap("../../TestMedia/videos/boatm10.mp4"); // open the default camera
    if(!cap.isOpened()) {  // check if we succeeded
        std::cout << "no vid" << std::endl;
        return -1;
    }

    for(;;)
    {
        Mat image;
        cap >> image; // get a new frame from camera

        if (image.rows == 0 || image.cols == 0)
            continue;

        float scale = 0.5;
        Size size(scale*image.cols, scale*image.rows);
        Mat scaledImage;
        resize(image, scaledImage, size);
#endif
        int64 t2 = getTickCount();

        //Blur image with 2 different gaussians and take the difference
        Mat gray_image;
        cvtColor( scaledImage, gray_image, CV_BGR2GRAY );

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

        //Find anomalies in the blur diff image
        Mat anomalies1;
        Mat meanM;
        Mat stdevM;
        meanStdDev(blurDiff1, meanM, stdevM);
        double mean = meanM.at<double>(0,0);
        double stdev = stdevM.at<double>(0,0);
        anomalies1 = (blurDiff1-mean)/stdev;
        anomalies1 = anomalies1.mul(anomalies1);

        //Linear motion blur on image
        Mat linearBlur;
        filter2D(gray_image, linearBlur, -1 , blurKernel);
        imshow("Linear Motion BLur", linearBlur);

        //Find anomalies for the motion blurred image
        GaussianBlur(linearBlur, blur1, Size(k1, k1), sigma1);
        GaussianBlur(linearBlur, blur2, Size(k2, k2), sigma2);
        Mat blurDiff2;
        addWeighted(blur1, 1.0, blur2, -1.0, 1.0, blurDiff2);

        Mat anomalies2;
        meanStdDev(blurDiff2, meanM, stdevM);
        mean = meanM.at<double>(0,0);
        stdev = stdevM.at<double>(0,0);
        anomalies2 = (blurDiff2-mean)/stdev;
        anomalies2 = anomalies2.mul(anomalies2);

        //Take the result of subtracting the two anomaly maps
        Mat result = abs(anomalies1 - anomalies2);
        threshold(result, result, 16, 255, THRESH_BINARY);
        imshow("anomThresh", result);
        waitKey(1);

        //Dilate the anomaly difference map to create the potential obstacle map
        int morph_size = 7;
        Mat element = getStructuringElement( MORPH_ELLIPSE, Size( 2*morph_size + 1, 2*morph_size+1 ), Point( morph_size, morph_size ) );
        Mat final;

        morphologyEx( result, final, MORPH_DILATE, element );

        //Find and draw bounding boxes
        imshow("Obstacles", final);
        findContoursAndWriteResults(final, scaledImage);

        imshow("final", scaledImage);
#if VIDEO == 0
        waitKey(0);
#elif VIDEO == 1
        waitKey(1);
    }
#endif
    return 0;
}
