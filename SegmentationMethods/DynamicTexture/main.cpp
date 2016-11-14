#include <iostream>
#include <stdio.h>
#include <opencv2/opencv.hpp>

using namespace cv;
//can we just use the optflow map instead of calculating entropy which takes a while and isn't particularly accurate?
//
static void drawOptFlowMap(const Mat& flow, Mat& cflowmap, int step,
                    double, const Scalar& color)
{
    float maxNorm = 0;
    for(int y = 0; y < cflowmap.rows; y += step) {
        for(int x = 0; x < cflowmap.cols; x += step)
        {
            const Point2f& fxy = flow.at<Point2f>(y, x);
            float n = norm(fxy);
            if (n > maxNorm)
                maxNorm = n;
        }
    }

    for(int y = 0; y < cflowmap.rows; y += step)
        for(int x = 0; x < cflowmap.cols; x += step)
        {
            const Point2f& fxy = flow.at<Point2f>(y, x);
               Point2f dir = fxy/maxNorm;
               if (norm(fxy)>0.5)
               // line(cflowmap, Point(x,y), Point(cvRound(x+fxy.x), cvRound(y+fxy.y)),
                // Scalar(abs(dir.x)*255, abs(dir.y)*255, 0));
                 circle(cflowmap, Point(x,y), 1, Scalar((dir.x*255+255)/2, (dir.y*255+255)/2, 0), -1);
           // circle(cflowmap, Point(x,y), 2, color, -1);
        }

}
//TODO: result here can be used to develop a model for water, from which all other water pixels can be classified
//or it can be used to initialize the gaussian distribution for the segmentation method

//TODO: is there really no way to check for out of bounds more effectively?

void calcDissimilarity(Mat& flow, Mat& dissimilarity, int radius) {
    if (radius == 0) {
        return;
    }
    Mat tempDis = Mat::zeros(dissimilarity.rows, dissimilarity.cols, CV_64F);
    int rows = flow.rows;
    int cols = flow.cols;
    for (int r = 0; r < rows; r++) {
        for (int c = 0; c < cols; c++) {

                float totalMag = 0;
                float average = 0;
             Point2f curPoint = flow.at<Point2f>(r, c);
             for (int rOff = -radius; rOff <= radius; rOff++) {
                 for (int cOff = -radius; cOff <= radius; cOff++) {
                     int rIndex = r+rOff;
                     int cIndex = c+cOff;
                     //in bounds
                     if (rIndex >= 0 && rIndex < rows && cIndex >= 0 && cIndex < cols) {
                        Point2f nPoint = flow.at<Point2f>(rIndex, cIndex);

                        average += norm(nPoint-curPoint);//acos(nPoint.dot(curPoint)/(norm(nPoint)*norm(curPoint)));
                        totalMag += norm(nPoint);
                     }
                 }
             }
           //  totalMag = 1.0;
            tempDis.at<double>(r,c) = average/totalMag;
        }
     }

    float maxDissimilarity = 0;
    for (int r = 0; r < rows; r++) {
        for (int c = 0; c < cols; c++) {
             float curD = tempDis.at<double>(r, c);
             for (int rOff = -radius; rOff <= radius; rOff++) {
                 for (int cOff = -radius; cOff <= radius; cOff++) {
                     int rIndex = r+rOff;
                     int cIndex = c+cOff;
                     //in bounds
                     if (rIndex >= 0 && rIndex < rows && cIndex >= 0 && cIndex < cols) {
                        float neighD = tempDis.at<double>(rIndex, cIndex);
                        if (neighD < curD)
                            curD = neighD;
                     }
                 }
             }
             if (curD > maxDissimilarity)
                 maxDissimilarity = curD;
            dissimilarity.at<double>(r,c) =curD;
        }
     }

    dissimilarity /= maxDissimilarity;

}

//strategies, variance in optical flow vectors
//or do the radius/netvec thing
void calcEntropyTime(std::vector<Mat>& flows, Mat& entropy) {
    //Mat diff;
    for (int f = 1; f < flows.size(); f++) {
        //diff = flows[f]-flows[f-1];
        Mat f0 = flows[f];
        Mat f1 = flows[f-1];
        for (int r = 0; r < entropy.rows; r++) {
            for (int c = 0; c < entropy.cols; c++) {
                //Point2f curDiff = diff.at<Point2f>(r,c);
                //float n = norm(curDiff);
                Point2f dir1 = f0.at<Point2f>(r,c);
                Point2f dir2 = f1.at<Point2f>(r,c);
                float n1 = norm(dir1);
                float n2 = norm(dir2);

                if(n1 > 0.5 && n2 > 0.5) {
                    entropy.at<double>(r,c) += acos(dir1.dot(dir2)/(n1*n2));
                }
            }
        }
    }
    float maxNorm = 0;
    for (int r = 0; r < entropy.rows; r++) {
        for (int c = 0; c < entropy.cols; c++) {
            float curNorm = entropy.at<double>(r,c);
            if (curNorm > maxNorm)
                maxNorm = curNorm;
        }
    }
    entropy/=maxNorm;

}

void calcEntropySpace(const Mat& flow, Mat& entropy, int boxRadius) {
    //This can be done separably
    int boxSize = 2*boxRadius+1;
    Mat length = Mat::zeros(entropy.rows, entropy.cols, CV_64F);
    Mat maxRadius = Mat::zeros(entropy.rows, entropy.cols, CV_64F);
    Mat netVecX = Mat::zeros(entropy.rows, entropy.cols, CV_64F);
    Mat netVecY = Mat::zeros(entropy.rows, entropy.cols, CV_64F);

    for(int rowF = boxRadius, rowE = 0; rowF < flow.rows-boxRadius; rowF++, rowE++) {
        for (int colF = boxRadius, colE = 0; colF < flow.cols-boxRadius; colF++, colE++) {
            for (int rowOffset = -boxRadius; rowOffset <= boxRadius; rowOffset++) {
                for (int colOffset = -boxRadius; colOffset <= boxRadius; colOffset++) {
                    Point2f flowVec = flow.at<Point2f>(rowF+rowOffset, colF+colOffset);
                    length.at<double>(rowE, colE) += norm(flowVec);
                    netVecX.at<double>(rowE, colE) += flowVec.x;
                    netVecY.at<double>(rowE, colE) += flowVec.y;
                    float radius = norm(Point2f(netVecX.at<double>(rowE, colE), netVecY.at<double>(rowE, colE)));
                    float curMaxR = maxRadius.at<double>(rowE, colE);
                    if (radius > curMaxR) {
                        maxRadius.at<double>(rowE, colE) = radius;
                    }
                }
            }
        }
    }

    double maxVal = 0;
    for (int i = 0; i < maxRadius.rows; i++) {
        for (int j = 0; j < maxRadius.cols; j++) {
            if(maxRadius.at<double>(i,j) > maxVal)
                maxVal = maxRadius.at<double>(i,j);
        }
    }

   Mat normalizedRadius = maxRadius/maxVal;
   entropy = (length/maxRadius).mul(normalizedRadius);

   /* original paper method
   cv::log(length/maxRadius,entropy);
   entropy = (entropy/log(boxSize*boxSize-1)).mul(normalizedRadius);
*/
   maxVal = 0;
   for (int i = 0; i < entropy.rows; i++) {
       for (int j = 0; j < entropy.cols; j++) {
           if(entropy.at<double>(i,j) > maxVal)
               maxVal = entropy.at<double>(i,j);
       }
   }

   entropy = entropy/maxVal;
    //fast way
    /*Mat length = Mat::zeros(entropy.rows, entropy.cols, CV_32F);
    Mat_<Point2f> vecs(entropy.rows, entropy.cols, Point2f(0,0));

    Mat lengthF = Mat::zeros(entropy.rows, entropy.cols, CV_32F);
    Mat_<Point2f> vecsF(entropy.rows, entropy.cols, Point2f(0,0));

    //first row of col sums
    for(int x = 0; x < flow.cols; x++) {
        for (int y = 0; y < 2*boxRadius+1; y++) {
            Point2f flowVec = flow.at<Point2f>(y, x);
            vecs.at<Point2f>(boxRadius,x) += flowVec;
            length.at<double>(boxRadius,x) += norm(flowVec);
            assert(length.at<double>(y,x) >= 0);
        }
    }

    //now incrementally populate down the rows
    for(int x = 0; x < flow.cols; x++) {
        for(int y = boxRadius+1; y < flow.rows-boxRadius; y++) {
            Point2f toRemove = flow.at<Point2f>(y-boxRadius-1, x);
            Point2f toAdd = flow.at<Point2f>(y+boxRadius, x);
            vecs.at<Point2f>(y, x) = vecs.at<Point2f>(y-1, x)-toRemove+toAdd;
            length.at<double>(y,x) = length.at<double>(y-1, x)-norm(toRemove)+norm(toAdd);
            if (length.at<double>(y,x) < 0)
                std::cout << length.at<double>(y, x) << std::endl;
            //assert(length.at<double>(y,x) >= 0);
        }
    }

    //now we have all vertical sums, so we need to sum horizontally in groups of boxSize
    for (int y = boxRadius; y < flow.rows-boxRadius; y++) {
        for (int x = boxRadius; x < flow.cols-boxRadius; x++) {
            for (int k = -boxRadius; k <= boxRadius; k++)
            {
                vecsF.at<Point2f>(y,x) += flowVec.at<double>(y,x+k);
                lengthF.at<double>(y,x) += length.at<double>(y,x+k);
            }
        }
    }
    //negative length issue
    //radius update?*/
}

//current issues:
//water too far away (false negatives)
//buildings and structures above water (false positives)

int main(int argc, char *argv[])
{
    VideoCapture cap("../media/test2.mp4"); // open the default camera
    if(!cap.isOpened()) {  // check if we succeeded
       std::cout << "no vid" << std::endl;
       return -1;
    }

    int numFrames = 5;


    std::vector<Mat> flows;

    Mat frame1, frame2;
    cap >> frame1;
    resize(frame1, frame1, Size(frame1.cols/4,frame1.rows/4));

    int framecounter = 0;
    Mat flow, flowOut, cflow;
    UMat uflow;
    Mat gray1;
    Mat gray2;



    for(;;)
    {
        cap >> frame2; // get a new frame fro m camera
        if (frame2.rows == 0 || frame2.cols == 0)
            continue;

        framecounter++;
        resize(frame2, frame2, Size(frame2.cols/4, frame2.rows/4));

        imshow("orig", frame2);
        cvtColor( frame1, gray1, CV_BGR2GRAY );
        cvtColor( frame2, gray2, CV_BGR2GRAY );

        //get y,r,g,b stats

/*CV_EXPORTS_W void calcOpticalFlowFarneback( InputArray prev, InputArray next, InputOutputArray flow,
        double pyr_scale, int levels, int winsize,
        int iterations, int poly_n, double poly_sigma,
        int flags );*/
        calcOpticalFlowFarneback(gray1,gray2, uflow, 0.0, 1, 3, 1, 5, 1.2, 0);
        uflow.copyTo(flow);
        Mat dissimiliarity = Mat::ones(flow.rows, flow.cols, CV_64F);
       calcDissimilarity(flow, dissimiliarity, 1);

        cvtColor(gray1, cflow, COLOR_GRAY2BGR);

        if (flows.size() == numFrames) {
            flows.erase(flows.begin());
        }

        flows.push_back(flow);

        drawOptFlowMap(flow, cflow, 1, 1.5, Scalar(255, 255, 255));
        imshow("flow",cflow);

        if(flows.size() == numFrames) {
            Mat entropy = Mat::zeros(gray1.rows, gray1.cols, CV_64F);
            calcEntropyTime(flows, entropy);
            entropy = entropy.mul(dissimiliarity);
            entropy *= 255;
            entropy.convertTo(entropy, CV_8UC1);
            threshold(entropy, entropy, 32, 255, THRESH_BINARY);

          //  GaussianBlur(entropy, entropy, Size(3, 3), 1);
          //  threshold(entropy, entropy, 16, 255, THRESH_BINARY);

            imshow("time entropy", entropy);

            //need to convert this to a Mat so we can use calcCovarMatrix and mahalbonois operator
              //give everything below the mean full weight, and do a linear fall off of rgb weights for everything above (half way up)
              //but then how do we calculate the covariance matrix (weighted right)
              //maybe start with uniform weights to make things easy

              //make sure we don't eliminate too much water
              //maybe do a local stats

              //once we have a proper rgb mean and covariance we can just use halabanois distance on final to remove pixels that are probably boats
              //then we can use this as a backup check with either of the other two methods to filter our shiny parts and reflections, which this method is more sensitve to
/*

             int numPix = 0;
             for (int r = 0; r < entropy.rows; r++) {
                 for (int c = 0; c < entropy.cols; c++) {
                     if (final.at<int>(r,c) == 255) {
                         std::vector<int> pixVals;
                         pixVals.resize(5);
                         pixVals[0] = r;
                         pixVals[1] = c;
                         cv::Vec3b rgb = frame2.at<cv::Vec3b>(r,c);
                         pixVals[2] = rgb[0];
                         pixVals[3] = rgb[1];
                         pixVals[4] = rgb[2];
                         waterPix.push_back(pixVals);
                         numPix++;
                     }
                 }
             }
             numWaterPix.push_back(numPix);
             if (frameCounter == numWaterFrames) {
                 frameCounter = 0;
                 waterPix.erase(waterPix.begin(), waterPix.begin()+numWaterPix[0]);
             }
             numWaterPix.erase(numWaterPix.begin());*/


        }


        /*
       int radius = 10;
        Mat entropy = Mat::zeros(gray1.rows-2*radius, gray1.cols-2*radius, CV_32F);
        calcEntropySpace(flow, entropy, radius);

        //entropy calculate, now we display
        for (int i = 0; i < entropy.rows; i ++) {
            for (int j = 0; j < entropy.cols; j++) {
                if(entropy.at<double>(i,j) <= 0.25)
                    entropy.at<double>(i,j)  = 0;
                else
                    entropy.at<double>(i,j)  = 1.0;
            }
        }
        imshow("entropy", entropy);*/

        swap(frame1, frame2);
        waitKey(1);
    }

    return 0;
}
