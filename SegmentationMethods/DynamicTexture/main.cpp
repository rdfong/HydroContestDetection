#include <iostream>
#include <stdio.h>
#include <opencv2/opencv.hpp>

using namespace cv;

/**
 * @brief drawOptFlowMap Draw the optical flow map for flows that above a certain threshold
 * @param flow The flow vectors to draw
 * @param cflowmap The output map to draw the vectors on
 * @param step
 * @param color
 */
static void drawOptFlowMap(const Mat& flow, Mat& cflowmap)
{
    float maxNorm = 0;
    for(int y = 0; y < cflowmap.rows; y ++) {
        for(int x = 0; x < cflowmap.cols; x ++)
        {
            const Point2f& fxy = flow.at<Point2f>(y, x);
            float n = norm(fxy);
            if (n > maxNorm)
                maxNorm = n;
        }
    }

    for(int y = 0; y < cflowmap.rows; y ++)
        for(int x = 0; x < cflowmap.cols; x ++)
        {
            const Point2f& fxy = flow.at<Point2f>(y, x);
               Point2f dir = fxy/maxNorm;
               if (norm(fxy)>0.5)
                 circle(cflowmap, Point(x,y), 1, Scalar((dir.x*255+255)/2, (dir.y*255+255)/2, 0), -1);
        }

}

/**
 * @brief calcDissimilarity Creates the dissimiliarity map as defined by the paper
 * @param flow              The flow map
 * @param dissimilarity     The output flow dissimiliarity map
 * @param radius            The radius to average
 */
void calcDissimilarity(Mat& flow, Mat& dissimilarity, int radius) {
    if (radius == 0) {
        return;
    }
    //This first loop calculat
    Mat averageDifference = Mat::zeros(dissimilarity.rows, dissimilarity.cols, CV_64F);
    int rows = flow.rows;
    int cols = flow.cols;
    for (int r = 0; r < rows; r++) {
        for (int c = 0; c < cols; c++) {
            float totalMag = 0;
            float sum = 0;
             Point2f curPoint = flow.at<Point2f>(r, c);
             for (int rOff = -radius; rOff <= radius; rOff++) {
                 for (int cOff = -radius; cOff <= radius; cOff++) {
                     int rIndex = r+rOff;
                     int cIndex = c+cOff;
                     if (rIndex >= 0 && rIndex < rows && cIndex >= 0 && cIndex < cols) {
                        Point2f nPoint = flow.at<Point2f>(rIndex, cIndex);
                        sum += norm(nPoint-curPoint);
                        totalMag += norm(nPoint);
                     }
                 }
             }
            averageDifference.at<double>(r,c) = sum/totalMag;
        }
     }

    //Minimum dissimiliarity post processsing
    float maxDissimilarity = 0;
    for (int r = 0; r < rows; r++) {
        for (int c = 0; c < cols; c++) {
             float curD = averageDifference.at<double>(r, c);
             for (int rOff = -radius; rOff <= radius; rOff++) {
                 for (int cOff = -radius; cOff <= radius; cOff++) {
                     int rIndex = r+rOff;
                     int cIndex = c+cOff;
                     //in bounds
                     if (rIndex >= 0 && rIndex < rows && cIndex >= 0 && cIndex < cols) {
                        float neighD = averageDifference.at<double>(rIndex, cIndex);
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

    //Normalize dissimiliarity
    dissimilarity /= maxDissimilarity;
}

/**
 * @brief calcEntropyTime Calculate the entropy across multiple frames by accumulating angle changes between the collected frames
 * @param flows Vector of numFrames flow maps
 * @param entropy Resulting entropy map
 */
void calcEntropyTime(std::vector<Mat>& flows, Mat& entropy) {
    //Mat diff;
    for (int f = 1; f < flows.size(); f++) {
        Mat f0 = flows[f];
        Mat f1 = flows[f-1];
        for (int r = 0; r < entropy.rows; r++) {
            for (int c = 0; c < entropy.cols; c++) {
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
    //Normalize the entropy map
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

//Weaknesses of the method:
//water too far away show no entropy
//Anything above the water line should not be considered though horizon line detection though this can be simply mitigated with horizon line information
int main(int argc, char *argv[])
{
    VideoCapture cap("../../TestMedia/videos/boatm30.mp4"); // open the default camera
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
        //Performance is really size dependent since we are using a dense optical flow
        resize(frame2, frame2, Size(frame2.cols/4, frame2.rows/4));

        imshow("orig", frame2);
        cvtColor( frame1, gray1, CV_BGR2GRAY );
        cvtColor( frame2, gray2, CV_BGR2GRAY );

        //Get dense optical flow
        calcOpticalFlowFarneback(gray1,gray2, uflow, 0.0, 1, 3, 1, 5, 1.2, 0);
        uflow.copyTo(flow);
        Mat dissimiliarity = Mat::ones(flow.rows, flow.cols, CV_64F);
        calcDissimilarity(flow, dissimiliarity, 1);

        cvtColor(gray1, cflow, COLOR_GRAY2BGR);
        //We only take into account numFrames number of previous frames to get the entropy
        if (flows.size() == numFrames) {
            flows.erase(flows.begin());
        }
        flows.push_back(flow);

        drawOptFlowMap(flow, cflow);
        imshow("flow",cflow);

        //When the number of desired frames is reached we calculate the entropy from all the flow matrices
        if(flows.size() == numFrames) {
            Mat entropy = Mat::zeros(gray1.rows, gray1.cols, CV_64F);
            calcEntropyTime(flows, entropy);
            //Multiply the entropy with the dissimiliarity map and threshold it to get our final entropy map
            entropy = entropy.mul(dissimiliarity);
            entropy *= 255;
            entropy.convertTo(entropy, CV_8UC1);
            threshold(entropy, entropy, 32, 255, THRESH_BINARY);

            imshow("Entropy", entropy);
        }

        //swap current frame container
        swap(frame1, frame2);
        waitKey(1);
    }

    return 0;
}
