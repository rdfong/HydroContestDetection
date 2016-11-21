#include <iostream>
#include <stdio.h>
#include <opencv2/opencv.hpp>
#include "custom_bitset.h"

using namespace cv;

#define MAX_DIFF UINT_FAST8_MAX

#define VIDEO 0

enum Edge{UP = 0, RIGHT = 1, DOWN = 2, LEFT = 3, NONE = 4};


struct vNode {
    //keeps track of both the current best edge for a node and consequently the final edge in the MST
    Edge parentEdge = NONE;
    //keeps track of existence of children in the final MST, in order, UP, RIGHT, DOWN, LEFT (cw from top)
    std::vector<vNode*> childEdges;

    vNode **mapPrev = 0;
    vNode **mapNext = 0;


    vNode *neighbours[4] = {0,0,0,0};
    //5th one is just for initialization purposes
    uint8_t weights[5] = {MAX_DIFF,MAX_DIFF,MAX_DIFF,MAX_DIFF,MAX_DIFF};
    uint8_t value;
    //for pass up traversal
    uint8_t numTimesVisited;

    int distance = UINT8_MAX+1;

    uint8_t pathMin = 0;
    uint8_t pathMax = 0;
    //for display and debugging purposes
    int row = 0;
    int col = 0;

    bool seedNode = false;
    bool inForest = false;
};

//root is vNodes.begin()
std::vector<vNode> vNodes;
std::vector<vNode*> leaves;
custom_bitset<256> diffMap;
std::vector<vNode*> diffBins(256);
std::vector<vNode> dummyNodes;

void createVertexGrid(int rows, int cols) {
    auto v_col = vNodes.begin();
    auto v_prev_row = vNodes.begin();
    int row, col;

    vNode *curNode;
    vNode *nextNode;
    for (col = 0; col < cols-1; col++) {
        curNode = &*v_col;
        curNode->row = 0;
        curNode->col = col;
        auto v_next = v_col+1;
        nextNode = &*v_next;
        curNode->neighbours[RIGHT] = nextNode;
        nextNode->neighbours[LEFT] = curNode;
        curNode->seedNode = true;
        v_col++;
    }
    //for the last element in the first row
    curNode = &*v_col;
    curNode->row = 0;
    curNode->col = col;
    curNode->seedNode = true;
    v_col++;

    //Loop through vertex grid, connecting up and across vertices
    for (row = 1; row < rows; row++) {
        for (col = 0; col < cols-1; col++) {
            curNode = &*v_col;
            if (col == 0 || row == rows-1)
                curNode->seedNode = true;
            curNode->row = row;
            curNode->col = col;
            //set vertical neighbour
            nextNode = &*v_prev_row;
            curNode->neighbours[UP] = nextNode;
            nextNode->neighbours[DOWN] = curNode;
            //set horizontal neighbour
            auto v_next = v_col+1;
            nextNode = &*v_next;
            curNode->neighbours[RIGHT] = nextNode;
            nextNode->neighbours[LEFT] = curNode;
            //advance to next column
            v_col++;
            v_prev_row++;
        }

        //set last element in row
        curNode = &*v_col;
        curNode->row = row;
        curNode->col = col;
        curNode->seedNode = true;

        nextNode = &*v_prev_row;
        curNode->neighbours[UP] = nextNode;
        nextNode->neighbours[DOWN] = curNode;

        //move both pointers to first element of next rows
        v_col++;
        v_prev_row++;
    }
}

void resetNodes() {
    auto v_it = vNodes.begin();
    vNode *node;
    while(v_it != vNodes.end()) {
        node = &*v_it;
        node->childEdges.clear();
        node->parentEdge = NONE;
        node->inForest = false;
        node->mapNext = node->mapPrev = 0;
        node->numTimesVisited = 0;
        node->value = 0;
        node->pathMax = 0;
        node->pathMin = 0;
        node->distance = UINT8_MAX+1;
        v_it++;
    }
}

void updateVertexGridWeights(Mat im) {
    //clear all the arrays but vNodes
    leaves.clear();
    assert(diffMap.none());
    resetNodes();

    auto v_col = vNodes.begin();
    auto v_prev_row = vNodes.begin();

    int row, col, diff;
    uint8_t curVal, neighbourVal;
    auto im_it = im.begin<uint8_t>();
    curVal = *im_it;

    vNode *curNode;
    vNode *nextNode;
    row = 0;
    for (col = 0; col < im.cols-1; col++) {
        curNode = &*v_col;
        auto v_next = v_col+1;
        nextNode = &*v_next;
        neighbourVal = *(++im_it);
        curNode->weights[RIGHT] = nextNode->weights[LEFT] = abs(neighbourVal-curVal);
        curNode->value = curVal;
        curVal = neighbourVal;
        v_col++;
    }
    //for the last element in the first row
    curNode = &*v_col;
    curNode->value = curVal;

    //now move to next row
    v_col++;
    im_it++;

    auto prev_row_im_it = im.begin<uint8_t>();
    curVal = *im_it;

    //Loop through vertex grid, connecting up and across vertices
    for (row = 1; row < im.rows; row++) {
        for (col = 0; col < im.cols-1; col++) {
            curNode = &*v_col;
            curNode->value = curVal;
            //set vertical neighbour
            nextNode = &*v_prev_row;
            neighbourVal = *prev_row_im_it;
            ++prev_row_im_it;
            curNode->weights[UP] = nextNode->weights[DOWN] = abs(neighbourVal-curVal);

            //set horizontal neighbour
            auto v_next = v_col+1;
            nextNode = &*v_next;
            neighbourVal = *(++im_it);
            curNode->weights[RIGHT] = nextNode->weights[LEFT] = abs(neighbourVal-curVal);

            //advance to next column
            curVal = neighbourVal;
            v_col++;
            v_prev_row++;
        }

        //set last element in row
        curNode = &*v_col;
        curNode->value = curVal;
        nextNode = &*v_prev_row;
        neighbourVal = *prev_row_im_it;
        curNode->weights[UP] = nextNode->weights[DOWN] = abs(neighbourVal - curVal);

        //move both pointers to first element of next rows
        v_col++;
        v_prev_row++;
        ++im_it;
        ++prev_row_im_it;
    }
}

void initializeDiffBins() {
    dummyNodes.resize(256);
    for (int i = 0; i < 256; i++) {
        diffBins[i] = &dummyNodes[i];
    }
}

void visualizeMST(Mat im) {
    Mat imZeros = Mat::zeros(im.rows*20, im.cols*20, CV_8U);
    Mat MBDimage = Mat::zeros(im.rows, im.cols, CV_8U);
    vNode *root = &*vNodes.begin();
    auto v_it = vNodes.begin();
    for (int row = 0; row < im.rows; row++) {
        for (int col = 0; col < im.cols; col++) {
            vNode *curNode = &*v_it;
            //draw a dot for the current position
            circle(imZeros, Point2i(col*20+5,row*20+5),2,Scalar(255,255,255));
            if (curNode == root) {
                MBDimage.at<uint8_t>(curNode->row, curNode->col) =  curNode->distance;
                v_it++;
                continue;
            }
            assert(curNode->parentEdge != NONE);
            vNode *parentNode = curNode->neighbours[curNode->parentEdge];
            arrowedLine(imZeros, Point2i(parentNode->col*20+5,parentNode->row*20+5), Point2i(curNode->col*20+5,curNode->row*20+5),Scalar(255,255,255),1,8,0,0.25);

           /* for (int i = 0; i < curNode->childEdges.size(); i++) {
                vNode *childNode = curNode->childEdges[i];
                arrowedLine(imZeros, Point2i(curNode->col*20+5,curNode->row*20+5), Point2i(childNode->col*20+5,childNode->row*20+5),Scalar(255,255,255),1,8,0,0.25);
            }*/
            //draw a line to the neighbour point, but make sure to offset so we can see how many edges there are

            MBDimage.at<uint8_t>(curNode->row, curNode->col) = curNode->distance == UINT8_MAX+1? 128: curNode->distance;
            v_it++;
        }
    }
    //imshow("test", imZeros);

    float scale = 1.0;
    Size size(scale*MBDimage.cols, scale*MBDimage.rows);
    Mat scaledImage;
    resize(MBDimage, scaledImage, size, 0,0, INTER_NEAREST);
    imshow("mbd", scaledImage);

    resize(im, scaledImage, size, 0,0, INTER_NEAREST);
    imshow("orig", scaledImage);
}

void insert(int weight, vNode** n) {
    assert(n);
    vNode *node = *n;
    diffMap[weight] = 1;
    vNode **existingNode = diffBins[weight]->mapNext;
    node->mapNext = existingNode;
    node->mapPrev = &diffBins[weight];
    if (existingNode) {
        (*existingNode)->mapPrev = n;
    }
    diffBins[weight]->mapNext = n;
}

void remove(vNode* node) {
    (*(node->mapPrev))->mapNext = node->mapNext;
    if (node->mapNext)
        (*(node->mapNext))->mapPrev = node->mapPrev;
    node->mapNext = node->mapPrev = nullptr;
    int weight = node->weights[node->parentEdge];
    diffMap.set(weight, (diffBins[weight]->mapNext != nullptr));
}

vNode* extractMin() {
    int first;// = diffMap._Find_first();
    for (int i = 0; i < 256; i++) {
        if (diffMap.test(i)) {
            first = i;
            break;
        }
    }
    assert(diffBins[first]->mapNext);
    vNode *min = *(diffBins[first]->mapNext);
    remove(min);
    return min;
}

void createMST(Mat im) {
    assert(im.rows >= 2 && im.cols >= 2);
    vNode *root = &*vNodes.begin();
    //special case the root since it has no parent node
    root->inForest = true;

    vNode *rightNode = root->neighbours[RIGHT];
    vNode *downNode = root->neighbours[DOWN];
    //now add left and right neighbours, to the priority queue
    insert(root->weights[RIGHT], &(root->neighbours[RIGHT]));
    insert(root->weights[DOWN], &(root->neighbours[DOWN]));
    rightNode->parentEdge = LEFT;
    downNode->parentEdge = UP;

    int neighbourWeight, i;
    vNode *neighbourNode, *min;
    while (diffMap.any()) {
        min = extractMin();
        min->inForest = true;
        assert(min->parentEdge != NONE);
        (min->neighbours[min->parentEdge])->childEdges.push_back(min);

        for (i = 0; i < 4; i++) {
            neighbourNode = min->neighbours[i];
            neighbourWeight = min->weights[i];
            if (neighbourNode && !neighbourNode->inForest && neighbourNode->weights[neighbourNode->parentEdge] > neighbourWeight) {
                if (neighbourNode->parentEdge != NONE)
                    remove(neighbourNode);
                insert(neighbourWeight,&(min->neighbours[i]));
                neighbourNode->parentEdge = (Edge)((i+2)%4);
            }
        }
    }
}

void passUp() {
    //get leaves first
    std::queue<vNode*> bfsQueue;
    vNode *root = &*vNodes.begin();
    bfsQueue.push(root);
    while (!bfsQueue.empty()) {
        vNode *curNode = bfsQueue.front();
        bfsQueue.pop();
        if (curNode->childEdges.empty()) {
            leaves.push_back(curNode);
        } else {
            for (int i = 0; i < curNode->childEdges.size(); i++) {
                bfsQueue.push(curNode->childEdges[i]);
            }
        }
    }

    //now we can do the actual pass up
    vNode *curNode, *parentNode;
    for(int i = 0; i < leaves.size(); i++) {
        curNode = leaves[i];
        while (curNode->parentEdge != NONE) {
            if(curNode->numTimesVisited > 0  && curNode->numTimesVisited < curNode->childEdges.size()) {
                break;
            }

            if (curNode->seedNode) {
                curNode->distance = 0;
                curNode->pathMin = curNode->pathMax = curNode->value;
            }
            parentNode = curNode->neighbours[curNode->parentEdge];
            //don't want to update distances for anything that isn't connected to a seed node from below

            //if path originated from a seed node
            if (curNode->distance != UINT8_MAX+1) {
                int pathMin, pathMax;
                if (parentNode->value < curNode->pathMin) {
                    pathMin = parentNode->value;
                    pathMax = curNode->pathMax;
                } else if (parentNode->value > curNode->pathMax) {
                    pathMax = parentNode->value;
                    pathMin = curNode->pathMin;
                }
                if (parentNode->distance > pathMax-pathMin) {
                    parentNode->pathMin = pathMin;
                    parentNode->pathMax = pathMax;
                    parentNode->distance = pathMax-pathMin;
                }
            }
            parentNode->numTimesVisited++;
            curNode = parentNode;
        }
    }
}

void passDown() {
    std::queue<vNode*> bfsQueue;
    vNode *curNode;
    bfsQueue.push(&(*vNodes.begin()));
    while (!bfsQueue.empty()) {
        curNode = bfsQueue.front();
        bfsQueue.pop();
        if (curNode->seedNode) {
            curNode->distance = 0;
            curNode->pathMin = curNode->pathMax = curNode->value;
        } else {
            vNode *parentNode = curNode->neighbours[curNode->parentEdge];
            int pathMin, pathMax;
            if (curNode->value < parentNode->pathMin) {
                pathMin = curNode->value;
                pathMax = parentNode->pathMax;
            }
            else if (curNode->value > parentNode->pathMax) {
                pathMax = curNode->value;
                pathMin = parentNode->pathMin;
            }
            else {
                pathMin = parentNode->pathMin;
                pathMax = parentNode->pathMax;
            }
            if (curNode->distance > pathMax-pathMin) {
                 curNode->pathMin = pathMin;
                 curNode->pathMax = pathMax;
                 curNode->distance = pathMax - pathMin;
            }
        }

        for (int i = 0; i < curNode->childEdges.size(); i++) {
            bfsQueue.push(curNode->childEdges[i]);
        }
    }
}

void getBoundingBoxes(Mat binIm) {

}

const int K = 3;
void getDissimiliarityImage(std::vector<cv::Point3f>& boundaryPixels, Mat&in, Mat& out) {
    Mat labels;
    int numPix = boundaryPixels.size();
    kmeans(boundaryPixels,K,labels,TermCriteria(TermCriteria::EPS+TermCriteria::COUNT, 3, 0.001), 1, KMEANS_RANDOM_CENTERS);
    uint labelCount[K] = {0,0,0};
    std::vector<std::vector<cv::Point3f> > clusterPoints(3);
    auto label_it = labels.begin<int>();
    for (int row = 0; row < labels.rows; row++) {
        int label = *label_it;
        labelCount[label]++;
        clusterPoints[label].push_back(boundaryPixels[row]);
        ++label_it;
    }

    std::vector<Mat> backgroundDisMaps(K);
    std::vector<Mat> backgroundMeans(K);
    std::vector<Mat> coVarMats(K);

    for (int k = 0; k < K; k++) {
        backgroundDisMaps[k] = Mat::zeros(out.rows, out.cols, CV_32F);
        Mat pointsMat = Mat::zeros(clusterPoints[k].size(), 3, CV_32F);
        auto points_it = pointsMat.begin<float>();
        for (int i = 0; i < clusterPoints[k].size(); i++) {
            cv::Point3f p = clusterPoints[k][i];
            *(points_it) = p.x;
            *(++points_it) = p.y;
            *(++points_it) = p.z;
             ++points_it;
        }
        Mat mean;
        reduce(pointsMat,mean, 0, CV_REDUCE_AVG);
        backgroundMeans[k] = mean;
    }

    auto out_it_0 = backgroundDisMaps[0].begin<float>();
    auto out_it_1 = backgroundDisMaps[1].begin<float>();
    auto out_it_2 = backgroundDisMaps[2].begin<float>();
    auto in_it = in.begin<cv::Vec3b>();
    cv::Point3f curColor, diff0, diff1, diff2;
   for (int r = 0; r < in.rows; r++) {
        for (int c = 0; c < in.cols; c++) {
            curColor = cv::Point3f(*in_it);
            diff0 = (curColor - cv::Point3f(backgroundMeans[0]));
            diff1 = (curColor - cv::Point3f(backgroundMeans[1]));
            diff2 = (curColor - cv::Point3f(backgroundMeans[2]));
            *out_it_0 = diff0.dot(diff0);
            *out_it_1 = diff1.dot(diff1);
            *out_it_2 = diff2.dot(diff2);

            ++in_it;
            ++out_it_0;
            ++out_it_1;
            ++out_it_2;
        }
    }

    for (int k = 0; k < K; k++) {
        double minVal;
        double maxVal;
        cv::minMaxLoc(backgroundDisMaps[k], &minVal, &maxVal);
        backgroundDisMaps[k] /= maxVal;
        out += ((float)labelCount[k]/(float)numPix*backgroundDisMaps[k]);
    }
}

void getMBDImageAndBoundaryPix(Mat& color_im, Mat& mbd_image, std::vector<cv::Point3f>& boundaryPixels, int boundarySize) {
    auto im_it = mbd_image.begin<float>();
    auto v_it = vNodes.begin();
    auto color_it = color_im.begin<cv::Vec3b>();
    vNode* curNode;
    cv::Vec3b color;
    int count = 0;
    for (int row = 0; row < mbd_image.rows; row++) {
       for (int col = 0; col < mbd_image.cols; col++) {
           curNode = &*v_it;
           (*im_it) = curNode->distance/255.0;
           if (row < boundarySize || row >= mbd_image.rows-boundarySize || col < boundarySize || col >= mbd_image.cols-boundarySize) {
               color = *color_it;
               boundaryPixels[count] = (cv::Point3f(color[0], color[1], color[2]));
               count++;
           }
           ++v_it;
           ++im_it;
           ++color_it;
       }
    }
    assert(boundaryPixels.size() == count);
}

void treeFilter(Mat& dis_image, Mat& mbd_image, int size, float sigD) {
    int row, col, rowStart, rowEnd, colStart, colEnd, fRow, fCol;
    float treeDiff, curIntensity, weight, curTreeDist, total_bilateral_weight, total_bilateral_result;
    Mat result = Mat::zeros(dis_image.rows, dis_image.cols, CV_32F);
    std::vector<float*>dis_image_row(dis_image.rows);
    std::vector<float*>mbd_image_row(mbd_image.rows);
    for (row = 0; row < dis_image.rows; row++) {
        dis_image_row[row] = (float*)dis_image.ptr(row);
        mbd_image_row[row] = (float*)mbd_image.ptr(row);
    }
    for (row = 0; row < dis_image.rows; row++) {
        for (col = 0; col < dis_image.cols; col++) {
            rowStart = max(row-size, 0);
            rowEnd = min(row+size, dis_image.rows-1);
            colStart = max(col-size, 0);
            colEnd = min(col+size, dis_image.cols-1);
            curTreeDist = mbd_image_row[row][col];
            total_bilateral_weight = 0;
            total_bilateral_result = 0;
            //filter each pixel and sum the total as we go so we can normalize at the end
            for (fRow = rowStart; fRow <= rowEnd; fRow++) {
                for (fCol = colStart; fCol <= colEnd; fCol++) {
                    treeDiff = std::abs(curTreeDist-mbd_image_row[fRow][fCol]);
                    curIntensity = dis_image_row[fRow][fCol];
                    weight = exp(-treeDiff*treeDiff/(2*sigD));
                    total_bilateral_result += weight*curIntensity;
                    total_bilateral_weight += weight;
                }
            }
            //normalize the result
            total_bilateral_result /= total_bilateral_weight;
            result.at<float>(row,col) = total_bilateral_result;
        }
    }
    dis_image = result;
}

int main(int argc, char *argv[])
{
    int erosion_size = 1;
    Mat element = getStructuringElement( MORPH_ELLIPSE,
                                         Size( 2*erosion_size + 1, 2*erosion_size+1 ),
                                         Point( erosion_size, erosion_size ) );


#if VIDEO == 0
    Mat image;
    image = imread("../../TestMedia/images/openwater.JPG", CV_LOAD_IMAGE_COLOR);
    if (!image.data)
    {
        printf("No image data \n");
        return -1;
    }


    //approximate size, 900 by 600
    float scale = 1.0;
    Size size(scale*image.cols, scale*image.rows);
    Mat scaledImage, gray_image, lab, mbd_image, dis_image, new_dis_image, combined, combined8, intermediate;

    resize(image, scaledImage, size);

    //-----------------------------------------------------------------
    //SETUP FREICHEN FILTER BANK
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
    Mat frei_image = Mat::zeros(scaledImage.rows, scaledImage.cols, CV_32F);
    Mat m_term = Mat::zeros(scaledImage.rows, scaledImage.cols, CV_32F);
    Mat s_term = Mat::zeros(scaledImage.rows, scaledImage.cols, CV_32F);

    //-----------------------------------------------------------------




    vNodes.resize(scaledImage.rows*scaledImage.cols);//should only ever call thisonce
    createVertexGrid(scaledImage.rows, scaledImage.cols);
    initializeDiffBins();

    int64 t1 = getTickCount();

    cvtColor(scaledImage, gray_image, CV_BGR2GRAY );

     GaussianBlur(gray_image, gray_image, Size(3, 3), 1);
    updateVertexGridWeights(gray_image);
    createMST(gray_image);
     passUp();
     passDown();

     cvtColor(scaledImage, lab, CV_BGR2Lab);
     int boundary_size = 10;
     int num_boundary_pixels = (boundary_size*2*(gray_image.cols+gray_image.rows)-4*boundary_size*boundary_size);
     std::vector<cv::Point3f> boundaryPixels(num_boundary_pixels);
     mbd_image = Mat::zeros(gray_image.rows, gray_image.cols, CV_32FC1);
     getMBDImageAndBoundaryPix(lab, mbd_image, boundaryPixels, boundary_size);

     dis_image = Mat::zeros(lab.rows, lab.cols, CV_32FC1);
     getDissimiliarityImage(boundaryPixels, lab, dis_image);

      treeFilter(dis_image, mbd_image, 3, 0.1);

     new_dis_image = Mat::zeros(lab.rows, lab.cols, CV_32FC1);
     bilateralFilter(dis_image, new_dis_image, 3, 0.1, 0.1);

     //FREICHEN: CONCLUSION: ONLY USE FOR TRACKING, STILL PRODUCES TOO MUCH NOISE IN WATER
    /* gray_image.convertTo(gray_image, CV_32F);
     for (int f = 0; f < 9; f++) {
       filter2D(gray_image, frei_image, -1 , fBank[f]);
       s_term += frei_image.mul(frei_image);
       if (f == 1) {
           m_term = s_term.clone();
       }
   }
   cv::sqrt(m_term/s_term, frei_image);

   cv::threshold(frei_image, frei_image, 0.1, 1.0, THRESH_BINARY);
   morphologyEx(frei_image, frei_image, MORPH_OPEN, element);
 morphologyEx(frei_image, frei_image, MORPH_DILATE, element);*/
     //combine images
     combined = mbd_image + new_dis_image;

     double minVal;
     double maxVal;
     cv::minMaxLoc(combined, &minVal, &maxVal);
     combined /= maxVal;



     // POST PROCESSING FROM PAPER
     combined8 = combined*255;
     combined8.convertTo(combined8, CV_8U);

     double tau = threshold(combined8, intermediate, 0, 255, THRESH_OTSU);
     int gamma = 20;
     cv::exp(-gamma*(combined-tau/255.0), intermediate);
     combined = 1.0/(1.0+intermediate);

      combined*=255;
      combined.convertTo(combined, CV_8U);

       morphologyEx(combined, combined, MORPH_OPEN, element);

     threshold(combined, combined, 0, 255, THRESH_OTSU);

     //MY POST PROCESSING
      morphologyEx(combined, combined, MORPH_DILATE, element);

     int64 t2 = getTickCount();
    imshow("mbd", mbd_image);
    imshow("dis_post", new_dis_image);
   // imshow("frei", frei_image);
    imshow("combined", combined);
    std::cout << "PER FRAME TIME: " << (t2 - t1)/getTickFrequency() << std::endl;

    //visualizeMST(gray_image);
    waitKey(0);
#elif VIDEO == 1
       VideoCapture cap("../media/boathm10.mp4"); // open the default camera
       if(!cap.isOpened()) {  // check if we succeeded
           std::cout << "no vid" << std::endl;
           return -1;
       }

       int framecounter = 0;
       vNodes.resize(image.rows*image.cols);

       Mat image, gray_image;
       cap >> image;
       createVertexGrid(image);
       initializeDiffBins();

       for(;;)
       {
           Mat image;
           cap >> image; // get a new frame from camera

           if (image.rows == 0 || image.cols == 0)
               continue;

           int64 t2 = getTickCount();

           cvtColor( image, gray_image, CV_BGR2HSV );

           updateVertexGridWeights(gray_image);

           GaussianBlur(gray_image, gray_image, Size(7, 7), 5);
           createMST(image);
           passUp();
           passDown();

           framecounter++;

           waitKey(1);
       }

#endif
    return 0;
}
