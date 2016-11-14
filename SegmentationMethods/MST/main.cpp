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
    uint8_t numTimesVisted;

    int distance = UINT8_MAX+1;

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
        node->numTimesVisted = 0;
        node->value = 0;
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
            MBDimage.at<uint8_t>(curNode->row, curNode->col) = curNode->distance;
            v_it++;
        }
    }
   // imshow("test", imZeros);

    float scale = 1.0;
    Size size(scale*MBDimage.cols, scale*MBDimage.rows);
    imshow("orig", im);
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
        totalExtractTime += (t2-t1);
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
    auto l_it = leaves.begin();
    vNode *curNode, *parentNode;
    int pathMin, pathMax;
    while (l_it != leaves.end()) {
        curNode = *l_it;
        pathMin = pathMax = curNode->value;
        bool seedFound = false;
        while (curNode->parentEdge != NONE) {
            //put curnode at back if still waiting on subtree traversal
            if(curNode->numTimesVisted > 0  && curNode->numTimesVisted < curNode->childEdges.size()) {
                if (curNode->numTimesVisted == 1)
                    leaves.push_back(curNode);
                ++l_it;
                break;
            }

            seedFound = (curNode->seedNode || curNode->distance != UINT8_MAX+1);
            if (curNode->seedNode) {
                curNode->distance = 0;
                pathMin = pathMax = curNode->value;
            }
            parentNode = curNode->neighbours[curNode->parentEdge];
            //don't want to update distances for anything that isn't connected to a seed node from below
            if (seedFound) {
                if (parentNode->value < pathMin)
                    pathMin = parentNode->value;
                else if (parentNode->value > pathMax)
                    pathMax = parentNode->value;
                parentNode->distance = min(parentNode->distance, pathMax-pathMin);
            }
            parentNode->numTimesVisted++;
            curNode = parentNode;
        }
        ++l_it;
    }
}

void passDown() {
    std::queue<vNode*> bfsQueue;
    std::queue<std::pair<int, int> > pathMinMax;
    vNode *curNode = &(*vNodes.begin());
    if (curNode->seedNode) curNode->distance = 0;
    for (int i = 0; i < curNode->childEdges.size(); i++) {
        bfsQueue.push(curNode->childEdges[i]);
        pathMinMax.push(std::pair<int, int>(curNode->value, curNode->value));
    }

    //do a BFS pass down the tree, storing the path bounds for each potential path in the queue as we go
    std::pair<int,int> pathBounds;
    int pathMin, pathMax;
    while (!bfsQueue.empty()) {
        curNode = bfsQueue.front();
        bfsQueue.pop();
        pathBounds = pathMinMax.front();
        pathMinMax.pop();

        if (curNode->seedNode) {
            curNode->distance = 0;
            pathMin = pathMax = curNode->value;
        } else {
            if (curNode->value < pathBounds.first)
                pathMin = curNode->value;
            else if (curNode->value > pathBounds.second)
                pathMax = curNode->value;
            else {
                pathMin = pathBounds.first;
                pathMax = pathBounds.second;
            }
            curNode->distance = min(curNode->distance, pathMax-pathMin);
        }

        for (int i = 0; i < curNode->childEdges.size(); i++) {
            bfsQueue.push(curNode->childEdges[i]);
            pathMinMax.push(std::pair<int,int>(pathMin, pathMax));
        }
    }
}

void getBoundingBoxes(Mat binIm) {

}

int main(int argc, char *argv[])
{
#if VIDEO == 0
    Mat image;
    image = imread("../../TestMedia/images/720test.jpg", CV_LOAD_IMAGE_COLOR);
    if (!image.data)
    {
        printf("No image data \n");
        return -1;
    }

    //imshow("orig", image);
    //approximate size, 900 by 600
    float scale = 1.0;
    Size size(scale*image.cols, scale*image.rows);
    Mat scaledImage;
    resize(image, scaledImage, size);

    vNodes.resize(scaledImage.rows*scaledImage.cols);//should only ever call thisonce
    createVertexGrid(scaledImage.rows, scaledImage.cols);
    initializeDiffBins();

    Mat gray_image;
    cvtColor(scaledImage, gray_image, CV_BGR2GRAY );
    GaussianBlur(gray_image, gray_image, Size(7, 7), 5);
    updateVertexGridWeights(gray_image);
    int64 t1 = getTickCount();
    createMST(gray_image);
    int64 t2 = getTickCount();
     passUp();
     passDown();
    std::cout << "PER FRAME TIME: " << (t2 - t1)/getTickFrequency() << std::endl;

    visualizeMST(gray_image);
    waitKey(0);
#elif VIDEO == 1
       VideoCapture cap("../media/boatm10.mp4"); // open the default camera
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

           cvtColor( image, gray_image, CV_BGR2GRAY );
           updateVertexGridWeights(gray_image);

           GaussianBlur(gray_image, gray_image, Size(5, 5), 3);
           createMST(image);
           passUp();
          // passDown();

           framecounter++;

           waitKey(1);
       }

#endif
    return 0;
}
