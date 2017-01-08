#include <iostream>
#include <fstream>
#include <stdio.h>
#include <opencv2/opencv.hpp>
#include "custom_bitset.h"

using namespace cv;

#define MAX_DIFF UINT_FAST8_MAX

//--------------BOX SELECTION AND MERGING CODE--------------//
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

//score output
ofstream scoreFile;
std::string name;
std::string output;
std::string waitString;
std::string imageFolder;

/**
 * @brief getNonZeroPix     Gets all values of pixels specified by the mask
 * @param mask              Mask of which pixels to get values from im
 * @param im                The image to retrieve values from
 * @param nonZeroSubset     The output of the desired values
 */
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

/**
 * @brief findContoursAndWriteResults   Finds all contours and surrounding bounding boxes, writes to file and displays
 * @param obstacleMap                   The binary map of obstacles
 * @param image                         The original image on top of which we'll write rectangles
 * @param scoreFile                     The file handle to write to
 * @param outputName                    The file name to write the bounding box coordinates and dimensions to
 */
void findContoursAndWriteResults(Mat& obstacleMap, Mat& image, bool display) {
    hierarchy.clear();
    contours.clear();
    boundRects.clear();
    originalRects.clear();
    findContours( obstacleMap.clone(), contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE, Point(0, 0) );
    //get bounding rects from contours
    int expand = 1;
    for (int i =0; i < contours.size(); i++) {
        curRect = boundingRect(contours[i]);
        //don't add a box if it is too long or too tall in aspect ratio
        if ((double)curRect.width/curRect.height < 0.05|| (double)curRect.height/curRect.width < 0.05)
            continue;
        Point2i newTL(max(curRect.tl().x-expand, 0), max(curRect.tl().y-expand,0));
        Point2i newBR(min(curRect.br().x+expand, obstacleMap.cols-1), min(curRect.br().y+expand,obstacleMap.rows-1));
        boundRects.push_back(Rect(newTL, newBR));
        originalRects.push_back(curRect);
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
                } else if (intersection.area() > 0) {
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
            scoreFile << "other\n" << curRect.tl().x << " " << curRect.tl().y << " "
                      << curRect.width << " " << curRect.height <<std::endl;
        }
    }

    imwrite(output+name, image);
}
//-------------------------------------------------------//

enum Edge{UP = 0, RIGHT = 1, DOWN = 2, LEFT = 3, NONE = 4};

/**
 * @brief The vNode struct  Keeps track of node information in the MST
 */
struct vNode {
    //keeps track of both the current best edge for a node and consequently the final edge in the MST
    Edge parentEdge = NONE;
    //keeps track of existence of children in the final MST, in order, UP, RIGHT, DOWN, LEFT (cw from top)
    std::vector<vNode*> childEdges;

    vNode **mapPrev = 0;
    vNode **mapNext = 0;


    vNode *neighbours[4] = {0,0,0,0};
    //5th entry is just for initialization purposes
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
const int K = 3;


/**
 * @brief createVertexGrid  Create the initial vertex grid which just defined
 * @param rows              Number of rows of the grid
 * @param cols              Number of columns of the grid
 */
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

/**
 * @brief initializeDiffBins    Initialize the priority queue used to efficiently extract the minimum weight vNode during Prim's algorithm
 */
void initializeDiffBins() {
    dummyNodes.resize(256);
    for (int i = 0; i < 256; i++) {
        diffBins[i] = &dummyNodes[i];
    }
}

/**
 * @brief resetNodes    Reset the node information
 */
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

/**
 * @brief updateVertexGridWeights   Based on the new image, update the weights between each node and its neighbours
 * @param im                        The new image
 */
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

/**
 * @brief insert    Insert a node into MST
 * @param weight    Weight of the node which represents it's current minimum weight connection with the rest of the tree
 * @param n         The node to be inserted
 */
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

/**
 * @brief remove    Remove a node from the diffMap
 * @param node      The node to be removed
 */
void remove(vNode* node) {
    (*(node->mapPrev))->mapNext = node->mapNext;
    if (node->mapNext)
        (*(node->mapNext))->mapPrev = node->mapPrev;
    node->mapNext = node->mapPrev = nullptr;
    int weight = node->weights[node->parentEdge];
    diffMap.set(weight, (diffBins[weight]->mapNext != nullptr));
}

/**
 * @brief extractMin    Extracting the node from diffMap that has the minimum edge weight from the existing tree
 * @return              The extracted node.
 */
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

/**
 * @brief createMST     Creates the minimum spanning tree from the nodes and edges previously calculated
 */
void createMST() {
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

/**
 * @brief visualizeMST  Display the MST with connections
 * @param im            The original image
 *                      Note: probably best not to run this on a large image since it will display a grid that is 20x larger
 */
void visualizeMST(Mat im) {
    Mat imZeros = Mat::zeros(im.rows*20, im.cols*20, CV_8U);
    auto v_it = vNodes.begin();
    for (int row = 0; row < im.rows; row++) {
        for (int col = 0; col < im.cols; col++) {
            vNode *curNode = &*v_it;
            //draw a dot for the current position
            circle(imZeros, Point2i(col*20+5,row*20+5),2,Scalar(255,255,255));
            for (int i = 0; i < curNode->childEdges.size(); i++) {
                vNode *childNode = curNode->childEdges[i];
                arrowedLine(imZeros, Point2i(curNode->col*20+5,curNode->row*20+5), Point2i(childNode->col*20+5,childNode->row*20+5),Scalar(255,255,255),1,8,0,0.25);
            }
            v_it++;
        }
    }
    imshow("MST", imZeros);
}

/**
 * @brief passUp Does a depth first pass up from the leaves of the tree to the root to calculate distance transform values
 *               passDown needs to be run after to complete the transform calculations
 */
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

/**
 * @brief passDown Does a root breadth first pass of a tree to finalize the tree distance transform values
 *                  This needs to be run after the passUp
 */
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

/**
 * @brief getDissimiliarityImage    Get the dissimiliarity map given the boundary pixels
 * @param boundaryPixels            A vector of pixels that comprise the boundary of the image
 * @param in                        The original full color image
 * @param out                       The dissimiliarity map to be outputted
 */
void getDissimiliarityImage(std::vector<cv::Point3f>& boundaryPixels, Mat&in, Mat& out) {
    Mat labels;
    int numPix = boundaryPixels.size();
    kmeans(boundaryPixels,K,labels,TermCriteria(TermCriteria::EPS+TermCriteria::COUNT, 3, 0.001), 1, KMEANS_RANDOM_CENTERS);
    uint labelCount[K] = {0,0,0};
    std::vector<std::vector<cv::Point3f> > clusterPoints(K);
    auto label_it = labels.begin<int>();
    for (int row = 0; row < labels.rows; row++) {
        int label = *label_it;
        labelCount[label]++;
        clusterPoints[label].push_back(boundaryPixels[row]);
        ++label_it;
    }

    std::vector<Mat> backgroundDisMaps(K);
    std::vector<Mat> backgroundMeans(K);

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
        backgroundMeans[k] = mean.clone();
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

/**
 * @brief getMSTImageAndBoundaryPix     Returns the tree distance transform image and boundary pixels in one loop
 * @param color_im                      The image to get the boundary pixels from
 * @param mbd_image                     The tree distance transform image to be returned
 * @param boundaryPixels                The boundary pixels vector to populate
 * @param boundarySize                  Size of boundary to be considered as background
 */
void getMSTImageAndBoundaryPix(Mat& color_im, Mat& mbd_image, std::vector<cv::Point3f>& boundaryPixels, int boundarySize) {
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

/**
 * @brief treeFilter    Blurs the image while retaining structural elements defined by the tree distance transform
 * @param dis_image     Dissimiliarity image to be blurred
 * @param mst_image     The distance transform image
 * @param size          Radius of the gaussian blur kernel
 * @param sigD          Sigma of the gaussian blur kernel
 */
void treeFilter(Mat& dis_image, Mat& mst_image, int size, float sigD) {
    int row, col, rowStart, rowEnd, colStart, colEnd, fRow, fCol;
    float treeDiff, curIntensity, weight, curTreeDist, total_bilateral_weight, total_bilateral_result;
    Mat result = Mat::zeros(dis_image.rows, dis_image.cols, CV_32F);
    std::vector<float*>dis_image_row(dis_image.rows);
    std::vector<float*>mbd_image_row(mst_image.rows);
    for (row = 0; row < dis_image.rows; row++) {
        dis_image_row[row] = (float*)dis_image.ptr(row);
        mbd_image_row[row] = (float*)mst_image.ptr(row);
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

/**
 * @brief customOtsuThreshold   Does otsu thresholding repeated until a certain percent of the image remains
 * @param im                    The grayscale image to be thresholded
 */
void customOtsuThreshold(Mat& im) {
    float percent = 1.0;
    int curThresh = 0;
    Mat threshed = Mat::ones(im.rows, im.cols, CV_8U);

    while (percent > .25 && curThresh <= 255) {
        Mat temp, nonZeroSubset;
        getNonZeroPix<unsigned char>(threshed, im, nonZeroSubset);
        int newThresh = (int)threshold(nonZeroSubset, temp, 0, 255, THRESH_OTSU);
        if (curThresh == newThresh)
            curThresh = 256;
        else {
            curThresh = newThresh;
        }
        threshed.setTo(0, im < curThresh);
        int countWhite = sum(threshed).val[0];
        percent = (float)countWhite/(im.rows*im.cols);
    }
    if (curThresh == 0)
        curThresh = 256;
    threshold(im, im, curThresh, 255, THRESH_BINARY);
}

int main(int argc, char *argv[])
{
    std::cout << "Path name: " << argv[1] <<std::endl;
    std::cout << "Image name: " << argv[2] <<std::endl;
    std::cout << "Output folder: " << argv[3] <<std::endl;
    std::cout << "Wait string: " << argv[4] << std::endl;
    imageFolder = std::string(argv[1]);
    name = std::string(argv[2]);
    output = std::string(argv[3]);
    waitString = std::string(argv[4]);

    Mat image;
    //const char* name = (const char *)argv[2];
    image = imread(strcat(argv[1], (const char*)name.c_str()), CV_LOAD_IMAGE_COLOR);
    scoreFile.open(strcat(argv[3],strcat((char *)name.data(),".txt")));
    if (!image.data)
    {
        printf("No image data \n");
        return -1;
    }

    float scale = 1.0;
    Size size(scale*image.cols, scale*image.rows);
    Mat scaledImage, gray_image, lab, mbd_image, dis_image, new_dis_image, combined, rawCombined, intermediate;

    resize(image, scaledImage, size);


    //SETUP FOR MST
    vNodes.resize(scaledImage.rows*scaledImage.cols);//should only ever call thisonce
    createVertexGrid(scaledImage.rows, scaledImage.cols);
    initializeDiffBins();

    cvtColor(scaledImage, gray_image, CV_BGR2GRAY );
    int64 t1 = getTickCount();
    //First blur image
    GaussianBlur(gray_image, gray_image, Size(3, 3), 2);
    //Take blurred image to update vertex grid weights and create the minimum spanning tree
    updateVertexGridWeights(gray_image);
    createMST();
    passUp();
    passDown();

    //Get boundary disimmiliarity image and blur it with the tree filter
    cvtColor(scaledImage, lab, CV_BGR2HSV);
    int boundary_size = 20;
    int num_boundary_pixels = (boundary_size*2*(gray_image.cols+gray_image.rows)-4*boundary_size*boundary_size);
    std::vector<cv::Point3f> boundaryPixels(num_boundary_pixels);
    mbd_image = Mat::zeros(gray_image.rows, gray_image.cols, CV_32FC1);
    getMSTImageAndBoundaryPix(lab, mbd_image, boundaryPixels, boundary_size);

    dis_image = Mat::zeros(lab.rows, lab.cols, CV_32FC1);
    getDissimiliarityImage(boundaryPixels, lab, dis_image);
    treeFilter(dis_image, mbd_image, 4, 0.5);
    new_dis_image = Mat::zeros(lab.rows, lab.cols, CV_32FC1);
    bilateralFilter(dis_image, new_dis_image, 4, 0.5, 0.5);


    //combine images
    rawCombined = mbd_image + new_dis_image;

    double minVal;
    double maxVal;
    cv::minMaxLoc(rawCombined, &minVal, &maxVal);
    rawCombined /= maxVal;


    // Post process and threshold the image
    combined = rawCombined*255;
    combined.convertTo(combined, CV_8U);

    double tau = threshold(combined, intermediate, 0, 255, THRESH_OTSU);
    int gamma = 20;
    cv::exp(-gamma*(rawCombined-tau/255.0), intermediate);
    combined = 1.0/(1.0+intermediate);

    combined*=255;
    combined.convertTo(combined, CV_8U);

    customOtsuThreshold(combined);

    //Find and write contours
    if (maxVal-minVal > 0.75)
        findContoursAndWriteResults(combined, scaledImage, true);

    int64 t2 = getTickCount();
    imshow("mbd", mbd_image);
    imshow("dis_post", new_dis_image);
    imshow("combined", combined);
    imshow("final result", scaledImage);
    std::cout << "PER FRAME TIME: " << (t2 - t1)/getTickFrequency() << std::endl;

    scoreFile.close();
    if (waitString == "WAIT")
        waitKey(0);
    return 0;
}
