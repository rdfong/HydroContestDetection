#include "MST.h"
#include "custom_bitset.h"

//root is vNodes.begin()
std::vector<vNode> vNodes;
std::vector<vNode*> leaves;
custom_bitset<256> diffMap;
std::vector<vNode*> diffBins(256);
std::vector<vNode> dummyNodes;
const int K = 3;

void createVertexGrid(int rows, int cols) {
    vNodes.resize(rows*cols);

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
        v_col++;
    }
    //for the last element in the first row
    curNode = &*v_col;
    curNode->row = 0;
    curNode->col = col;
    v_col++;

    //Loop through vertex grid, connecting up and across vertices
    for (row = 1; row < rows; row++) {
        for (col = 0; col < cols-1; col++) {
            curNode = &*v_col;
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
        node->seedNode = false;
        node->distance = UINT8_MAX+1;
        v_it++;
    }
}

void updateVertexGridWeights(Mat& im) {
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

void setSeedNodes(Mat &seedNodeMap) {
    auto v_it = vNodes.begin();
    vNode *node = &*v_it;
    //We have to set upper left corner to a seed node
    node->seedNode = true;
    for (int row = 0; row < seedNodeMap.rows; row++) {
        unsigned char* seedNodeRow = seedNodeMap.ptr<uint8_t>(row);
        for (int col = 0; col < seedNodeMap.cols; col++) {
            node = &*v_it;
            if (seedNodeRow[col] == 255) {
                node->seedNode = true;
            }
            v_it++;
        }
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

void createMST(Mat& im) {
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
        } else if (curNode->parentEdge != NONE){
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

void getMBDImage(Mat& mbd_image) {
    auto im_it = mbd_image.begin<float>();
    auto v_it = vNodes.begin();
    vNode* curNode;
    for (int row = 0; row < mbd_image.rows; row++) {
       for (int col = 0; col < mbd_image.cols; col++) {
           curNode = &*v_it;
           (*im_it) = curNode->distance/255.0;
           ++v_it;
           ++im_it;
       }
    }
}

void getBoundaryPix(Mat& color_im, std::vector<cv::Point3f>& boundaryPixels, int boundarySize) {
    auto color_it = color_im.begin<cv::Vec3b>();
    cv::Vec3b color;
    int count = 0;
    for (int row = 0; row < color_im.rows; row++) {
       for (int col = 0; col < color_im.cols; col++) {
           if (row < boundarySize || row >= color_im.rows-boundarySize || col < boundarySize || col >= color_im.cols-boundarySize) {
               color = *color_it;
               boundaryPixels[count] = (cv::Point3f(color[0], color[1], color[2]));
               count++;
           }
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

void postProcessing(Mat& combined) {
    Mat intermediate;
    Mat rawCombined = combined.clone();
    combined = rawCombined*255;
    combined.convertTo(combined, CV_8U);

    double tau = threshold(combined, intermediate, 0, 255, THRESH_OTSU);
    int gamma = 20;
    cv::exp(-gamma*(rawCombined-tau/255.0), intermediate);
    combined = 1.0/(1.0+intermediate);
    combined*=255;
    combined.convertTo(combined, CV_8U);
}
