#include <opencv2/opencv.hpp>
#include <eigen3/Eigen/Dense>
#include <opencv2/core/eigen.hpp>
using namespace cv;

#define MAX_DIFF UINT_FAST8_MAX
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


void createVertexGrid(int rows, int cols);
void resetNodes();
void updateVertexGridWeights(Mat& im);
void setSeedNodes(Mat& seedNodeMap);
void initializeDiffBins();
void visualizeMST(Mat im);
void insert(int weight, vNode** n);
void remove(vNode* node);
vNode* extractMin();
void createMST(Mat& im);
void passUp();
void passDown();
void getDissimiliarityImage(std::vector<cv::Point3f>& boundaryPixels, Mat& in, Mat& out);
void getBoundaryPix(Mat& color_im, std::vector<cv::Point3f>& boundaryPixels, int boundarySize);
void getMBDImage(Mat& mbd_image);
void treeFilter(Mat& dis_image, Mat& mbd_image, int size, float sigD);
void postProcessing(Mat& combined);
