#include <opencv2/opencv.hpp>
#include <eigen3/Eigen/Dense>
#include <opencv2/core/eigen.hpp>
using namespace cv;

#define MAX_DIFF UINT_FAST8_MAX
enum Edge{UP = 0, RIGHT = 1, DOWN = 2, LEFT = 3, NONE = 4};

//Node structure for the minimum spanning tree
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

//Creates the initial vertex grid structure, defining neighbourhoods between nodes (only run once initialization)
void createVertexGrid(int rows, int cols);

//Resets all information on the nodes besides neighbourhood info from createVertexGrid
void resetNodes();

//Initialize a map of intensity differences between pixels (edges) to be use in creating the minimum spanning tree
void initializeDiffBins();

//Sets which pixels are seed nodes based on a binary map (see findSeedNodes in GMM.cpp)
void setSeedNodes(Mat& seedNodeMap);

//Based on the new frame, update the intensity differences between neighbouring pixels
void updateVertexGridWeights(Mat& im);

//Based on the updated weights, create a minimum spanning tree, populating child and parent fields of each node
void createMST(Mat& im);

//Edge insertion, removal and extraction methods used for creating MST via Prim's algorithm
void insert(int weight, vNode** n);
void remove(vNode* node);
vNode* extractMin();

//Method to visualize the MST
void visualizeMST(Mat im);

//Pass up and down the tree to assign distance transform values to each node of the tree
void passUp();
void passDown();

//Get the MST distance transform image to be used with the boundary dissimiliarity map in retrieving the saliency map
void getMSTDistanceImage(Mat& mst_image);

//Get all the pixels on the boundary of the image
void getBoundaryPix(Mat& color_im, std::vector<cv::Point3f>& boundaryPixels, int boundarySize);

//Get the boundary dissimiliarity map
void getDissimiliarityImage(std::vector<cv::Point3f>& boundaryPixels, Mat& in, Mat& out);

//To be used to blur the dissimiliarity map while retaining structural information as defined by the MST distance transform image
void treeFilter(Mat& dis_image, Mat& mbd_image, int size, float sigD);

//Post process the resulting combination of the two maps as defined by the MST paper
void postProcessing(Mat& combined);
