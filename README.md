# HydroContestDetection
EPFL Hydro Contest Competitor Detection Project

HydroTestSuite:
    Contains all the code to generate annotations for the test images and to generate scores.
    Also contains other Matlab scripts to generate weak priors and set horizon lines.
    See HydroTestSuite/README.md for more details.

SegmentationMethods:
    GMM-MST implementation.

faster_rcnn_pytorch:

    -> Copy of https://github.com/longcw/faster_rcnn_pytorch.git adapted to HydroContest project 
    
    -> Setup details in faster_rcnn_pytorch/readme.md from original project



yolo-pytorch:

    -> Copy of https://github.com/longcw/yolo2-pytorch.git adapted to HydroContest project 
    
    -> Setup details in yolo-pytorch/readme.md from original project



faster_rcnn_pytorch/yolo-pytorch:

    -> For both methods, main script is runExperiment.py which performs 4-folds CV, chooses best model, outputs scores, and produces output images.
    
    -> Before running runExperiment, setup details for each project must first be followed
    
    -> One run of the script takes anywhere between 4 and 8 hours (recommended to run with tmux)
    
    -> To run:   python runExperiment.py GPU# Experiment_Name DELETE_TRAINED_MODELS? TEST_WITH_EXISTING_MODELS?
    
    -> Example: "python runExperiment.py  1    anchor_test              YES                     NO"
    
    -> Proposals end up in HydroTestSuite/proposals/(rcnn/yolo)

results:
    Contains all results for all CNN based methods outputted by runExperiment.py script.
    
    
TestMedia:
    Contains images and videos used in testing implementation during development process.
    Due to lack of images in the training/test set there is some overlap with the annotated images.

VOC2007:
    Contains VOC2007 train and test data + annotations.
