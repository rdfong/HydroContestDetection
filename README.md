# HydroContestDetection
EPFL Hydro Contest Competitor Detection Project

HydroTestSuite:
    Contains all the code to generate annotations for the test images and to generate scores.
    Also contains other Matlab scripts to generate weak priors and set horizon lines.
    See HydroTestSuite/README.md for more details.

SegmentationMethods:
    Main implementation is contained here. Holds all the different strategies attempted including the final model.

TestMedia:
    Contains images and videos used in testing implementation during development process.
    Due to lack of images in the training/test set there is some overlap with the annotated images.

VOC2007:
    Contains VOC2007 train and test data + annotations. Only used by BING method.
