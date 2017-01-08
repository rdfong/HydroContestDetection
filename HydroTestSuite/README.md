HydroTestSuite:

images/
    Contains all 100 images to be used for training and testing.

annotations/
    Contains all annotations for the images, marked using the annotations.m script.

proposals/
    Contains all proposals created from the genProposals script.

scores/
    A folder to store score files, score files themselves were not included in the repository.

annotate.m: 
    Runs through all images in the images folder and let's the user annotate bounding boxes.
    Annotations are outputted in the annotations folder.

genProposals.m
    Calls one of the bounding box proposal methods. 
    The executable itself is expected to output the proposal files, you should change the paths to the executables.
    Read comments for information on how to use it.

getWeakPriors.m
    A script to specify weak priors for land, sea and sky gaussian model parameters.
    In the script you draw the dividing lines between each zone.
    Note that the final implementation does not use weak priors do to their lack of generalisability.

score.m
    Generates scores for your proposals.

setHorizons.m
    A script to get manually specified horizon line data for each image, outputted to images/ folder.

