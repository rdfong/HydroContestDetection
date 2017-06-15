import numpy as np


# trained model
h5_fname = 'yolo-voc.weights.h5'

# VOC
label_names = ('boat', 'buoy', 'person', 'bird', 'other')
#label_names = ('aeroplane', 'bicycle', 'bird', 'boat',
#               'bottle', 'bus', 'car', 'cat', 'chair',
#               'cow', 'diningtable', 'dog', 'horse',
#               'motorbike', 'person', 'pottedplant',
#               'sheep', 'sofa', 'train', 'tvmonitor')
num_classes = len(label_names)

#anchors = np.asarray([(1.08, 1.19), (3.42, 4.41), (6.63, 11.38), (9.42, 5.11), (16.62, 10.52)], dtype=np.float)

#Top 5
#anchors = np.asarray([(0.5320, 0.5006), (0.8193, 2.2671), (3.6940, 1.8372), (1.7838, 0.7866), (8.4074, 4.1325)], dtype=np.float)

#Top 9
#anchors = np.asarray([(1.236,3.021), (4.836, 4.609), (9.590, 4.546), (3.837, 1.680), (1.479, 0.629), (0.443, 0.411), (2.326, 1.139), (0.658, 1.565), (7.174, 2.714)], dtype=np.float)

#Top 9 with half 2x grid res
anchors = np.asarray([(2.472, 6.042), (9.672, 9.218), (19.18, 9.092), (7.674, 3.360), (2.958, 1.258), (0.886, 0.822), (4.652, 2.278), (1.316, 3.13), (14.348, 5.428)], dtype=np.float)

#Top 7
#anchors = np.asarray([(0.4413, 0.5685), (1.4532, 0.9536), (0.5719, 2.3570), (1.2073, 3.7385), (8.1486, 5.5385), (5.1543, 4.2308) ,(2.9918, 1.9657)], dtype=np.float)

#Top 11
#anchors = np.asarray([(0.6259, 4.0836), (8.1486, 5.5385), (0.8486, 0.6580), (0.2993, 0.4985), (0.9394, 1.9142), (2.0996, 1.4085), (0.4649, 1.6457), (1.5133, 0.8510), (1.7864, 3.2289), (3.3889, 2.0389), (5.1610, 4.2713)], dtype=np.float)

#Top 3
#anchors = np.asarray([(6.8206, 4.9738), (0.7945, 1.2101), (2.7131, 2.1925)], dtype=np.float)


num_anchors = len(anchors)

