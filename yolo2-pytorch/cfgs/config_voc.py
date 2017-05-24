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

anchors = np.asarray([(1.08, 1.19), (3.42, 4.41), (6.63, 11.38), (9.42, 5.11), (16.62, 10.52)], dtype=np.float)

#anchors = np.asarray([(0.5320, 0.5006), (0.8193, 2.2671), (3.6940, 1.8372), (1.7838, 0.7866), (8.4074, 4.1325)], dtype=np.float)

num_anchors = len(anchors)

