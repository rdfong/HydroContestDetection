exp_name = 'yolo_boat_models'

pretrained_fname = 'pretrained_models/darknet19.weights.npz'

start_step = 0
lr_decay_epochs = {60, 90}
lr_decay = 1./10

max_epoch = 160

weight_decay = 0.0005
momentum = 0.9
init_learning_rate = 1e-3

# for training yolo2
object_scale = 5.
noobject_scale = 1.
class_scale = 1.
coord_scale = 1.
iou_thresh = 0.5

# dataset
imdb_train = 'voc_boat_train'
imdb_test = 'voc_boat_test'
batch_size = 1
train_batch_size = 8
#train_batch_size = 16
