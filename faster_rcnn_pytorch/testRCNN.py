import os
import cv2
import torch
import cPickle
import numpy as np
import sys

from faster_rcnn import network
from faster_rcnn.faster_rcnn import FasterRCNN, RPN
from faster_rcnn.utils.timer import Timer
from faster_rcnn.fast_rcnn.nms_wrapper import nms

from faster_rcnn.fast_rcnn.bbox_transform import bbox_transform_inv, clip_boxes
from faster_rcnn.datasets.factory import get_imdb
from faster_rcnn.fast_rcnn.config import cfg, cfg_from_file, get_output_dir


# hyper-parameters
# ------------
imdb_name = sys.argv[1]
cfg_file = 'experiments/cfgs/faster_rcnn_end2end.yml'
#trained_model = 'models/VGGnet_fast_rcnn_iter_70000.h5'
#trained_model = 'models/training/darknet19_voc07trainval_exp1/darknet19_voc07trainval_exp1_1.h5'
trained_model = 'models/saved_model3/stub.h5'

rand_seed = 1024

save_name = 'faster_rcnn_100000'
max_per_image = 300
thresh = 0.05
vis = False
test_boats = True
# ------------

if rand_seed is not None:
    np.random.seed(rand_seed)

if rand_seed is not None:
    np.random.seed(rand_seed)

# load config
cfg_from_file(cfg_file)


def vis_detections(im, class_name, dets, thresh=0.8):
    """Visual debugging of detections."""
    for i in range(np.minimum(10, dets.shape[0])):
        bbox = tuple(int(np.round(x)) for x in dets[i, :4])
        score = dets[i, -1]
        if score > thresh:
            cv2.rectangle(im, bbox[0:2], bbox[2:4], (0, 204, 0), 2)
            cv2.putText(im, '%s: %.3f' % (class_name, score), (bbox[0], bbox[1] + 15), cv2.FONT_HERSHEY_PLAIN,
                        1.0, (0, 0, 255), thickness=1)
    return im


def im_detect(net, image):
    """Detect object classes in an image given object proposals.
    Returns:
        scores (ndarray): R x K array of object class scores (K includes
            background as object category 0)
        boxes (ndarray): R x (4*K) array of predicted bounding boxes
    """

    im_data, im_scales = net.rpn.get_image_blob(image)
    im_info = np.array(
        [[im_data.shape[1], im_data.shape[2], im_scales[0]]],
        dtype=np.float32)

    cls_prob, bbox_pred, rois = net(im_data, im_info)
    scores = cls_prob.data.cpu().numpy()
    boxes = rois.data.cpu().numpy()[:, 1:5] / im_info[0][2]

    if cfg.TEST.BBOX_REG:
        # Apply bounding-box regression deltas
        box_deltas = bbox_pred.data.cpu().numpy()
        pred_boxes = bbox_transform_inv(boxes, box_deltas)
        pred_boxes = clip_boxes(pred_boxes, image.shape)
    else:
        # Simply repeat the boxes, once for each class
        pred_boxes = np.tile(boxes, (1, scores.shape[1]))

    return scores, pred_boxes


def max_index(a):
    return [int(a.tolist().index(max(a))), max(a)]

def test_net(name, net, imdb, max_per_image=300, thresh=0.05, vis=False):
    """Test a Fast R-CNN network on an image database."""
    num_images = len(imdb.image_index)
    # all detections are collected into:
    #    all_boxes[cls][image] = N x 5 array of detections in
    #    (x1, y1, x2, y2, score)
    all_boxes = [[] for _ in xrange(num_images)]

    output_dir = get_output_dir(imdb, name)
    print output_dir
    # timers
    _t = {'im_detect': Timer(), 'misc': Timer()}
    det_file = os.path.join(output_dir, 'detections.pkl')

    for i in range(num_images):
        im = cv2.imread(imdb.image_path_at(i))
        _t['im_detect'].tic()
        scores, boxes = im_detect(net, im)
        #
        #scores (ndarray): R x K array of object class scores (K includes
        #                                          background as object category 0)
        #                                          boxes (ndarray): R x (4*K) array of predicted bounding boxes
        detect_time = _t['im_detect'].toc(average=False)

        _t['misc'].tic()
        if vis:
            # im2show = np.copy(im[:, :, (2, 1, 0)])
            im2show = np.copy(im)


        scores = scores[:, 1:imdb.num_classes]
        boxes = boxes[:,4:4*imdb.num_classes]
        
        scores = np.reshape(scores, (np.product(scores.shape), 1))
        boxes = np.reshape(boxes, (np.product(boxes.shape)/4,4))
        
        if len(scores) > 0:
            inds = np.where(scores > thresh)[0]
            scores = scores[inds]
            boxes = boxes[inds,:]
        
            dets = np.hstack((boxes, scores)) \
                .astype(np.float32, copy=False)

            keep = nms(dets, cfg.TEST.NMS)
            dets = dets[keep, :]
        else:
            dets = []
         
        if vis:
            im2show = vis_detections(im2show, 'Object', dets)

        if len(dets) > 0:
            all_boxes[i] = dets
            # Limit to max_per_image detections *over all classes*
            if max_per_image > 0:
                image_scores = np.hstack([all_boxes[i][:, -1]])
                if len(image_scores) > max_per_image:
                    image_thresh = np.sort(image_scores)[-max_per_image]
                    keep = np.where(all_boxes[i][:, -1] >= image_thresh)[0]
                    all_boxes[i] = all_boxes[i][keep, :]
            nms_time = _t['misc'].toc(average=False)

        print 'im_detect: {:d}/{:d} {:.3f}s {:.3f}s' \
            .format(i + 1, num_images, detect_time, nms_time)
    
        if test_boats:
            path = imdb.image_path_at(i)
            pathArray = path.split('/')
            f = open('../HydroTestSuite/proposals/' + pathArray[len(pathArray)-1]+'.txt', 'w')
            print '../HydroTestSuite/proposals/' + pathArray[len(pathArray)-1]
            # Write to boat detection format
            for d in range(len(all_boxes[i])):
                f.write('obstacle\n')
                f.write('{}\n'.format(all_boxes[i][d][4]))
                x1 = all_boxes[i][d][0]
                y1 = all_boxes[i][d][1]
                x2 = all_boxes[i][d][2]
                y2 = all_boxes[i][d][3]
                f.write('{} {} {} {}\n'.format(x1, y1, x2-x1+1, y2-y1+1))
            f.close()

        if vis:
            cv2.imshow('test', im2show)
            cv2.waitKey(1)

    with open(det_file, 'wb') as f:
        cPickle.dump(all_boxes, f, cPickle.HIGHEST_PROTOCOL)

    print 'Evaluating detections'
    imdb.evaluate_detections(all_boxes, output_dir)


if __name__ == '__main__':
    # load data
    imdb = get_imdb(imdb_name)
    imdb.competition_mode(on=True)

    # load net
    full_net = FasterRCNN(classes=imdb.classes, debug=False)
    network.load_net(trained_model, full_net)
    print('load model successfully!')

    full_net.cuda()
    full_net.eval()

    # evaluation
    test_net(save_name, full_net, imdb, max_per_image, thresh=thresh, vis=vis)
