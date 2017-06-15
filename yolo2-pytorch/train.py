import os
import cv2
import torch
import numpy as np
import sys
from torch.multiprocessing import Pool

from darknet import Darknet19

from datasets.pascal_voc import VOCDataset
from datasets.boat_voc import BoatDataset
import utils.yolo as yolo_utils
import utils.network as net_utils
from utils.timer import Timer
import cfgs.config as cfg

try:
    from pycrayon import CrayonClient
except ImportError:
    CrayonClient = None


imdb_name = sys.argv[1]
meta = imdb_name.split('_')
imdb = None
if meta[1] == 'boat':
    imdb = BoatDataset(imdb_name, cfg.DATA_DIR, cfg.train_batch_size,
                  yolo_utils.preprocess_train, processes=2, shuffle=True, dst_size=cfg.inp_size)
else:
    imdb = VOCDataset(imdb_name, cfg.DATA_DIR, cfg.train_batch_size,
                  yolo_utils.preprocess_train, processes=2, shuffle=True, dst_size=cfg.inp_size)
print('load data succ...')

net = Darknet19()
#net_utils.load_net(cfg.trained_model, net)
#pretrained_model = os.path.join(cfg.train_output_dir, 'voc_boat_trainval_70.h5')
#net_utils.load_net(pretrained_model, net)
#pretrained_model = cfg.trained_model

net_utils.weights_normal_init(net, dev=0.01)
#net_utils.load_net(pretrained_model, net)
net.load_from_npz(cfg.pretrained_model, num_conv=18)
net.cuda()
net.train()
print('load net succ...')

# optimizer
start_epoch = 0
lr = cfg.init_learning_rate
optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=cfg.momentum, weight_decay=cfg.weight_decay)
#optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=cfg.weight_decay)

# tensorboad
use_tensorboard = cfg.use_tensorboard and CrayonClient is not None
# use_tensorboard = False
remove_all_log = False
if use_tensorboard:
    cc = CrayonClient(hostname='127.0.0.1')
    if remove_all_log:
        print('remove all experiments')
        cc.remove_all_experiments()
    if start_epoch == 0:
        try:
            cc.remove_experiment(cfg.exp_name)
        except ValueError:
            pass
        exp = cc.create_experiment(cfg.exp_name)
    else:
        exp = cc.open_experiment(cfg.exp_name)

train_loss = 0
bbox_loss, iou_loss, cls_loss = 0., 0., 0.
cnt = 0
t = Timer()
save_model_count = 0

old_epoch = imdb.epoch
step = 0
while imdb.epoch <= cfg.max_epoch:
    t.tic()
    # batch
    batch = imdb.next_batch()
    #if epoch has advanced then save the current network before we do anything else
    #also update the optimizer if necessary
    if old_epoch > 0 and (imdb.epoch > old_epoch):
        if imdb.epoch in cfg.lr_decay_epochs:
            lr *= cfg.lr_decay
            optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=cfg.momentum, weight_decay=cfg.weight_decay)    
            #optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=cfg.weight_decay)
        if imdb.epoch % 10 == 0:
            save_name = os.path.join(cfg.train_output_dir, '{}_{}.h5'.format(imdb_name, imdb.epoch))
            net_utils.save_net(save_name, net)
            print('save model: {}'.format(save_name))

    im = batch['images']
    gt_boxes = batch['gt_boxes']
    gt_classes = batch['gt_classes']
    dontcare = batch['dontcare']
    orgin_im = batch['origin_im']

    # forward
    im_data = net_utils.np_to_variable(im, is_cuda=True, volatile=False).permute(0, 3, 1, 2)
    net(im_data, gt_boxes, gt_classes, dontcare)

    # backward
    loss = net.loss
    bbox_loss += net.bbox_loss.data.cpu().numpy()[0]
    iou_loss += net.iou_loss.data.cpu().numpy()[0]
    cls_loss += net.cls_loss.data.cpu().numpy()[0]
    train_loss += loss.data.cpu().numpy()[0]
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    cnt += 1

    duration = t.toc()
    if step % 1 == 0:
        train_loss /= cnt
        bbox_loss /= cnt
        iou_loss /= cnt
        cls_loss /= cnt
        print('epoch: %d, step: %d, loss: %.3f, bbox_loss: %.3f, iou_loss: %.3f, cls_loss: %.3f (%.2f s/batch)' % (
            imdb.epoch, step, train_loss, bbox_loss, iou_loss, cls_loss, duration))
        
        if use_tensorboard and step % cfg.log_interval == 0:
            exp.add_scalar_value('loss_train', train_loss, step=step)
            exp.add_scalar_value('loss_bbox', bbox_loss, step=step)
            exp.add_scalar_value('loss_iou', iou_loss, step=step)
            exp.add_scalar_value('loss_cls', cls_loss, step=step)
            exp.add_scalar_value('learning_rate', lr, step=step)

        train_loss = 0
        bbox_loss, iou_loss, cls_loss = 0., 0., 0.
        cnt = 0
        t.clear()
    step += 1
    old_epoch = imdb.epoch
imdb.close()

