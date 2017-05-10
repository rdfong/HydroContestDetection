# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

import xml.dom.minidom as minidom

import os
# import PIL
import numpy as np
import scipy.sparse
import subprocess
import cPickle
import math
import glob
import uuid
import scipy.io as sio
import xml.etree.ElementTree as ET

from .imdb import imdb
from .imdb import ROOT_DIR
import ds_utils
from .voc_eval import voc_eval

# TODO: make fast_rcnn irrelevant
# >>>> obsolete, because it depends on sth outside of this project
from ..fast_rcnn.config import cfg


# <<<< obsolete


class boat_voc(imdb):
    def __init__(self, image_set, year, devkit_path=None):
        imdb.__init__(self, 'voc_' + year + '_' + image_set)
        #year here is 'boat'
        self._year = year
        self._image_set = image_set
        self._devkit_path = self._get_default_path() if devkit_path is None \
            else devkit_path
        self._data_path = os.path.join(self._devkit_path, 'images')
        
        self._classes = ('__background__',  # always index 0
                         'boat', 'buoy', 'person', 'bird', 'other')
        self._class_to_ind = dict(zip(self.classes, xrange(self.num_classes)))       #will convert images to lower case later
        self._image_ext = '.JPG'
        
        self._image_index = self._load_image_set_index()
        # Default to roidb handler
        # self._roidb_handler = self.selective_search_roidb
        self._roidb_handler = self.gt_roidb
        self._salt = str(uuid.uuid4())
        self._comp_id = 'comp4'

        assert os.path.exists(self._devkit_path), \
            'VOCboat path does not exist: {}'.format(self._devkit_path)
        assert os.path.exists(self._data_path), \
            'Path does not exist: {}'.format(self._data_path)

    def image_path_at(self, i):
        """
        Return the absolute path to image i in the image sequence.
        """
        return self.image_path_from_index(self._image_index[i])

    def image_path_from_index(self, index):
        """
        Construct an image path from the image's "index" identifier.
        """
        image_path = os.path.join(self._data_path,
                                  index + self._image_ext)
        assert os.path.exists(image_path), \
            'Path does not exist: {}'.format(image_path)
        return image_path

    def _load_image_set_index(self):
        """
        Load the indexes listed in this dataset's image set file.
        """
        # Example path to image set file:
        # self._devkit_path + /VOCdevkit2007/VOC2007/ImageSets/Main/val.txt
        image_set_file = os.path.join(self._data_path,
                                      self._image_set + '.txt')

        with open(image_set_file) as f:
            image_index = [x.strip() for x in f.readlines()]
        return image_index

    def _get_default_path(self):
        """
        Return the default path where PASCAL VOC is expected to be installed.
        """
        return os.path.join(cfg.DATA_DIR, '../../HydroTestSuite/')

    def gt_roidb(self):
        """
        Return the database of ground-truth regions of interest.

        This function loads/saves from/to a cache file to speed up future calls.
        """
        cache_file = os.path.join(self.cache_path, self.name + '_gt_roidb.pkl')
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as fid:
                roidb = cPickle.load(fid)
            print '{} gt roidb loaded from {}'.format(self.name, cache_file)
            return roidb

        gt_roidb = [self._load_pascal_annotation(index)
                    for index in self.image_index]
        with open(cache_file, 'wb') as fid:
            cPickle.dump(gt_roidb, fid, cPickle.HIGHEST_PROTOCOL)
        print 'wrote gt roidb to {}'.format(cache_file)

        return gt_roidb
    
    def _get_comp_id(self):
        comp_id = (self._comp_id + '_' + self._salt if self.config['use_salt']
                   else self._comp_id)
        return comp_id

    def _get_voc_results_file_template(self):
        # VOCdevkit/results/VOC2007/Main/<comp_id>_det_test_aeroplane.txt
        filename = self._get_comp_id() + '_det_' + self._image_set + '_{:s}.txt'
        filedir = os.path.join(self._devkit_path, 'results', 'VOC' + self._year, 'Main')
        if not os.path.exists(filedir):
            os.makedirs(filedir)
        path = os.path.join(filedir, filename)
        return path

    def _write_voc_results_file(self, all_boxes):
        print 'Writing VOC results file'
        filename = 'FasterRCNN_boat_results'
        with open(filename, 'wt') as f:
            for im_ind, index in enumerate(self.image_index):
                dets = all_boxes[im_ind]
                if dets == []:
                    continue
                # the VOCdevkit expects 1-based indices
                for k in xrange(dets.shape[0]):
                    f.write('{:s} {:.3f} {:.1f} {:.1f} {:.1f} {:.1f}\n'.
                            format(index, dets[k, -1],
                                   dets[k, 0] + 1, dets[k, 1] + 1,
                                   dets[k, 2] + 1, dets[k, 3] + 1))

    def _do_python_eval(self, output_dir='output'):
        annopath = os.path.join(
            self._data_path,
            '../annotations/VOC_annotations/',
            '{:s}.xml')
        imagesetfile = os.path.join(
            self._data_path,
            self._image_set + '.txt')
        cachedir = os.path.join(self._devkit_path, 'annotations_cache')
        aps = []
        # The PASCAL VOC metric changed in 2010
        use_07_metric = True
        print 'VOC07 metric? ' + ('Yes' if use_07_metric else 'No')
        if not os.path.isdir(output_dir):
            os.mkdir(output_dir)
        
        filename = 'FasterRCNN_boat_results'
        rec, prec, ap = voc_eval(
            filename, annopath, imagesetfile,'object', cachedir, ovthresh=0.5,
            use_07_metric=use_07_metric)
        print('AP for Object Detection = {:.4f}'.format(ap))
        with open(os.path.join(output_dir + '_pr.pkl'), 'w') as f:
            cPickle.dump({'rec': rec, 'prec': prec, 'ap': ap}, f)
        print('--------------------------------------------------------------')

    def _do_matlab_eval(self, output_dir='output'):
        print '-----------------------------------------------------'
        print 'Computing results with the official MATLAB eval code.'
        print '-----------------------------------------------------'
        path = os.path.join(cfg.ROOT_DIR, 'lib', 'datasets',
                            'VOCdevkit-matlab-wrapper')
        cmd = 'cd {} && '.format(path)
        cmd += '{:s} -nodisplay -nodesktop '.format(cfg.MATLAB)
        cmd += '-r "dbstop if error; '
        cmd += 'voc_eval(\'{:s}\',\'{:s}\',\'{:s}\',\'{:s}\'); quit;"' \
            .format(self._devkit_path, self._get_comp_id(),
                    self._image_set, output_dir)
        print('Running:\n{}'.format(cmd))
        status = subprocess.call(cmd, shell=True)

    def evaluate_detections(self, all_boxes, output_dir):
        self._write_voc_results_file(all_boxes)
        self._do_python_eval(output_dir)
        #if self.config['matlab_eval']:
        #    self._do_matlab_eval(output_dir)
            #if self.config['cleanup']:
            #for cls in self._classes:
            #    if cls == '__background__':
            #        continue
            #    filename = self._get_voc_results_file_template().format(cls)
            #    os.remove(filename)

    def competition_mode(self, on):
        if on:
            self.config['use_salt'] = False
            self.config['cleanup'] = False
        else:
            self.config['use_salt'] = True
            self.config['cleanup'] = True


if __name__ == '__main__':
    d = boat_voc('train', 'boat')
    res = d.roidb
    from IPython import embed;

    embed()
