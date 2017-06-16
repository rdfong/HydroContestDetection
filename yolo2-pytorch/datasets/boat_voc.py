import cPickle
import os
import uuid
import cv2
import xml.etree.ElementTree as ET

import numpy as np
import scipy.sparse

# from functools import partial

from imdb import ImageDataset
from voc_eval import voc_eval
# from utils.yolo import preprocess_train


class BoatDataset(ImageDataset):
    def __init__(self, imdb_name, datadir, batch_size, im_processor, processes=3, shuffle=True, dst_size=None):
        super(BoatDataset, self).__init__(imdb_name, datadir, batch_size, im_processor, processes, shuffle, dst_size)
        meta = imdb_name.split('_')
        self._year = meta[1]
        self._image_set = meta[2]
        self._data_path = self._get_default_path()

        self._classes = ('boat', 'buoy', 'person', 'bird', 'other')
        self._class_to_ind = dict(zip(self.classes, range(self.num_classes)))
        self._image_ext = '.JPG'

        self._salt = str(uuid.uuid4())
        self._comp_id = 'comp4'

        # PASCAL specific config options
        self.config = {'cleanup': True,
                       'use_salt': True}

        self.load_dataset()
        # self.im_processor = partial(process_im, image_names=self._image_names, annotations=self._annotations)
        # self.im_processor = preprocess_train

    def load_dataset(self):
        # set self._image_index and self._annotations
        self._image_indexes = self._load_image_set_index()
        self._image_names = [self.image_path_from_index(index) for index in self.image_indexes]
        self._annotations = self._load_pascal_annotations()

    def _load_pascal_annotations(self):
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

        gt_roidb = [self._annotation_from_index(index)
                    for index in self.image_indexes]
        with open(cache_file, 'wb') as fid:
            cPickle.dump(gt_roidb, fid, cPickle.HIGHEST_PROTOCOL)
        print 'wrote gt roidb to {}'.format(cache_file)

        return gt_roidb

    def _annotation_from_index(self, index):
        """
        Load image and bounding boxes info from XML file in the PASCAL VOC
        format.
        """
        filename = os.path.join(self._data_path,'../annotations/VOC_annotations/', index + '.JPG.xml')
        
        tree = ET.parse(filename)
        objs = tree.findall('object')
        # if not self.config['use_diff']:
        #     # Exclude the samples labeled as difficult
        #     non_diff_objs = [
        #         obj for obj in objs if int(obj.find('difficult').text) == 0]
        #     # if len(non_diff_objs) != len(objs):
        #     #     print 'Removed {} difficult objects'.format(
        #     #         len(objs) - len(non_diff_objs))
        #     objs = non_diff_objs
        num_objs = len(objs)

        boxes = np.zeros((num_objs, 4), dtype=np.uint16)
        gt_classes = np.zeros((num_objs), dtype=np.int32)
        overlaps = np.zeros((num_objs, self.num_classes), dtype=np.float32)
        # "Seg" area for pascal is just the box area
        seg_areas = np.zeros((num_objs), dtype=np.float32)
        ishards = np.zeros((num_objs), dtype=np.int32)


        hfilename = os.path.join(self._data_path,'../images/', index + '.JPG_horizon.txt')
        horizon = np.loadtxt(open(hfilename, "rb"), delimiter=" ");
        
        # Load object bounding boxes into a data frame.
        for ix, obj in enumerate(objs):
            bbox = obj.find('bndbox')
            # Make pixel indexes 0-based
            x1 = float(bbox.find('xmin').text)
            y1 = float(bbox.find('ymin').text)
            x2 = float(bbox.find('xmax').text)
            y2 = float(bbox.find('ymax').text)

            diffc = obj.find('difficult')
            difficult = 0 if diffc == None else int(diffc.text)
            ishards[ix] = difficult

            cls = self._class_to_ind[obj.find('name').text.lower().strip()]
            boxes[ix, :] = [x1, y1, x2, y2]
            gt_classes[ix] = cls
            overlaps[ix,cls] = 1.0
            seg_areas[ix] = (x2 - x1 + 1) * (y2 - y1 + 1)

        overlaps = scipy.sparse.csr_matrix(overlaps)

        return {'boxes': boxes,
                'horizon': horizon,
                'gt_classes': gt_classes,
                'gt_ishard': ishards,
                'gt_overlaps': overlaps,
                'flipped': False,
                'seg_areas': seg_areas}

    def image_path_at(self, i):
        """
        Return the absolute path to image i in the image sequence.
        """
        return self.image_path_from_index(self._image_indexes[i])

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
        return '../HydroTestSuite/images/'

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
        filename = 'YOLO_boat_results'
        with open(filename, 'wt') as f:
            for im_ind, index in enumerate(self._image_indexes):
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
        cachedir = os.path.join(self._data_path, '../annotations_cache')

        aps = []
        # The PASCAL VOC metric changed in 2010
        use_07_metric = True
        print 'VOC07 metric? ' + ('Yes' if use_07_metric else 'No')
        if not os.path.isdir(output_dir):
            os.mkdir(output_dir)
        
        filename = 'YOLO_boat_results'
        rec, prec, ap = voc_eval(
            filename, annopath, imagesetfile,'object', cachedir, ovthresh=0.5,
            use_07_metric=use_07_metric)
        print('AP for Object Detection = {:.4f}'.format(ap))
        with open(os.path.join(output_dir + '_pr.pkl'), 'w') as f:
            cPickle.dump({'rec': rec, 'prec': prec, 'ap': ap}, f)
        print('--------------------------------------------------------------')
        return ap

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
        return self._do_python_eval(output_dir)
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
    d = pascal_voc('test', 'boat')
    res = d.roidb
    from IPython import embed;

    embed()
