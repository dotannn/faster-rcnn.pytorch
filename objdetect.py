from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import _init_paths # don't remove this, it enables nms imports

import numpy as np
import cv2
import torch
from torch.autograd import Variable

from model.nms.nms_wrapper import nms
from model.utils.config import cfg, cfg_from_file, cfg_from_list, get_output_dir
from model.utils.blob import im_list_to_blob
from model.faster_rcnn.vgg16 import vgg16
from model.faster_rcnn.resnet import resnet


xrange = range



class DetectionEvent:
    def __init__(self, bbox, score, label, features):
        self._bbox = bbox
        self._score = score
        self._label = label
        self._features = features

        # calc center of mass
        self._com = np.array(( self._bbox[0] + self._bbox[2]/2.0, self._bbox[1] + self._bbox[3]/2.0 ))


class ObjectDetector:
    _classes = np.asarray(['__background__',
                           'aeroplane', 'bicycle', 'bird', 'boat',
                           'bottle', 'bus', 'car', 'cat', 'chair',
                           'cow', 'diningtable', 'dog', 'horse',
                           'motorbike', 'person', 'pottedplant',
                           'sheep', 'sofa', 'train', 'tvmonitor'])


    def __init__(self, checkpoint_file, detection_thresh=0.5, arch='vgg16', class_agnostic=False, cuda=True):

        if arch=='vgg16':
            self._fasterRCNN = vgg16(self._classes, pretrained=False, class_agnostic=class_agnostic)
        elif arch=='resnet101':
            self._fasterRCNN = resnet(self._classes, 101, pretrained=False, class_agnostic=class_agnostic)
        else:
            raise NotImplementedError()

        self._class_agnostic = class_agnostic
        self._detection_thresh = detection_thresh

        self._fasterRCNN.create_architecture()
        checkpoint = torch.load(checkpoint_file)
        self._fasterRCNN.load_state_dict(checkpoint['model'])

        self._im_data = torch.FloatTensor(1)
        self._im_info = torch.FloatTensor(1)
        self._num_boxes = torch.LongTensor(1)
        self._gt_boxes = torch.FloatTensor(1)

        # ship to cuda
        if cuda:
            self._im_data = self._im_data.cuda()
            self._im_info = self._im_info.cuda()
            self._num_boxes = self._num_boxes.cuda()
            self._gt_boxes = self._gt_boxes.cuda()

        # make variable
        self._im_data = Variable(self._im_data, volatile=True)
        self._im_info = Variable(self._im_info, volatile=True)
        self._num_boxes = Variable(self._num_boxes, volatile=True)
        self._gt_boxes = Variable(self._gt_boxes, volatile=True)

        if cuda:
            self._fasterRCNN = self._fasterRCNN.cuda()


        # set default to evaluation mode.
        self._fasterRCNN.eval()

    @staticmethod
    def _get_image_blob(im):
        """Converts an image into a network input.
            Arguments:
              im (ndarray): a color image in BGR order
            Returns:
              blob (ndarray): a data blob holding an image pyramid
              im_scale_factors (list): list of image scales (relative to im) used
                in the image pyramid
            """
        im_orig = im.astype(np.float32, copy=True)
        im_orig -= cfg.PIXEL_MEANS

        im_shape = im_orig.shape
        im_size_min = np.min(im_shape[0:2])
        im_size_max = np.max(im_shape[0:2])

        processed_ims = []
        im_scale_factors = []

        for target_size in cfg.TEST.SCALES:
            im_scale = float(target_size) / float(im_size_min)
            # Prevent the biggest axis from being more than MAX_SIZE
            if np.round(im_scale * im_size_max) > cfg.TEST.MAX_SIZE:
                im_scale = float(cfg.TEST.MAX_SIZE) / float(im_size_max)
            im = cv2.resize(im_orig, None, None, fx=im_scale, fy=im_scale,
                            interpolation=cv2.INTER_LINEAR)
            im_scale_factors.append(im_scale)
            processed_ims.append(im)

        # Create a blob to hold the input images
        blob = im_list_to_blob(processed_ims)

        return blob, np.array(im_scale_factors)

    def detect(self, frame, motion_boxes=None, detection_labels=['person']):
        im_in = np.array(frame)
        im = im_in[:, :, ::-1]
        blobs, im_scales = self._get_image_blob(im)
        im_blob = blobs
        im_info_np = np.array([[im_blob.shape[1], im_blob.shape[2], im_scales[0]]], dtype=np.float32)

        im_data_pt = torch.from_numpy(im_blob)
        im_data_pt = im_data_pt.permute(0, 3, 1, 2)
        im_info_pt = torch.from_numpy(im_info_np)

        self._im_data.data.resize_(im_data_pt.size()).copy_(im_data_pt)
        if motion_boxes:
            # TODO concat motion_boxes with im_info
            pass
        else:
            self._im_info.data.resize_(im_info_pt.size()).copy_(im_info_pt)

        self._gt_boxes.data.resize_(1, 1, 5).zero_()
        self._num_boxes.data.resize_(1).zero_()

        rois, cls_prob, bbox_pred, \
        rpn_loss_cls, rpn_loss_box, \
        RCNN_loss_cls, RCNN_loss_bbox, \
        rois_label, pooled_feat = self._fasterRCNN(self._im_data, self._im_info, self._gt_boxes, self._num_boxes)

        scores = cls_prob.data
        boxes = rois.data[:, :, 1:5]

        pred_boxes = np.tile(boxes, (1, scores.shape[1]))

        pred_boxes /= im_scales[0]

        scores = scores.squeeze()
        pred_boxes = pred_boxes.squeeze()

        labels = np.argmax(scores.cpu().numpy(), axis=1)

        # now do:
        # 1. filter by min width & height
        # 2. filter using NMS
        # 3. wrap to list of detection events

        events = []

        # scan all labels we care about
        for label in detection_labels:
            # get label idx
            j = int(np.squeeze(np.where(label == self._classes)))

            # check which region-proposal detect this label
            inds = torch.nonzero(scores[:, j] > self._detection_thresh).view(-1)

            # if at least one detection of this label
            if inds.numel() > 0:

                # sort by score to start with best detection
                cls_scores = scores[:, j][inds]
                _, order = torch.sort(cls_scores, 0, True)

                # pull regions boxes
                cls_boxes = pred_boxes[inds][:, j * 4:(j + 1) * 4]
                cls_features = pooled_feat[inds]

                # concat boxes + score  for the nms
                cls_dets = torch.cat((torch.cuda.FloatTensor(cls_boxes), cls_scores.unsqueeze(1)), 1)
                # cls_dets = torch.cat((cls_boxes, cls_scores), 1)

                # reorder based on score
                cls_dets = cls_dets[order]
                cls_features = cls_features[order]

                # run nms
                keep = nms(cls_dets, 0.2)

                keep = keep.squeeze()
                keep = keep.cpu().tolist()
                for idx in keep:
                    events.append(DetectionEvent(bbox=cls_dets[idx,:4].cpu().numpy(), score=cls_dets[idx, 4],
                                                 label=label, features=cls_features[idx].data.cpu().numpy()))

        return events

