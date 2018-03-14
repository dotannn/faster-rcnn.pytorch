from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths
import os
import sys
import numpy as np
import argparse
import pprint
import pdb
import time
import cv2
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim

import torchvision.transforms as transforms
import torchvision.datasets as dset
from scipy.misc import imread
from roi_data_layer.roidb import combined_roidb
from roi_data_layer.roibatchLoader import roibatchLoader
from model.utils.config import cfg, cfg_from_file, cfg_from_list, get_output_dir
from model.rpn.bbox_transform import clip_boxes
from model.nms.nms_wrapper import nms
from model.rpn.bbox_transform import bbox_transform_inv
from model.utils.net_utils import save_net, load_net, vis_detections
from model.utils.blob import im_list_to_blob
from model.faster_rcnn.vgg16 import vgg16
from model.faster_rcnn.resnet import resnet
import pdb

xrange = range

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


pascal_classes = np.asarray(['__background__',
                             'aeroplane', 'bicycle', 'bird', 'boat',
                             'bottle', 'bus', 'car', 'cat', 'chair',
                             'cow', 'diningtable', 'dog', 'horse',
                             'motorbike', 'person', 'pottedplant',
                             'sheep', 'sofa', 'train', 'tvmonitor'])

class_agnostic = False

fasterRCNN = vgg16(pascal_classes, pretrained=False, class_agnostic=class_agnostic)

fasterRCNN.create_architecture()
checkpoint_file = "/media/dotan/Data/models/faster_rcnn_1_10_625.pth"

checkpoint = torch.load(checkpoint_file)

fasterRCNN.load_state_dict(checkpoint['model'])

print(checkpoint['pooling_mode'])

im_data = torch.FloatTensor(1)
im_info = torch.FloatTensor(1)
num_boxes = torch.LongTensor(1)
gt_boxes = torch.FloatTensor(1)

# ship to cuda
im_data = im_data.cuda()
im_info = im_info.cuda()
num_boxes = num_boxes.cuda()
gt_boxes = gt_boxes.cuda()

# make variable
im_data = Variable(im_data, volatile=True)
im_info = Variable(im_info, volatile=True)
num_boxes = Variable(num_boxes, volatile=True)
gt_boxes = Variable(gt_boxes, volatile=True)

fasterRCNN = fasterRCNN.cuda()

fasterRCNN.eval()

start = time.time()
max_per_image = 100
thresh = 0.05
vis = True

cap = cv2.VideoCapture(0)

while True:
    total_tic = time.time()
    ok, frame = cap.read()
    if ok is None:
        break

    im_in = np.array( frame )
    im = im_in[:, :, ::-1]
    blobs, im_scales = _get_image_blob( im )
    im_blob = blobs
    im_info_np = np.array([[im_blob.shape[1], im_blob.shape[2], im_scales[0]]], dtype=np.float32)

    im_data_pt = torch.from_numpy( im_blob )
    im_data_pt = im_data_pt.permute( 0, 3, 1, 2 )
    im_info_pt = torch.from_numpy( im_info_np )

    im_data.data.resize_( im_data_pt.size() ).copy_( im_data_pt )
    im_info.data.resize_( im_info_pt.size() ).copy_( im_info_pt )
    gt_boxes.data.resize_( 1, 1, 5 ).zero_()
    num_boxes.data.resize_( 1 ).zero_()

    # pdb.set_trace()
    det_tic = time.time()

    rois, cls_prob, bbox_pred, \
    rpn_loss_cls, rpn_loss_box, \
    RCNN_loss_cls, RCNN_loss_bbox, \
    rois_label, pooled_feat = fasterRCNN( im_data, im_info, gt_boxes, num_boxes )

    scores = cls_prob.data
    boxes = rois.data[:, :, 1:5]

    pred_boxes = np.tile( boxes, (1, scores.shape[1]) )

    pred_boxes /= im_scales[0]

    scores = scores.squeeze()
    pred_boxes = pred_boxes.squeeze()
    det_toc = time.time()
    detect_time = det_toc - det_tic
    misc_tic = time.time()

    im2show = np.copy( im )

    for j in xrange( 1, len( pascal_classes ) ):
        inds = torch.nonzero( scores[:, j] > thresh ).view( -1 )
        # if there is det
        if inds.numel() > 0:
            cls_scores = scores[:, j][inds]
            _, order = torch.sort( cls_scores, 0, True )
            if class_agnostic:
                cls_boxes = pred_boxes[inds, :]
            else:
                cls_boxes = pred_boxes[inds][:, j * 4:(j + 1) * 4]

            cls_dets = torch.cat( (torch.cuda.FloatTensor(cls_boxes), cls_scores.unsqueeze( 1 )), 1)
            # cls_dets = torch.cat((cls_boxes, cls_scores), 1)
            cls_dets = cls_dets[order]
            keep = nms( cls_dets, cfg.TEST.NMS )
            cls_dets = cls_dets[keep.view( -1 ).long()]
            if vis:
                im2show = vis_detections( im2show, pascal_classes[j], cls_dets.cpu().numpy(), 0.5 )

    misc_toc = time.time()
    nms_time = misc_toc - misc_tic
    im2showRGB = cv2.cvtColor( im2show, cv2.COLOR_BGR2RGB )
    cv2.imshow( "frame", im2showRGB )
    total_toc = time.time()
    total_time = total_toc - total_tic
    frame_rate = 1 / total_time
    print( 'Frame rate:', frame_rate )
    if cv2.waitKey( 1 ) & 0xFF == ord( 'q' ):
        break


cap.release()
cv2.destroyAllWindows()