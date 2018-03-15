from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths # don't remove this, it enables nms imports

import cv2
import torch
import numpy as np
import time

from model.nms.nms_wrapper import nms
from model.utils.net_utils import save_net, load_net, vis_detections

from objdetect import ObjectDetector
from objtrack import ObjectsTracker

class FPSEstimator(object):
    def __init__(self, name=None):
        self.name = name

    def __enter__(self):
        self.tstart = time.time()

    def __exit__(self, type, value, traceback):
        print ('%s: %.3fFPS' % (self.name, 1.0/(time.time() - self.tstart)))

class Timer(object):
    def __init__(self, name=None):
        self.name = name

    def __enter__(self):
        self.tstart = time.time()

    def __exit__(self, type, value, traceback):
        print ('%s: %.3fs' % (self.name, (time.time() - self.tstart)))

def draw_detections(frame, insts):
    for inst in insts:
        if inst.stability < 1:
            continue
        p1 = (int(inst._bbox[0]), int(inst._bbox[1]))
        p2 = (int(inst._bbox[2]), int(inst._bbox[3]))
        name = inst._name
        frame = cv2.rectangle(frame, p1, p2, color=(0,0,255), thickness=2 )
        text = "%s %.2f | %s (%.2fm)" % (name, inst._score, inst._distance_level, inst._distance)
        frame = cv2.putText(frame, text, (p1[0], p1[1]+30), fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=1,color=(0, 255, 0))
    return frame

checkpoint_file = "models/faster_rcnn_1_10_14657_resnet101_coco.pth"
detection_threshold = 0.3

objdetect = ObjectDetector(checkpoint_file, detection_thresh=detection_threshold, arch='resnet101')

tracker = ObjectsTracker()


cap = cv2.VideoCapture(1)

def detect_motion(prev_frame, frame):
    return []


old_proposals = []
prev_frame = None

cv2.namedWindow("frame", cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty("frame",cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)


while True:
    with FPSEstimator('-total'):
        ok, frame = cap.read()

        with Timer('--detect'):
            motion_boxes = detect_motion(prev_frame, frame)

            if motion_boxes or old_proposals:
                events = objdetect.detect(frame, motion_boxes=np.vstack(motion_boxes, old_proposals))
            else:
                events = objdetect.detect(frame)

        with Timer('--track'):
            tracker.track(events)

        frame = draw_detections(frame, tracker.instances)

        old_proposals = tracker.current_proposals()


        cv2.imshow( "frame", frame )
        if cv2.waitKey( 1 ) & 0xFF == ord( 'q' ):
            break

        prev_frame = frame.copy()


cap.release()
cv2.destroyAllWindows()

