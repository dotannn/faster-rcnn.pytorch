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
        p1 = (int(inst._bbox[0]), int(inst._bbox[1]))
        p2 = (int(inst._bbox[2]), int(inst._bbox[3]))
        name = inst._name
        frame = cv2.rectangle(frame, p1, p2, color=(0,0,255), thickness=2 )
        text = "%s %.2f | %s" % (name, inst._score, inst._distance_level)
        frame = cv2.putText(frame, text, (p1[0], p1[1]+30), fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=1,color=(0, 255, 0))
    return frame

checkpoint_file = "models/faster_rcnn_1_10_625.pth"
detection_threshold = 0.5

objdetect = ObjectDetector(checkpoint_file)

tracker = ObjectsTracker()


cap = cv2.VideoCapture(0)

def detect_motion(frame):
    return []


old_proposals = []
while True:
    with FPSEstimator('-frame'):
        ok, frame = cap.read()

        with Timer('--detect'):
            motion_boxes = detect_motion(frame)

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


cap.release()
cv2.destroyAllWindows()

