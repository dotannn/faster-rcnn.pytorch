from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths # don't remove this, it enables nms imports

import cv2
import numpy as np

from model.nms.nms_wrapper import nms
from model.utils.net_utils import save_net, load_net, vis_detections

from utils.context_managers import FPSEstimator, Timer
from utils.helpers import draw_detections
from objdetect import ObjectDetector
from objtrack import ObjectsTracker
from settings import DETECTION_THRESHOLD, CHECKPOINT_FILE, ARCH


def detect_motion(prev_frame, frame):
    return []


def main():
    objdetect = ObjectDetector(CHECKPOINT_FILE, detection_thresh=DETECTION_THRESHOLD, arch=ARCH)
    tracker = ObjectsTracker()

    cap = cv2.VideoCapture(-1)
    if not cap.isOpened():
        return

    old_proposals = []
    prev_frame = None

    cv2.namedWindow("frame", cv2.WND_PROP_FULLSCREEN)
    # cv2.setWindowProperty("frame",cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)

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
                tracker.track(events, frame)

            frame = draw_detections(frame, tracker.instances)

            old_proposals = tracker.current_proposals()

            cv2.imshow( "frame", frame )
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            prev_frame = frame.copy()

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
