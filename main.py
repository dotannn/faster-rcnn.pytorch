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



RESCALE = 0.33
FORCE_FPS = 3

SAVE_VIDEO = True
outdir = "/home/tal/Data/results"
input_dir = "/home/tal/Data/"

filename = "confrence/far/scene2/normal.mp4"

def detect_motion(prev_frame, frame):
    return []


def main():
    objdetect = ObjectDetector(CHECKPOINT_FILE, detection_thresh=DETECTION_THRESHOLD, arch=ARCH)
    tracker = ObjectsTracker()


    cap = cv2.VideoCapture("%s/%s" % ( input_dir, filename))
    if not cap.isOpened():
        return

    old_proposals = []
    prev_frame = None

    cv2.namedWindow("frame", cv2.WND_PROP_FULLSCREEN)
    # cv2.setWindowProperty("frame",cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)
    frame_count = 0

    if SAVE_VIDEO:
        _, frame = cap.read()
        if RESCALE < 1:
            frame = cv2.resize(frame, (0, 0), fx=RESCALE, fy=RESCALE)

        (h, w) = frame.shape[:2]

        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        writer = cv2.VideoWriter("%s/%s" % (outdir, filename.replace("/", "_")), fourcc, FORCE_FPS,
                                 (w, h), True)
    while True:
        ok, frame = cap.read()
        if RESCALE < 1:
            frame = cv2.resize(frame, (0, 0), fx=RESCALE, fy=RESCALE)
        if ok is None:
            break

        frame_count += 1
        if (frame_count - 1) % int(25.0 / FORCE_FPS) != 0:
            continue

        with FPSEstimator('-total'):
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
            if SAVE_VIDEO:
                writer.write(frame)
            prev_frame = frame.copy()

    writer.release()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
