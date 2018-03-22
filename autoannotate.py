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
from glob import glob
from pathlib import Path
import json

RESCALE = 0.33

DISPLAY = True

input_dir = "/home/tal/Data/11_floor"
output_dir = "/home/tal/Data/11_floor_annotated"
def main():
    objdetect = ObjectDetector(CHECKPOINT_FILE, detection_thresh=DETECTION_THRESHOLD, arch=ARCH)


    videos = glob("%s/**/*.mp4" % input_dir, recursive=True)
    for filename in videos:
        frame_count = 0
        cap = cv2.VideoCapture(filename)

        filepath = Path(filename)

        while True:
            try:
                ok, frame = cap.read()
                if ok is None:
                    break

                if RESCALE < 1:
                    frame = cv2.resize(frame, (0, 0), fx=RESCALE, fy=RESCALE)

                events = objdetect.detect(frame)

                cv2.imwrite("%s/%s_%08d.jpg" %(output_dir, filepath.stem, frame_count), frame)
                with open("%s/%s_%08d.json" % (output_dir, filepath.stem, frame_count), 'w') as fp:
                    json.dump([ev.to_dict() for ev in events], fp)


                if DISPLAY is True:
                    cv2.imshow("disp", draw_detections(frame, events))
                    cv2.waitKey(1)
                frame_count += 1
            except:
                break






if __name__ == '__main__':
    main()
