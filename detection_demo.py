#================================================================
#
#   File name   : detection_demo.py
#   Author      : PyLessons
#   Created date: 2020-09-27
#   Website     : https://pylessons.com/
#   GitHub      : https://github.com/pythonlessons/TensorFlow-2.x-YOLOv3
#   Description : object detection image and video example
#
#================================================================
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import cv2
import numpy as np
import tensorflow as tf
from yolov3.utils import detect_image, detect_realtime, detect_video, Load_Yolo_model, detect_video_realtime_mp
from yolov3.configs import *
from pathlib import Path

# image_path   = "./IMAGES/kite.jpg"
image_path   = "/home/plamedi/Documents/repos/final-year-project/yolov1/assets/dog-cycle-car.png"
# video_path   = "/home/plamedi/Documents/repos/pocket/TensorFlow-2.x-YOLOv3/IMAGES/test.mp4"
video_path   = Path("/home/plamedi/Downloads/canadian_road.mp4")

yolo = Load_Yolo_model()
# detect_image(yolo, image_path, "./IMAGES/kite_pred.jpg", input_size=YOLO_INPUT_SIZE, show=True, rectangle_colors=(255,0,0))
# detect_video(yolo, str(video_path), "./IMAGES/detected.mp4", input_size=YOLO_INPUT_SIZE, show=False, rectangle_colors=(255,0,0))
detect_realtime(yolo, '', input_size=YOLO_INPUT_SIZE, show=True, rectangle_colors=(255, 0, 0))

#detect_video_realtime_mp(video_path, "Output.mp4", input_size=YOLO_INPUT_SIZE, show=False, rectangle_colors=(255,0,0), realtime=False)
