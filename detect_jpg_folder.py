from object_detection.inference_utils import detector_utils as detector_utils
from object_detection.inference_utils.classes_names_and_colors import classes_dict

import cv2
import tensorflow as tf
import time
import argparse
import glob
import numpy as np
import math
import os
import random

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-i','--images_path', dest='images_path',
        help='Path to images folder', 
        type=str)
    parser.add_argument('-f', '--frozen_graph_path', dest='frozen_graph_path', 
        help='Path to frozen graph folder', 
        type=str)
    parser.add_argument('-s', '--score_thresh', dest='score_thresh',
        default=0.2, 
        help='Score threshold for displaying bounding boxes', 
        type=float)
    parser.add_argument('-w', '--width', dest='width', 
        default=640, 
        help='Width of the frames in the video stream.', 
        type=int)
    parser.add_argument('--height', dest='height', 
        default=480, 
        help='Height of the frames in the video stream.', 
        type=int)  
    parser.add_argument('-d', '--delay', dest='delay', 
        default=25, 
        help='Delay in milliseconds. Default=25. Put 0 to wait for key press', 
        type=int)   
    parser.add_argument('-r', '--random', dest='random', 
        default=False, 
        help='Random shuffle the images in the folder', 
        type=int)   
    args = parser.parse_args()

    detection_graph, sess = detector_utils.load_inference_graph(args.frozen_graph_path)
    h,w = args.height, args.width

    img_list = glob.glob(os.path.join(args.images_path,'*.jpg'))

    if args.random:
        random.shuffle(img_list)
    else:
        img_list = sorted(img_list)

    for img_path in img_list:

        image = cv2.imread(img_path)
        if image.shape[0] != h or image.shape[1] != w: image = cv2.resize(image, (w,h), interpolation=cv2.INTER_LINEAR)
        image_np = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Detection
        boxes, scores, classes = detector_utils.detect_objects(image_np, detection_graph, sess)

        num_good_boxes = sum(scores > args.score_thresh)

        # Draw bounding boxes
        for i in range(num_good_boxes):
            (left, right, top, bottom) = (int(boxes[i][1]*w), int(boxes[i][3]*w), int(boxes[i][0]*h), int(boxes[i][2]*h))
            str_score = ':0.' + str(math.modf(scores[i])[0])[2:4]
            c = str(int(classes[i]))
            cv2.rectangle(image_np, (left,top),(right,bottom), classes_dict[c]['color'], 2, 1)
            cv2.putText(image_np,classes_dict[c]['name'] + str_score,(left,top-3),cv2.FONT_HERSHEY_DUPLEX,0.75,classes_dict[c]['color'],1)

        cv2.namedWindow("Detection", cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty("Detection",cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)
        cv2.imshow('Detection', cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR))

        if cv2.waitKey(args.delay) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break
