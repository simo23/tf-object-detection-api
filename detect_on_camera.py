from object_detection.inference_utils import detector_utils as detector_utils
from object_detection.inference_utils.classes_names_and_colors import classes_dict

import cv2
import tensorflow as tf
import datetime
import argparse
import math
import numpy as np
import os

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-a', '--stream_addr', dest='stream_addr',
        default='0', 
        help='Path to streaming address', 
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
        help='Delay in milliseconds. Default=25ms. Put 0 to wait for key press at each frame', 
        type=int)     
    args = parser.parse_args()

    h,w = args.height,args.width
    detection_graph, sess = detector_utils.load_inference_graph(args.frozen_graph_path)

    cap = cv2.VideoCapture(int(args.stream_addr) if args.stream_addr=='0' else args.stream_addr)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)


    while True:

        ret, image_np = cap.read()
        image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)

        # Detection
        boxes, scores, classes = detector_utils.detect_objects(image_np, detection_graph, sess)

        # Draw bounding boxes
        for i in range(sum(scores > args.score_thresh)):
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
