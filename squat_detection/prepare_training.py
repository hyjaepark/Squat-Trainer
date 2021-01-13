
import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile

from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image
import time

import cv2 as cv

sys.path.append("C:/Users/user/Desktop/tensorflowapi/models/research/object_detection")

from utils import label_map_util
from utils import visualization_utils as vis_util

def get_inference_graph_and_labels(pb,pbtxt,num_class):

    PATH_TO_CKPT = pb
    PATH_TO_LABELS = pbtxt
    NUM_CLASSES = num_class

    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')

    label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
    categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
    category_index = label_map_util.create_category_index(categories)
    return detection_graph,category_index

def detecting_mode(frame,detection_graph,sess,threshold_dic,detected_dic,points,tracker_dic):

    print("detecting")
    classes_dic = {'head': 1, 'knee': 2, 'hip': 3, 'feet': 4, 'back': 5}

    frame_new_shape = frame.reshape(1, frame.shape[0], frame.shape[1], 3)

    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
    boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
    scores = detection_graph.get_tensor_by_name('detection_scores:0')
    classes = detection_graph.get_tensor_by_name('detection_classes:0')
    num_detections = detection_graph.get_tensor_by_name('num_detections:0')

    (boxes, scores, classes, num_detections) = sess.run(
        [boxes, scores, classes, num_detections],
        feed_dict={image_tensor: frame_new_shape})

    rows = frame.shape[0]
    cols = frame.shape[1]

    for i in range(int(num_detections[0])):
        for j in classes_dic.keys():
            if classes[0][i] == classes_dic[j] and scores[0][i] >= threshold_dic[j]:
                x = boxes[0][i][1] * cols
                y = boxes[0][i][0] * rows
                right = boxes[0][i][3] * cols
                bottom = boxes[0][i][2] * rows
                threshold_dic[j] = min(scores[0][i], 0.3)
                points[j] = (x,y,right-x,bottom-y)
                detected_dic[j] = True
                make_tracker(frame,points,tracker_dic)

    return threshold_dic, points, detected_dic, boxes, scores, classes, tracker_dic

def visualize(frame,boxes,classes,scores,category_index):
    vis_util.visualize_boxes_and_labels_on_image_array(
      frame,
      np.squeeze(boxes),
      np.squeeze(classes).astype(np.int32),
      np.squeeze(scores),
      category_index,
      use_normalized_coordinates=True,
      line_thickness=5,min_score_thresh=.1)


def make_tracker(frame,points,tracker_dic):
    for keys, values in points.items():
        tracker_dic[keys] = cv.TrackerKCF_create()
        tracker_dic[keys].init(frame, points[keys])

    return tracker_dic


def tracker_update_get_center(frame,tracker_dic,tracker_points_dic,center_points_dic):
    check_xys = {}

    for keys, values in tracker_dic.items():
        xys = [int(v) for v in values.update(frame)[1]]
        # print(xys)
        check_xys[keys] = xys
        if xys != [0, 0, 0, 0]:
            tracker_points_dic[keys] = xys


    for keys_c, values_c in tracker_points_dic.items():
        center_point = (values_c[0] + round(values_c[2] / 2), values_c[1] + round(values_c[3] / 2))
        center_points_dic[keys_c] = center_point
        cv.rectangle(frame, (values_c[0], values_c[1]), (values_c[0] + values_c[2], values_c[1] + values_c[3]),
                     (0, 255, 255),
                     thickness=3)
        cv.circle(frame, center_point, 5, (0, 255, 255), thickness=5)

    return check_xys, center_points_dic
