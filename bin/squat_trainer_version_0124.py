import numpy as np
import sys
import tensorflow as tf
import cv2 as cv
import time

sys.path.append("C:/Users/user/Desktop/tensorflowapi/models/research/object_detection")

from utils import label_map_util
from utils import visualization_utils as vis_util

from squat_detection import prepare_training

# get inference graph and labels
detection_graph,category_index = prepare_training.get_inference_graph_and_labels(
    "frozen_inference_graph59246.pb",'object-detection.pbtxt',5)


## Webcam/video/image
##################
cap = cv.VideoCapture("squat_vid2.mp4")
# cap = cv.VideoCapture(0)
##################

slope_hip_knee_frame_before = 100
flag = True

## SQL에 올릴 데이터들
back_error = 0
knee_error = 0
hip_error = 0
success_count = 0
all_squat_count = 0



with detection_graph.as_default():
    with tf.Session(graph=detection_graph) as sess:

        classes_dic = {'head': 1, 'knee': 2, 'hip': 3, 'feet': 4, 'back': 5}
        threshold_dic = {'head': 0.5, 'knee': 0.5, 'hip': 0.5, 'feet': 0.5, 'back': 0.5}
        tracker_dic = {}
        points_dic = {}
        detected_dic = {'head': False, 'knee': False, 'hip': False, 'feet': False, 'back': False}
        save_point_dic = {}
        tracker_points_dic = {}
        check_xys = {'head': [0,0,0,0], 'knee': [0,0,0,0], 'hip': [0,0,0,0], 'feet': [0,0,0,0], 'back': [0,0,0,0]}
        flag_for_detection = False


        while (cap.isOpened()):

            ret, frame = cap.read()
            frame = np.rot90(frame,3)
            frame = frame.copy()

            def detecting_mode(frame):

                frame_new_shape = frame.reshape(1,frame.shape[0],frame.shape[1],3)

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
                        if classes[0][i] == classes_dic[j] and scores[0][i] >= threshold_dic[j] :

                            x = boxes[0][i][1] * cols
                            y = boxes[0][i][0] * rows
                            right = boxes[0][i][3] * cols
                            bottom = boxes[0][i][2] * rows
                            points_dic[j] = [x,y,right,bottom]
                            threshold_dic[j] = min(scores[0][i],0.6)
                            tracker_dic[j] = cv.TrackerKCF_create()
                            tracker_dic[j].init(frame, (x, y, right - x, bottom - y))
                            detected_dic[j] = True

                if False in detected_dic.values() :
                    print("not all detected, return to detection")
                    continue



            for keys,values in tracker_dic.items():
                xys = [int(v) for v in values.update(frame)[1]]
                print(xys)
                check_xys[keys] = xys
                if xys != [0, 0, 0, 0] :
                    tracker_points_dic[keys] = xys
                    save_point_dic[keys] = xys

                else :

                    tracker_points_dic[keys] = save_point_dic[keys]

            center_points_dic={}
            for keys_c,values_c in tracker_points_dic.items():
                center_point = (values_c[0]+round(values_c[2]/2),values_c[1]+round(values_c[3]/2))
                center_points_dic[keys_c] = center_point
                cv.rectangle(frame, (values_c[0], values_c[1]), (values_c[0] + values_c[2], values_c[1] + values_c[3]), (0, 255, 255),
                             thickness=3)
                cv.circle(frame,center_point,5,(0,255,255),thickness=5)



            try :
                slope_hip_knee = (center_points_dic["knee"][1] - center_points_dic["hip"][1]) / \
                                 (center_points_dic["knee"][0] - center_points_dic["hip"][0])
                slope_hip_back = (center_points_dic["hip"][1] - center_points_dic["back"][1]) / \
                                 (center_points_dic["hip"][0] - center_points_dic["back"][0])
                slope_back_head = (center_points_dic["back"][1] - center_points_dic["head"][1]) / \
                                  (center_points_dic["back"][0] - center_points_dic["head"][0])


                ## if slope_hip_knee of this frame is bigger than before and below threshold, evaluate
                if slope_hip_knee >= slope_hip_knee_frame_before and slope_hip_knee < 0.5 and flag == True:
                    all_squat_count += 1
                    print("가장 내려간 지점 엉덩이와 무릎의 기울기 : ", round(slope_hip_knee,2))
                    lowest_slope = slope_hip_knee
                    k = 0
                    ## 엉덩이가 충분히 내려갔는지 확인

                    if lowest_slope > 0.2:
                        print("엉덩이를 더 내리세요")
                        flag = False
                        hip_error += 1

                    else :
                        print("엉덩이 ok")
                        flag = False
                        k += 1

                    ## 등이 펴져있는지 확인

                    if abs(slope_hip_back - slope_back_head) > 2:
                        print("등을 더 피세요")
                        back_error += 1

                    else :
                        print("등 ok")
                        k += 1
                    ## 무릎이 너무 나가있는지 확인

                    if tracker_points_dic["feet"][0]+tracker_points_dic["feet"][3]- center_points_dic["knee"][0] < 20 :
                        print("무릎이 발끝 밖으로 나가있는지 확인하세요\n\n")
                        knee_error += 1

                    else :
                        print("무릎 ok\n\n")
                        k += 1

                    if k == 3:
                        print("올바른 스쿼트 성공!!")

                ## reset when squat starts again
                if slope_hip_knee < slope_hip_knee_frame_before and slope_hip_knee > 2:
                    flag = True


                ## this is to save the previous frame slope
                slope_hip_knee_frame_before = slope_hip_knee

            except:
                pass




            # vis_util.visualize_boxes_and_labels_on_image_array(
            #   frame,
            #   np.squeeze(boxes),
            #   np.squeeze(classes).astype(np.int32),
            #   np.squeeze(scores),
            #   category_index,
            #   use_normalized_coordinates=True,
            #   line_thickness=5,min_score_thresh=.3)

            cv.imshow('frame',frame)
            k = cv.waitKey(1) & 0xFF

            if  k == ord('q'):
                print("총 스쿼트 한 횟수 : {}, 올바른 자세 : {}".format(all_squat_count,success_count))
                print("이번 스쿼트 세션에서의 범한 오류 허리 : {}, 무릎 : {}, 엉덩이 : {} : "
                      .format(back_error,knee_error,hip_error))
                break

            ## threshold 초기화
            elif k == ord("b"):
                threshold_dic = {'head': 0.3, 'knee': 0.3, 'hip': 0.2, 'feet': 0.2, 'back': 0.2}
