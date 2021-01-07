import numpy as np
import sys
import tensorflow as tf
import cv2 as cv
import time
import paho.mqtt.client as mqtt

# mqttc = mqtt.Client("python_pub")  # MQTT Client
# mqttc.connect("test.mosquitto.org", 1883)    # MQTT
# mqttc.connect("192.168.101.101", 1883)  # iCORE-SDP Brokerq

sys.path.append("C:/Users/user/Desktop/공유폴더 자료/tensorflowapi/models/research/object_detection")

from utils import label_map_util
from utils import visualization_utils as vis_util


### 이부분 디렉토리 맞춰서 변경해야함
from squat_detection import prepare_training

def main():

    # get inference graph and labels
    detection_graph,category_index = prepare_training.get_inference_graph_and_labels(
        "frozen_inference_graph100973.pb",'object-detection.pbtxt',5)


    ## Webcam/video/image
    ##################
    cap = cv.VideoCapture(".\\videos\\test7_2.mp4")
    # cap = cv.VideoCapture(0)
    ##################
    ret, frame = cap.read()
    h, w = frame.shape[:2]
    fourcc = cv.VideoWriter_fourcc(*'DIVX')
    video_write = cv.VideoWriter('saved_out.avi', fourcc, 25.0, (w, h))

    y_value_of_head_before = 1000
    flag = True
    frame_num = 0

    ## SQL에 올릴 데이터들
    back_error = 0
    knee_error = 0
    hip_error = 0
    success_count = 0
    all_squat_count = 0



    with detection_graph.as_default():
        with tf.Session(graph=detection_graph) as sess:

            classes_dic = {'head': 1, 'knee': 2, 'hip': 3, 'feet': 4, 'back': 5}
            threshold_dic = {'head': 0.2, 'knee': 0.1, 'hip': 0.1, 'feet': 0.2, 'back': 0.2}
            tracker_dic = {}
            points_dic = {}
            detected_dic = {'head': False, 'knee': False, 'hip': False, 'feet': False, 'back': False}
            tracker_points_dic = {}

            fgbg = cv.createBackgroundSubtractorMOG2(varThreshold=400)
            fgmask = fgbg.apply(frame)
            nlabels, labels, stats, centroids = cv.connectedComponentsWithStats(fgmask)
            start = 0
            while (cap.isOpened()):
                ret, frame = cap.read()
                # frame = np.rot90(frame,3)
                # frame = cv.resize(frame,(600,800))
                # frame = frame.copy()
                frame_num += 1

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
                            threshold_dic[j] = min(scores[0][i],0.4)
                            tracker_dic[j] = cv.TrackerMOSSE_create()
                            tracker_dic[j].init(frame, (x, y, right - x, bottom - y))
                            detected_dic[j] = True

                for j in detected_dic.keys():
                    if detected_dic[j] == False:
                        if frame_num % 20 == 0:
                            print("not all detected, return to detection")
                        continue


                for keys,values in tracker_dic.items():
                    xys = [int(v) for v in values.update(frame)[1]]

                    if xys != [0, 0, 0, 0] :
                        tracker_points_dic[keys] = xys

                center_points_dic = {}
                for keys_c,values_c in tracker_points_dic.items():
                    center_point = (values_c[0]+round(values_c[2]/2),values_c[1]+round(values_c[3]/2))
                    center_points_dic[keys_c] = center_point
                    cv.rectangle(frame, (values_c[0], values_c[1]), (values_c[0] + values_c[2], values_c[1] + values_c[3]), (0, 255, 255),
                                 thickness=3)
                    cv.circle(frame,center_point,5,(0,255,255),thickness=5)

                cv.putText(frame, "squat_total : {}".format(all_squat_count), (100, 50), 2, 1, (0, 0, 0),2)
                cv.putText(frame, "squat_success : {}".format(success_count), (100, 100), 2, 1, (0, 0, 0),2)


                try :
                    slope_hip_knee = (center_points_dic["knee"][1] - center_points_dic["hip"][1]) / \
                                     (center_points_dic["knee"][0] - center_points_dic["hip"][0])
                    slope_hip_back = (center_points_dic["hip"][1] - center_points_dic["back"][1]) / \
                                     (center_points_dic["hip"][0] - center_points_dic["back"][0])
                    slope_back_head = (center_points_dic["back"][1] - center_points_dic["head"][1]) / \
                                      (center_points_dic["back"][0] - center_points_dic["head"][0])


                    ## 올라가는 순간 머리의 y 값이 커지므로, y값이 커지는 시점에 자세를 판별함
                    if center_points_dic["head"][1] < y_value_of_head_before and slope_hip_knee < 1 and flag == True \
                            and all_squat_count <= success_count+hip_error+back_error+knee_error:

                        all_squat_count += 1

                        k=0
                        print("가장 내려간 지점 엉덩이와 무릎의 기울기 : ", round(slope_hip_knee, 2))
                        if slope_hip_knee > 0.3:
                            print("엉덩이를 더 내리세요")
                            hip_error += 1
                            # mqttc.publish("hip", "lower hip!!")
                        else :
                            print("엉덩이 ok")
                            k += 1

                        if abs(slope_hip_back - slope_back_head) > 3:
                            print("등을 더 피세요")
                            back_error += 1
                            # mqttc.publish("back", "straighten back")
                        else :
                            print("등 ok")
                            k += 1
                        ## 무릎이 너무 나가있는지 확인

                        if center_points_dic["knee"][0] > tracker_points_dic["feet"][0]+tracker_points_dic["feet"][2] :
                            print("무릎이 발끝 밖으로 나가있는지 확인하세요\n\n")
                            knee_error += 1

                            # mqttc.publish("knee", "keep knee behind feet")

                        else :
                            print("무릎 ok\n\n")
                            k += 1

                        if k == 3:
                            print("올바른 스쿼트 성공!!\n\n")
                            success_count += 1

                        flag = False

                    ## 내려가는 순간 머리의 y 값이 작아지므로, y값이 작아지는 시점에 리셋함
                    if center_points_dic["head"][1] > y_value_of_head_before :
                        flag = True

                    ## 5 프레임전의 y값 저장
                    if frame_num % 10 == 0:
                        y_value_of_head_before = center_points_dic["head"][1]

                    else :
                        y_value_of_head_before = -100
                    # slope_hip_knee_frame_before = slope_hip_knee

                    # mqttc.publish("allsquat", str(all_squat_count))
                    # mqttc.publish("success", str(success_count))

                except:
                    pass




                vis_util.visualize_boxes_and_labels_on_image_array(
                  frame,
                  np.squeeze(boxes),
                  np.squeeze(classes).astype(np.int32),
                  np.squeeze(scores),
                  category_index,
                  use_normalized_coordinates=True,
                  line_thickness=5,min_score_thresh=.3)
                video_write.write(frame)
                cv.imshow('frame',frame)
                k = cv.waitKey(1) & 0xFF

                if  k == ord('q'):
                    print("총 스쿼트 한 횟수 : {}, 올바른 자세 : {}".format(all_squat_count,success_count))
                    print("이번 스쿼트 세션에서의 범한 오류 허리 : {}, 무릎 : {}, 엉덩이 : {} : "
                          .format(back_error,knee_error,hip_error))
                    cv.destroyAllWindows()
                    break


                ## threshold 초기화
                elif k == ord("b"):
                    threshold_dic = {'head': 0.3, 'knee': 0.3, 'hip': 0.2, 'feet': 0.2, 'back': 0.2}
            cap.release()
            video_write.release()





main()

