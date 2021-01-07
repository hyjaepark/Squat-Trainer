


import numpy as np
import tensorflow as tf
import cv2 as cv
import matplotlib.pyplot as plt

import cv2

cap = cv2.VideoCapture(0)

# Read the graph.
with tf.gfile.FastGFile('frozen_inference_graph.pb', 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())

with tf.Session() as sess:
    while True:
        ret,img = cap.read()
        img = np.array(img)
        # Restore session
        sess.graph.as_default()
        tf.import_graph_def(graph_def, name='')


        # Read and preprocess an image.
        # img = cv.imread('prediction1_all.jpg')
        rows = img.shape[0]
        cols = img.shape[1]
        inp = cv.resize(img, (300, 300))
        inp = inp[:, :, [2, 1, 0]]  # BGR2RGB

        # Run the model
        out = sess.run([sess.graph.get_tensor_by_name('num_detections:0'),
                        sess.graph.get_tensor_by_name('detection_scores:0'),
                        sess.graph.get_tensor_by_name('detection_boxes:0'),
                        sess.graph.get_tensor_by_name('detection_classes:0')],
                       feed_dict={'image_tensor:0': inp.reshape(1, inp.shape[0], inp.shape[1], 3)})

        fig, ax = plt.subplots()
        # Visualize detected bounding boxes.
        num_detections = int(out[0][0])
        for i in range(num_detections):
            classId = int(out[3][0][i])
            score = float(out[1][0][i])
            bbox = [float(v) for v in out[2][0][i]]
            if score > 0.3:
                x = bbox[1] * cols
                y = bbox[0] * rows
                right = bbox[3] * cols
                bottom = bbox[2] * rows
                cv.rectangle(img, (int(x), int(y)), (int(right), int(bottom)), (125, 255, 51), thickness=2)

                if classId == 1:
                    print("head", np.round(score,3)*100,"%")
                    # pointhead = [x,y,right,bottom]
                    # print("왼쪽 위 꼭지점 x,y좌표:  {:.2f}/{:.2f}".format(x,y),"오른쪽 아래 꼭지점 x,y좌표:   {:.2f}/{:.2f}".format(right,bottom),"\n")
                    centerpoint_h = [(int(x)+int(right))/2,(int(y)+int(bottom))/2]
                    print("중앙 x,y좌표 {}/{}:".format(centerpoint_h[0],centerpoint_h[1]),"\n")
                    circle1 = plt.Circle((centerpoint_h[0],centerpoint_h[1]), 3, color="red")
                    ax.add_artist(circle1)
                if classId == 2:
                    print("knee", np.round(score,3)*100,"%")
                    pointknee = [x,y,right,bottom]
                    centerpoint_k = [(int(x)+int(right))/2,(int(y)+int(bottom))/2]
                    print("중앙 x,y좌표 {}/{}:".format(centerpoint_k[0],centerpoint_k[1]),"\n")
                    circle1 = plt.Circle((centerpoint_k[0],centerpoint_k[1]), 3, color="red")
                    ax.add_artist(circle1)
                if classId == 3:
                    print("hip", np.round(score,3)*100,"%")
                    pointhip = [x,y,right,bottom]
                    centerpoint_p = [(int(x)+int(right))/2,(int(y)+int(bottom))/2]
                    print("중앙 x,y좌표 {}/{}:".format(centerpoint_p[0],centerpoint_p[1]),"\n")
                    circle1 = plt.Circle((centerpoint_p[0],centerpoint_p[1]), 3, color="red")
                    ax.add_artist(circle1)
                if classId == 4:
                    print("feet", np.round(score,3)*100,"%")
                    pointfeet = [x,y,right,bottom]
                    centerpoint_f = [(int(x)+int(right))/2,(int(y)+int(bottom))/2]
                    endpoint_f = [int(right)-10,(int(y)+int(bottom))/2]
                    print("중앙 x,y좌표 {}/{}:".format(centerpoint_f[0],centerpoint_f[1]),"\n")
                    circle1 = plt.Circle((centerpoint_f[0],centerpoint_f[1]), 3, color="red")
                    circle2 = plt.Circle((endpoint_f[0],endpoint_f[1]), 3, color="red")
                    ax.add_artist(circle1)
                    ax.add_artist(circle2)
                if classId == 5:
                    print("back", np.round(score,3)*100,"%")
                    pointback = [x,y,right,bottom]
                    centerpoint_b = [(int(x)+int(right))/2,(int(y)+int(bottom))/2]
                    print("중앙 x,y좌표 {}/{}:".format(centerpoint_b[0],centerpoint_b[1]),"\n")
                    circle1 = plt.Circle((centerpoint_b[0],centerpoint_b[1]), 3, color="red")
                    ax.add_artist(circle1)

# plt.plot([0,250],[0,250],[250,250],[250,0],marker="o")


                    # if centerpoint_k[1]-centerpoint_p[1] > 10:
                    #     print("엉덩이를 더 내리시오")
                    #
                    # if centerpoint_f[0] > endpoint_f[0] :
                    #     print("무릎이 너무 앞으로 나가있습니다.")
                    #
                    #     slope_head_back = (centerpoint_h[1]-centerpoint_b[1])/(centerpoint_h[0]-centerpoint_b[0])
                    #     slope_back_hip = (centerpoint_b[1]-centerpoint_p[1])/(centerpoint_b[0]-centerpoint_p[0])
                    #
                    #     print(slope_head_back)
                    #     print(slope_back_hip)
                    #     if abs(slope_back_hip-slope_head_back) > 2:
                    #         print("허리를 피세요")

        cv2.imshow('object detection', cv2.resize(img, (800, 600)))
        if cv2.waitKey(25) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break



# plt.imshow(img)
# plt.show()
# cv.imshow('TensorFlow MobileNet-SSD', img)
# cv.waitKey()

