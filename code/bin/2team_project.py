

import numpy as np
import tensorflow as tf
import cv2 as cv
import time
import datetime as dt
import sys
from random import randint


# from datetime import datetime


def read_camera():
    global img, cam
    ret, img = cam.read()
    # img = np.rot90(img, 3)
    # img = cv.resize(img, (?, ?))
    # img = img.copy()
    return img


def detect_bodyparts(*file_name):

    global out
    inp = cv.resize(img, (300, 300))
    inp = inp[:, :, [2, 1, 0]]  # BGR2RGB
    out = sess.run([sess.graph.get_tensor_by_name('num_detections:0'),
                    sess.graph.get_tensor_by_name('detection_scores:0'),
                    sess.graph.get_tensor_by_name('detection_boxes:0'),
                    sess.graph.get_tensor_by_name('detection_classes:0')],
                   feed_dict={'image_tensor:0': inp.reshape(1, inp.shape[0], inp.shape[1], 3)})
    return out

def output():
    global old, point, rect, classId, score, old2, rows, cols, num_detections
    # point = [['head'], ['knee'], ['hip'], ['feet'], ['back']]
    old = [0.1, 0.1, 0.1, 0.1, 0.1]  # parameter
    rows = img.shape[0]
    cols = img.shape[1]
    num_detections = int(out[0][0])
    for i in range(num_detections):
        score = float(out[1][0][i])
        classId = int(out[3][0][i])

        if score > old[classId - 1]:
            bbox = [float(v) for v in out[2][0][i]]
            x = bbox[1] * cols
            y = bbox[0] * rows
            right = bbox[3] * cols
            bottom = bbox[2] * rows
            old[classId - 1] = score
            point[classId - 1] = [[x, y, right, bottom]]
            point2 = np.squeeze(point)
            try:
                for i in range(len(point2)):
                    rect[i] = (int(point2[i][0]), int(point2[i][1]), int(point2[i][2] - point2[i][0]),
                               int(point2[i][3] - point2[i][1]))
            except:
                pass
    return score, point, img, rect, old


def init_tracking():
    global tracker_name, tracker_value

    try:
        tracker_name = ['head', 'knee', 'hip', 'feet', 'back']
        tracker_value = {}
        for names in tracker_name:
            tracker_value[names] = cv.TrackerMOSSE_create()
        for a, i in enumerate(tracker_value.values()):
            i.init(img, rect[a])
    except:
        pass

def rect_makin(i):
    if classId2 == 1:
        bbox = [float(v) for v in out[2][0][i]]
        x = bbox[1] * cols
        y = bbox[0] * rows
        right = bbox[3] * cols
        bottom = bbox[2] * rows
        # print('head : ' , x, y, right, bottom, '\t')
        rect[classId2 - 1] = (int(x), int(y), int(right - x),
                              int(bottom - y))
        init_tracking_head(rect)
    if classId2 == 2:
        bbox = [float(v) for v in out[2][0][i]]
        x = bbox[1] * cols
        y = bbox[0] * rows
        right = bbox[3] * cols
        bottom = bbox[2] * rows
        # print('knee : ' , x, y, right, bottom, '\t')
        rect[classId2 - 1] = (int(x), int(y), int(right - x),
                              int(bottom - y))
        init_tracking_knee(rect)
    if classId2 == 3:
        bbox = [float(v) for v in out[2][0][i]]
        x = bbox[1] * cols
        y = bbox[0] * rows
        right = bbox[3] * cols
        bottom = bbox[2] * rows
        # print('hip : ' , x, y, right, bottom, '\t')

        rect[classId2 - 1] = (int(x), int(y), int(right - x),
                              int(bottom - y))
        init_tracking_hip(rect)
    if classId2 == 4:
        bbox = [float(v) for v in out[2][0][i]]
        x = bbox[1] * cols
        y = bbox[0] * rows
        right = bbox[3] * cols
        bottom = bbox[2] * rows
        # print('feet : ' , x, y, right, bottom, '\t')

        rect[classId2 - 1] = (int(x), int(y), int(right - x),
                              int(bottom - y))
        init_tracking_feet(rect)
    if classId2 == 5:
        bbox = [float(v) for v in out[2][0][i]]
        x = bbox[1] * cols
        y = bbox[0] * rows
        right = bbox[3] * cols
        bottom = bbox[2] * rows
        # print('back : ' , x, y, right, bottom, '\n')

        rect[classId2 - 1] = (int(x), int(y), int(right - x),
                              int(bottom - y))
        init_tracking_back(rect)

def init_tracking_head(rect):

    tracker_value['head'] = cv.TrackerMOSSE_create()
    tracker_value['head'].init(img, rect[0])

def init_tracking_knee(rect):
    tracker_value['knee'] = cv.TrackerMOSSE_create()
    tracker_value['knee'].init(img, rect[1])

def init_tracking_hip(rect):
    tracker_value['hip'] = cv.TrackerMOSSE_create()
    tracker_value['hip'].init(img, rect[2])

def init_tracking_feet(rect):
    tracker_value['feet'] = cv.TrackerMOSSE_create()
    tracker_value['feet'].init(img, rect[3])

def init_tracking_back(rect):
    tracker_value['back'] = cv.TrackerMOSSE_create()
    tracker_value['back'].init(img, rect[4])


def tracking():
    global listco, listcoco, counting_flag, up, counting, listco5, high_end, initin, hum
    listco = {'head':[0,0,0,0],'knee':[0,0,0,0],'hip':[0,0,0,0],'feet':[0,0,0,0],'back':[0,0,0,0]}
    for k, values in tracker_value.items():
        co = [int(v) for v in values.update(img)[1]]
        if co != [0, 0, 0, 0]:
            listco[k] = co

        if co == [0, 0, 0, 0] and k == 'head':
            tracker_value['head'] = cv.TrackerKCF_create()
            tracker_value['head'].init(img, tuple(listcoco['head']))
            listco['head'] = [int(v) for v in tracker_value['head'].update(img)[1]]
        if co == [0, 0, 0, 0] and k == 'knee':
            tracker_value['knee'] = cv.TrackerKCF_create()
            tracker_value['knee'].init(img, tuple(listcoco['knee']))
            listco['knee'] = [int(v) for v in tracker_value['knee'].update(img)[1]]
        if co == [0, 0, 0, 0] and k == 'hip':
            tracker_value['hip'] = cv.TrackerKCF_create()
            tracker_value['hip'].init(img, tuple(listcoco['hip']))
            listco['hip'] = [int(v) for v in tracker_value['hip'].update(img)[1]]

        if co == [0, 0, 0, 0] and k == 'feet':
            tracker_value['feet'] = cv.TrackerKCF_create()
            tracker_value['feet'].init(img, tuple(listcoco['feet']))
            listco['feet'] = [int(v) for v in tracker_value['feet'].update(img)[1]]

        if co == [0, 0, 0, 0] and k == 'back':
            tracker_value['back'] = cv.TrackerKCF_create()
            tracker_value['back'].init(img, tuple(listcoco['back']))
            listco['back'] = [int(v) for v in tracker_value['back'].update(img)[1]]

        if listco['head'][1] < listco5['head'][1] and up == False and hum == True:
            up = True
        if ((int(listco['head'][1]) - 30) < high_end and (int(listco['head'][1]) + 30 > high_end)):
            up = False
            counting_flag = False
            initin = True

    if frame_num % 15 == 0:
        listco5 = listco
    listcoco = listco

def draw_line():
    global line_color
    line_color = (0, 0, 0)
    # select_line_color()
    try:
        cv.line(img, tuple(head_in[0]), tuple(back_in[0]), line_color, 2, lineType=cv.LINE_AA)
        cv.line(img, tuple(back_in[0]), tuple(hip_in[0]), line_color, 2, lineType=cv.LINE_AA)
        cv.line(img, tuple(hip_in[0]), tuple(knee_in[0]), line_color, 2, lineType=cv.LINE_AA)
        cv.line(img, tuple(knee_in[0]), tuple(feet_in[0]), line_color, 2, lineType=cv.LINE_AA)
    except:
        pass


def draw_spot_tracking():
    global spot_color, font, akk, ak, hum
    font = cv.FONT_HERSHEY_SIMPLEX
    thickness = 5
    ak = []
    spot_color = (0,0,255)

    for i, v in listco.items():
        ak.append(((int((2 * listco[i][0] + listco[i][2]) / 2), int((2 * listco[i][1] + listco[i][3]) / 2))))

    for i in range(len(ak)):
        if i == 0 and (listco['head'][0] < 0 or (listco['head'][0]+listco['head'][2]) > img.shape[1]) and listco['head'][2] > int((img.shape)[1]/2):
            continue
        if i == 1 and (listco['knee'][0] < 0 or (listco['knee'][0]+listco['knee'][2]) > img.shape[1]):
            continue
        if i == 2 and (listco['hip'][1] <= listco['head'][1] or listco['hip'][1] <= listco['back'][1]):
            hum = False
            detect_dic[i+1] = 0
            continue
        elif i == 4 and (listco['back'][1] <= listco['head'][1] or listco['back'][1] >= listco['hip'][1]):
            hum = False
            detect_dic[i+1] = 0
            continue
        else:
            pass
        if (detect_dic[i+1] == 0 and (abs(ak[i][0]-akk[i][0])+abs(ak[i][1]-akk[i][1])) < 30) and hum != True:
            continue
        cv.circle(img, ak[i], 3, spot_color, thickness=thickness)
    if frame_num % 30 == 0:
        akk = ak

def draw_box_tracking():
    for c, e in enumerate(listco.values()):
        if ((detect_dic[c + 1] == 0 and abs(ak[c][0] - akk[c][0]) + abs(ak[c][1] - akk[c][1]) < 30) and hum != True):
            continue
        elif c == 0 and (listco['head'][0] < 0 or (listco['head'][0]+listco['head'][2]) > img.shape[1]):
            continue
        elif c == 1 and (listco['knee'][0] < 0 or (listco['knee'][0]+listco['knee'][2]) > img.shape[1]):
            continue
        elif c == 2 :
            cv.rectangle(img, (e[0], e[1]), (e[0] + e[2], e[1] + e[3]),
                         (randint(0, 255), randint(0, 255), randint(0, 255)), 3)
        elif c == 3 and (listco['feet'][0] < 0 or (listco['feet'][0]+listco['feet'][2]) > img.shape[1]):
            continue
        elif c == 4 and (listco['back'][0] < 0 or (listco['back'][0]+listco['back'][2]) > img.shape[1]):
            continue
        else:
            cv.rectangle(img, (e[0], e[1]), (e[0] + e[2], e[1] + e[3]),
                         (0, 0, 0), 3)

def get_slope_tracking():
    global c1, c2, c3, c4
    cc1,cc2,cc3 = 0,0,0
    try:
        c1 = abs((((2 * listco['back'][1] + listco['back'][3]) / 2) - ((2 * listco['head'][1] + listco['head'][3]) / 2)) / (
                    ((2 * listco['back'][0] + listco['back'][2]) / 2) - ((2 * listco['head'][0] + listco['head'][2]) / 2)))
    except ZeroDivisionError:
        c1 = cc1
    finally:
        cc1 = c1
    try:
        c2 = abs((((2 * listco['back'][1] + listco['back'][3]) / 2) - ((2 * listco['hip'][1] + listco['hip'][3]) / 2)) / (
                    ((2 * listco['back'][0] + listco['back'][2]) / 2) - ((2 * listco['hip'][0] + listco['hip'][2]) / 2)))
    except ZeroDivisionError:
        c2 = cc2
    finally:
        cc2 = c2
    try:
        c3 = abs((((2 * listco['knee'][1] + listco['knee'][3]) / 2) - ((2 * listco['hip'][1] + listco['hip'][3]) / 2)) / (
                    ((2 * listco['knee'][0] + listco['knee'][2]) / 2) - ((2 * listco['hip'][0] + listco['hip'][2]) / 2)))
    except ZeroDivisionError:
        c3 = cc3
    finally:
        cc3 = c3
    try:
        c4 = abs((((2 * listco['knee'][1] + listco['knee'][3]) / 2) - ((2 * listco['feet'][1] + listco['feet'][3]) / 2)) / (
                    ((2 * listco['knee'][0] + listco['knee'][2]) / 2) - ((2 * listco['feet'][0] + listco['feet'][2]) / 2)))
    except ZeroDivisionError:
        c4 = cc3
    finally:
        cc3 = c4

    return c1, c2, c3, c4

def react_chest_char():
    try:
        if c1 < c2:
            cv.putText(img, 'Chest UP', (130, 30), font, 1, (0, 0, 0), 1, cv.LINE_AA)
    except:
        pass
        # cv.putText(img, 'Good!', (130, 30), font, 1, (0, 0, 0), 1, cv.LINE_AA)
    return img


def react_chest():
    if (s1 >= 1.5) & (s1 < 3):
        return 1
    else:
        return 0


def react_back():
    if (s2 < s1) & (s2 > 0.8):
        return 1
    else:
        return 0


def react_thigh():
    if (s3 < 0.6):
        return 1
    else:
        return 0


# def count(time):
#     cv.putText(img, 'Count: {}',time,)

def feed_back():
    global count, swt, startt, ns, old_c
    get_slope()
    react_chest()
    react_back()
    react_thigh()
    old_c = count
    a, b, c = react_chest(), react_back(), react_thigh()
    # abc = a,b,c
    startt = -1.001
    # ns = startt + 2

    if a and b and c == 1:
        startt = time.localtime().tm_sec
        ns = startt + 2
        swt = 1
        if startt == 58:
            ns = 0
        if startt == 59:
            ns = 1
        import winsound
        try:
            winsound.PlaySound('good.wav', winsound.SND_FILENAME)
        except:
            pass

    # if isinstance(ns, int) and time.localtime().tm_sec < ns:
    if swt == 1:
        # if time.localtime().tm_sec == startt+1:
        #     count += 2
        if time.localtime().tm_sec != ns:
            cv.putText(img, 'Good pose!', (30, 60), font, 1, (0, 0, 0), 1, cv.LINE_AA)
        else:
            startt = time.localtime().tm_sec
            swt = 0
            count += 1

    else:
        # pass
        swt = 0

        # if old_c+1 == count:
        #
        #     old_c =count

        # old_c = count
    # if startt+1 == time.localtime().tm_sec:
    # count != old_c
    # count = old_c + 1
    cv.putText(img, 'Count: {}'.format(count), (30, 25), font, 1, (0, 0, 0), 1, cv.LINE_AA)

    # else:
    #     pass
    return img, count


def main():
    # file_name = 'pp.jpg'#input('확장자를 포함한 파일명을 입력하세요.')
    # read_camera()
    detect_bodyparts()
    # init_tracking()
    # get_bodyparts()
    # draw_line()
    # draw_spot()
    # feed_back()


if __name__ == '__main__':
    count = 0
    counting = 0
    up = False
    counting_flag = False
    hum = False
    listco5 = {'head':(0,0,0,0)}
    listco51 = {'head':(0,0,0,0)}
    swt = 0
    st = []
    high_end = 1000
    high_dend = 0
    tts = []
    countings = []
    initin = True
    c1, c2, c3 = 0, 0, 0

    frame_num = 0
    frame_num_tt1 = 0
    old_thigh = 0
    akk = [(0, 0), (0, 0), (0, 0), (0, 0), (0, 0)]

    point = [[], [], [], [], []]
    rect = [[], [], [], [], []]
    detect_dic = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}

    listco = {'head': [0, 0, 0, 0], 'knee': [0, 0, 0, 0], 'hip': [0, 0, 0, 0], 'feet': [0, 0, 0, 0],
              'back': [0, 0, 0, 0]}
    listcoco = {'head': (0, 0, 0, 0), 'knee': (0,0,0,0), 'hip': (0,0,0,0),'feet': (0,0,0,0),'back': (0,0,0,0)}
    listco3 = {'head':[1,1,1,1],'knee':[1,1,1,1],'hip':[1,1,1,1],'feet':[1,1,1,1],'back':[1,1,1,1]}

    # cam = cv.VideoCapture(0)

    # cam = cv.VideoCapture("test9.mp4")
    # cam = cv.VideoCapture("squat_vid5_2.mp4")
    cam = cv.VideoCapture("./videos/test7_2.mp4")
    ret, frame = cam.read()
    h, w = frame.shape[:2]
    fourcc = cv.VideoWriter_fourcc(*'DIVX')
    video_write = cv.VideoWriter('saved_out.avi', fourcc, 25.0, (w, h))
    # cam = cv.VideoCapture("squat_vid2.mp4")

    # cam.set(cv.CAP_PROP_FPS,1)
    with tf.gfile.FastGFile('frozen_inference_graph100973.pb', 'rb') as f:
    # with tf.gfile.FastGFile('frozen_inference_graph74714.pb', 'rb') as f:
        # with tf.gfile.FastGFile('frozen_inference_graph25324.pb', 'rb') as f:
        # with tf.gfile.FastGFile('frozen_inference_graph15451.pb', 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    with tf.Session() as sess:
        sess.graph.as_default()
        tf.import_graph_def(graph_def, name='')
        read_camera()
        main()
        output()
        init_tracking()
        while 1:
            # try:
            read_camera()

            # st = time.strftime('%H:%M:%S')
            # print(st)

            # print(detect_dic)
            # cv.putText(img,)
            old_e = [0.3, 0.2, 0.2, 0.2, 0.2]  # pp2
            old_h = [0.8, 0.5, 0.5, 0.5, 0.5]  # pp2

            # if counting != counting2:
                # print(counting)
            # if get_slope_tracking()[2] < 1 and up == True and counting == counting2:
            if frame_num % 60 == 0 and hum==False:

                out2 = detect_bodyparts()
                num_detections2 = int(out2[0][0])
                scores2 = []
                for i in range(num_detections2):
                    score2 = float(out2[1][0][i])
                    scores2.append(score2)
                    classId2 = int(out2[3][0][i])
                    if score2 > old_e[classId2 - 1]:
                        # print(classId2)

                        old_e[classId2 - 1] = score2
                        rect_makin(i)

                        detect_dic[classId2] = 1

                        # print(detect_dic.values())

                        # detect_dic.values()[] = 1
                if 0 not in detect_dic.values():
                    hum = True
                    up = False
                    if st == []:
                        st = dt.datetime.now()  # .strftime('%M:%S')

                    print(hum)


                elif max(scores2) <= 0.8:
                    hum = False
                    st = []
                    print('no hum')
                    detect_dic = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}

            elif (listco['head'][0] < 0 or (listco['head'][0]+listco['head'][2]) > img.shape[1]) and frame_num % 60 == 0:
                hum = False
                nnt = dt.datetime.now()
                countings.append(counting)
                counting = 0
                if st != []:
                    tt1 = str(nnt - st)
                    print('total', tt1)
                    frame_num_tt1 = frame_num
                detect_dic = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
                out2 = detect_bodyparts()
                num_detections2 = int(out2[0][0])
                scores2 = []
                for i in range(num_detections2):
                    score2 = float(out2[1][0][i])
                    scores2.append(score2)
                    classId2 = int(out2[3][0][i])
                    if score2 > old_e[classId2 - 1]:
                        old_e[classId2 - 1] = score2
                        rect_makin(i)
                        detect_dic[classId2] = 1
                if 0 not in detect_dic.values():
                    hum = True
                    up = False
                    if st == []:
                        st = dt.datetime.now()

                    print(hum)


                elif max(scores2) <= 0.8:
                    hum = False
                    st = []
                    print('no hum')
                    detect_dic = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}

            elif get_slope_tracking()[2] < 1 and up == True and counting_flag == False and hum == True and initin == True:
                # print('squat!!')
                counting_flag = True
                out2 = detect_bodyparts()
                num_detections2 = int(out2[0][0])
                scores2 = []
                for i in range(num_detections2):
                    score2 = float(out2[1][0][i])
                    scores2.append(score2)
                    classId2 = int(out2[3][0][i])
                    if score2 > old_h[classId2 - 1]:
                        # print(classId2)
                        old_h[classId2 - 1] = score2
                        rect_makin(i)
                        hum = True
                if max(scores2) <= 0.9:
                    hum = False
                    nnt = dt.datetime.now()
                    tt1 = str(nnt -st)
                    countings.append(counting)
                    counting = 0
                    counting_flag = False
                    if st != []:
                        print('total',tt1)
                        frame_num_tt1 = frame_num

                    st = []
                    # print(hum)
                    detect_dic = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
                # print('aa',get_slope_tracking())

                if 0.95 > get_slope_tracking()[2] and 0.8 < get_slope_tracking()[1] and 0.7 < get_slope_tracking()[0] and 2 > get_slope_tracking()[0]\
                        and  2 < get_slope_tracking()[3]:
                    counting += 1
                    initin = False

                # else:
                #     counting2 = 1000
            # init_tracking()
            # t = time.time()
            if hum == True and high_end > listco5['head'][1] and listco5['head'][1] != 0:
                high_end = listco5['head'][1]
                # high_dend = listco5['head'][0]
                print(high_end)
            tracking()
            if frame_num % 15 == 0:
                old_thigh = get_slope_tracking()[2]
            draw_spot_tracking()
            draw_box_tracking()
            # get_slope_tracking()
            # react_chest_char()
            # time.sleep(0.03)
            frame_num += 1

            kk = cv.waitKey(3)

            # cv.resizeWindow('Squat', 480, 852)
            nt = dt.datetime.now()#.strftime('%M:%S')

            if st != []:
                ddt = (nt - st)
                # print(type(ddt))
                # print('{}',dt.datetime.minute,':{}')
                # ddt = ddt.__format__('%M:%S:%ms')
                # ddt = ddt
                ddt = str(ddt)
                # print(ddt)
            # try:
                cv.putText(img, 'Squat Time : '+ddt, (30,30), cv.FONT_HERSHEY_SIMPLEX,0.5,(0,0,0),2, cv.LINE_AA)
            # except:
            #     pass
            if frame_num_tt1 != 0 and frame_num_tt1 + 50 >= frame_num:
                if frame_num % 10 < 9:
                    cv.putText(img, 'Squat time saved : ' + tt1, (80, 60), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 105, 255),2)
            else:
                pass
            cv.putText(img,'COUNT: '+str(counting),(int(img.shape[1]*5/8),int(img.shape[0]*0.5/10)), cv.FONT_HERSHEY_SIMPLEX, 1, (0,0,0),4)
            video_write.write(frame)
            cv.imshow('Squat', img)
            time.sleep(0.018)
            if kk == 27:
                cv.destroyAllWindows()
                cam.release()
                video_write.release()
                countings.append(counting)
                print('아임 그루트')
                break
            elif kk == ord("b"):
                old = [0.2, 0.2, 0.2, 0.2, 0.2]
            else:
                pass
            # except:
            #     e,f = sys.exc_info()[0], sys.exc_info()[1]
            #     e1 = sys.exc_info()
            #     print(e,f)
            #     cv.destroyAllWindows()
            #     cam.release()
            #     break
            # finally:
            #     pass
        print(frame_num)
        print(countings)

