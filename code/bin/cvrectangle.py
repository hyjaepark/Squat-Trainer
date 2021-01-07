import cv2 as cv
import numpy as np

# img = cv.imread('''C:\\Users\\user\\Desktop\\workspace\\workspace_new_image\\images\\test\\pic_011.jpg''')
# print(type(img))
# print(img.shape)
#
# cv.rectangle(img,(0,0),(100,100),(255,255,255))
# cv.imshow("ss",img)
# cv.waitKey()

dicti = {1:"head",2:"feet"}

inv_dicti = {i:c for c,i in dicti.items()}

print(inv_dicti)

classes_dic = {1:"head",2:"knee",3:"hip",4:"feet",5:"back"}
classes_dic = {i:c for c,i in classes_dic.items()}
print(classes_dic)