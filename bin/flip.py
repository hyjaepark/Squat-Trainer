## for all images in folder, flip files and write

import glob
import cv2 as cv
import os
# files = glob.glob("C:\\\\Users\\\\user\\\\Desktop\\\\workspace\\\\workspace_new_image\\\\*.jpg")
for i in ["train","test"]:
    files = glob.glob("C:\\Users\\user\\Desktop\\workspace\\workspace_new_image\\images\\{}\\*.jpg".format(i))
    print(files)
    path = "C:\\Users\\user\\Desktop\\workspace\\workspace_new_image\\{}f".format(i)
    # test = 0
    for i in files:
        img = cv.imread(i)
        img = cv.flip(img,1)
        # cv.imshow("test",img)
        cv.imwrite(os.path.join(path,"flipped_{}.jpg".format(i[-11:-4])),img)
        cv.waitKey()
        cv.destroyAllWindows()
        # test += 1
        # if test == 5:
        #     break


# img = cv.imread("pic_011.jpg")
# img = cv.flip(img,1)
# cv.imshow("test",img)
# cv.imwrite("test_flip.jpg",img)
# cv.waitKey()
# cv.destroyAllWindows()
