# import cv2
# import glob
# import math
# import numpy as np
# import os
# def cal_dis(pts1,pts2):
#     dist = math.sqrt(int(pts2[0]-pts1[0])**2 + int(pts2[1]-pts1[1])**2)
#     return dist
# def resized_img(img,percent):
#     width = int(img.shape[1]*percent/100)
#     height = int(img.shape[0]*percent/100)
#     dim = (width,height)
#     resized = cv2.resize(img,dim,interpolation = cv2.INTER_AREA)
#     return resized
# count = 6000
# #Top 0.3 0.9
# #240 0.2 0.8
# if __name__ == "__main__":
#     if not os.path.exists("./images"):
#         os.mkdir('./images')
#     path_im = sorted(glob.glob('./original_10_5/*_4_L.jpg'))
#     lower = [0, 52, 44]
#     upper = [179, 255, 255]
#     HSV = np.vstack((lower, upper))
#     for file in path_im:

#         image_RBG = cv2.imread(file)
#         y, x, _ = (image_RBG.shape)
#         top = int(y * 0.3)  # 0.3
#         left = int(x * 0.3)  # 0.3
#         bot = int(y * 0.9)
#         right = int(x * 0.9)
#         crop_img = image_RBG[top:bot, left:right]
#         # cv2.imshow("image", resized_img(crop_img,30))
#         # cv2.waitKey(0)
#         image_blur = cv2.GaussianBlur(crop_img, (7, 7), 0)
#         kernel = np.ones((7, 7), np.uint8)
#         lower = np.asarray(HSV[0])
#         upper = np.asarray(HSV[1])
#         # Convert to HSV format and color threshold
#         image_hsv = cv2.cvtColor(image_blur, cv2.COLOR_BGR2HSV)
#         mask = cv2.inRange(image_hsv, lower, upper)
#         mask = cv2.dilate(mask, kernel, iterations=3)
#         kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
#         mask_closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
#         mask_clean = cv2.morphologyEx(mask_closed, cv2.MORPH_OPEN, kernel)
#         contours, _ = cv2.findContours(mask_clean, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
#         max_area1 = -1
#         for i in contours:
#             area = cv2.contourArea(i)
#             if area > max_area1:
#                 max_area1 = area
#                 contour = i
#         cnt = contour
#         x, y, w, h = cv2.boundingRect(cnt)
#         cx = int((2 * x + w) / 2)
#         cy = int((2 * y + h) / 2)
#         # cv2.rectangle(image_RBG, (x, y), (x + w, y + h), (0, 255, 0), 2)
#         # cv2.drawContours(image_RBG, cnt, -1, (0, 0, 255), 3)
#         # cv2.imshow("image", resized_img(image_RBG,30))
#         # cv2.waitKey(0)
#         d = int(cal_dis((x, y), (cx, cy)))
#         # cv2.circle(image_RBG,(cx,cy), 5, (0,255,0), 1)
#         crop_img = crop_img[cy - d:cy + d, cx - int(d * 80 / 57):cx + int(d * 80 / 57)]
#         # print(crop_img)
#         if len(crop_img) == 0:
#             print("This list is empty")
#             print("order path", file)
#             continue
#         else:
#             crop_resize = cv2.resize(crop_img, (640, 456))  # (w,h)
#             cv2.imwrite("./images/" + '{}.jpg'.format(count), crop_resize)  # fix here
#         count +=1
#MASKRCNN don't need root, darken, only using single bud and angle imforamtion.
import cv2
import glob
import math
import numpy as np
import os
def cal_dis(pts1,pts2):
    dist = math.sqrt(int(pts2[0]-pts1[0])**2 + int(pts2[1]-pts1[1])**2)
    return dist
def resized_img(img,percent):
    width = int(img.shape[1]*percent/100)
    height = int(img.shape[0]*percent/100)
    dim = (width,height)
    resized = cv2.resize(img,dim,interpolation = cv2.INTER_AREA)
    return resized

if __name__ == "__main__":
    count = 1820
    if not os.path.exists("./images_3"):
        print("True")
        os.mkdir('./images_3')
    path_im = sorted(glob.glob('/home/airlab/Desktop/Code_label_for_new_student/DATASET/subimage_15/*.jpg'))
    for file in path_im:
        file_im = cv2.imread(file)
        # file_im = cv2.resize(file_im,(640,456))
        cv2.imwrite("./images_3/" + '{}.jpg'.format(count),file_im) #fix here
        count +=1
    print("Done!")