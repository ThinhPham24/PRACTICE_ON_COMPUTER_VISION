import cv2
import numpy as np
import os

def resize_image(image, percent):
    x = int((image.shape[1]*percent)/100)
    y = int((image.shape[0]*percent)/100)
    print("x, y", (x,y))
    image_resize = cv2.resize(image,(x, y), interpolation = cv2.INTER_LINEAR)
    return image_resize
def img_process(img):
    ### Binary by HSV inrange ###
    low_H, low_S, low_V, high_H, high_S, high_V = [15, 0, 5, 180, 216, 232]
    image_HSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    image_threshold = cv2.inRange(image_HSV, (low_H, low_S, low_V), (high_H, high_S, high_V))
    # kernel_size = 8
    # kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (kernel_size, kernel_size))
    # frame_erode = cv.erode(frame_threshold, kernel)
    # kernel_size = 12
    # kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (kernel_size, kernel_size))
    # frame_dilate = cv.dilate(frame_erode, kernel)

    return image_threshold