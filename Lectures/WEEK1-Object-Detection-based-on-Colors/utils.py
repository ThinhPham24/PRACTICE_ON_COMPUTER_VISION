import cv2
import numpy as np
import os

def resize_image(image, percent):
    x = int((image.shape[1]*percent)/100)
    y = int((image.shape[0]*percent)/100)
    image_resize = cv2.resize(image,(x, y), interpolation = cv2.INTER_LINEAR)
    return image_resize
def img_process(img):
    ### Binary by HSV inrange ###
    low_H, low_S, low_V, high_H, high_S, high_V = [0, 77, 185, 180, 255, 255]
    image_HSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    image_threshold = cv2.inRange(image_HSV, (low_H, low_S, low_V), (high_H, high_S, high_V))
    kernel_size = 9
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    frame_erode = cv2.erode(image_threshold, kernel)
    kernel_size = 11
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    frame_dilate = cv2.dilate(frame_erode, kernel)
    closing = cv2.morphologyEx(frame_dilate,cv2.MORPH_CLOSE, kernel)

    return image_threshold, closing