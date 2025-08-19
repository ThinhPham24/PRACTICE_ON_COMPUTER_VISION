from cmath import cos
import cv2
import os
import sys
import random
import math
import time
import numpy as np
length_arrow = 50
split = 11
import argparse
import colorsys
import glob
import datetime
import re
import logging
import scipy
import skimage.color
import skimage.io
import skimage.transform
import urllib.request
import shutil
import warnings
from distutils.version import LooseVersion
from collections import OrderedDict
def random_colors(N, bright=True):
    """
    Generate random colors.
    To get visually distinct colors, generate them in HSV space then
    convert to RGB.
    """
    brightness = 1.0 if bright else 0.7
    hsv = [(i / N, 1, brightness) for i in range(N)]
    colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
    random.shuffle(colors)
    return colors

def find_moment(contour):
    M = cv2.moments(contour)
    if M['m00'] != 0:
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])
    else:
        cx = 0
        cy = 0
    return cx,cy

def apply_masks(image, mask, color, alpha=0.5):
    """Apply the given mask to the image.
    """
    for c in range(3):
        image[:, :, c] = np.where(mask == 1,
                                  image[:, :, c] *
                                  (1 - alpha) + alpha * color[c] * 255,
                                  image[:, :, c])
    return image

#----------------NTD----------------------------
def cart2pol(x, y):
    theta = np.arctan2(y, x)
    rho = np.hypot(x, y)
    return theta, rho


def pol2cart(theta, rho):
    x = rho * np.cos(theta)
    y = rho * np.sin(theta)
    return x, y


def rotate_contour(cnt, angle, cen_point = None):
    if cen_point != None:
        cx, cy = cen_point
    else:
        M = cv2.moments(cnt)
        if M['m00'] != 0:
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
        else:
            cx = 0
            cy = 0
    cnt_norm = cnt - [cx, cy]
    
    coordinates = cnt_norm[:, 0, :]
    xs, ys = coordinates[:, 0], coordinates[:, 1]
    thetas, rhos = cart2pol(xs, ys)
    
    thetas = np.rad2deg(thetas)
    thetas = (thetas + angle) % 360
    thetas = np.deg2rad(thetas)
    
    xs, ys = pol2cart(thetas, rhos)
    
    cnt_norm[:, 0, 0] = xs
    cnt_norm[:, 0, 1] = ys
    # print("xs,ys",(xs,ys))
    # cv2.circle(img,(xs,ys),6,[0,0,255],-1)
    # cv2.imshow("points",img)
    # cv2.waitKey(0)
    cnt_rotated_1 = cnt_norm + [cx, cy]
    cnt_rotated = cnt_rotated_1.astype(np.int32)

    return cnt_rotated, [cx, cy], cnt_rotated_1
def resized_img(img,percent):
    width = int(img.shape[1] * percent / 100)
    height = int(img.shape[0] * percent / 100)
    dim = (width, height)
    # resize image
    resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
    return resized
def find_rbbox(img,mask, angle):
    ret, thresh = cv2.threshold(mask, 127, 255,0)
    contours,hierarchy = cv2.findContours(thresh,2,1)
    # cnt = contours[0]
    cnt = max(contours, key = cv2.contourArea)
    # cv2.drawContours(img,cnt,-1,(0,255,0),10)
    #-----------S1 rect, rect_rotated--------------
    cnt_rotated, center_point,_ = rotate_contour(cnt, angle,None)
    # print("asdafdaf",cnt_rotated)
    # cv2.drawContours(img,cnt_rotated,-1,(255,0,0),10)
    x,y,w,h = cv2.boundingRect(cnt_rotated)
    rect_rotated = np.array([[[x, y]],[[x+w, y]],[[x+w, y+h]],[[x, y+h]]])
    # cv2.circle(img,(x, y),5,[255,0,0],-1)
    # cv2.circle(img,(x+w, y),5,[255,0,0],-1)
    # cv2.circle(img,(x+w, y+h),5,[255,0,0],-1)
    # cv2.circle(img,[x, y+h],5,[255,0,0],-1)
    rect_out, _,_ = rotate_contour(rect_rotated, -angle, center_point)    # rotate back
    #-----------S2 origin_rotated-------------------
    length = w
    # convex_hull ==> far_points
    hull = cv2.convexHull(cnt_rotated,returnPoints = False) # returnPoints = False while finding convex hull, in order to find convexity defects.
    try:
        defects = cv2.convexityDefects(cnt_rotated,hull)
    except:
        cnt = cv2.approxPolyDP(cnt_rotated,0.01*cv2.arcLength(cnt,True),True)
        hull = cv2.convexHull(cnt, returnPoints=False)
        defects = cv2.convexityDefects(cnt, hull)
        # pass
    far_points = []
    for i in range(defects.shape[0]):
        s,e,f,d = defects[i,0]     #[ start point, end point, farthest point, approximate distance to farthest point ].
        far = tuple(cnt_rotated[f][0])
        far_points.append(far)
        # cv2.circle(img,far,5,[0,0,255],-1)
    # print(far_points)

    # filter convex_hull in 1/3 rect_rotated (x,x+length *1/3), (y,y+h) percent =33%
    percent = 10
    far_points_filtered = []
    for point in far_points:
        if (x <= point[0] <= x+int(length*percent/100)) and (y <= point[1] <= y+h):
            far_points_filtered.append(point)
        # elif 
        #     far_points_filtered.append(np.max(point))
    y_min, y_max = 10000, 0
    for point in far_points_filtered:
        if point[1] < y_min:
            y_min = point[1]
        if point[1] > y_max:
            y_max = point[1]
    if not far_points_filtered: 
        origin_rotated = np.array([[[x, center_point[1]]]])
        print("cannot defect hull")
    else:
        origin_rotated = np.array([[[x, (y_min + y_max)//2]]])
    # print("center point", center_point)
    origin,_,diem = rotate_contour(origin_rotated, -angle, center_point)
    # print("diem",origin)
    # cv2.drawContours(img, [diem], 0, (0, 255, 255), 20)
    origin = tuple(origin.squeeze())

    # origin + angle + length ==> end_point
    x_end = int(origin[0] + length * math.cos(math.radians(angle)))
    y_end = int(origin[1] - length * math.sin(math.radians(angle)))
    end_point = (x_end, y_end)      
    # Draw vector
    # -----------Other part---------
    # y1, x1, y2, x2 = roi
    # cv2.rectangle(img,(x1,y1),(x2,y2),(0,0,0),2)
    caption = str(angle)
    x_add = int(30*math.cos(math.radians(angle)))
    y_add = int(10*math.sin(math.radians(angle)))
    cv2.putText(img, caption, (x_end + x_add, y_end -y_add), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 1)
    # ------------------------------  
    # cv2.drawContours(img, [rect_out], 0, (255, 0, 0), 2)
    cv2.circle(img, origin,3,[0,0,255],-1)
    cv2.arrowedLine(img, origin, end_point, (255,0,0), 2)

    # cv2.drawContours(img, contours, 0, (255, 0, 0), 3)
    # cv2.drawContours(img, [cnt_rotated], 0, (0, 255, 0), 3)
    # cv2.drawContours(img, [rect_rotated], 0, (0, 255, 0), 2)
    
    # cv2.imwrite('out.png',img)
    # cv2.imshow('img',img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return origin, length
def save_bud_info(output_folder, image_name, angles, points, lengths):
    f = open(f"{output_folder}/TXT/{image_name}.txt", "w")
    for i in range(len(points)):
        # start_point + end_point
        x,y = points[i]
        x_tip = int(x + lengths[i] * math.cos(math.radians(angles[i])))
        y_tip = int(y - lengths[i] * math.sin(math.radians(angles[i])))
        
        # save to .txt file
        one_line = ' '.join([str(m) for m in [x,y,x_tip,y_tip]])
        f.write(f"{one_line}\n")
    f.close()

    # new txt
    f = open(f"{output_folder}/TXT/{image_name}_1.txt", "w")
    for i in range(len(points)):
        # start_point + end_point
        x,y = points[i]
        
        # save to .txt file
        one_line = ' '.join([str(m) for m in [x,y,angles[i],lengths[i]]])
        f.write(f"{one_line}\n")
    f.close()