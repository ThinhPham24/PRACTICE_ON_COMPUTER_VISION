from enum import EnumMeta
import sys
import os
import json
import datetime
from tkinter import image_names
from turtle import title
import numpy as np
import skimage.draw
import cv2
import os
import sys
import random
import itertools
import colorsys
from PIL import Image, ImageDraw
import numpy as np
from skimage.measure import find_contours
import matplotlib.pyplot as plt
from matplotlib import patches,  lines
from matplotlib.patches import Polygon
import IPython.display
import matplotlib.pyplot as plt
import random
#***************
import datetime
import json
import os
import re
import fnmatch
from PIL import Image
import numpy as np
import math
import time
import argparse
import glob
from inference import find_rbbox, resized_img
#***************
# folder = "/TANG/"
# annotations_dir = "annotations"
# current_dir = os.getcwd()
# path= ''.join([current_dir, folder])

# annotations_path = os.path.join(path, annotations_dir)

# print("annotations_savepath ",annotations_path )
# print("annotations_savepath = ", annotations_savepath)
# if not os.path.isdir(os.path.abspath(annotations_path)):
#     os.mkdir(annotations_path)
#*********************
def filter_for_jpeg(root, files):
    file_types = ['*.jpeg', '*.jpg']
    file_types = r'|'.join([fnmatch.translate(x) for x in file_types])
    files = [os.path.join(root, f) for f in files]
    files = [f for f in files if re.match(file_types, f)]
    
    return files

def filter_for_annotations(root, files, image_filename):
    # print("image_filename:",image_filename)
    file_types = ['*.png']
    file_types = r'|'.join([fnmatch.translate(x) for x in file_types])
    basename_no_extension = os.path.splitext(os.path.basename(image_filename))[0]
    file_name_prefix = basename_no_extension + '_' + '.*'
    files = [os.path.join(root, f) for f in files]
    files = [f for f in files if re.match(file_types, f)]
    files = [f for f in files if re.match(file_name_prefix, os.path.splitext(os.path.basename(f))[0])]
    return files
# constants
def magic(numList):
    s = ''.join(map(str, numList))
    return int(s)
def find_annotation(image_filename,TRAIN_ANNOTATION_DIR):
    total_angle = []
    for root, _, files in os.walk(TRAIN_ANNOTATION_DIR):
        annotation_files = filter_for_annotations(root, files, image_filename)
        # print("annotation_files:",annotation_files)
        # go through each associated annotation
        for annotation_filename in annotation_files:
            angle_list = []
            
            check = 0
            # ----- Xử lý dựa trên tên của annotation file (binary image) ----------#
            #print(annotation_filename)
            file, ext = os.path.splitext(annotation_filename)  # split filename and extension
            # print("file = ", file)
            #print("ext = ", ext)
            name = os.path.basename(file)
            s = len(os.path.basename(file))
            #print("len = ", s)
            l = list(name)
            for i in range(s):
                if l[i] == '_':
                    check = check + 1  
                if l[i] != '_' and check==3:
                    angle_list.append(l[i])
                else:
                    continue
            #print("angle_list = ", angle_list)

            angle_int = list(map(int, angle_list))
            angle = magic(angle_int)
            # Clasify the angles (18 CLASSES - 20 degrees each part)

            if (angle > 350 and angle <=360) or (angle >= 0 and angle <= 10):
                angle_cl = 0

            elif (angle > 10 and angle <= 30):
                angle_cl = 1

            elif (angle > 30 and angle <= 50):
                angle_cl = 2 

            elif (angle > 50 and angle <= 70):
                angle_cl = 3

            elif (angle > 70 and angle <= 90):
                angle_cl = 4  

            elif (angle > 90 and angle <= 110):
                angle_cl = 5

            elif (angle > 110 and angle <= 130):
                angle_cl = 6  

            elif (angle > 130 and angle <= 150):
                angle_cl = 7

            elif (angle > 150 and angle <= 170):
                angle_cl = 8  

            elif (angle > 170 and angle <= 190):
                angle_cl = 9

            elif (angle > 190 and angle <= 210):
                angle_cl = 10  

            elif (angle > 210 and angle <= 230):
                angle_cl = 11    

            elif (angle > 230 and angle <= 250):
                angle_cl = 12

            elif (angle > 250 and angle <= 270):
                angle_cl = 13  

            elif (angle > 270 and angle <= 290):
                angle_cl = 14

            elif (angle > 290 and angle <= 310):
                angle_cl = 15  

            elif (angle > 310 and angle <= 330):
                angle_cl = 16

            elif (angle > 330 and angle <= 350):
                angle_cl = 17  
            else:
                continue
            # angle_s = angle_cl
            total_angle.append(angle_cl)
        return annotation_files, total_angle
def color_radom(i): 
    r = random.randint(50,255)
    g = random.randint(50,100)
    b = random.randint(50,255)
    rgb = [r,g,b]
    return rgb
if __name__ =="__main__":
    '''
    python check_only_annotation.py
    '''
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--path_img', default= "C:\\Users\\ptthi\\OneDrive\\Desktop\\Image_processing\\new_data\\new_sub20_train\\IMG", type=str, required=False,
                        help= 'direction of image folder') #path to the folder that contain image


    parser.add_argument('--path_an', default= "C:\\Users\\ptthi\\OneDrive\\Desktop\\Image_processing\\new_data\\new_sub20_train\\ANNOTATION", type=str, required=False,
                        help= 'direction of annotation folder') #annotation path
    args = parser.parse_args()
    print("this is path", args.path_img)
    filenames = sorted(glob.glob(args.path_img + "/*.jpg")) #read all files in the path mentioned
    number_image = 1
    segmentation_id = 1
    for index_1,image_name in enumerate(filenames):
        print("image_filename:",image_name)            
        image_original = cv2.imread(image_name)
        image_color = cv2.imread(image_name)
        file, ext = os.path.splitext(image_name)
        file = str(file).split('/')[-1]
        annotation_files, angles_all = find_annotation(image_filename=image_name, TRAIN_ANNOTATION_DIR= args.path_an)
        print("annotation_file:",annotation_files)
        print("angle of model:", angles_all)
        angles, points, lengths = [], [], []
        overlay = image_color.copy()
        image_only_contour = overlay.copy()
        for i in range(len(annotation_files)):
            # print("current mask", annotation_files[i])
            image_mask = cv2.imread(annotation_files[i])
            gray = cv2.cvtColor(image_mask, cv2.COLOR_BGR2GRAY)
            ret, thresh = cv2.threshold(gray,127,255,cv2.THRESH_BINARY)
            cnts, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
            # for i in range(len(point_cnt)):cv2.fillPoly(image_backgorund, point_cnt[i], color_radom(len(point_cnt)))  
            # img_with_overlay = cv2.normalize(np.int64(image_color) * image_backgorund, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
            cv2.drawContours(image_only_contour, cnts, -1, (0, 0, 255), 1) 
            for (i_x,cnt) in enumerate(cnts):
                cv2.fillPoly(overlay, [cnt] ,color_radom(i))
                cv2.addWeighted(overlay, 0.5, image_color, 1 - 0.5,0,image_color)
            # print("angle", angles_all[i])
            find_rbbox(image_color,gray,angles_all[i]*20)
            cv2.drawContours(image_color, cnts, -1, (0, 0, 0), 1) 
            find_rbbox(image_only_contour,gray,angles_all[i]*20)
        concate_img = np.concatenate((image_color,image_only_contour),axis=1)
        image_white = np.zeros((concate_img.shape[0],concate_img.shape[1],3),np.uint8)
        image_white.fill(2555)
        image_white[:,int(int(concate_img.shape[1]/2)/2):int(int(concate_img.shape[1]/2)/2) + int(int(concate_img.shape[1]/2))] = image_original
        concate_img_v = np.concatenate((image_white,concate_img),axis = 0)
        print("number of image:", number_image)
        number_image += 1
        # cv2.imwrite(check_annotation_path + '/'+ f"{file}_check.png",concate_img_v)
        cv2.imshow("mask_image", resized_img(concate_img,70))
        k = cv2.waitKey(0)
        if k ==ord('q') or k ==ord('Q'):          #press 'Q' to stop the program
            break
    print("CHECKING ANNOTAION IS TO BE FINISH!")

