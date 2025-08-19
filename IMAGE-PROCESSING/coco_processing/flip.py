# import numpy as np
# a = np.ones([9, 5, 7, 4])
# c = np.ones([9, 5, 4, 3])
# b = np.dot(a, c)
# print(b)
# import the necessary packages

## Run this in conda activate pc environment

import numpy as np
import argparse
import imutils
import cv2
import os, os.path
import glob


import random
import matplotlib.pyplot as plt
import albumentations as A
import numpy as np
import argparse
import imutils
import cv2
import os, os.path
import glob

# *********************** parameters *****************************#
# Offet value - change manually
offset1 = 10000
offset2 = 20000
# No. of new image from one original image
# k = 4; # So luong anh muon tao ra tu anh goc
# k = 3; # So luong anh muon tao ra tu anh goc
# k = 30

k = 1
# Input folder
folder = "/DATA/"
# *********************** parameters *****************************#

rename_list = []
remained_list = []
check = 0

current_dir = os.getcwd()
path = ''.join([current_dir, folder])
print("path = ", path)

#type 0:train, 1:test
type=1

if type==0: #train folder
    # -----------------------------------------------------------------------#
    images_dir = "train/img_new_rot"
    annotations_dir = "train/an_new_rot"
    images_path = os.path.join(path, images_dir)
    annotations_savepath = os.path.join(path, annotations_dir)
    if not os.path.isdir(os.path.abspath(annotations_savepath)):
        os.mkdir(annotations_savepath)
    # -----------------------------------------------------------------------#
    new_image_save = "train/img_rot_flip"
    new_annotations_save = "train/an_rot_flip"
    images_new_savepath = os.path.join(path, new_image_save)
    annotations_new_savepath = os.path.join(path, new_annotations_save)
    if not os.path.isdir(os.path.abspath(images_new_savepath)):
        os.mkdir(images_new_savepath)
    if not os.path.isdir(os.path.abspath(annotations_new_savepath)):
        os.mkdir(annotations_new_savepath)
    # -----------------------------------------------------------------------#
    images_path = os.path.join(path, images_dir)
    annotations_savepath = os.path.join(path, annotations_dir)
    print("images_path = ", images_path)
    if not os.path.isdir(os.path.abspath(annotations_savepath)):
        os.mkdir(annotations_savepath)

if type==1: #test folder
  # -----------------------------------------------------------------------#
    images_dir = "test/img_new_rot"
    annotations_dir = "test/an_new_rot"
    images_path = os.path.join(path, images_dir)
    annotations_savepath = os.path.join(path, annotations_dir)
    if not os.path.isdir(os.path.abspath(annotations_savepath)):
        os.mkdir(annotations_savepath)
    # -----------------------------------------------------------------------#
    new_image_save = "test/img_rot_flip"
    new_annotations_save = "test/an_rot_flip"
    images_new_savepath = os.path.join(path, new_image_save)
    annotations_new_savepath = os.path.join(path, new_annotations_save)
    if not os.path.isdir(os.path.abspath(images_new_savepath)):
        os.mkdir(images_new_savepath)
    if not os.path.isdir(os.path.abspath(annotations_new_savepath)):
        os.mkdir(annotations_new_savepath)
    # -----------------------------------------------------------------------#
    images_path = os.path.join(path, images_dir)
    annotations_savepath = os.path.join(path, annotations_dir)
    print("images_path = ", images_path)
    if not os.path.isdir(os.path.abspath(annotations_savepath)):
        os.mkdir(annotations_savepath)

# constants
def magic(numList):
    s = ''.join(map(str, numList))
    return int(s)


filenames = glob.glob(images_path + "/*.*")  # read all files in the path mentioned
for n, image_file in enumerate(filenames):
    file, ext = os.path.splitext(image_file)  # split filename and extension
    name = os.path.basename(file)
    s = len(os.path.basename(file))

    l = list(name)
    for i in range(s):
        if l[i] != '_' and check == 0:
            rename_list.append(l[i])
        else:
            check = 1
            remained_list.append(l[i])
    rename_int = list(map(int, rename_list))
    remained_name = ''.join(map(str, remained_list))
    num = magic(rename_int)
    num_modify = num * k
    new_value1 = num_modify + offset1
    new_value2 = num_modify + offset2
    rename_list = []
    remained_list = []
    check = 0

    file, ext = os.path.splitext(image_file)  # split filename and extension
    # image = cv2.imread(image_file.path)
    image = cv2.imread(image_file)  #
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    transform = A.VerticalFlip(p=1)
    augmented_image = transform(image=image)
    filename = os.path.join(images_new_savepath, '{}{}.jpg'.format(new_value1, remained_name))
    cv2.imwrite(filename, augmented_image['image'])

    transform = A.HorizontalFlip(p=1)
    augmented_image = transform(image=image)
    filename = os.path.join(images_new_savepath, '{}{}.jpg'.format(new_value2, remained_name))
    cv2.imwrite(filename, augmented_image['image'])

filenames = glob.glob(annotations_savepath + "/*.*")  # read all files in the path mentioned
for n, image_file in enumerate(filenames):
    check = 0
    angle_list = []
    file, ext = os.path.splitext(image_file)  # split filename and extension
    name = os.path.basename(file)
    s = len(os.path.basename(file))
    # print("name's shape:",name)
    # print("Length:",s)
    l = list(name)

    for i in range(s):
        if l[i] != '_' and check == 0:
            rename_list.append(l[i])
        elif l[i] == '_':
            check = check + 1
            remained_list.append(l[i])
        elif l[i] != '_' and check == 3:
            angle_list.append(l[i])
        else:
            remained_list.append(l[i])

    # print("rename_int = ", rename_int)
    # print("remained_list = ", remained_list)
    # print("angle_list = ", angle_list)
    remained_name = ''.join(map(str, remained_list))

    # ----------------------------------------------#
    # Prefix
    rename_int = list(map(int, rename_list))
    num = magic(rename_int)
    num_modify = num * k
    new_value1 = num_modify + offset1
    new_value2 = num_modify + offset2
    # ----------------------------------------------#
    # Suffix
    angle_int = list(map(int, angle_list))
    angle_ori = magic(angle_int)
    # ----------------------------------------------#
    rename_list = []
    remained_list = []

    file, ext = os.path.splitext(image_file)  # split filename and extension
    # image = cv2.imread(image_file.path)
    image = cv2.imread(image_file)  #

    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    transform = A.VerticalFlip(p=1)
    augmented_image = transform(image=image)
    if angle_ori == 0 or angle_ori == 180 or angle_ori == 360:
        new_angle = angle_ori
    else:
        new_angle = 360 - angle_ori
    filename = os.path.join(annotations_new_savepath, '{}{}{}.png'.format(new_value1, remained_name, new_angle))
    cv2.imwrite(filename, augmented_image['image'])



    transform = A.HorizontalFlip(p=1)
    augmented_image = transform(image=image)

    if 0 <= angle_ori <= 180:
        new_angle = 180 - angle_ori
    elif angle_ori == 360:
        new_angle = 180
    else:
        new_angle = 540 - angle_ori
    filename = os.path.join(annotations_new_savepath, '{}{}{}.png'.format(new_value2, remained_name, new_angle))
    cv2.imwrite(filename, augmented_image['image'])