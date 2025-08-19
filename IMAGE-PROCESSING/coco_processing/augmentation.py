import numpy as np
import os, os.path
import glob
import random
import matplotlib.pyplot as plt
import albumentations as A
import numpy as np
import argparse
import imutils
import cv2
import time
import aug
from utility import  alb_library,aug_library
import random
import shutil
import re
from tqdm import tqdm 

folder = "/DATA/"
current_dir = os.getcwd()
path_1 = ''.join([current_dir, folder])
print("path = ", path_1)

# constants
def magic(numList):
    s = ''.join(map(str, numList))
    return int(s)
def generate_path(path):
    if not os.path.exists(path):
        os.makedirs(path)
#type 0:train, 1:test
type = 0
if type ==0:    #train folder
    #-----------------------------------------------------------------------#
    images_dir = "IMG/rot_train"
    annotations_dir = "ANNOTATION/rot_train"
    images_path = os.path.join(path_1, images_dir)
    annotations_path = os.path.join(path_1, annotations_dir)
    #-----------------------------------------------------------------------#
    #-----------------------------------------------------------------------#
    new_image_save = "IMG/aug_train"
    new_annotations_save = "ANNOTATION/aug_train"
    images_new_savepath = os.path.join(path_1, new_image_save)
    annotations_new_savepath = os.path.join(path_1, new_annotations_save)
    if not os.path.isdir(os.path.abspath(images_new_savepath)):
        os.mkdir(images_new_savepath)
    # if not os.path.isdir(os.path.abspath(annotations_new_savepath)):
    #     os.mkdir(annotations_new_savepath)
    #-----------------------------------------------------------------------#
    #-----------------------------------------------------------------------#

if type==1:     #test folder
    #-----------------------------------------------------------------------#
    images_dir = "IMG/rot_val"
    annotations_dir = "ANNOTATION/rot_val"
    images_path = os.path.join(path_1, images_dir)
    annotations_path = os.path.join(path_1, annotations_dir)
    #-----------------------------------------------------------------------#
    #-----------------------------------------------------------------------#
    new_image_save = "IMG/aug_val_rot"
    new_annotations_save = "ANNOTATION/aug_val_rot"
    images_new_savepath = os.path.join(path_1, new_image_save)
    annotations_new_savepath = os.path.join(path_1, new_annotations_save)
    if not os.path.isdir(os.path.abspath(images_new_savepath)):
        os.mkdir(images_new_savepath)
    # if not os.path.isdir(os.path.abspath(annotations_new_savepath)):
    #     os.mkdir(annotations_new_savepath)
    #-----------------------------------------------------------------------#
    #-----------------------------------------------------------------------#

if __name__ == "__main__": 
    filenames = glob.glob(images_path + "/*.*") #read all files in the path mentioned
    # print("ALL image", filenames)
    pipeline = aug_library()
    pipeline2 = alb_library()
    # -----GRADIENT---------
    ration = 0.2# %
    total_files = len(filenames)
    remain = int(ration*total_files)
    for filename in tqdm(random.sample(filenames,remain),total=remain):
        path_img = os.path.join(images_path, filename)
        name = os.path.basename(path_img)
        new_img = cv2.imread(path_img)
        sample = aug.Sample(new_img)
        image_1 = pipeline.apply(sample)
        filename_image = os.path.join(images_new_savepath, '{}'.format(name))
        cv2.imwrite(filename_image,image_1.image)
        # cv2.imshow("image",np.concatenate((new_img ,image_1.image),axis= 1))
        # k = cv2.waitKey(0)
        # if  k== ord('q'):
        #     break
        # cv2.destroyAllWindows()
        os.remove(path_img) # delete file
    #------------------COONTRAST, BRIGHTNESS, SATURATION----------
    filenames = glob.glob(images_path + "/*.*") # RE-LOAD
    ration = 0.375# %
    total_files = len(filenames)
    remain = int(ration*total_files)
    for filename in tqdm(random.sample(filenames,remain),total=remain):
        path_img = os.path.join(images_path, filename)
        name = os.path.basename(path_img)
        new_img = cv2.imread(path_img)
        image_2, index = pipeline2.select_aug(new_image=new_img)
        filename_image = os.path.join(images_new_savepath, '{}'.format(name))
        cv2.imwrite(filename_image,image_2['image'])
        os.remove(path_img) # delete file
    # -------------REMOVE INTO FOLDER---------
    filenames = glob.glob(images_path + "/*.*") # RE-LOAD
    for filename in tqdm(random.sample(filenames,len(filenames)),total=len(filenames)):
        path_img = os.path.join(images_path, filename)
        shutil.move(path_img, images_new_savepath)