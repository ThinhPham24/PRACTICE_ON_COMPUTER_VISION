#!/usr/bin/env python3
# Version 1: Put TRAIN data into	: DATA/TRAIN/images 	& DATA/TRAIN/annotations
#	     Put VALIDATE data into	: DATA/VALIDATE/images & DATA/VALIDATE/annotations
#	     Put TEST data into	: DATA/TEST/images 	& DATA/TEST/annotations

# Output file's name: train.json - validate.json - test.json

import datetime
from glob import glob
import json
import os
import re
import fnmatch
from PIL import Image
import numpy as np
from pycococreatortools import pycococreatortools
import math
import scandir

folder = "/DATA/"
annotations_dir = "annotations"
current_dir = os.getcwd()
path = ''.join([current_dir, folder])

annotations_savepath = os.path.join(path, annotations_dir)
# print("annotations_savepath = ", annotations_savepath)
if not os.path.isdir(os.path.abspath(annotations_savepath)):
    os.mkdir(annotations_savepath)


ROOT_DIR = 'DATA'

IMG_DIR = '/IMG'
ANNOTATION_DIR = '/ANNOTATION'

TRAIN_IMAGE_DIR = ROOT_DIR + IMG_DIR + "/train"
TRAIN_ANNOTATION_DIR = ROOT_DIR +  ANNOTATION_DIR + "/train"

VALIDATE_IMAGE_DIR = ROOT_DIR + IMG_DIR + "/validate"
VALIDATE_ANNOTATION_DIR = ROOT_DIR +  ANNOTATION_DIR + "/validate"

TEST_IMAGE_DIR = ROOT_DIR + IMG_DIR + "/test"
TEST_ANNOTATION_DIR = ROOT_DIR +  ANNOTATION_DIR + "/test"


INFO = {
    "description": "Training Dataset",
    "url": "https://github.com/waspinator/pycococreator",
    "version": "0.1.0",
    "year": 2018,
    "contributor": "waspinator",
    "date_created": datetime.datetime.utcnow().isoformat(' ')
}

LICENSES = [
    {
        "id": 1,
        "name": "Attribution-NonCommercial-ShareAlike License",
        "url": "http://creativecommons.org/licenses/by-nc-sa/2.0/"
    }
]

CATEGORIES = [
    {
        'id': 1,
        'name': 'single_bud', #3'single_bud'
    },

]

def filter_for_jpeg(root, files):
    file_types = ['*.jpeg', '*.jpg']
    file_types = r'|'.join([fnmatch.translate(x) for x in file_types])
    files = [os.path.join(root, f) for f in files]
    files = [f for f in files if re.match(file_types, f)]
    
    return files
#Use test and validate
# def filter_for_annotations(root, files, image_filename):
#     print("image_filename:",image_filename)
#     file_types = ['*.png']
#     file_types = r'|'.join([fnmatch.translate(x) for x in file_types])
#     basename_no_extension = os.path.splitext(os.path.basename(image_filename))[0]
#     file_name_prefix = basename_no_extension + '.*'
#     files = [os.path.join(root, f) for f in files]
#     files = [f for f in files if re.match(file_types, f)]
#     files = [f for f in files if re.match(file_name_prefix, os.path.splitext(os.path.basename(f))[0])]

#     return files
###Use training
def filter_for_annotations(root, files, image_filename):
    # print("image_filename:",image_filename)
    file_types = ['*.png']
    file_types = r'|'.join([fnmatch.translate(x) for x in file_types])
    basename_no_extension = os.path.splitext(os.path.basename(image_filename))[0]
    file_name_prefix = basename_no_extension + '_' + '.*'
    # print("file_name_prefix = ", file_name_prefix)
    # files = [os.path.join(root, f) for f in files]
    files = [f for f in files if re.match(file_types, f)]
    files = [f for f in files if re.match(file_name_prefix, os.path.splitext(os.path.basename(f))[0])]

    return files
# def filter_for_annotations_1(files,image_filename):

# constants
def magic(numList):
    s = ''.join(map(str, numList))
    return int(s)
def main():
    #**********************************************************************************************#
    # TRAINING DATA
    coco_output = {
        "info": INFO,
        "licenses": LICENSES,
        "categories": CATEGORIES,
        "images": [],
        "annotations": []
    }

    image_id = 1
    segmentation_id = 1
#######################################33
    # filter for jpeg images
    # for root, _, files in os.walk(TRAIN_IMAGE_DIR):
    # image_sorted = np.sort(glob(TRAIN_IMAGE_DIR + '/*.jpg'))
    # annotation_sorted = np.sort(glob(TRAIN_ANNOTATION_DIR + '/*.png'))
    # print("sorted image", image_sorted.shape)
    # remove_nam = image_sorted
    # # print("lenght",len(remove_nam))
    # for i in remove_nam:
    #     print("image", i)
    #     indx = np.where(remove_nam == i )[0][0]
    #     print("where",indx)
    #     remove_nam = np.delete(remove_nam,indx)
    #     print("remove image",remove_nam.shape)
#######################
    #***********************TRAIN******************************#
    root1 = 'DATA/ANNOTATION/train'
    annotation_sorted = np.sort(glob(TRAIN_ANNOTATION_DIR + '/*.png'))
    remove_anno = annotation_sorted
    print("original shape",remove_anno.shape)
    for root, _, files in scandir.walk(TRAIN_IMAGE_DIR):
        image_files = filter_for_jpeg(root, files)
        # print("image_files:", image_files)
        
        # go through each image
        for image_filename in image_files:
            print("file image",image_filename)
            print("shape of file was deleted",remove_anno.shape)
            image = Image.open(image_filename)
            image_info = pycococreatortools.create_image_info(
                image_id, os.path.basename(image_filename), image.size)
            coco_output["images"].append(image_info)
            annotation_files = filter_for_annotations(root1, remove_anno, image_filename)
            print("annotation_files:",annotation_files)
            index =[]
            for j in annotation_files:
                indx = np.where(remove_anno == j)[0][0]
                index.append(indx)
                index_totall = np.asarray(index)
            # print("index", index_totall)
            remove_anno = np.delete(remove_anno,index_totall)
            for annotation_filename in annotation_files:
                    angle_list = []
                    angle_s = []
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
                    # print("angle_list = ", angle_list)
    
                    angle_int = list(map(int, angle_list))
                    angle = magic(angle_int)
                    # print("angle = ", angle)
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
                    angle_s = angle_cl
    
                    # -------------------- END OF Clasify the angles -----------------------#
                    if 'bud' in annotation_filename:
                        class_id = 1
                    else:
                        continue
                    category_info = {'id': class_id, 'is_crowd': 'crowd' in image_filename}
                    # print("category_info", category_info)
                    binary_mask = np.asarray(Image.open(annotation_filename).convert('1')).astype(np.uint8)
                    #print(binary_mask) # For testing only. 0 is OK -> found the problem at this point then can solve it
    
                    annotation_info = pycococreatortools.create_annotation_info_direction(
                        segmentation_id, image_id, category_info, binary_mask,
                        image.size, angle_s, tolerance=2) # Voi anh size lon thi phai sua cho nay lai
    
                    if annotation_info is not None:
                        coco_output["annotations"].append(annotation_info)
    
                    segmentation_id = segmentation_id + 1
    
            image_id = image_id + 1
    
    with open('{}/train.json'.format(annotations_savepath), 'w') as output_json_file:
        json.dump(coco_output, output_json_file)
    print("TRAIN DATA FINISH!")
    #*****************************************************************#

if __name__ == "__main__":
    main()