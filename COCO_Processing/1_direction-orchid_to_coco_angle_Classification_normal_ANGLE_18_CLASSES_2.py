#!/usr/bin/env python3
# Version 1: Put TRAIN data into	: DATA/TRAIN/images 	& DATA/TRAIN/annotations
#	     Put VALIDATE data into	: DATA/VALIDATE/images & DATA/VALIDATE/annotations
#	     Put TEST data into	: DATA/TEST/images 	& DATA/TEST/annotations

# Output file's name: train.json - validate.json - test.json

import datetime
import json
import os
import re
import fnmatch
from PIL import Image
import numpy as np
from pycococreatortools import pycococreatortools
import math


folder = "/DATA_TEST_AUGMENTATION/"
annotations_dir = "annotations_or"
current_dir = os.getcwd()
path = ''.join([current_dir, folder])

annotations_savepath = os.path.join(path, annotations_dir)
# print("annotations_savepath = ", annotations_savepath)
if not os.path.isdir(os.path.abspath(annotations_savepath)):
    os.mkdir(annotations_savepath)


ROOT_DIR = 'DATA_TEST_AUGMENTATION'

IMG_DIR = '/IMG'
ANNOTATION_DIR = '/ANNOTATION'

TRAIN_IMAGE_DIR = ROOT_DIR + IMG_DIR + "/img_train"
TRAIN_ANNOTATION_DIR = ROOT_DIR +  ANNOTATION_DIR + "/an_train"

VALIDATE_IMAGE_DIR = ROOT_DIR + IMG_DIR + "/img_test"
VALIDATE_ANNOTATION_DIR = ROOT_DIR +  ANNOTATION_DIR + "/an_test"

TEST_IMAGE_DIR = ROOT_DIR + IMG_DIR + "/img_test"
TEST_ANNOTATION_DIR = ROOT_DIR +  ANNOTATION_DIR + "/an_test"


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

def filter_for_annotations(root, files, image_filename):
    print("image_filename:",image_filename)
    file_types = ['*.png']
    file_types = r'|'.join([fnmatch.translate(x) for x in file_types])
    basename_no_extension = os.path.splitext(os.path.basename(image_filename))[0]
    file_name_prefix = basename_no_extension + '.*'
    files = [os.path.join(root, f) for f in files]
    files = [f for f in files if re.match(file_types, f)]
    files = [f for f in files if re.match(file_name_prefix, os.path.splitext(os.path.basename(f))[0])]

    return files
def magic(numList):
    s = ''.join(map(str, numList))
    return int(s)


def main():
    #**********************************************************************************************#
    # # TRAINING DATA
    coco_output = {
        "info": INFO,
        "licenses": LICENSES,
        "categories": CATEGORIES,
        "images": [],
        "annotations": []
    }
    
    image_id = 1
    segmentation_id = 1
    
    # filter for jpeg images
    for root, _, files in os.walk(TRAIN_IMAGE_DIR):
        image_files = filter_for_jpeg(root, files)
        print("image_files:", image_files)
    
        # go through each image
        for image_filename in image_files:
            # print("file image",image_filename)
            image = Image.open(image_filename)
            image_info = pycococreatortools.create_image_info(
                image_id, os.path.basename(image_filename), image.size)
            coco_output["images"].append(image_info)
    
            # filter for associated png annotations
            for root, _, files in os.walk(TRAIN_ANNOTATION_DIR):
                annotation_files = filter_for_annotations(root, files, image_filename)
                print("annotation_files:",annotation_files)
    
                # go through each associated annotation
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
                    #print("angle_list = ", angle_list)
    
                    angle_int = list(map(int, angle_list))
                    angle = magic(angle_int)
                    # print("angle = ", angle)
                    # --- END OF "Xử lý dựa trên tên của annotation file (binary image)"----#
                    #------------ PHẦN TÍNH TOÁN TỪ ANGLE TÍNH RA SIN & COS ----------------#
                    # Clasify the angles (8 CLASSES - 45 degrees each part)
                    '''
                    if (angle > 337.5 and angle <=360) or (angle >= 0 and angle <= 22.5):
                        angle_cl = 1
    
                    elif (angle > 22.5 and angle <= 67.5):
                        angle_cl = 2
    
                    elif (angle > 67.5 and angle <= 112.5):
                        angle_cl = 3
    
                    elif (angle > 112.5 and angle <= 157.5):
                        angle_cl = 4
    
                    elif (angle > 157.5 and angle <= 202.5):
                        angle_cl = 5
    
                    elif (angle > 202.5 and angle <= 247.5):
                        angle_cl = 6
    
                    elif (angle > 247.5 and angle <= 292.5):
                        angle_cl = 7
    
                    elif (angle > 292.5 and angle <= 337.5):
                        angle_cl = 8
    
                    else:
                        continue
                    angle_s = angle_cl
    
                    '''
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
    #**********************************************************************************************#
    # VALIDATE DATA
    coco_output = {
        "info": INFO,
        "licenses": LICENSES,
        "categories": CATEGORIES,
        "images": [],
        "annotations": []
    }

    image_id = 1
    segmentation_id = 1
    
    # filter for jpeg images
    for root, _, files in os.walk(VALIDATE_IMAGE_DIR):
        image_files = filter_for_jpeg(root, files)
        print("image_files:", image_files)

        # go through each image
        for image_filename in image_files:
            image = Image.open(image_filename)
            image_info = pycococreatortools.create_image_info(
                image_id, os.path.basename(image_filename), image.size)
            coco_output["images"].append(image_info)

            # filter for associated png annotations
            for root, _, files in os.walk(VALIDATE_ANNOTATION_DIR):
                annotation_files = filter_for_annotations(root, files, image_filename)
                print("annotation_files:",annotation_files)

                # go through each associated annotation
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
                    #print("angle_list = ", angle_list)

                    angle_int = list(map(int, angle_list))
                    angle = magic(angle_int)
                    # print("angle = ", angle)
                    # --- END OF "Xử lý dựa trên tên của annotation file (binary image)"----#
                    #------------ PHẦN TÍNH TOÁN TỪ ANGLE TÍNH RA SIN & COS ----------------#
                    # Clasify the angles (8 CLASSES - 45 degrees each part)
                    '''
                    if (angle > 337.5 and angle <=360) or (angle >= 0 and angle <= 22.5):
                        angle_cl = 1

                    elif (angle > 22.5 and angle <= 67.5):
                        angle_cl = 2

                    elif (angle > 67.5 and angle <= 112.5):
                        angle_cl = 3

                    elif (angle > 112.5 and angle <= 157.5):
                        angle_cl = 4

                    elif (angle > 157.5 and angle <= 202.5):
                        angle_cl = 5

                    elif (angle > 202.5 and angle <= 247.5):
                        angle_cl = 6

                    elif (angle > 247.5 and angle <= 292.5):
                        angle_cl = 7

                    elif (angle > 292.5 and angle <= 337.5):
                        angle_cl = 8

                    else:
                        continue
                    angle_s = angle_cl

                    '''
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
                    #print("category_info", category_info)               
                    binary_mask = np.asarray(Image.open(annotation_filename)
                        .convert('1')).astype(np.uint8)
                    #print(binary_mask) # For testing only. 0 is OK -> found the problem at this point then can solve it
         
                    annotation_info = pycococreatortools.create_annotation_info_direction(
                        segmentation_id, image_id, category_info, binary_mask,
                        image.size, angle_s, tolerance=2) # Voi anh size lon thi phai sua cho nay lai

                    if annotation_info is not None:
                        coco_output["annotations"].append(annotation_info)

                    segmentation_id = segmentation_id + 1

            image_id = image_id + 1
    with open('{}/validate.json'.format(annotations_savepath), 'w') as output_json_file:
        json.dump(coco_output, output_json_file)
    print("VALIDATE DATA FINISH!")
    #**********************************************************************************************#
    # TESTING DATA
    coco_output = {
        "info": INFO,
        "licenses": LICENSES,
        "categories": CATEGORIES,
        "images": [],
        "annotations": []
    }

    image_id = 1
    segmentation_id = 1
    
    # filter for jpeg images
    for root, _, files in os.walk(TEST_IMAGE_DIR):
        image_files = filter_for_jpeg(root, files)
        #print("image_files:", image_files)

        # go through each image
        for image_filename in image_files:
            image = Image.open(image_filename)
            image_info = pycococreatortools.create_image_info(
                image_id, os.path.basename(image_filename), image.size)
            coco_output["images"].append(image_info)

            # filter for associated png annotations
            for root, _, files in os.walk(TEST_ANNOTATION_DIR):
                annotation_files = filter_for_annotations(root, files, image_filename)
                # print("annotation_files:",annotation_files)

                # go through each associated annotation
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
                    #print("angle_list = ", angle_list)

                    angle_int = list(map(int, angle_list))
                    angle = magic(angle_int)
                    # print("angle = ", angle)
                    # --- END OF "Xử lý dựa trên tên của annotation file (binary image)"----#
                    #------------ PHẦN TÍNH TOÁN TỪ ANGLE TÍNH RA SIN & COS ----------------#
                    # Clasify the angles (8 CLASSES - 45 degrees each part)
                    '''
                    if (angle > 337.5 and angle <=360) or (angle >= 0 and angle <= 22.5):
                        angle_cl = 1

                    elif (angle > 22.5 and angle <= 67.5):
                        angle_cl = 2

                    elif (angle > 67.5 and angle <= 112.5):
                        angle_cl = 3

                    elif (angle > 112.5 and angle <= 157.5):
                        angle_cl = 4

                    elif (angle > 157.5 and angle <= 202.5):
                        angle_cl = 5

                    elif (angle > 202.5 and angle <= 247.5):
                        angle_cl = 6

                    elif (angle > 247.5 and angle <= 292.5):
                        angle_cl = 7

                    elif (angle > 292.5 and angle <= 337.5):
                        angle_cl = 8

                    else:
                        continue
                    angle_s = angle_cl

                    '''
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
                    #print("category_info", category_info)               
                    binary_mask = np.asarray(Image.open(annotation_filename)
                        .convert('1')).astype(np.uint8)
                    #print(binary_mask) # For testing only. 0 is OK -> found the problem at this point then can solve it
         
                    annotation_info = pycococreatortools.create_annotation_info_direction(
                        segmentation_id, image_id, category_info, binary_mask,
                        image.size, angle_s, tolerance=2) # Voi anh size lon thi phai sua cho nay lai

                    if annotation_info is not None:
                        coco_output["annotations"].append(annotation_info)

                    segmentation_id = segmentation_id + 1

            image_id = image_id + 1
    with open('{}/test.json'.format(annotations_savepath), 'w') as output_json_file:
        json.dump(coco_output, output_json_file)
    print("TEST DATA FINISH!")

if __name__ == "__main__":
    main()
