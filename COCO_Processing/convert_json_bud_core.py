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

# - Saving the json format after convert image to COCO format dataset
folder = "/DATA_fullbud/"
annotations_dir = "annotations"
current_dir = os.getcwd()
path = ''.join([current_dir, folder])

annotations_savepath = os.path.join(path, annotations_dir)
print("annotations_savepath = ", annotations_savepath)
if not os.path.isdir(os.path.abspath(annotations_savepath)):
    os.mkdir(annotations_savepath)
#---- End of saving Json format --------------------------------------#



ROOT_DIR = 'DATA_fullbud'

# IMG_DIR = '/IMG'
# ANNOTATION_DIR = '/ANNOTATION'

TRAIN_IMAGE_DIR ="/home/airlab/Desktop/IMAGE_CROP/DATA_fullbud/train/img_test"
TRAIN_ANNOTATION_DIR = "/home/airlab/Desktop/IMAGE_CROP/DATA_fullbud/train/an_test"

VALIDATE_IMAGE_DIR = "/home/airlab/Desktop/IMAGE_CROP/DATA_fullbud/val/img_test"
VALIDATE_ANNOTATION_DIR = "/home/airlab/Desktop/IMAGE_CROP/DATA_fullbud/val/an_test"

# TEST_IMAGE_DIR = ROOT_DIR + IMG_DIR + "/test"
# TEST_ANNOTATION_DIR = ROOT_DIR +  ANNOTATION_DIR + "/test"



INFO = {
    "description": "Training Dataset",
    "url": "https://github.com/waspinator/pycococreator",
    "version": "0.1.0",
    "year": 2017,
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
        'name': 'core',
    },
    {
        'id': 2,
        'name': 'bud',
    },
     {
        'id': 3,
        'name': 'leaves',
    },
     {
        'id': 4,
        'name': 'fullbud',
    }

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
    file_name_prefix = basename_no_extension + '_' + '.*'
    #print("file_name_prefix = ", file_name_prefix)
    files = [os.path.join(root, f) for f in files]
    files = [f for f in files if re.match(file_types, f)]
    files = [f for f in files if re.match(file_name_prefix, os.path.splitext(os.path.basename(f))[0])]

    return files

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
    
    # filter for jpeg images
    for root, _, files in os.walk(TRAIN_IMAGE_DIR):
        image_files = filter_for_jpeg(root, files)
        #print("image_files:", image_files)

        # go through each image
        for image_filename in image_files:
            image = Image.open(image_filename)
            image_info = pycococreatortools.create_image_info(
                image_id, os.path.basename(image_filename), image.size)
            coco_output["images"].append(image_info)

            # filter for associated png annotations
            for root, _, files in os.walk(TRAIN_ANNOTATION_DIR):
                annotation_files = filter_for_annotations(root, files, image_filename)
                #print("annotation_files:",annotation_files)

                # go through each associated annotation
                for annotation_filename in annotation_files:
                    
                    # print("annotion ",annotation_filename)
                    if '_core_' in annotation_filename:
                        class_id = 1
                    elif '_bud_' in annotation_filename:
                        class_id = 2
                    elif '_leaves_' in annotation_filename:
                        class_id = 3
                    elif '_fullbud_' in annotation_filename:
                        class_id = 4
                    else:
                        continue
                    category_info = {'id': class_id, 'is_crowd': 'crowd' in image_filename}
                    # print("category_info", category_info)              
                    binary_mask = np.asarray(Image.open(annotation_filename)
                        .convert('1')).astype(np.uint8)
                    #print(binary_mask) # For testing only. 0 is OK -> found the problem at this point then can solve it
                    
                    annotation_info = pycococreatortools.create_annotation_info(
                        segmentation_id, image_id, category_info, binary_mask,
                        image.size, tolerance=2) # Voi anh size lon thi phai sua cho nay lai

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
        #print("image_files:", image_files)

        # go through each image
        for image_filename in image_files:
            image = Image.open(image_filename)
            image_info = pycococreatortools.create_image_info(
                image_id, os.path.basename(image_filename), image.size)
            coco_output["images"].append(image_info)

            # filter for associated png annotations
            for root, _, files in os.walk(VALIDATE_ANNOTATION_DIR):
                annotation_files = filter_for_annotations(root, files, image_filename)
                #print("annotation_files:",annotation_files)

                # go through each associated annotation
                for annotation_filename in annotation_files:
                    
                    # print("annotion ",annotation_filename)
                    if '_core_' in annotation_filename:
                        class_id = 1
                    elif '_bud_' in annotation_filename:
                        class_id = 2
                    elif '_leaves_' in annotation_filename:
                        class_id = 3
                    elif '_fullbud_' in annotation_filename:
                        class_id = 4
                    else:
                        continue
                    category_info = {'id': class_id, 'is_crowd': 'crowd' in image_filename}
                    # print("category_info", category_info)               
                    binary_mask = np.asarray(Image.open(annotation_filename)
                        .convert('1')).astype(np.uint8)
                    #print(binary_mask) # For testing only. 0 is OK -> found the problem at this point then can solve it
                    
                    annotation_info = pycococreatortools.create_annotation_info(
                        segmentation_id, image_id, category_info, binary_mask,
                        image.size, tolerance=2) # Voi anh size lon thi phai sua cho nay lai

                    if annotation_info is not None:
                        coco_output["annotations"].append(annotation_info)
                    segmentation_id = segmentation_id + 1

            image_id = image_id + 1
    with open('{}/validate.json'.format(annotations_savepath), 'w') as output_json_file:
        json.dump(coco_output, output_json_file)
    print("VALIDATE DATA FINISH!")
    #**********************************************************************************************#
    # # TESTING DATA
    # coco_output = {
    #     "info": INFO,
    #     "licenses": LICENSES,
    #     "categories": CATEGORIES,
    #     "images": [],
    #     "annotations": []
    # }

    # image_id = 1
    # segmentation_id = 1
    
    # # filter for jpeg images
    # for root, _, files in os.walk(TEST_IMAGE_DIR):
    #     image_files = filter_for_jpeg(root, files)
    #     #print("image_files:", image_files)

    #     # go through each image
    #     for image_filename in image_files:
    #         image = Image.open(image_filename)
    #         image_info = pycococreatortools.create_image_info(
    #             image_id, os.path.basename(image_filename), image.size)
    #         coco_output["images"].append(image_info)

    #         # filter for associated png annotations
    #         for root, _, files in os.walk(TEST_ANNOTATION_DIR):
    #             annotation_files = filter_for_annotations(root, files, image_filename)
    #             #print("annotation_files:",annotation_files)

    #             # go through each associated annotation
    #             for annotation_filename in annotation_files:
                    
    #                 print("annotion ",annotation_filename)
    #                 if '_bud_' in annotation_filename:
    #                     class_id = 1
    #                 elif 'root' in annotation_filename:
    #                     class_id = 2
    #                 elif 'darkenpart' in annotation_filename:
    #                     class_id = 3
    #                 else:
    #                     continue
    #                 category_info = {'id': class_id, 'is_crowd': 'crowd' in image_filename}
    #                 print("category_info", category_info)               
    #                 binary_mask = np.asarray(Image.open(annotation_filename)
    #                     .convert('1')).astype(np.uint8)
    #                 #print(binary_mask) # For testing only. 0 is OK -> found the problem at this point then can solve it
                    
    #                 annotation_info = pycococreatortools.create_annotation_info(
    #                     segmentation_id, image_id, category_info, binary_mask,
    #                     image.size, tolerance=2) # Voi anh size lon thi phai sua cho nay lai

    #                 if annotation_info is not None:
    #                     coco_output["annotations"].append(annotation_info)

    #                 segmentation_id = segmentation_id + 1

    #         image_id = image_id + 1
    # with open('{}/test.json'.format(annotations_savepath), 'w') as output_json_file:
    #     json.dump(coco_output, output_json_file)
    # print("TEST DATA FINISH!")

if __name__ == "__main__":
    main()
