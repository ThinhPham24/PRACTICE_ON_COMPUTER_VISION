import datetime
import json
import os
import re
import fnmatch
from PIL import Image
import numpy as np
from pycococreatortools import pycococreatortools
import cv2
import glob
import os.path as osp
# - Saving the json format after convert image to COCO format dataset
folder = "/BUD_CORE/"
annotations_dir = "annotations_solo"
current_dir = os.getcwd()
path = ''.join([current_dir, folder])

annotations_savepath = os.path.join(path, annotations_dir)
print("annotations_savepath = ", annotations_savepath)
if not os.path.isdir(os.path.abspath(annotations_savepath)):
    os.mkdir(annotations_savepath)
#---- End of saving Json format --------------------------------------#



ROOT_DIR = 'BUD_CORE'

IMG_DIR = 'IMG'
ANNOTATION_DIR = 'ANNOTATION'

TRAIN_IMAGE_DIR = "/home/airlab/Desktop/IMAGE_CROP/BUD_CORE/train/img_test"
TRAIN_ANNOTATION_DIR = "/home/airlab/Desktop/IMAGE_CROP/BUD_CORE/train/an_test"

VALIDATE_IMAGE_DIR = "/home/airlab/Desktop/IMAGE_CROP/BUD_CORE/val/img_test"
VALIDATE_ANNOTATION_DIR = "/home/airlab/Desktop/IMAGE_CROP/BUD_CORE/val/an_test"
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
def magic(numList):
    s = ''.join(map(str, numList))
    return int(s)
# CATEGORIES = [
#     {
#         'id': 1,
#         'name': 'core',
#     },
#     {
#         'id': 2,
#         'name': 'bud',
#     },
#      {
#         'id': 3,
#         'name': 'leaves',
#     },
#      {
#         'id': 4,
#         'name': 'fullbud',
#     }
# ]
CATEGORIES = [
    {
        'id': 1,
        'name': 'core',
    },
    {
        'id': 2,
        'name': 'bud',
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
    file_name_prefix = basename_no_extension + '_' + '.*'
    #print("file_name_prefix = ", file_name_prefix)
    files = [os.path.join(root, f) for f in files]
    files = [f for f in files if re.match(file_types, f)]
    files = [f for f in files if re.match(file_name_prefix, os.path.splitext(os.path.basename(f))[0])]
    return files

images_path_train = os.path.join(path, TRAIN_IMAGE_DIR)
annotations_path_train = os.path.join(path, TRAIN_ANNOTATION_DIR)
# print("annotation path", annotations_path_train)
images_path_val = os.path.join(path, VALIDATE_IMAGE_DIR)
annotations_path_val = os.path.join(path, VALIDATE_ANNOTATION_DIR)
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
    filenames = sorted(glob.glob(images_path_train + "/*.jpg"))
    # print("file names", len(filenames))
    filenames_an = sorted(glob.glob(annotations_path_train + "/*.png"))
    # print("all annotation", filenames_an)
    ind_num = 0
    for n, image_file in enumerate(filenames):
        annotation_files =[]
        # print("so image", n +1 )
        ind_image = str(os.path.basename(image_file)).split(".")[0]
        # print("index", (ind_image,n +1))
        image = Image.open(image_file)
        image_info = pycococreatortools.create_image_info(
            image_id, os.path.basename(image_file), image.size)
        coco_output["images"].append(image_info)
        for m in range(len(filenames_an)):
            if ind_num < (len(filenames_an)):
                ind_anno = str(os.path.basename(filenames_an[ind_num])).split(".")[0].split('_')[0]
            # print("indx ann", ind_anno)
            if ind_num < (len(filenames_an)) and ind_anno == ind_image:
                print("file annotation", os.path.basename(filenames_an[ind_num]))
                # print("file names", len(filenames))
                # print("file names", len(filenames_an))
                # print("so luong",ind_num )
                annotation_files.append(filenames_an[ind_num])
                ind_num +=1
            else: 
                if None in annotation_files :
                    continue
                else:
                    # print(annotation_files)
                    for annotation_filename in annotation_files:
                        base = osp.splitext(osp.basename(annotation_filename))[0]
                        # print("annotation_filename",annotation_filename)
                        if '_core_' in base:
                            class_id = 1
                        # elif '_bud_' in annotation_filename:
                        #     class_id = 2
                        elif '_bud_' in base:
                            class_id = 2
                        # elif '_fullbud_' in annotation_filename:
                        #     class_id = 4
                        else:
                            continue
                        print("class of image", class_id)
                        category_info = {'id': class_id, 'is_crowd': 'crowd' in image_file}
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
                break
        image_id = image_id + 1
    with open('{}/train.json'.format(annotations_savepath), 'w') as output_json_file:
        json.dump(coco_output, output_json_file)
    print("TRAIN DATA FINISH!")
#*****************************************************
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
    filenames = sorted(glob.glob(images_path_val + "/*.jpg"))
    # print("file names", len(filenames))
    filenames_an = sorted(glob.glob(annotations_path_val + "/*.png"))
    ind_num = 0
    for n, image_file in enumerate(filenames):
        annotation_files =[]
        # print("no of image", n+1)
        ind_image = str(os.path.basename(image_file)).split(".")[0]
        print("ind image", ind_image)
        image = Image.open(image_file)
        image_info = pycococreatortools.create_image_info(
            image_id, os.path.basename(image_file), image.size)
        coco_output["images"].append(image_info)
        for m in range(len(filenames_an)):
            if ind_num < (len(filenames_an)):
                ind_anno = str(os.path.basename(filenames_an[ind_num])).split(".")[0].split('_')[0]
            print("indx ann", ind_anno)
            if ind_num < (len(filenames_an)) and ind_anno == ind_image:
                # print("file annotation", os.path.basename(filenames_an[ind_num]))
                # print("file names", len(filenames))
                # print("file names", len(filenames_an))
                # print("so luong",ind_num )
                annotation_files.append(filenames_an[ind_num])
                ind_num +=1
            else: 
                if None in annotation_files :
                    continue
                else:
                    print(annotation_files)
                    for annotation_filename in annotation_files:
                        base = osp.splitext(osp.basename(annotation_filename))[0]
                        print("annotion ",base)
                        if '_core_' in base:
                            class_id = 1
                        # elif '_bud_' in annotation_filename:
                        #     class_id = 2
                        elif '_bud_' in base:
                            class_id = 2
                        # elif '_fullbud_' in annotation_filename:
                        #     class_id = 4
                        else:
                            continue
                        print("class of image", class_id)
                        category_info = {'id': class_id, 'is_crowd': 'crowd' in image_file}
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
                    # print("id of image", image_id)
                break
        image_id = image_id + 1
    with open('{}/val.json'.format(annotations_savepath), 'w') as output_json_file:
        json.dump(coco_output, output_json_file)
    print("VALIDATE DATA FINISH!")
if __name__ == "__main__":
    main()