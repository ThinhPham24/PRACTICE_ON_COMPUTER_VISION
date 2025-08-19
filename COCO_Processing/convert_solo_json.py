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
# - Saving the json format after convert image to COCO format dataset
folder = "/DATA_TEST_AUGMENTATION/"
annotations_dir = "annotations_solo"
current_dir = os.getcwd()
path = ''.join([current_dir, folder])

annotations_savepath = os.path.join(path, annotations_dir)
print("annotations_savepath = ", annotations_savepath)
if not os.path.isdir(os.path.abspath(annotations_savepath)):
    os.mkdir(annotations_savepath)
#---- End of saving Json format --------------------------------------#



ROOT_DIR = 'DATA_TEST_AUGMENTATION'

IMG_DIR = 'IMG'
ANNOTATION_DIR = 'ANNOTATION'

TRAIN_IMAGE_DIR = IMG_DIR + "/img_train"
TRAIN_ANNOTATION_DIR =  ANNOTATION_DIR + "/an_train"

VALIDATE_IMAGE_DIR = IMG_DIR + "/img_test"
VALIDATE_ANNOTATION_DIR = ANNOTATION_DIR + "/an_test"

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
CATEGORIES = [
     {
        'id': 1,
        'name': 'class_1',
    },
    {
        'id': 2,
        'name': 'class_2',
    },
    {
        'id': 3,
        'name': 'class_3',
    },
    {
        'id': 4,
        'name': 'class_4',
    },
    {
        'id': 5,
        'name': 'class_5',
    },
    {
        'id': 6,
        'name': 'class_6',
    },
    {
        'id': 7,
        'name': 'class_7',
    },
    {
        'id': 8,
        'name': 'class_8',
    },
    {
        'id': 9,
        'name': 'class_9',
    }, 
    {
        'id': 10,
        'name': 'class_10',
    },
    {
        'id': 11,
        'name': 'class_11',
    },
    {
        'id': 12,
        'name': 'class_12',
    },
    {
        'id': 13,
        'name': 'class_13',
    },
    {
        'id': 14,
        'name': 'class_14',
    },
    {
        'id': 15,
        'name': 'class_15',
    },
    {
        'id': 16,
        'name': 'class_16',
    },
    {
        'id': 17,
        'name': 'class_17',
    },
    {
        'id': 18,
        'name': 'class_18',
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
    ind_num = 0
    for n, image_file in enumerate(filenames):
        annotation_files =[]
        print("so image", n)
        ind_image = str(os.path.basename(image_file)).split(".")[0]
        print("index", ind_image)
        image = Image.open(image_file)
        image_info = pycococreatortools.create_image_info(
            image_id, os.path.basename(image_file), image.size)
        coco_output["images"].append(image_info)
        for m in range(len(filenames_an)):
            if ind_num < (len(filenames_an)):
                ind_anno = str(os.path.basename(filenames_an[ind_num])).split(".")[0].split('_bud')[0]
            # print("indx ann", filenames_an[ind_num])
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
                    for annotation_filename in annotation_files:
                        print("annotation_filename",annotation_filename)
                        angle_list = []
                        angle_s = []
                        check = 0
                        file, ext = os.path.splitext(annotation_filename)  # split filename and extension
                        name = os.path.basename(file)
                        # print("NAME: ",name)
                        s = len(os.path.basename(file))
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
                        print("angle:",angle)
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
                        if '_bud_' in annotation_filename:
                            class_id = angle_s + 1
                        else:
                            continue
                        print("class of image", class_id)
                        category_info = {'id': class_id, 'is_crowd': 'crowd' in image_file}
                        print("category_info", category_info)               
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
        print("no of image", n+1)
        ind_image = str(os.path.basename(image_file)).split(".")[0]
        image = Image.open(image_file)
        image_info = pycococreatortools.create_image_info(
            image_id, os.path.basename(image_file), image.size)
        coco_output["images"].append(image_info)
        for m in range(len(filenames_an)):
            if ind_num < (len(filenames_an)):
                ind_anno = str(os.path.basename(filenames_an[ind_num])).split(".")[0].split('_bud')[0]
            # print("indx ann", filenames_an[ind_num])
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
                    for annotation_filename in annotation_files:
                        print("annotation_filename",annotation_filename)
                        angle_list = []
                        angle_s = []
                        check = 0
                        file, ext = os.path.splitext(annotation_filename)  # split filename and extension
                        name = os.path.basename(file)
                        # print("NAME: ",name)
                        s = len(os.path.basename(file))
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
                        print("angle:",angle)
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

                        # print("annotion ",annotation_filename)
                        if '_bud_' in annotation_filename:
                            class_id = angle_s +1
                        else:
                            continue
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