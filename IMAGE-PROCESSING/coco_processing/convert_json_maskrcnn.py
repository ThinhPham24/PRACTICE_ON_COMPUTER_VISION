import datetime
import json
import os
import re
import fnmatch
from PIL import Image
import numpy as np
from pycococreatortools import pycococreatortools
import math
import glob

folder = "/DATA_Clean/"
annotations_dir = "annotations"
current_dir = os.getcwd()
path = ''.join([current_dir, folder])
print("PATH",path)

annotations_savepath = os.path.join(path, annotations_dir)
print("annotations_savepath = ", annotations_savepath)
if not os.path.isdir(os.path.abspath(annotations_savepath)):
    os.mkdir(annotations_savepath)


ROOT_DIR = 'CONV'

# IMG_DIR = 'TEST'
# ANNOTATION_DIR = 'TEST'

TRAIN_IMAGE_DIR = 'train\img_new_rot'
TRAIN_ANNOTATION_DIR = 'train\an_new_rot'

VALIDATE_IMAGE_DIR = "\test\img_new_rot"
VALIDATE_ANNOTATION_DIR ="\test\an_new_rot"

# TEST_IMAGE_DIR = ROOT_DIR + IMG_DIR + "/test"
# TEST_ANNOTATION_DIR = ROOT_DIR +  ANNOTATION_DIR + "/test"
IMG_DIR1 = '/IMG_1'
ANNOTATION_DIR1 = '/ANNOTATION_1'

TRAIN_IMAGE_DIR1 = ROOT_DIR + IMG_DIR1 + "/train"
TRAIN_ANNOTATION_DIR1 = ROOT_DIR + ANNOTATION_DIR1 + "/train"

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
        'name': 'single_bud',
    },
]
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
    file_name_prefix = basename_no_extension + '.*'
    files = [os.path.join(root, f) for f in files]
    files = [f for f in files if re.match(file_types, f)]
    files = [f for f in files if re.match(file_name_prefix, os.path.splitext(os.path.basename(f))[0])]
    return files
def magic(numList):
    s = ''.join(map(str, numList))
    return int(s)
images_path = os.path.join(path, TRAIN_IMAGE_DIR)
annotations_path = os.path.join(path, TRAIN_ANNOTATION_DIR)
images_path_val = os.path.join(path, VALIDATE_IMAGE_DIR)
annotations_path_val = os.path.join(path, VALIDATE_ANNOTATION_DIR)
print("this is path", images_path)
print("path of annotation", annotations_path)
def main():
    # #**********************************************************************************************#
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
    filenames = sorted(glob.glob(images_path + "/*.jpg"))
    print("file names", len(filenames))
    filenames_an = sorted(glob.glob(annotations_path + "/*.png"))
    print("file names", len(filenames_an))
    ind_num = 0
    # print("igsg", (len(filenames_an)-1))
    for n, image_file in enumerate(filenames):
        annotation_files =[]
        print("so image", n)
        # print("image name", os.path.basename(image_file))
        ind_image = str(os.path.basename(image_file)).split(".")[0]
        print("index", ind_image)
        image = Image.open(image_file)
        image_info = pycococreatortools.create_image_info(
                image_id, os.path.basename(image_file), image.size)
        coco_output["images"].append(image_info)
        for m in range(len(filenames_an)):
            if ind_num < (len(filenames_an)):
                ind_anno = str(os.path.basename(filenames_an[ind_num])).split(".")[0].split('_')[0]
            # print("indx ann", filenames_an[ind_num])
            if ind_num < (len(filenames_an)) and ind_anno == ind_image:
                print("file annotation", os.path.basename(filenames_an[ind_num]))
                # print("file names", len(filenames))
                # print("file names", len(filenames_an))
                # print("so luong",ind_num )
                annotation_files.append(filenames_an[ind_num])
                ind_num +=1
            else: 
                # print("BREAK")
                # print("list annotation",list_an)
                if None in annotation_files :
                    continue
                else:
                    for annotation_filename in annotation_files:
                        # print("afnnaf", annotation_filename)
                        angle_list = []
                        angle_s = []
                        check = 0
                        # print(annotation_filename)
                        file, ext = os.path.splitext(annotation_filename)  # split filename and extension
                        # print("file = ", file)
                        # print("ext = ", ext)
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

                        angle_int = list(map(int, angle_list))
                        angle = magic(angle_int)
                        print("angle = ", angle)
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
                        category_info = {'id': class_id, 'is_crowd': 'crowd' in image_file}
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
                # print("reaf",( ind_anno,ind_image))
                break
        image_id = image_id + 1
    # print("coco",coco_output)
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
    filenames = sorted(glob.glob(images_path_val + "/*.jpg"))
    # print("file names", len(filenames))
    filenames_an = sorted(glob.glob(annotations_path_val + "/*.png"))
    # print("file names", len(filenames_an))
    ind_num = 0
    # print("igsg", (len(filenames_an)-1))
    for n, image_file in enumerate(filenames):
        annotation_files =[]
        print("so image", n)
        # print("image name", os.path.basename(image_file))
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
                print("file annotation", os.path.basename(filenames_an[ind_num]))
                # print("file names", len(filenames))
                # print("file names", len(filenames_an))
                # print("so luong",ind_num )
                annotation_files.append(filenames_an[ind_num])
                ind_num +=1
            else: 
                # print("BREAK")
                # print("list annotation",list_an)
                if None in annotation_files :
                    continue
                else:
                    for annotation_filename in annotation_files:
                        # print("afnnaf", annotation_filename)
                        angle_list = []
                        angle_s = []
                        check = 0
                        # print(annotation_filename)
                        file, ext = os.path.splitext(annotation_filename)  # split filename and extension
                        # print("file = ", file)
                        # print("ext = ", ext)
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

                        angle_int = list(map(int, angle_list))
                        angle = magic(angle_int)
                        print("angle = ", angle)
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
                        category_info = {'id': class_id, 'is_crowd': 'crowd' in image_file}
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
                # print("reaf",( ind_anno,ind_image))
                break
        image_id = image_id + 1
    # print("coco",coco_output)
    with open('{}/validate.json'.format(annotations_savepath), 'w') as output_json_file:
        json.dump(coco_output, output_json_file)
    print("VALIDATW DATA FINISH!")
if __name__ == "__main__":
    main()