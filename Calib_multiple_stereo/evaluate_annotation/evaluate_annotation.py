from enum import EnumMeta
import sys
import os
import json
import datetime
from tkinter import image_names
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
from pycococreatortools import pycococreatortools
import math
import time
import argparse
import glob
import coco
from inference import find_rbbox, resized_img
#***************
# # Root directory of the project
# ROOT_DIR = os.path.abspath("../../")
# ROOT_TEMP = os.path.abspath("/home/airlab/Downloads/MaskRCNN_ORCHID")
# print("ROOT_TEMP = ", ROOT_TEMP)
# sys.path.append(ROOT_TEMP)  # To find local version of the library
# from config import Config
# import model as modellib, utils
# import coco      # T add
# # import visualize # T add
# print(os.path.abspath(modellib.__file__))
# import numpy as np
# from visualize import display_instances
# HOME_DIR = '/home/airlab/Downloads/MaskRCNN_ORCHID/maskrcnn'
# DATA_DIR = os.path.join(HOME_DIR, "data/shapes")
#***********
#***********
folder = "/TEST_CHECK/"
annotations_dir = "ANNOTATION_subimage_13"
current_dir = os.getcwd()
path= ''.join([current_dir, folder])

annotations_path = os.path.join(path, annotations_dir)
# print("annotations_savepath ",annotations_path )
# print("annotations_savepath = ", annotations_savepath)
if not os.path.isdir(os.path.abspath(annotations_path)):
    os.mkdir(annotations_path)
image_dir = "subimage_13"
image_path = os.path.join(path, image_dir)
# print("image_path",image_path)
# print("annotations_savepath = ", annotations_savepath)
if not os.path.isdir(os.path.abspath(image_path)):
    os.mkdir(image_path)
#**********
json_dir = "annotations"
json_path = os.path.join(path, json_dir)
# print("json_path",json_path)
# print("annotations_savepath = ", annotations_savepath)
if not os.path.isdir(os.path.abspath(json_path)):
    os.mkdir(json_path)
#***************
# HOME_DIR ='/home/airlab/Desktop/Latest_Training_dataset/MaskRCNN_Nov_11th_2021'
# DATA_DIR = os.path.join(HOME_DIR, "shapes")
# dataset_train = coco.CocoDataset()
# dataset_train.load_coco(DATA_DIR, subset="train", year="")
# dataset_train.prepare() # cái này check utils.Dataset --> prepare()
# Load and display random samples
# print("dataset_train.image_ids:", len(dataset_train.image_ids))
# image_ids = np.random.choice(dataset_train.image_ids, 1)
# print(image_ids)

#*********************
TRAIN_ANNOTATION_DIR = os.path.join(path, annotations_dir)
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
    # print("image_filename:",image_filename)
    file_types = ['*.png']
    file_types = r'|'.join([fnmatch.translate(x) for x in file_types])
    basename_no_extension = os.path.splitext(os.path.basename(image_filename))[0]
    file_name_prefix = basename_no_extension + '.*'
    files = [os.path.join(root, f) for f in files]
    files = [f for f in files if re.match(file_types, f)]
    files = [f for f in files if re.match(file_name_prefix, os.path.splitext(os.path.basename(f))[0])]
    return files
# constants
def magic(numList):
    s = ''.join(map(str, numList))
    return int(s)

def convert_coco_format(image_filename,idx):
    image_id = 1 + idx
    # print("name of image", image_id)
    coco_output = {
        "info": INFO,
        "licenses": LICENSES,
        "categories": CATEGORIES,
        "images": [],
        "annotations": []
    }
    segmentation_id = 1
    image = Image.open(image_filename)
    image_info = pycococreatortools.create_image_info(image_id, os.path.basename(image_filename), image.size)
    coco_output["images"].append(image_info)

    # filter for associated png annotations
    for root, _, files in os.walk(TRAIN_ANNOTATION_DIR):
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
            binary_mask = np.asarray(Image.open(annotation_filename).convert('1')).astype(np.uint8)
            #print(binary_mask) # For testing only. 0 is OK -> found the problem at this point then can solve it
    
            annotation_info = pycococreatortools.create_annotation_info_direction(
                segmentation_id, image_id, category_info, binary_mask,
                image.size, angle_s, tolerance=2) # Voi anh size lon thi phai sua cho nay lai
            if annotation_info is not None:
                coco_output["annotations"].append(annotation_info)
            segmentation_id = segmentation_id + 1
        return coco_output
def find_annotation(image_filename):
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
def apply_mask(image, mask, color, alpha=0.5):
    """Apply the given mask to the image.
    """
    for c in range(3):
        image[:, :, c] = np.where(mask == 1,
                                  image[:, :, c] *
                                  (1 - alpha) + alpha * color[c] * 255,
                                  image[:, :, c])
    return image
def display_top_masks(image, mask, class_ids, class_names, limit=1):
    """Display the given image and the top few class masks."""
    to_display = []
    titles = []
    to_display.append(image)
    titles.append("H x W={}x{}".format(image.shape[0], image.shape[1]))
    # Pick top prominent classes in this image
    unique_class_ids = np.unique(class_ids)
    mask_area = [np.sum(mask[:, :, np.where(class_ids == i)[0]])
                 for i in unique_class_ids]
    top_ids = [v[0] for v in sorted(zip(unique_class_ids, mask_area),
                                    key=lambda r: r[1], reverse=True) if v[1] > 0]
    # Generate images and titles
    for i in range(limit):
        class_id = top_ids[i] if i < len(top_ids) else -1
        # print("classs", class_id)
        # Pull masks of instances belonging to the same class.
        m = mask[:, :, np.where(class_ids == class_id)[0]]
        m = np.sum(m * np.arange(1, m.shape[-1] + 1), -1)
        to_display.append(m)
        # print("lenght", class_names[class_id])
        # titles.append(class_names[class_id] if class_id != -1 else "-")
        titles.append(class_names)
    # display_images(to_display, titles=titles, cols=limit + 1, cmap="Blues_r")
    return  to_display
def display_images(images, titles=None, cols=4, cmap=None, norm=None,
                   interpolation=None):
    """Display the given set of images, optionally with titles.
    images: list or array of image tensors in HWC format.
    titles: optional. A list of titles to display with each image.
    cols: number of images per row
    cmap: Optional. Color map to use. For example, "Blues".
    norm: Optional. A Normalize instance to map values to colors.
    interpolation: Optional. Image interpolation to use for display.
    """
    titles = titles if titles is not None else [""] * len(images)
    rows = len(images) // cols + 1
    plt.figure(figsize=(54, 54 * rows // cols))
    i = 1
    for image, title in zip(images, titles):
        plt.subplot(rows, cols, i)
        plt.title(title, fontsize=9)
        plt.axis('off')
        plt.imshow(image.astype(np.uint8), cmap=cmap,
                   norm=norm, interpolation=interpolation)
        i += 1
    plt.show()
def color_radom(i): 
    r = random.randint(50,255)
    g = random.randint(50,100)
    b = random.randint(50,255)
    rgb = [r,g,b]
    return rgb
if __name__ =="__main__":
    dataset_train = coco.CocoDataset()
    dataset_train.prepare()
    filenames = sorted(glob.glob(image_path + "/*.jpg")) #read all files in the path mentioned
    for index_1,image_name in enumerate(filenames):
        print("image_filename:",image_name)
        image_color = cv2.imread(image_name)
        file, ext = os.path.splitext(image_name)
        file = str(file).split('/')[-1]
        # out_coco = convert_coco_format(image_filename=image_name,idx= index_1)
        # with open('{}/train_{}.json'.format(json_path,file), 'w') as output_json_file:
        #     json.dump(out_coco, output_json_file)
        ''' # Display json files  
        # dataset_train.load_coco(path, subset_ann = "train_{}".format(file),subset_image = image_dir, year="")
        # image = dataset_train.load_image(index_1)
        # image_1 = cv2.cvtColor(image,cv2.COLOR_RGB2BGR)
        # mask, class_ids, angle_ids = dataset_train.load_mask(index_1)
        # print("class_id",mask.shape)
        # print('name of class', dataset_train.class_names)
        # dis_images = display_top_masks(image, mask, class_ids, dataset_train.class_names)
        # image_merge = cv2.bitwise_and(dis_images[0],dis_images[0],mask = dis_images[1].astype(np.uint8))
        # image_merge_1 = cv2.cvtColor(image_merge,cv2.COLOR_RGB2BGR)
        # cv2.imshow("image",image_merge_1)
        # k = cv2.waitKey(0)
        # if k ==27:
        #     break
        ''' 
        annotation_files, angles_all = find_annotation(image_filename=image_name)
        angles, points, lengths = [], [], []
        for i in range(len(annotation_files)):
            print("current mask", annotation_files[i])
            image_mask = cv2.imread(annotation_files[i])
            gray = cv2.cvtColor(image_mask, cv2.COLOR_BGR2GRAY)
            ret, thresh = cv2.threshold(gray,127,255,cv2.THRESH_BINARY)
            cnts, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
            overlay = image_color.copy()
            # for i in range(len(point_cnt)):cv2.fillPoly(image_backgorund, point_cnt[i], color_radom(len(point_cnt)))  
            # img_with_overlay = cv2.normalize(np.int64(image_color) * image_backgorund, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
            for (i_x,cnt) in enumerate(cnts):
                cv2.fillPoly(overlay, [cnt] ,color_radom(i))
                cv2.addWeighted(overlay, 0.5, image_color, 1 - 0.5,0,image_color)
            print("angle", angles_all[i])
            find_rbbox(image_color,gray,angles_all[i]*20)
        cv2.imshow("image", resized_img(image_color,150))
        k = cv2.waitKey(0)
        if k ==ord('q'):
            break
        print(annotation_files)
    print("TRAIN DATA FINISH!")




