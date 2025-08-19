from enum import EnumMeta
import sys
import os
import json
import datetime
from tkinter import image_names
from turtle import title
import skimage.draw
import cv2
import os
import sys
import random
import itertools
import colorsys
from PIL import Image, ImageDraw
from skimage.measure import find_contours
import matplotlib.pyplot as plt
from matplotlib import patches,  lines
import IPython.display
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
import random
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
#***************
import os
import sys
import time
import numpy as np
import imgaug  # https://github.com/aleju/imgaug (pip3 install imageaug)
# Download and install the Python COCO tools from https://github.com/waleedka/coco
# That's a fork from the original https://github.com/pdollar/coco with a bug
# fix for Python 3.
# I submitted a pull request https://github.com/cocodataset/cocoapi/pull/50
# If the PR is merged then use the original repo.
# Note: Edit PythonAPI/Makefile and replace "python" with "python3".
from pycocotools.coco import COCO
from pycocotools import mask as maskUtils
#*********************
def color_radom(i): 
    r = random.randint(50,255)
    g = random.randint(50,100)
    b = random.randint(50,255)
    rgb = [r,g,b]
    return rgb
def annToRLE( ann, height, width):
        """
        Convert annotation which can be polygons, uncompressed RLE to RLE.
        :return: binary mask (numpy 2D array)
        """
        segm = ann['segmentation']
        if isinstance(segm, list):
            # polygon -- a single object might consist of multiple parts
            # we merge all parts into one mask rle code
            rles = maskUtils.frPyObjects(segm, height, width)
            rle = maskUtils.merge(rles)
        elif isinstance(segm['counts'], list):
            # uncompressed RLE
            rle = maskUtils.frPyObjects(segm, height, width)
        else:
            # rle
            rle = ann['segmentation']
        return rle

def annToMask(ann, height, width):
        """
        Convert annotation which can be polygons, uncompressed RLE, or RLE to binary mask.
        :return: binary mask (numpy 2D array)
        """
        rle = annToRLE(ann, height, width)
        m = maskUtils.decode(rle)
        
        return m

def load_mask(annotations, flag = True):
        """Load instance masks for the given image.

        Different datasets use different ways to store masks. This
        function converts the different mask format to one format
        in the form of a bitmap [height, width, instances].

        Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.    """
        # print("information",image_info)
        instance_masks = []
        class_ids = []
        #******************************* MOI ADD ***************************************************************#
        angle_ids = []
        #******************************* MOI ADD ***************************************************************#
        # print("lenght of annotation", len(annotations))

        # Build mask of shape [height, width, instance_count] and list
        # of class IDs that correspond to each channel of the mask.
        for annotation in annotations:
            # class_id = self.map_source_class_id("coco.{}".format(annotation['category_id']))
            class_id = annotation['category_id']
            #******************************* MOI ADD ***************************************************************#
            if flag == True:
                angle_id = annotation['angle']
            #angle_id = annotation['angle']
            #******************************* MOI ADD ***************************************************************#
            if class_id:
                m = annToMask(annotation, int(annotation["height"]), int(annotation["width"]))
                # Some objects are so small that they're less than 1 pixel area
                # and end up rounded out. Skip those objects.
                if m.max() < 1:
                    continue
                # Is it a crowd? If so, use a negative class ID.
                if annotation['iscrowd']:
                    # Use negative class ID for crowds
                    class_id *= -1
                    # For crowd masks, annToMask() sometimes returns a mask
                    # smaller than the given dimensions. If so, resize it.
                    if m.shape[0] != int(annotation["height"]) or m.shape[1] != int(annotation["width"]):
                        m = np.ones([int(annotation["height"]), int(annotation["width"])], dtype=bool)
                instance_masks.append(m)
                class_ids.append(class_id)
                #******************************* MOI ADD ***************************************************************#
                if flag == True:
                    angle_ids.append(angle_id)
                #******************************* MOI ADD ***************************************************************#

        # Pack instance masks into an array
        if class_ids:
            # mask = np.stack(instance_masks, axis=2).astype(np.bool)
            # print("instance", instance_masks)
            class_ids = np.array(class_ids, dtype=np.int32)
            #******************************* MOI ADD ***************************************************************#
            if flag == True:
                angle_ids = np.array(angle_ids, dtype=np.int32)
                #print("angle_ids = ", angle_ids)
                return instance_masks, class_ids, angle_ids
            else:
                return instance_masks, class_ids, angle_ids
def getmask(ids_img,class_ids,flag):
    annotations=coco.loadAnns(coco.getAnnIds(imgIds=ids_img, catIds=class_ids, iscrowd=None))
    mask, cls_id, angle = load_mask(annotations=annotations, flag=flag)
    return  mask, cls_id, angle
def display_mask(image_color,image_mask):
    # merge_instant_masks = np.sum(instant_masks,axis = 0).astype(np.uint8)*255
    instant_masks = np.array(image_mask).astype(np.uint8)*255
    angles, points, lengths = [], [], []
    # mask_merge = []
    overlay = image_color.copy()
    image_only_contour = overlay.copy()
    for i, mask in enumerate(instant_masks):
    # gray = cv2.cvtColor(image_mask, cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(mask,127,255,cv2.THRESH_BINARY)
        cnts, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        # for i in range(len(point_cnt)):cv2.fillPoly(image_backgorund, point_cnt[i], color_radom(len(point_cnt)))  
        # img_with_overlay = cv2.normalize(np.int64(image_color) * image_backgorund, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
        cv2.drawContours(image_only_contour, cnts, -1, (0, 0, 255), 1) 
        for i_x,cnt in enumerate(cnts):
            cv2.fillPoly(overlay, [cnt] ,color_radom(i+100))
            cv2.addWeighted(overlay, 0.5, image_color,1-0.5,0,image_color)
    # print("angle", angles_all[i])
    # find_rbbox(image_color,gray,angles_all[i]*20)
    # cv2.drawContours(image_color, cnts, -1, (0, 0, 0), 1) 
    # find_rbbox(image_only_contour,gray,angles_all[i]*20)
    return overlay,image_color


if __name__ =="__main__":
    '''
    python check_only_json.py
    '''
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--path_img', default= "C:\\Users\\ptthi\\OneDrive\\Desktop\\Image_processing\\THINH_4\\IMG\\validate", type=str, required=False,
                        help= 'direction of image folder') #path to the folder that contain image
    parser.add_argument('--path_json', default= "C:\\Users\\ptthi\\OneDrive\\Desktop\\Image_processing\\THINH_4\\annotations\\validate.json", type=str, required=False,
                        help= 'path of json') #annotation path
    
    args = parser.parse_args()
    print("this is path", args.path_img)

    coco = COCO(f"{args.path_json}")
    class_ids = sorted(coco.getCatIds())
    ids_img = coco.getImgIds(catIds=[class_ids[0]])
    for i in ids_img:
        path=os.path.join(args.path_img, coco.imgs[i]['file_name'])
        name= coco.imgs[i]['file_name']
        print("name of image:", name)
        image = cv2.imread(path)
        instant_masks, cls_id, _ = getmask(i,class_ids,flag=False)
        print("LENDTH OF MASK", len(instant_masks))
        image_mask,image_color = display_mask(image, instant_masks)
        image_mask = cv2.putText(image_mask, f'Number of mask: {len(instant_masks)}',(50, 50), cv2.FONT_HERSHEY_DUPLEX, 1, (255,0,0), 2, cv2.LINE_AA)
        image_mask = cv2.putText(image_mask, f'Name of image: {name}',(50, 100), cv2.FONT_HERSHEY_DUPLEX, 1, (255,0,255), 2, cv2.LINE_AA)  
        cv2.imshow('IMAGE',image_mask)
        k = cv2.waitKey(0)
        if k ==ord('q') or k == ord('Q'):
                break
        
    