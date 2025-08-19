import os
import cv2
import glob
from PIL import Image, ImageDraw
import numpy as np
import json
import labelme
import os.path as osp
import uuid
import math

import argparse

# folder = "/Image_processing/"
# annotations_dir = "ANNOTATION"

# current_dir = os.getcwd()
# path = ''.join([current_dir, folder])

# annotations_savepath = os.path.join(path, annotations_dir)
# # print("path", annotations_savepath)
# if not os.path.isdir(os.path.abspath(annotations_savepath)):
#     os.mkdir(annotations_savepath)
def calculate_angle(sp,ep):
    sp = np.array(sp)
    ep = np.array(ep)
    angle =  math.atan2(int(ep[1])-int(sp[1]),int(ep[0])-int(sp[0]))*180/math.pi
    if angle < 0:
        angle = - (angle)
    else:
        angle = 360- angle
    return int(angle) 
if __name__ == "__main__":
    '''
    python3 convert_json_labelme2mask_with_angle.py 
    '''
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--path_json', default= "C:\\Users\\ptthi\OneDrive\\Desktop\\Image_processing\\data_test\\validate", type=str, required=False,
                        help= 'choose the function to check annotation or json') #path to folder that contain boths image and json files
    args = parser.parse_args()

    path_jsons = glob.glob(args.path_json + "/*.json")
    print("Path jsons:", path_jsons)

    annotations_dir = "ANNOTATION" #name of the folder or the new folder that will contain the mask from json
    path = args.path_json

    annotations_savepath = os.path.join(path, annotations_dir)
    # print("path", annotations_savepath)
    if not os.path.isdir(os.path.abspath(annotations_savepath)):
        os.makedirs(annotations_savepath)


    for i, path_json in enumerate(path_jsons):
        label_file = labelme.LabelFile(filename=path_json)
        base = osp.splitext(osp.basename(path_json))[0]
        # out_img_file = osp.join(args.output_dir, "JPEGImages", base + ".jpg")
        img = labelme.utils.img_data_to_arr(label_file.imageData)
        masks = {} 
        angles = {}
        for shape in label_file.shapes:
            points = shape["points"]
            label = shape["label"]
            group_id = shape.get("group_id")
            shape_type = shape.get("shape_type", "polygon")
            mask = labelme.utils.shape_to_mask(
                img.shape[:2], points, shape_type
            )

            if group_id is None:
                group_id = uuid.uuid1()

            instance = (label, group_id)

            if instance in masks:
                masks[instance] = masks[instance] | mask
            else:
                masks[instance] = mask
            angle = calculate_angle(points[0],points[1])
            index = (label, group_id)
            if index in  angles:
                angles[index] = angles[index] | angle
            else:
                angles[index] = angle
        i = 0
        name_temp = "None"
        for instance, mask in masks.items():
            cls_name, group_id = instance
            for instance_angle, angle in angles.items():
                _, group_id_angle = instance_angle
                if group_id == group_id_angle:
                    # print("class_name", cls_name)
                    mask = np.array(mask)
                    mask_temp = (mask*255).astype('uint8')
                    im = Image.fromarray(mask_temp)
                    imrgb = Image.merge('RGB', (im,im,im))
                    if cls_name == name_temp:
                        i +=1
                        name_temp = cls_name
                    else:
                        i =0
                        name_temp = cls_name
                    # print("I",i)
                    if cls_name == "core" or cls_name == "bud" or cls_name == "leaf" or cls_name == "single":
                        filename = os.path.join(annotations_savepath, '{}_{}_{}_{}.png'.format(base,cls_name,i,angle))
                        imrgb.save(filename)

        