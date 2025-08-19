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
from collections import Counter
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
    python3 convert_json_labelme2mask_with_angle_fill.py 
    python convert_json_labelme2mask_with_angle_fill.py
    '''
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--path_json', default= "C:\\Users\\ptthi\OneDrive\\Desktop\\Image_processing\\new_data\\new_sub20_train", type=str, required=False,
                        help= 'choose the function to check annotation or json') #path to folder that contain boths image and json files
    args = parser.parse_args()

    path_jsons = sorted(glob.glob(args.path_json + "/*.json"))
    # print("Path jsons:", path_jsons)
    out_folder = "IMG"
    annotations_dir = "ANNOTATION" #name of the folder or the new folder that will contain the mask from json
    path = args.path_json
    annotations_savepath = os.path.join(path, annotations_dir)
    image_savepath = os.path.join(path, out_folder)
    # print("path", annotations_savepath)
    if not os.path.isdir(os.path.abspath(annotations_savepath)):
        os.makedirs(annotations_savepath)
    if not os.path.isdir(os.path.abspath(image_savepath)):
        os.makedirs(image_savepath)

    name_base = 659
    for num, path_json in enumerate(path_jsons):
        label_file = labelme.LabelFile(filename=path_json)
        image_base = osp.splitext(osp.basename(path_json))[0]
        # out_img_file = osp.join(args.output_dir, "JPEGImages", base + ".jpg")
        #------------Load image, change the name of its image and save it to another folder------------
        print("name image",(path + f"\{image_base}.jpg"))
        load_image = cv2.imread(path + f"\{image_base}.jpg")
        filename_image = os.path.join(image_savepath, f'{name_base + num}.jpg')
        cv2.imwrite(f"{filename_image}", load_image)
        #------------------------------------
        img = labelme.utils.img_data_to_arr(label_file.imageData)
        masks = {} 
        angles = {}
        for shape in label_file.shapes:
            points = shape["points"]
            #-----add-------
            i = len(points)
            label = shape["label"]
            group_id = shape.get("group_id")
            shape_type = shape.get("shape_type", "polygon")
            mask = labelme.utils.shape_to_mask(
                img.shape[:2], points[1:i], shape_type # you should change this point--> points[1:i] when you label bud according to angle
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
        count_id = {}
        for instance, mask in masks.items():
            cls_name, group_id = instance
            # number_bud = []
            print("group of image", cls_name)
            for instance_angle, angle in angles.items():
                _, group_id_angle = instance_angle
                if group_id == group_id_angle:
                    # print("class_name", cls_name)
                    mask = np.array(mask)
                    mask_temp = (mask*255).astype('uint8')
                    im = Image.fromarray(mask_temp)
                    # imrgb = Image.merge('RGB', (im,im,im))
                    # image_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
                    kernel = np.ones((3, 3), np.uint8)
                    cls = cv2.morphologyEx(mask_temp, cv2.MORPH_CLOSE, kernel, iterations=1)                 
                    # imrgb = Image.merge('RGB', (cls,cls,cls))
                    if cls_name == name_temp or cls_name in count_id:
                        print("number ", count_id[cls_name])
                        count_id[cls_name] +=1
                        name_temp = cls_name
                        # count_id[cls_name] = i
                        # print("count", count_id)
                    if cls_name not in count_id:
                        i = 0
                        name_temp = cls_name
                        count_id[name_temp] = i
                    print("I",count_id)
                    # if  cls_name == "Root" or cls_name == "root":
                    #     filename = os.path.join(annotations_savepath, '{}_{}_{}_{}.png'.format(name_base+num,'darken',count_id[cls_name],angle))
                    #     # print("file name", filename)
                    #     # cv2.imshow("image", cls)
                    #     # cv2.waitKey(0)
                    #     # imrgb.save(filename)
                    #     cv2.imwrite(f"{filename}", cls)
                    if  cls_name == "Bud" or cls_name == "bud" or cls_name == "BUD" or  cls_name == "buds" :
                        filename = os.path.join(annotations_savepath, '{}_{}_{}_{}.png'.format(name_base+num,'bud',count_id[cls_name],angle))
                        # print("file name", filename)
                        # cv2.imshow("image", cls)
                        # cv2.waitKey(0)
                        # imrgb.save(filename)
                        cv2.imwrite(f"{filename}", cls)


        