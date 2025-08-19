import os
import cv2
import glob
from PIL import Image, ImageDraw
import numpy as np
import json
import labelme
import os.path as osp
import uuid

folder = "/LEAF_BUD_CORE_SINGLE/"
annotations_dir = "ANNOTATION"

current_dir = os.getcwd()
path = ''.join([current_dir, folder])

annotations_savepath = os.path.join(path, annotations_dir)
# print("path", annotations_savepath)
if not os.path.isdir(os.path.abspath(annotations_savepath)):
    os.mkdir(annotations_savepath)

if __name__ == "__main__":
    path_jsons = glob.glob("/home/airlab/Desktop/IMAGE_CROP/LEAF_BUD_CORE_SINGLE/Lableme/*.json")
    # sub_f = open(path_json)
    # data_sub = json.load(sub_f)
    # print(data_sub.shapes["label"])
    for i, path_json in enumerate(path_jsons):
        label_file = labelme.LabelFile(filename=path_json)
        base = osp.splitext(osp.basename(path_json))[0]
        # out_img_file = osp.join(args.output_dir, "JPEGImages", base + ".jpg")
        img = labelme.utils.img_data_to_arr(label_file.imageData)
        masks = {} 
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
        i = 0
        name_temp = "None"
        for instance, mask in masks.items():
            cls_name, group_id = instance
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
            if cls_name == "core" or cls_name == "bud":
                filename = os.path.join(annotations_savepath, '{}_{}_{}_0.png'.format(base,cls_name,i))
                imrgb.save(filename)
            # cv2.imshow("mask", mask_temp)
            # k = cv2.waitKey(0)
            # if  k == ord("b"):
            #     break

        