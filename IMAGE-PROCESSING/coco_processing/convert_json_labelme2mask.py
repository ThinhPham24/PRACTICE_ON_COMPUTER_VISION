import os
import cv2
import glob
from PIL import Image, ImageDraw
import numpy as np
import json
import labelme
import os.path as osp
import uuid

folder = "/Single_bud_dataset/datas3_singlebud/"
annotations_dir = "ANNOTATION"

current_dir = os.getcwd()
path = ''.join([current_dir, folder])

annotations_savepath = os.path.join(path, annotations_dir)
# print("path", annotations_savepath)
if not os.path.isdir(os.path.abspath(annotations_savepath)):
    os.makedirs(annotations_savepath)

if __name__ == "__main__":
    path_jsons = glob.glob("/home/airlab/Desktop/IMAGE_CROP/Single_bud_dataset/datas3_singlebud/datas3/*.json")
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
            i = len(points)
            label = shape["label"]
            group_id = shape.get("group_id")
            shape_type = shape.get("shape_type", "polygon")
            mask = labelme.utils.shape_to_mask(
                img.shape[:2], points[1:i], shape_type
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
            # im = Image.fromarray(mask_temp)
            # imrgb = Image.merge('RGB', (im,im,im))
            kernel = np.ones((3, 3), np.uint8)
            cls = cv2.morphologyEx(mask_temp, cv2.MORPH_CLOSE, kernel, iterations=1)        
            if cls_name == name_temp:
                i +=1
                name_temp = cls_name
            else:
                i =0
                name_temp = cls_name
            # print("I",i)
            # if cls_name == "bud" or cls_name == "dark":
            #     filename = os.path.join(annotations_savepath, '{}_{}_{}_0.png'.format(base,cls_name,i))
            #     imrgb.save(filename)
            # cv2.imshow("mask", mask_temp)
            # k = cv2.waitKey(0)
            # if  k == ord("b"):
            #     break
            if cls_name == "core" or cls_name == "bud" or cls_name == "leaf" or cls_name == "single":
                filename = os.path.join(annotations_savepath, '{}_{}_{}_0.png'.format(base,cls_name,i))
                # print("file name", filename)
                # cv2.imshow("image", cls)
                # cv2.waitKey(0)
                # imrgb.save(filename)
                cv2.imwrite(f"{filename}", cls)

        