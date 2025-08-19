import numpy as np
import cv2
import glob
import os
import PIL as Image
import random
import shutil
import re
from tqdm import tqdm 
from PIL import Image, ImageDraw
import os.path as osp

folder = "/BUD_ARROW/IMG"
folder_AN = "/BUD_ARROW/ANNOTATION"
current_dir = os.getcwd()
path_im= ''.join([current_dir, folder])
path_an = ''.join([current_dir, folder_AN])

folder_img_train = "/home/airlab/Desktop/IMAGE_CROP/BUD_ARROW/TRAIN"
folder_an_train = "/home/airlab/Desktop/IMAGE_CROP/BUD_ARROW/ANNOTATION_TRAIN"
# folder_img_fix = "/home/airlab/Desktop/IMAGE_CROP/LEAF_BUD_CORE_SINGLE/fix_CORE_LEAF"
# folder_an_fix = "/home/airlab/Desktop/IMAGE_CROP/LEAF_BUD_CORE_SINGLE/fix_AN_CORE_LEAF"
# folder_img_test = "/home/airlab/Desktop/IMAGE_CROP/LEAF_BUD_CORE_SINGLE/TEST_CORE_LEAF"
# folder_an_test = "/home/airlab/Desktop/IMAGE_CROP/LEAF_BUD_CORE_SINGLE/TEST_AN_CORE_LEAF"
def generate_path(path):
    if not os.path.exists(path):
        os.makedirs(path)
if __name__ == "__main__":
    # path_an = sorted(glob.glob('/home/airlab/Desktop/IMAGE_CROP/ANNOTATION/*.png'))

    # path_im = sorted(glob.glob('/home/airlab/Desktop/IMAGE_CROP/IMAGE/*.jpg'))
    generate_path(folder_img_train)
    generate_path(folder_an_train)
    # generate_path(folder_img_test)
    # generate_path(folder_an_test)
    # generate_path(folder_img_fix)
    # generate_path(folder_an_fix)
    files_img = os.listdir(path_im)
    files_an = os.listdir(path_an)

    
    for filename in tqdm(random.sample(files_img,372),total=372):
        path_img = os.path.join(path_im, filename)
        order_im_compare = str(filename).split(".")[0]
        shutil.move(path_img, folder_img_train)
        for an in files_an:
            order_bud_compare = str(an).split('_core_')[0]  
            order_bud_compare_leaves = str(an).split('_bud_')[0]     
            if order_bud_compare == order_im_compare or order_bud_compare_leaves == order_im_compare:
                path_anan = os.path.join(path_an, an)
                shutil.move(path_anan, folder_an_train)
                
    # name =1000
    # for _,file_name1 in enumerate(os.listdir(folder_img_train)):
    #     path_img = os.path.join(folder_img_train, file_name1)
    #     order_im_compare = str(file_name1).split(".")[0]
    #     for _, filean in enumerate(os.listdir(folder_an_train)):
    #         order_bud_compare = str(filean).split('_core_')[0]
    #         order_bud_compare_leaves = str(filean).split('_leaves_')[0]

    #         path_anan = os.path.join(folder_an_train, filean)
    #         if order_bud_compare == order_im_compare or order_bud_compare_leaves == order_im_compare:
    #             base_an = osp.splitext(osp.basename(filean))[0].split(order_bud_compare)[-1]
    #             filename1 = os.path.join(folder_img_fix, f'{name}.jpg')
    #             imrgb = Image.open(path_img)
    #             imrgb.save(filename1)
    #             filename_mask = os.path.join(folder_an_fix, f'{name}{base_an}_0.png')
    #             imrgb_mask = Image.open(path_anan)
    #             imrgb_mask.save(filename_mask)
    #             name +=1
    # files_img1 = os.listdir(path_im)
    # files_an1 = os.listdir(path_an)
    # for _,file_name2 in enumerate(files_img1):
    #     path_img = os.path.join(path_im, file_name2)
    #     order_im_compare = str(file_name2).split(".")[0]
    #     for _, filean in enumerate(files_an1):
    #         order_bud_compare = str(filean).split('_core_')[0]
    #         order_bud_compare_leaves = str(filean).split('_leaves_')[0]

    #         path_anan = os.path.join(path_an, filean)
    #         if order_bud_compare == order_im_compare or order_bud_compare_leaves == order_im_compare:
    #             base_an = osp.splitext(osp.basename(filean))[0].split(order_bud_compare)[-1]
    #             filename1 = os.path.join(folder_img_test, f'{name}.jpg')
    #             imrgb = Image.open(path_img)
    #             imrgb.save(filename1)
    #             filename_mask = os.path.join(folder_an_test, f'{name}{base_an}_0.png')
    #             imrgb_mask = Image.open(path_anan)
    #             imrgb_mask.save(filename_mask)
    #             name +=1


    
        # # print("oafaf", order_im_compare)
        # shutil.move(path_img, folder_img_train)
        # # regex = re.compile(r'^A\d{5}', re.IGNORECASE)
        # for an in files_an:
        #     order_bud_compare = str(an).split('_dark_')[0]
        #     # print(" comare",an)
        #     if order_bud_compare == order_im_compare:
        #         path_anan = os.path.join(path_an, an)
        #         shutil.move(path_anan, folder_an_train)
    




