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

folder = "\\DATA\\IMG"
folder_AN = "\\DATA\\ANNOTATION"
current_dir = os.getcwd()
path_im= ''.join([current_dir, folder])
path_an = ''.join([current_dir, folder_AN])

folder_img_train = "C:\\Users\\ptthi\\OneDrive\\Desktop\\Image_processing\\DATA_PROCESS\\DATA\\train"
folder_an_train = "C:\\Users\\ptthi\\OneDrive\\Desktop\\Image_processing\\DATA_PROCESS\\DATA\\train_anno"
def generate_path(path):
    if not os.path.exists(path):
        os.makedirs(path)
if __name__ == "__main__":
    # path_an = sorted(glob.glob('/home/airlab/Desktop/IMAGE_CROP/ANNOTATION/*.png'))

    # path_im = sorted(glob.glob('/home/airlab/Desktop/IMAGE_CROP/IMAGE/*.jpg'))
    generate_path(folder_img_train)
    generate_path(folder_an_train)
    files_img = os.listdir(path_im)
    files_an = os.listdir(path_an)

    
    for filename in tqdm(random.sample(files_img,1000),total=1000):   
        path_img = os.path.join(path_im, filename)
        order_im_compare = str(filename).split(".")[0]
        shutil.move(path_img, folder_img_train)
        for an in files_an:
            order_bud_compare = str(an).split('_darken_')[0]  
            order_bud_compare_leaves = str(an).split('_bud_')[0]     
            if order_bud_compare == order_im_compare or order_bud_compare_leaves == order_im_compare:
                path_anan = os.path.join(path_an, an)
                shutil.move(path_anan, folder_an_train)
                

