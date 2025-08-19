import numpy as np
import cv2
import glob
import os
import PIL as Image
import os.path as osp

from tqdm import tqdm
current_dir = os.getcwd()
print("path", current_dir)
# path= ''.join([current_dir, folder])
# Folder Name:
name = "Images"
# Tạo folder
# if not os.path.exists(current_dir + '/' + "{}".format(name)):
os.makedirs(current_dir + '/' + '{}'.format(name),  exist_ok=True)
path_folder = current_dir + '/' + "{}".format(name)
number_image = 0
if __name__ == "__main__":
    path_im = sorted(glob.glob('C:\\Users\\ptthi\\OneDrive\\Desktop\\PRACTICE_ON_COMPUTER_VISION\WEEK3-Calibration-Camera\\Single_Camera\\*.png'))
    
    desired_size = (640,458)
    for im_path  in tqdm(path_im):
        image = cv2.imread(im_path)
        resized_img = cv2.resize(image, desired_size, cv2.INTER_AREA)
        image_name = path_folder + '/' + 'image_{}.png'.format(number_image)
        cv2.imwrite(image_name, resized_img)
        number_image += 1
        print("image")