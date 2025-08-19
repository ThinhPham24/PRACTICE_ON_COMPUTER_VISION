# Custom deep learning developmentation 
import os
import numpy as np
import glob
from tqdm import tqdm
import cv2
import time
# import pyyaml module
import yaml
from yaml.loader import SafeLoader
from PIL import Image
import pandas as pd

import csv
from ops_utils import Net_detection
# Input folder
folder = '/Shuffle_Images_v1'
# folder = '/image_test'
current_dir = os.getcwd()
path = ''.join([current_dir, folder])
print("path = ", path)
folder_out = '/OUTPUT'
path_out = ''.join([current_dir, folder_out])
if not os.path.isdir(os.path.abspath(path_out)):
        os.mkdir(path_out)
def enter_number():
        number_bud = int(input("Insert the number of bud: "))
        status = int(input("Insert the Status: "))
        visible = int(input("Insert the Visible: "))
        return [number_bud,status,visible]
if __name__ == '__main__':
        new_version = input("CREATE NEW (Press C) OR REPEAT (Press R): ")
        if new_version == "C":
                path_img = sorted(glob.glob(path + '/' + '*.jpg'))
        else:
                num_con = int(input("ORDER TO CONTINUNE (Press number): "))
                path_img = sorted(glob.glob(path + '/' + '*.jpg'))
                path_img = path_img[num_con:]
        # Configure of model
        with open(f'{current_dir}\\orchid_config.yaml', 'r') as f:
                data = yaml.load(f, Loader=SafeLoader)
        CLASSES_NAME = data['names']
        #------CSV FILE-------
        all_result = []
        header = ['name of image','number of bud','status','visible'] 
        #init model
        model_path = f"{current_dir}\\best_yolov8.onnx"
        config_path = f"{current_dir}\\orchid_config.yaml"
        net_model = Net_detection(path=model_path,file=config_path,cls=CLASSES_NAME)
        for i, image in enumerate(path_img):
                print("image:", image)
                t1 = time.time()
                base_name = os.path.basename(image)
                merge_img = net_model.apply(image_path=image)
                t2 = time.time()
                print('PLEASE TYPE ANSWER:', (t2-t1))
                cv2.imshow('image', merge_img)
                k = cv2.waitKey(0)
                check = 0 
                while(check == 0):
                        result = enter_number()
                        print('BUD, STATUS, VISIBLE: ', result)
                        Ok = input("OK(press o) OR NOT: ")
                        if Ok == "O" or Ok =='o':
                                check =1
                                all_result.append([base_name,result[0],result[1],result[2]])
                                cv2.destroyAllWindows()
                        else:
                                check = 0
                if new_version == "C":
                        with open(f'{current_dir}\\data_check.csv', 'w', encoding='UTF8', newline='') as f:
                                writer = csv.writer(f)
                                # write the header
                                writer.writerow(header)   
                                        # write multiple rows
                                writer.writerows(all_result)
                else: 
                        with open(f'{current_dir}\\data_check.csv', "a", encoding='UTF8', newline='') as f:
                                writer = csv.writer(f)
                                # write multiple rows
                                writer.writerows(all_result)
                                all_result = []
                if new_version == "C":
                        print("Number of image:", (i+ 1 ))
                else:
                        print("Number of image:", (i+ 1 + num_con))
                if k == ord('e'):
                        break


