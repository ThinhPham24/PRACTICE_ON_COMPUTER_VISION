# Custom deep learning developmentation 
import os
import numpy as np
import glob
from tqdm import tqdm
import cv2
import time
import pandas as pd
import csv

# Input folder
folder = '/OUTPUT_YOLO'
# folder = '/fruit'
current_dir = os.getcwd()
path = ''.join([current_dir, folder])
folder_out = '/OUTPUT'
path_out = ''.join([current_dir, folder_out])


if __name__ == '__main__':
        path_img = glob.glob(path + '/' + '*.jpg')
        # Configure of model
        df = pd.read_csv(f'{current_dir}\\data_check.csv')
        # print("HEADER", df['name of image'])
        list = []
        for i in range(len(df)):
                print(df.iloc[i, 1], df.iloc[i, 4])
                if df.iloc[i, 1] == df.iloc[i, 4]:
                        list.append(1)
                else:
                        list.append(0)
        df['yolact_status'] = list
        df.to_csv(f'{current_dir}\\data_check.csv')
        # for name in df["name of image"]:
        #         out_path = os.path.join(path_out, name)
        #         image_out = os.path.join(path, name)
        #         image = cv2.imread(image_out)
        #         # cv2.imshow("image", image)
        #         # cv2.waitKey(0)
        #         cv2.imwrite(out_path, image)
