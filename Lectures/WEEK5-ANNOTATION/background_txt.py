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
count = 1
if __name__ == "__main__":
    # path_an = sorted(glob.glob('/home/airlab/Desktop/New_training/01_Annotation/ANNOTATION/test/*.png'))
    # path_im = sorted(glob.glob('/home/airlab/Desktop/New_training/01_Annotation/IMG/test/*.jpg'))
    name = 'train'
    path_im = sorted(glob.glob('C:\\Users\\ptthi\\OneDrive\\Desktop\\Image_processing\\DATA_PROCESS\\BACKGROUND\\*.png'))
    print("path", path_im)
    # tao folder moi
    if not os.path.exists(current_dir + '/' + "{}_new".format(name)):
        os.makedirs(current_dir + '/' + '{}_new'.format(name))
    for i,im in enumerate(path_im):
        base_an = osp.splitext(osp.basename(im))[0]
        print("name of image:", i)
        if i % 2 == 0:
            print("True")
            file_im = cv2.imread(im)
            cv2.imwrite(current_dir + '/' + '{}_new'.format(name) + '/' + '{}.jpg'.format(count),file_im)
            count += 1
        print("count",count)


