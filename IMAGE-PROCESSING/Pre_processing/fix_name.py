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
count = 100
if __name__ == "__main__":
    # path_an = sorted(glob.glob('/home/airlab/Desktop/New_training/01_Annotation/ANNOTATION/test/*.png'))
    # path_im = sorted(glob.glob('/home/airlab/Desktop/New_training/01_Annotation/IMG/test/*.jpg'))
    name = 'train'

    path_an = sorted(glob.glob('C:\\Users\\ptthi\\OneDrive\\Desktop\\Image_processing\\THINH_7\\ANNOTATION\\train\\*.png'))
    path_im = sorted(glob.glob('C:\\Users\\ptthi\\OneDrive\\Desktop\\Image_processing\\THINH_7\\IMG\\train\\*.jpg'))
   
    print("path", path_im)
    print("all annotation", path_an)
    # tao folder moi
    if not os.path.exists(current_dir + '/' + "{}_new".format(name)):
        os.makedirs(current_dir + '/' + '{}_new'.format(name))
    if not os.path.exists(current_dir + '/' +'an_{}_new'.format(name)):
        os.makedirs(current_dir + '/' +'an_{}_new'.format(name))
    for im in tqdm(path_im):
        print("image", im)
        order_im = str(im).split("\\")
        order_im_compare = str(im).split("\\")[-1].split(".")[0]
        print("order image", order_im_compare)
        file_im = cv2.imread(im)
        cv2.imwrite(current_dir + '/' + '{}_new'.format(name) + '/' + '{}.jpg'.format(count),file_im)
        # count += 1
        # print("count",count)
        for an in path_an:
            order_darken_compare = str(an).split("\\")[-1].split('_Bud_')[0]
            order_bud_compare = str(an).split("\\")[-1].split('_core_')[0]
            print("order image", order_im_compare)
            print("name LEAVES", order_darken_compare)
            # print("name CORE", order_bud_compare)
            if order_darken_compare == order_im_compare:
                order_darken = str(an).split("/")[-1].split('{}_'.format(order_darken_compare))[1]
                base_an = osp.splitext(osp.basename(an))[0].split(order_darken_compare)[-1]
                print("base", base_an)
                # print("number of darken", order_darken)
                file_an = cv2.imread(an)
                # print('order of darken', order_darken_compare)
                filename_mask = os.path.join(current_dir + '/' + f'an_{name}_new',  f'{count}{base_an}.png')
                cv2.imwrite(filename_mask, file_an)
            # if order_bud_compare == order_im_compare:
            #     # print("name",order_bud_compare)
            #     order_bud = str(an).split("/")[-1].split('{}_'.format(order_bud_compare))[1]
            #     base_an = osp.splitext(osp.basename(an))[0].split(order_bud_compare)[-1]
            #     # print("orger bud",order_bud)
            #     # print("number of bud", order_bud)
            #     file_an = cv2.imread(an)
            #     # print('order of bud', order_bud_compare)
            #     filename_mask = os.path.join(f'./Single_bud_dataset/an_{name}_new', f'{count}{base_an}.png')
            #     cv2.imwrite(filename_mask, file_an)
        count += 1
        print("count",count)


