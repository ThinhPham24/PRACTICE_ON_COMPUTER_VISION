import numpy as np
import cv2
import glob
import os
import PIL as Image
import os.path as osp

from tqdm import tqdm
current_dir = os.getcwd()
path = os.path.join(current_dir,'runs/segment')
new_dir = os.path.join(current_dir,'runs/image')
print("path", path)
if __name__ == "__main__":
 
    folder = os.listdir(path)
    print("path", folder)
    if not os.path.exists(new_dir):
        os.makedirs(new_dir)
    for i, subfolder in tqdm(enumerate(folder), total = len(folder), desc="ALL FOLDER"):
        path_sub = glob.glob(path + '/'+ subfolder + '/' + "*.jpg")
        for  image in path_sub:
            head_tail = os.path.split(image)
            file = cv2.imread(image)
            base_name = head_tail[1]
            print("name", base_name)
            filename = os.path.join(new_dir + '/' + base_name)
            print("filename", filename)
            cv2.imwrite(filename, file)

    # print("path", path_im)
    # print("all annotation", path_an)
    # # tao folder moi
    # if not os.path.exists(current_dir + '/' + "{}_new".format(name)):
    #     os.makedirs(current_dir + '/' + '{}_new'.format(name))
    # if not os.path.exists(current_dir + '/' +'an_{}_new'.format(name)):
    #     os.makedirs(current_dir + '/' +'an_{}_new'.format(name))
    # for im in tqdm(path_im):
    #     print("image", im)
    #     order_im = str(im).split("\\")
    #     order_im_compare = str(im).split("\\")[-1].split(".")[0]
    #     print("order image", order_im_compare)
    #     file_im = cv2.imread(im)
    #     cv2.imwrite(current_dir + '/' + '{}_new'.format(name) + '/' + '{}.jpg'.format(count),file_im)
    #     # count += 1
    #     # print("count",count)
    #     for an in path_an:
    #         order_darken_compare = str(an).split("\\")[-1].split('_Bud_')[0]
    #         order_bud_compare = str(an).split("\\")[-1].split('_core_')[0]
    #         print("order image", order_im_compare)
    #         print("name LEAVES", order_darken_compare)
    #         # print("name CORE", order_bud_compare)
    #         if order_darken_compare == order_im_compare:
    #             order_darken = str(an).split("/")[-1].split('{}_'.format(order_darken_compare))[1]
    #             base_an = osp.splitext(osp.basename(an))[0].split(order_darken_compare)[-1]
    #             print("base", base_an)
    #             # print("number of darken", order_darken)
    #             file_an = cv2.imread(an)
    #             # print('order of darken', order_darken_compare)
    #             filename_mask = os.path.join(current_dir + '/' + f'an_{name}_new',  f'{count}{base_an}.png')
    #             cv2.imwrite(filename_mask, file_an)



