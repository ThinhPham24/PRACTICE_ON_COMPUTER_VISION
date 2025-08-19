import enum
from glob import glob
import random
import glob
import os
import os.path as osp
from PIL import Image, ImageDraw
folder = "/LEAF_BUD_CORE_SINGLE"
annotations_dir = "IMG_dark"
current_dir = os.getcwd()
path = ''.join([current_dir, folder])
annotations_savepath = os.path.join(path, annotations_dir)
if not os.path.isdir(os.path.abspath(annotations_savepath)):
    os.mkdir(annotations_savepath) 
if __name__ == "__main__":
    path_imgs = glob.glob("/home/airlab/Desktop/IMAGE_CROP/LEAF_BUD_CORE_SINGLE/IMG/*.jpg")
    path_ans =  glob.glob("/home/airlab/Desktop/IMAGE_CROP/LEAF_BUD_CORE_SINGLE/ANNOTATION_dark/*.png")
    for i, path_an in enumerate(path_ans):
        base_an = osp.splitext(osp.basename(path_an))[0].split('_dark_')[0]
        for j, path_img in enumerate(path_imgs):
            base_img = osp.splitext(osp.basename(path_img))[0]
            if base_an == base_img:
                filename = os.path.join(annotations_savepath, f'{base_img}.jpg')
                imrgb = Image.open(path_img)
                imrgb.save(filename)

        # if 'leaves' in base or 'core' in base:
            
        # # else:
        # #     print("base")

