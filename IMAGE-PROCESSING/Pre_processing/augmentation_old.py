import numpy as np
import argparse
import imutils
import cv2
import os, os.path
import glob

import random
import matplotlib.pyplot as plt
import albumentations as A
import numpy as np
import argparse
import imutils
import cv2
import os, os.path
import glob

rename_list = []
remained_list = []
check = 0
offset1 = 20000000000
offset2 = 30000000000
offset3 = 40000000000
offset4 = 50000000000
offset5 = 60000000000
offset6 = 70000000000
offset7 = 80000000000
offset8 = 90000000000
offset9 = 95000000000
k = 1 

folder = "/Single_bud_dataset/"
current_dir = os.getcwd()
path = ''.join([current_dir, folder])
print("path = ", path)


#type 0:train, 1:test
type=0

if type==0:
    #-----------------------------------------------------------------------#
    images_dir = "train/img_rot_flip"
    annotations_dir = "train/an_rot_flip"
    images_path = os.path.join(path, images_dir)
    annotations_savepath = os.path.join(path, annotations_dir)
    if not os.path.isdir(os.path.abspath(annotations_savepath)):
        os.mkdir(annotations_savepath)
    #-----------------------------------------------------------------------#
    new_image_save = "train/img"
    new_annotations_save = "train/an"
    images_new_savepath = os.path.join(path, new_image_save)
    annotations_new_savepath = os.path.join(path, new_annotations_save)
    if not os.path.isdir(os.path.abspath(images_new_savepath)):
        os.mkdir(images_new_savepath)
    if not os.path.isdir(os.path.abspath(annotations_new_savepath)):
        os.mkdir(annotations_new_savepath)
    #-----------------------------------------------------------------------#


if type==1:
    #-----------------------------------------------------------------------#
    images_dir = "test/img_rot_flip"
    annotations_dir = "test/an_rot_flip"
    images_path = os.path.join(path, images_dir)
    annotations_savepath = os.path.join(path, annotations_dir)
    if not os.path.isdir(os.path.abspath(annotations_savepath)):
        os.mkdir(annotations_savepath)
    #-----------------------------------------------------------------------#
    new_image_save = "test/img"
    new_annotations_save = "test/an"
    images_new_savepath = os.path.join(path, new_image_save)
    annotations_new_savepath = os.path.join(path, new_annotations_save)
    if not os.path.isdir(os.path.abspath(images_new_savepath)):
        os.mkdir(images_new_savepath)
    if not os.path.isdir(os.path.abspath(annotations_new_savepath)):
        os.mkdir(annotations_new_savepath)
    #-----------------------------------------------------------------------#

images_path = os.path.join(path, images_dir)
annotations_savepath = os.path.join(path, annotations_dir)
print("images_path = ", images_path)
print("ANNO_path = ", annotations_savepath)
if not os.path.isdir(os.path.abspath(annotations_savepath)):
    os.mkdir(annotations_savepath)
def adjust_gamma(image, gamma=1.0):
	# build a lookup table mapping the pixel values [0, 255] to
	# their adjusted gamma values
	invGamma = 1.0 / gamma
	table = np.array([((i / 255.0) ** invGamma) * 255
		for i in np.arange(0, 256)]).astype("uint8")
	# apply gamma correction using the lookup table
	return cv2.LUT(image, table)
# constants
def magic(numList):
    s = ''.join(map(str, numList))
    return int(s)
def scale_random(img):
    s_x = - 0.25
    s_y = - 0.25
    img_shape = img.shape
    resize_scale_x = 1 + s_x
    resize_scale_y = 1 + s_y
    img=  cv2.resize(img, None, fx = resize_scale_x, fy = resize_scale_y)
    canvas = np.zeros(img_shape, dtype = np.uint8)
    y_lim = int(min(resize_scale_y,1)*img_shape[0])
    x_lim = int(min(resize_scale_x,1)*img_shape[1])
    canvas[:y_lim,:x_lim,:] =  img[:y_lim,:x_lim,:]
    img = canvas
    return img

filenames = glob.glob(images_path + "/*.*") #read all files in the path mentioned
# random_file = random.sample(filenames,int(len(filenames)*0.2))
for n, image_file in enumerate(filenames):
    file, ext = os.path.splitext(image_file)  # split filename and extension
    name = os.path.basename(file)
    s = len(os.path.basename(file))
    # print("name's shape img:",name)
    #print("Length:",s)
    l = list(name)
    for i in range(s):
        if l[i] != '_' and check==0:
            rename_list.append(l[i])
        else:
            check = 1  
            remained_list.append(l[i])
    rename_int = list(map(int, rename_list))
    remained_name = ''.join(map(str, remained_list))
    num = magic(rename_int)
    num_modify = num*k
    new_value1 = num_modify + offset1
    new_value2 = num_modify + offset2
    new_value3 = num_modify + offset3
    new_value4 = num_modify + offset4
    new_value5 = num_modify + offset5
    new_value6 = num_modify + offset6
    new_value7 = num_modify + offset7
    new_value8 = num_modify + offset8
    new_value9 = num_modify + offset9
    rename_list = []
    remained_list = []
    check = 0

    file, ext = os.path.splitext(image_file)  # split filename and extension
    #image = cv2.imread(image_file.path)
    new_img = cv2.imread(image_file) # Thay đổi tương ứng!
    #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # loop over the rotation angles
    
    filename = os.path.join(images_new_savepath, '{}{}.jpg'.format(new_value1,remained_name))
    cv2.imwrite(filename,new_img)

    #
    # light = A.Compose([A.CLAHE(p=0.5),A.RandomGamma(p=0.8)], p=1)
    # random.seed(42)
    # transformed = light(image=new_img)
    # filename = os.path.join(images_new_savepath, '{}{}.jpg'.format(new_value2,remained_name))
    # cv2.imwrite(filename,transformed['image'])
    #
    transform = A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, brightness_by_max=True, always_apply=True, p=1)
    random.seed(42)
    augmented_image = transform(image=new_img)
    filename = os.path.join(images_new_savepath, '{}{}.jpg'.format(new_value3,remained_name))
    cv2.imwrite(filename,augmented_image['image'])
    # Scale
    augmented_image = scale_random(new_img)
    filename = os.path.join(images_new_savepath, '{}{}.jpg'.format(new_value4,remained_name))
    cv2.imwrite(filename,augmented_image)
    #
    transform = A.JpegCompression(quality_lower=19, quality_upper=20, p=1)
    random.seed(42)
    augmented_image = transform(image=new_img)
    filename = os.path.join(images_new_savepath, '{}{}.jpg'.format(new_value5,remained_name))
    cv2.imwrite(filename,augmented_image['image'])

filenames = glob.glob(annotations_savepath + "/*.*") #read all files in the path mentioned
for n, image_file in enumerate(filenames):
    file, ext = os.path.splitext(image_file)  # split filename and extension
    name = os.path.basename(file)
    s = len(os.path.basename(file))
    #print("name's shape:",name)
    #print("Length:",s)
    l = list(name)
    for i in range(s):
        if l[i] != '_' and check==0:
            rename_list.append(l[i])
        else:
            check = 1
            remained_list.append(l[i])
    # ----------------------------------------------#
    #print("number",num)
    rename_int = list(map(int, rename_list))
    remained_name = ''.join(map(str, remained_list))
    num = magic(rename_int)
    num_modify = num*k
    new_value1 = num_modify + offset1
    new_value2 = num_modify + offset2
    new_value3 = num_modify + offset3
    new_value4 = num_modify + offset4
    new_value5 = num_modify + offset5
    new_value6 = num_modify + offset6
    new_value7 = num_modify + offset7
    new_value8 = num_modify + offset8
    new_value9 = num_modify + offset9
    rename_list = []
    remained_list = []
    check = 0

    file, ext = os.path.splitext(image_file)  # split filename and extension
    #image = cv2.imread(image_file.path)
    new_img = cv2.imread(image_file) # Thay đổi tương ứng!
    #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # loop over the rotation angles

    filename = os.path.join(annotations_new_savepath, '{}{}.png'.format(new_value1,remained_name))
    cv2.imwrite(filename,new_img)
    #
    # filename = os.path.join(annotations_new_savepath, '{}{}.png'.format(new_value2,remained_name))
    # cv2.imwrite(filename,new_img)
    #
    filename = os.path.join(annotations_new_savepath, '{}{}.png'.format(new_value3,remained_name))
    cv2.imwrite(filename,new_img)

    # Scale
    augmented_image = scale_random(new_img)
    filename = os.path.join(annotations_new_savepath, '{}{}.png'.format(new_value4,remained_name))
    cv2.imwrite(filename,augmented_image)
    #
    filename = os.path.join(annotations_new_savepath, '{}{}.png'.format(new_value5,remained_name))
    cv2.imwrite(filename,new_img)