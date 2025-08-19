# USAGE
# python rotate_simple_full.py 

# import the necessary packages
import numpy as np
import argparse
import imutils
import cv2
import os, os.path
import glob

import random
import matplotlib.pyplot as plt
import albumentations as A

rename_list = []
remained_list = []
check = 0
offset1 = 200000
offset2 = 300000
offset3 = 400000
offset4 = 500000
offset5 = 600000
offset6 = 700000
offset7 = 800000
offset8 = 900000
offset9 = 950000
#k = 4; # So luong anh muon tao ra tu anh goc
#k = 3; # So luong anh muon tao ra tu anh goc
#k = 30 ;
k = 1 

folder = "/DATA_TEST_AUGMENTATION/"
current_dir = os.getcwd()
path = ''.join([current_dir, folder])
print("path = ", path)
#-----------------------------------------------------------------------#
images_dir = "test_img"
annotations_dir = "test_an"
images_path = os.path.join(path, images_dir)
annotations_savepath = os.path.join(path, annotations_dir)
if not os.path.isdir(os.path.abspath(annotations_savepath)):
    os.mkdir(annotations_savepath)
#-----------------------------------------------------------------------#
new_image_save = "img_test_rot"
new_annotations_save = "an_test_rot"
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

   
#for n, image_file in enumerate(os.scandir(images_path)): 
    #file, ext = os.path.splitext(image_file)  # split filename and extension

    #print("name:",os.path.basename(file))


filenames = glob.glob(images_path + "/*.*") #read all files in the path mentioned
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
    
    # new_img = cv2.flip(image, 0)
    
    #cv2.imshow("Rotated (Problematic)", new_img)
    #filename = os.path.join(images_new_savepath, '{}{}.jpg'.format(new_value2,remained_name))
    #cv2.imwrite(filename,new_img)



    # transform = A.Blur (blur_limit= 10, always_apply=True, p=1)
    # random.seed(42)
    # augmented_image = transform(image=new_img)
    # filename = os.path.join(images_new_savepath, '{}{}.jpg'.format(new_value2,remained_name))
    # cv2.imwrite(filename,augmented_image['image'])


    # light = A.Compose([A.RandomBrightnessContrast(p=1),A.RandomGamma(p=1),A.CLAHE(p=1),], p=1)
    
    light = A.Compose([A.RandomBrightnessContrast(p=0.5),A.RandomGamma(p=0.5),A.CLAHE(p=0.5),], p=1)
    random.seed(42)
    transformed = light(image=new_img)
    filename = os.path.join(images_new_savepath, '{}{}.jpg'.format(new_value3,remained_name))
    cv2.imwrite(filename,transformed['image'])
#********************
# A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=50, val_shift_limit=50, p=1),]
    # medium = A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=50, val_shift_limit=50, p=0.65)
    # random.seed(42)
    # transformed = medium(image=new_img)
    # filename = os.path.join(images_new_savepath, '{}{}.jpg'.format(new_value5,remained_name))
    # cv2.imwrite(filename,transformed['image'])
#*********************
   # strong = A.Compose([ A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=40, val_shift_limit=40, p=1)], p=1)
    #random.seed(42)
    #transformed = strong(image=image)
   # filename = os.path.join(images_new_savepath, '{}{}.jpg'.format(new_value5,remained_name))
    #cv2.imwrite(filename,transformed['image'])

    
    #transform = A.Compose([A.RandomCrop(320, 320), A.OneOf([A.RGBShift(), A.HueSaturationValue()]),])
    #random.seed(42)
    #transformed = transform(image=image)
    #filename = os.path.join(images_new_savepath, '{}{}.jpg'.format(new_value6,remained_name))
    #cv2.imwrite(filename,transformed['image'])


    #transform = A.Compose([A.RandomCrop(320, 320), A.JpegCompression(quality_lower=19, quality_upper=20, p=1),])
    #random.seed(60)
    #augmented_image = transform(image=image)
    #filename = os.path.join(images_new_savepath, '{}{}.jpg'.format(new_value7,remained_name))
    #cv2.imwrite(filename,augmented_image['image'])

    #transform = A.MultiplicativeNoise(multiplier=[0.5, 1.5], elementwise=True, p=1)

#******************


    # transform = A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, brightness_by_max=True, always_apply=True, p=1)
    # random.seed(42)
    # augmented_image = transform(image=new_img)
    # filename = os.path.join(images_new_savepath, '{}{}.jpg'.format(new_value3,remained_name))
    # cv2.imwrite(filename,augmented_image['image'])

    # transform = A.JpegCompression(quality_lower=19, quality_upper=20, p=1)
    # random.seed(42)
    # augmented_image = transform(image=new_img)
    # filename = os.path.join(images_new_savepath, '{}{}.jpg'.format(new_value4,remained_name))
    # cv2.imwrite(filename,augmented_image['image'])

#***************************

#for n, image_file in enumerate(os.scandir(annotations_savepath)): 
#    file, ext = os.path.splitext(image_file)  # split filename and extension
    #print("name:",os.path.basename(file))
# CHANGED HERE
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
   
    # new_img = cv2.flip(image, 0)
    #cv2.imshow("Rotated (Problematic)", new_img)
    # filename = os.path.join(annotations_new_savepath, '{}{}.jpg'.format(new_value2,remained_name))
    # cv2.imwrite(filename,new_img)
    

    #cv2.imshow("Rotated (Problematic)", new_img)
    # filename = os.path.join(annotations_new_savepath, '{}{}.png'.format(new_value2,remained_name))
    # cv2.imwrite(filename,new_img)
    
    #new_img = image
    #cv2.imshow("Rotated (Problematic)", new_img)
    #filename = os.path.join(annotations_new_savepath, '{}{}.jpg'.format(new_value3,remained_name))
    #cv2.imwrite(filename,new_img)

    #new_img = image
    #cv2.imshow("Rotated (Problematic)", new_img)
    #filename = os.path.join(annotations_new_savepath, '{}{}.jpg'.format(new_value4,remained_name))
    #cv2.imwrite(filename,new_img)


    #new_img = image
    #cv2.imshow("Rotated (Problematic)", new_img)
    # filename = os.path.join(annotations_new_savepath, '{}{}.jpg'.format(new_value5,remained_name))
    # cv2.imwrite(filename,new_img)


    #transform = A.Compose([A.RandomCrop(320, 320), A.OneOf([A.RGBShift(), A.HueSaturationValue()]),])
    #random.seed(42)
    #transformed = transform(image=image)
    #filename = os.path.join(annotations_new_savepath, '{}{}.jpg'.format(new_value6,remained_name))
    #cv2.imwrite(filename,transformed['image'])

    #transform = A.Compose([A.RandomCrop(320, 320), A.OneOf([A.RGBShift(), A.HueSaturationValue()]),])
    #random.seed(60)
    #transformed = transform(image=image)
    #filename = os.path.join(annotations_new_savepath, '{}{}.jpg'.format(new_value7,remained_name))
    #cv2.imwrite(filename,transformed['image'])
  



    #new_img = image
    # #cv2.imshow("Rotated (Problematic)", new_img)
    # filename = os.path.join(annotations_new_savepath, '{}{}.png'.format(new_value3,remained_name))
    # cv2.imwrite(filename,new_img)

    # new_img = image
    #cv2.imshow("Rotated (Problematic)", new_img)
    # filename = os.path.join(annotations_new_savepath, '{}{}.png'.format(new_value4,remained_name))
    # cv2.imwrite(filename,new_img)

    '''
    # loop over the rotation angles again, this time ensuring
    # no part of the image is cut off
    for angle in np.arange(0, 360, 15):
        rotated = imutils.rotate_bound(image, angle)
        cv2.imshow("Rotated (Correct)", rotated)
        cv2.waitKey(1000)
    '''

