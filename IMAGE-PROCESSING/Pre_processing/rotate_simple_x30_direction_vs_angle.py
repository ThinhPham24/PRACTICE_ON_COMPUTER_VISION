# Step Oder, Rotate, Blur, Flip. 
# USAGE
# step 1: python rotate_simple_x30_direction_vs_angle.py 
# step 2: python flip.py
# step 3: python augmentation.py
# import the necessary packages
import numpy as np
import argparse
import imutils
import cv2
import os, os.path
import glob

#*********************** parameters *****************************#
# Offet value - change manually
offset = 1000

# No. of new image from one original image
# k = 4; # So luong anh muon tao ra tu anh goc
# k = 3; # So luong anh muon tao ra tu anh goc
k = 30

# Input folder
folder = "/THINH_1/"
#*********************** parameters *****************************#

rename_list = []
remained_list = []
check = 0

current_dir = os.getcwd()
path = ''.join([current_dir, folder])
print("path = ", path)

#type 0:train, 1:test
type = 0

if type ==0:    #train folder
    #-----------------------------------------------------------------------#
    images_dir = "IMG/scale_train"
    annotations_dir = "ANNOTATION/scale_train"
    images_path = os.path.join(path, images_dir)
    annotations_path = os.path.join(path, annotations_dir)
    # if not os.path.isdir(os.path.abspath(annotations_path)):
    #     os.mkdir(annotations_path)
    #-----------------------------------------------------------------------#
    #-----------------------------------------------------------------------#
    new_image_save = "IMG/rot_train"
    new_annotations_save = "ANNOTATION/rot_train"
    images_new_savepath = os.path.join(path, new_image_save)
    annotations_new_savepath = os.path.join(path, new_annotations_save)
    if not os.path.isdir(os.path.abspath(images_new_savepath)):
        os.mkdir(images_new_savepath)
    if not os.path.isdir(os.path.abspath(annotations_new_savepath)):
        os.mkdir(annotations_new_savepath)
    #-----------------------------------------------------------------------#
    # images_path = os.path.join(path, images_dir)
    # annotations_savepath = os.path.join(path, annotations_dir)
    # print("images_path = ", images_path)
    # if not os.path.isdir(os.path.abspath(annotations_savepath)):
    #     os.mkdir(annotations_savepath)

if type==1:     #test folder
    #-----------------------------------------------------------------------#
    images_dir = "IMG/validate"
    annotations_dir = "ANNOTATION/validate"
    images_path = os.path.join(path, images_dir)
    annotations_path = os.path.join(path, annotations_dir)
    # if not os.path.isdir(os.path.abspath(annotations_savepath)):
    #     os.mkdir(annotations_savepath)
    #-----------------------------------------------------------------------#
    #-----------------------------------------------------------------------#
    new_image_save = "IMG/rot_validate"
    new_annotations_save = "ANNOTATION/rot_validate"
    images_new_savepath = os.path.join(path, new_image_save)
    annotations_new_savepath = os.path.join(path, new_annotations_save)
    if not os.path.isdir(os.path.abspath(images_new_savepath)):
        os.mkdir(images_new_savepath)
    if not os.path.isdir(os.path.abspath(annotations_new_savepath)):
        os.mkdir(annotations_new_savepath)
    #-----------------------------------------------------------------------#
    # images_path = os.path.join(path, images_dir)
    # annotations_savepath = os.path.join(path, annotations_dir)
    # print("images_path = ", images_path)
    # if not os.path.isdir(os.path.abspath(annotations_savepath)):
    #     os.mkdir(annotations_savepath)



# constants
def magic(numList):
    s = ''.join(map(str, numList))
    return int(s)

   
filenames = glob.glob(images_path + "/*.jpg") #read all files in the path mentioned
for n, image_file in enumerate(filenames):
    file, ext = os.path.splitext(image_file)  # split filename and extension
    name = os.path.basename(file)
    s = len(os.path.basename(file))

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
    new_value = num_modify + offset
    rename_list = []
    remained_list = []
    check = 0

    file, ext = os.path.splitext(image_file)  # split filename and extension
    #image = cv2.imread(image_file.path)
    image = cv2.imread(image_file) # 
    #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # loop over the rotation angles
    for i in range(k):
        if k == 3:
            angle = (i+1)*90
        if k == 4: 
            angle = (i+1)*90+45
        if k == 30: 
            angle = (i+1)*12
        rotated = imutils.rotate(image, angle)
        #cv2.imshow("Rotated (Problematic)", rotated)
        filename = os.path.join(images_new_savepath, '{}{}.jpg'.format(new_value+i,remained_name))
        cv2.imwrite(filename,rotated)
        #cv2.waitKey(1000)

filenames = glob.glob(annotations_path + "/*.png") #read all files in the path mentioned
for n, image_file in enumerate(filenames):
    check = 0
    angle_list = []
    file, ext = os.path.splitext(image_file)  # split filename and extension
    name = os.path.basename(file)
    s = len(os.path.basename(file))
    #print("name's shape:",name)
    #print("Length:",s)
    l = list(name)

    for i in range(s):
        if l[i] != '_' and check==0:
            rename_list.append(l[i])
        elif l[i] == '_':
            check = check + 1  
            remained_list.append(l[i])
        elif l[i] != '_' and check==3:
            angle_list.append(l[i])
        else:
            remained_list.append(l[i])

    #print("rename_int = ", rename_int)
    #print("remained_list = ", remained_list)
    #print("angle_list = ", angle_list)
    remained_name = ''.join(map(str, remained_list))

    #----------------------------------------------#
    # Prefix
    rename_int = list(map(int, rename_list))
    num = magic(rename_int)
    num_modify = num*k
    new_value = num_modify + offset
    #----------------------------------------------#
    # Suffix
    angle_int = list(map(int, angle_list))
    angle_ori = magic(angle_int)
    #----------------------------------------------#
    rename_list = []
    remained_list = []



    file, ext = os.path.splitext(image_file)  # split filename and extension
    #image = cv2.imread(image_file.path)
    image = cv2.imread(image_file) # 
    #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # loop over the rotation angles
    for i in range(k):
        if k == 3:
            angle = (i+1)*90
        if k == 4: 
            angle = (i+1)*90+45
        if k == 30: 
            angle = (i+1)*12
        rotated = imutils.rotate(image, angle)
        #----------- New angle calculation ------------------#
        new_angle = angle + angle_ori
        #------------Change here-----------------------------#
        while (new_angle > 360):
            new_angle = new_angle - 360 
        #----- End of New angle calculation -----------------#
        filename = os.path.join(annotations_new_savepath, '{}{}{}.png'.format(new_value+i,remained_name,new_angle))
        cv2.imwrite(filename,rotated)



