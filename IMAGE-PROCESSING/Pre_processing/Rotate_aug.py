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
import random
import re
import fnmatch
#*********************** parameters *****************************#
# Offet value - change manually
offset = 1000

# No. of new image from one original image
# k = 4; # So luong anh muon tao ra tu anh goc
# k = 3; # So luong anh muon tao ra tu anh goc
k = 30

# Input folder
folder = "/DATA/"
#*********************** parameters *****************************#

rename_list = []
remained_list = []
check = 0

current_dir = os.getcwd()
path = ''.join([current_dir, folder])
print("path = ", path)

#type 0:train, 1:test
type = 1

if type ==0:    #train folder
    #-----------------------------------------------------------------------#
    images_dir = "IMG/train"
    annotations_dir = "ANNOTATION/train"
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
    new_image_save = "IMG/rot_val"
    new_annotations_save = "ANNOTATION/rot_val"
    images_new_savepath = os.path.join(path, new_image_save)
    annotations_new_savepath = os.path.join(path, new_annotations_save)
    if not os.path.isdir(os.path.abspath(images_new_savepath)):
        os.mkdir(images_new_savepath)
    if not os.path.isdir(os.path.abspath(annotations_new_savepath)):
        os.mkdir(annotations_new_savepath)
    #-----------------------------------------------------------------------#

# constants
def magic(numList):
    s = ''.join(map(str, numList))
    return int(s)
def filter_for_jpeg(root, files):
    file_types = ['*.jpeg', '*.jpg']
    file_types = r'|'.join([fnmatch.translate(x) for x in file_types])
    files = [os.path.join(root, f) for f in files]
    files = [f for f in files if re.match(file_types, f)]
    
    return files

def filter_for_annotations(root, files, image_filename):
    # print("image_filename:",image_filename)
    file_types = ['*.png']
    file_types = r'|'.join([fnmatch.translate(x) for x in file_types])
    basename_no_extension = os.path.splitext(os.path.basename(image_filename))[0]
    file_name_prefix = basename_no_extension + '_' + '.*'
    files = [os.path.join(root, f) for f in files]
    files = [f for f in files if re.match(file_types, f)]
    files = [f for f in files if re.match(file_name_prefix, os.path.splitext(os.path.basename(f))[0])]
    return files
def find_annotation(image_filename,TRAIN_ANNOTATION_DIR):
    for root, _, files in os.walk(TRAIN_ANNOTATION_DIR):
        annotation_files = filter_for_annotations(root, files, image_filename)
        # print("annotation_files:",annotation_files)
        # go through each associated annotation
        return annotation_files
if __name__ == "__main__":
    filenames = glob.glob(images_path + "/*.jpg") #read all files in the path mentioned

    for n, image_file in enumerate(filenames):
        file, ext = os.path.splitext(image_file)  # split filename and extension
        name = os.path.basename(file)
        filenames_an = find_annotation(image_filename=image_file, TRAIN_ANNOTATION_DIR = annotations_path)
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
                angle = (i+1)*12 + random.randint(-8,8)
            # # # --------------ANNOTAION FILE------------
            for n, image_file_an in enumerate(filenames_an):
                check_an = 0
                angle_list_an = []
                file_an, ext = os.path.splitext(image_file_an)  # split filename and extension
                name_an = os.path.basename(file_an)
                s_an = len(os.path.basename(file_an))
                #print("name's shape:",name)
                l_an = list(name_an)
                rename_list_an = []
                remained_list_an = []
                for j in range(s_an):
                    if l_an[j] != '_' and check_an==0:
                        rename_list_an.append(l_an[j])
                    elif l_an[j] == '_':
                        check_an = check_an + 1  
                        remained_list_an.append(l_an[j])
                    elif l_an[j] != '_' and check_an==3:
                        angle_list_an.append(l_an[j])
                    else:
                        remained_list_an.append(l_an[j])
                remained_name_an = ''.join(map(str, remained_list_an))
                # Suffix
                angle_int_an = list(map(int, angle_list_an))
                angle_ori_an = magic(angle_int_an)
                #----------------------------------------------#
                rename_list_an = []
                remained_list_an = []
                file_an, ext = os.path.splitext(image_file_an)  # split filename and extension
                #image = cv2.imread(image_file.path)
                image_an = cv2.imread(image_file_an) # 
                #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                rotated_an = imutils.rotate(image_an, angle)
                #----------- New angle calculation ------------------#
                new_angle_an = angle + angle_ori_an
                #------------Change here-----------------------------#
                while (new_angle_an > 360):
                    new_angle_an = new_angle_an - 360 
                #----- End of New angle calculation -----------------#
                filename_an_re = os.path.join(annotations_new_savepath, '{}{}{}.png'.format(new_value+i,remained_name_an,new_angle_an))
                cv2.imwrite(filename_an_re,rotated_an)
            # print("ANGLE:",angle)
            rotated = imutils.rotate(image, angle)
            #cv2.imshow("Rotated (Problematic)", rotated)
            filename = os.path.join(images_new_savepath, '{}{}.jpg'.format(new_value+i,remained_name))
            cv2.imwrite(filename,rotated)
            #cv2.waitKey(1000)
           

