import numpy as np
import cv2
import os, os.path
import glob
folder = "/THINH_1/"
current_dir = os.getcwd()
path = ''.join([current_dir, folder])
print("path = ", path)
#type 0:train, 1:test
type= 0

if type==0:
    #-----------------------------------------------------------------------#
    images_dir = "IMG/train"
    annotations_dir = "ANNOTATION/train"
    images_path = os.path.join(path, images_dir)
    annotations_savepath = os.path.join(path, annotations_dir)
    if not os.path.isdir(os.path.abspath(annotations_savepath)):
        os.mkdir(annotations_savepath)
    #-----------------------------------------------------------------------#
    new_image_save = "IMG/train_resize"
    new_annotations_save = "ANNOTATION/train_resize"
    images_new_savepath = os.path.join(path, new_image_save)
    annotations_new_savepath = os.path.join(path, new_annotations_save)
    if not os.path.isdir(os.path.abspath(images_new_savepath)):
        os.mkdir(images_new_savepath)
    if not os.path.isdir(os.path.abspath(annotations_new_savepath)):
        os.mkdir(annotations_new_savepath)
    #-----------------------------------------------------------------------#
if type==1:
    #-----------------------------------------------------------------------#
    images_dir = "IMG/validate"
    annotations_dir = "ANNOTATION/validate"
    images_path = os.path.join(path, images_dir)
    annotations_savepath = os.path.join(path, annotations_dir)
    if not os.path.isdir(os.path.abspath(annotations_savepath)):
        os.mkdir(annotations_savepath)
    #-----------------------------------------------------------------------#
    new_image_save = "IMG/validate_resize"
    new_annotations_save = "ANNOTATION/validate_resize"
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
# random_file = random.sample(filenames,int(len(filenames)*0.5))
# print("random", len(random_file))
size = (640, 640)
for n, image_file in enumerate(filenames):
    file, ext = os.path.splitext(image_file)  # split filename and extension
    name = os.path.basename(file)
    #image = cv2.imread(image_file.path)
    new_img = cv2.imread(image_file) # Thay đổi tương ứng!
    re_img = cv2.resize(new_img, size)
 
    # loop over the rotation angles
    filename = os.path.join(images_new_savepath, '{}.jpg'.format(name))
    cv2.imwrite(filename,re_img)
#-----------------------------APPLY MASK------------------------------------#
filenames = glob.glob(annotations_savepath + "/*.*") #read all files in the path mentioned
for n, image_file in enumerate(filenames):
    file, ext = os.path.splitext(image_file)  # split filename and extension
    name = os.path.basename(file)
    #image = cv2.imread(image_file.path)
    new_img = cv2.imread(image_file) # Thay đổi tương ứng!
    re_img = cv2.resize(new_img, size)
    # loop over the rotation angles
    filename = os.path.join(annotations_new_savepath, '{}.png'.format(name))
    cv2.imwrite(filename,re_img)