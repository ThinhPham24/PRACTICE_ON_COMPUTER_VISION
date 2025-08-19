


#***************
import os
import re
import fnmatch
import argparse
import glob

import random
import cv2
import numpy as np
def resized_img(img,percent):
    width = int(img.shape[1] * percent / 100)
    height = int(img.shape[0] * percent / 100)
    dim = (width, height)
    # resize image
    resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
    return resized
#***************
# folder = "/TANG/"
# annotations_dir = "annotations"
# current_dir = os.getcwd()
# path= ''.join([current_dir, folder])

# annotations_path = os.path.join(path, annotations_dir)

# print("annotations_savepath ",annotations_path )
# print("annotations_savepath = ", annotations_savepath)
# if not os.path.isdir(os.path.abspath(annotations_path)):
#     os.mkdir(annotations_path)
#*********************
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
    file_name_prefix = basename_no_extension + '.*'
    files = [os.path.join(root, f) for f in files]
    files = [f for f in files if re.match(file_types, f)]
    files = [f for f in files if re.match(file_name_prefix, os.path.splitext(os.path.basename(f))[0])]
    return files
# constants
def magic(numList):
    s = ''.join(map(str, numList))
    return int(s)
def find_annotation(image_filename,TRAIN_ANNOTATION_DIR):
    total_angle = []
    for root, _, files in os.walk(TRAIN_ANNOTATION_DIR):
        annotation_files = filter_for_annotations(root, files, image_filename)
        return annotation_files
def color_radom(i): 
    r = random.randint(50,255)
    g = random.randint(50,100)
    b = random.randint(50,255)
    rgb = [r,g,b]
    return rgb
if __name__ =="__main__":
    ''' Run code: 
    python check_only_annotation_without_angle.py
    '''
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--path_img', default= "/home/airlab/Desktop/DATA/IMAGE_ANNOTATION", type=str, required=False,
                        help= 'choose the function to check annotation or json')
    parser.add_argument('--path_ann', default= "/home/airlab/Desktop/DATA/IMAGE_ANNOTATION", type=str, required=False,
                        help= 'choose the function to check annotation or json')               
    args = parser.parse_args()

    TRAIN_ANNOTATION_DIR = args.path_ann
    print("this is path", args.path_img)
    filenames = sorted(glob.glob(args.path_img + "/*.jpg")) #read all files in the path mentioned
    number_image = 1
    segmentation_id = 1
    for index_1,image_name in enumerate(filenames):
        print("image_filename:",image_name)            
        image_original = cv2.imread(image_name)
        image_color = cv2.imread(image_name)
        file, ext = os.path.splitext(image_name)
        file = str(file).split('/')[-1]
        annotation_files = find_annotation(image_filename=image_name, TRAIN_ANNOTATION_DIR=TRAIN_ANNOTATION_DIR)
        print("annotation_file:",annotation_files)
        angles, points, lengths = [], [], []
        overlay = image_color.copy()
        image_only_contour = overlay.copy()
        for i in range(len(annotation_files)):
            # print("current mask", annotation_files[i])
            image_mask = cv2.imread(annotation_files[i])
            gray = cv2.cvtColor(image_mask, cv2.COLOR_BGR2GRAY)
            ret, thresh = cv2.threshold(gray,127,255,cv2.THRESH_BINARY)
            cnts, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
            # for i in range(len(point_cnt)):cv2.fillPoly(image_backgorund, point_cnt[i], color_radom(len(point_cnt)))  
            # img_with_overlay = cv2.normalize(np.int64(image_color) * image_backgorund, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
            cv2.drawContours(image_only_contour, cnts, -1, (0, 0, 255), 1) 
            for (i_x,cnt) in enumerate(cnts):
                cv2.fillPoly(overlay, [cnt] ,color_radom(i))
                cv2.addWeighted(overlay, 0.5, image_color, 1 - 0.5,0,image_color)
            cv2.drawContours(image_color, cnts, -1, (0, 0, 0), 1) 
        concate_img = np.concatenate((image_color,image_only_contour),axis=1)
        image_white = np.zeros((960,concate_img.shape[1],3),np.uint8)
        image_white.fill(2555)
        image_white[:,int(int(concate_img.shape[1]/2)/2):int(int(concate_img.shape[1]/2)/2) + int(int(concate_img.shape[1]/2))] = image_original
        concate_img_v = np.concatenate((image_white,concate_img),axis = 0)
        print("number of image:", number_image)
        number_image += 1
        # cv2.imwrite(check_annotation_path + '/'+ f"{file}_check.png",concate_img_v)
        cv2.imshow("mask_image", resized_img(concate_img,30))
        k = cv2.waitKey(0)
        if k ==ord('q') or k ==ord('Q'):
            break
    print("CHECKING ANNOTAION IS TO BE FINISH!")

