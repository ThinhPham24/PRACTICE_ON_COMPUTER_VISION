import numpy as np
import os, os.path
import glob
import random
import matplotlib.pyplot as plt
import albumentations as A
import numpy as np
import argparse
import imutils
import cv2
import time
import aug
import re
import fnmatch
class ComplexExamplePipeline(aug.Pipeline):

    def __init__(self):
        super(ComplexExamplePipeline, self).__init__()
        self.seq1 = aug.Sequential(
            self.affine_ops(),
            aug.Choice(
                aug.Stretch(p=.5, x_scale=aug.uniform(.25, .5), y_scale=aug.uniform(.25, .5)),
                aug.Rotation(p=.5, angle=aug.truncnorm(0., 5., 5., 10.))),
            aug.GaussianBlur(p=1),
        )

        self.seq2 = aug.Sequential(aug.GaussianBlur(p=1), aug.GaussianBlur(p=1))

    def affine_ops(self):
        return aug.Sequential(
            aug.Stretch(p=.5, x_scale=aug.uniform(.25, .5), y_scale=aug.uniform(.25, .5)),
            aug.Rotation(p=.5, angle=aug.truncnorm(0., 5., 5., 10.)))

    def apply(self, sample):
        sample = self.seq1.apply(sample)
        sample = self.seq2.apply(sample)

        return sample
# -----------
class aug_library(aug.Pipeline):
    def __init__(self):
        super(aug_library, self).__init__()

        self.seq1 = aug.Sequential(
        aug.Choice(
                aug.LinearGradient(orientation="horizontal", edge_brightness=(.1, .3)),
                aug.LinearGradient(orientation="vertical", edge_brightness=(.2, .3)), 
                # aug.RatialGradient(orientation="vertical", edge_brightness=(.2, .3)), 
        ),
        aug.Brightness(p =1.,change = random.uniform(0.1, 0.4))
        )
    def apply(self, sample):
        sample = self.seq1.apply(sample)
        return  sample
    
class alb_library():
    def __init__(self):
        super(alb_library, self).__init__()
        self.list_mode = [self.saturation(),self.bright_ness(), self.contrast()]
    def bright_ness(self):
        range_radom = random.Random()
        light = A.RandomBrightness(limit= range_radom.uniform(-0.4, 0.4),always_apply=True,p=1)
        return light
    def contrast(self):
        range_radom = random.Random()
        light = A.RandomContrast(limit= range_radom.uniform(-0.2,0.2),always_apply=True,p=1)
        return light
    def saturation(sefl):
        range_radom = random.Random()
        light = A.ColorJitter(brightness=0,contrast= 0, saturation= range_radom.uniform(0, 10),hue=0, always_apply=True, p=1)
        return light
    def select_aug(self, new_image):
        x = random.Random()
        index = x.randint(0, 2)
        light= self.list_mode[index]
        random.seed(42)
        seq2 = light(image=new_image)
        return seq2, index
def adjust_gamma(image, gamma=1.0):
	# build a lookup table mapping the pixel values [0, 255] to
	# their adjusted gamma values
	invGamma = 1.0 / gamma
	table = np.array([((i / 255.0) ** invGamma) * 255
		for i in np.arange(0, 256)]).astype("uint8")
	# apply gamma correction using the lookup table
	return cv2.LUT(image, table)


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
def hsv2colorjitter(h, s, v):
    """Map HSV (hue, saturation, value) jitter into ColorJitter values (brightness, contrast, saturation, hue)"""
    return v, v, s, h
#  T += [A.ColorJitter(*hsv2colorjitter(hsv_h, hsv_s, hsv_v))]
def generate_path(path):
    if not os.path.exists(path):
        os.makedirs(path)
def generate_folder(path,name):
    path_add = path + '/' + "{}".format(name)
    if not os.path.exists(path_add):
        os.makedirs(path_add)
    return path_add
class seek_file_in_folder():
    def __init__(self,need_root):
        self.need_root = need_root
    def filter_for_jpeg(self, root, files):
        file_types = ['*.jpeg', '*.jpg']
        file_types = r'|'.join([fnmatch.translate(x) for x in file_types])
        files = [os.path.join(root, f) for f in files]
        files = [f for f in files if re.match(file_types, f)]   
        return files
    def filter_for_annotations(self, files, image_filename):
        file_types = ['*.png']
        file_types = r'|'.join([fnmatch.translate(x) for x in file_types])
        basename_no_extension = os.path.splitext(os.path.basename(image_filename))[0]
        file_name_prefix = basename_no_extension + '_' + '.*'
        #print("file_name_prefix = ", file_name_prefix)
        files = [os.path.join(self.need_root, f) for f in files]
        files = [f for f in files if re.match(file_types, f)]
        files = [f for f in files if re.match(file_name_prefix, os.path.splitext(os.path.basename(f))[0])]
        return files
class image_prosessing():
    def __init__(self):
        super(image_prosessing, self).__init__()
    def morphological_geometric(self,image):
        color_image = np.zeros((image.shape[1],image.shape[0],3),dtype=np.uint8)
        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        kernel = np.ones((3, 3), np.uint8)
        erode = cv2.erode(image_gray, kernel, iterations=1)
        kernel = np.ones((3, 3), np.uint8)
        opening = cv2.morphologyEx(erode, cv2.MORPH_OPEN, kernel, iterations=3)
        kernel_di = np.ones((3, 3), np.uint8)
        dilation = cv2.dilate(opening, kernel_di, iterations=3)  # 11
        kernel = np.ones((3, 3), np.uint8)
        closing = cv2.morphologyEx(dilation, cv2.MORPH_CLOSE, kernel, iterations=1)
        kernel = np.ones((3, 3), np.uint8)
        erode_second = cv2.erode(closing, kernel, iterations=4)
        result = cv2.bitwise_not(color_image,color_image,mask=erode_second)
        return result
    def color_radom(self,i): 
            r = random.randint(50,255)
            g = random.randint(50,100)
            b = random.randint(50,255)
            rgb = [r,g,b]
            return rgb
    def display_addweight(self, src_image, des_image, flag = True):
        if flag == True:
            image_color = src_image.copy()
            overlap = src_image.copy()
            # gray = cv2.cvtColor(des_image, cv2.COLOR_BGR2GRAY)
            gray = des_image
            ret, thresh = cv2.threshold(gray,127,255,cv2.THRESH_BINARY)
            cnts, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
            # for i in range(len(point_cnt)):cv2.fillPoly(image_backgorund, point_cnt[i], color_radom(len(point_cnt)))  
            # img_with_overlay = cv2.normalize(np.int64(image_color) * image_backgorund, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
            cv2.drawContours(des_image, cnts, -1, (0, 0, 255), 1) 
            for (i_x,cnt) in enumerate(cnts):
                cv2.fillPoly(overlap, [cnt] ,self.color_radom(i_x))
                img_with_overlay = cv2.normalize(np.int64(image_color) * overlap, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
                # cv2.addWeighted(image_color, 0.1, overlap, 0.9,0)
        else:
            alpha = 0.4
            beta = (1.0 - alpha)
            img_with_overlay = np.uint8(alpha*(src_image)+beta*(des_image))
        con_image = np.concatenate((img_with_overlay,src_image),axis = 1)  
        return con_image
    def save_image(self,path, name, image):
        filename_mask = os.path.join(path, f'{name}.png')
        cv2.imwrite(filename_mask,image)
    def calculate_area(self, image):
        if len(image.shape) > 2 :
            gray_image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
            _, thresholded_image = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY)
        else: 
            thresholded_image = gray_image 
        # Find contours in the thresholded image
        contours, _ = cv2.findContours(thresholded_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # Find max contours
        max_contour = max(contours, key = cv2.contourArea)
        # Find the area of object
        area = cv2.contourArea(max_contour)
        return (area,max_contour)
    def draw_contour(self,image,contour,text,area):
        i = random.randint(1,255)
        # cv2.drawContours(image,[contour],-1,self.color_radom(i),1)
        x,y,w,h = cv2.boundingRect(contour)
        # draw the book contour (in green)
        cv2.rectangle(image,(x,y),(x+w,y+h),self.color_radom(i),2)
        # Put text on image
        cv2.putText(image,text=text,org=(x,y-20),fontFace=cv2.FONT_ITALIC,
                    fontScale=1,color=self.color_radom(i),thickness=3,lineType=cv2.LINE_8)
        area_txt = str(np.round(area,0))
        cv2.putText(image,text=area_txt,org=(x+50,y-20),fontFace=cv2.FONT_ITALIC,
                    fontScale=1,color=self.color_radom(i),thickness=2,lineType=cv2.LINE_AA)
        return image

    
        
