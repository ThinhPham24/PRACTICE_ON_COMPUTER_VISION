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