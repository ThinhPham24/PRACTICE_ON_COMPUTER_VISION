import cv2
import os
import numpy as np
cv_file_top = cv2.FileStorage('./Calibrated1/'  + '0degree/stereoMap.txt',cv2.FileStorage_READ)
# cv_file.open('/home/airlab/Desktop/Transformation_matrix/result_image/Calibrated/' + str(address) + '/stereoMap.txt',cv2.FileStorage_READ)
stereoMapR_x_top = cv_file_top.getNode('stereoMapR_x').mat()
stereoMapR_y_top= cv_file_top.getNode('stereoMapR_y').mat()
stereoMapL_x_top = cv_file_top.getNode('stereoMapL_x').mat()
stereoMapL_y_top = cv_file_top.getNode('stereoMapL_y').mat()
Q1_top = cv_file_top.getNode('q').mat()
print("Q",Q1_top)