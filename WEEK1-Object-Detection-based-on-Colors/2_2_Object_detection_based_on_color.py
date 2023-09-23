''' 
Author: Ms. Pham The Thinh
Title: OBJECT DETECTION BASED ON COLORS - Phát hiện đối tượng dựa vào màu sắc
'''
#---------------------------
# Import the nesseary library : Khai báo các thư viện cần thiết để sử dụng trong chương trình
import cv2 as cv
import os
import numpy as np 
import imutils
import argparse
from utils import img_process
#-----------------------------      
# MAIN PROGRAM: CHƯƠNG TRÌNH CHÍNH
if __name__ =="__main__":

    parser = argparse.ArgumentParser(description='Code for Thresholding Operations using inRange tutorial.')
    parser.add_argument('--path_image', help='Path of detected image', default="C:\\Users\\ptthi\\OneDrive\\Desktop\\PRACTICE_OF_COMPUTER_VISION\\Object-Detection-based-on-Colors\\images\\ballread.png", type=str)
    args = parser.parse_args()
    # READ IMAGE: Đọc ảnh
    img = cv.imread(args.path_image)
    # Find object based on HSV value  (Object detection based on color): Tìm các đối tượng theo ngưỡng
    imgproc = img_process(img)
    inv_mask =  np.invert(imgproc)
    # BITWISE METHOD: Sử dụng phương pháp and hai ảnh
    red=cv.bitwise_and(img,img,mask=inv_mask)
    # Show the image: Hiển thị ảnh
    cv.imshow("image", red)
    key = cv.waitKey(0)
    
    