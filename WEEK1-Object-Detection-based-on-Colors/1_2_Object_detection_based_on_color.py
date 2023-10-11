''' 
Author: Master. Pham The Thinh
Title: OBJECT DETECTION BASED ON COLORS - Phát hiện đối tượng dựa vào màu sắc
'''
#---------------------------
# Import the nesseary library : Khai báo các thư viện cần thiết để sử dụng trong chương trình
import cv2 
import os
import numpy as np 
import imutils
import argparse
from utils import img_process, resize_image
#-----------------------------      
# MAIN PROGRAM: CHƯƠNG TRÌNH CHÍNH
if __name__ =="__main__":

    parser = argparse.ArgumentParser(description='Code for Thresholding Operations using inRange tutorial.')
    parser.add_argument('--path_image', help='Path of detected image', default="C:\\Users\\ptthi\\OneDrive\\Desktop\\PRACTICE_ON_COMPUTER_VISION\\WEEK1-Object-Detection-based-on-Colors\\images\\redball.png", type=str)
    args = parser.parse_args()
    # READ IMAGE: Đọc ảnh
    img = cv2.imread(args.path_image)
    # Find object based on HSV value  (Object detection based on color): Tìm các đối tượng theo ngưỡng
    mask, imgproc = img_process(img)
    cv2.imshow("imgproc:", imgproc)
    cv2.imshow("image_threshold", mask)
    inv_mask =  np.invert(imgproc)
    # BITWISE METHOD: Sử dụng phương pháp and hai ảnh
    cnts,_ = cv2.findContours(imgproc,cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    max_cnt = max(cnts,key=cv2.contourArea)
    # Calculating the area, Perimeter and finding center of object from contour
    area = round(cv2.contourArea(max_cnt),0)
    perimeter = round(cv2.arcLength(max_cnt,True),0)
    M = cv2.moments(max_cnt)
    cx = int(M['m10']/M['m00'])
    cy = int(M['m01']/M['m00'])
    cv2.circle(img,(cx,cy),2,(0,255,0),1)
    cv2.putText(img,"red ball",(cx,cy),cv2.FONT_HERSHEY_COMPLEX,1,(10,100,20),2,cv2.LINE_4)
    cv2.putText(img,"area: {}".format(area),(cx-30,cy-50),cv2.FONT_HERSHEY_COMPLEX,1,(10,100,200),2,cv2.LINE_8)
    cv2.putText(img,"p: {}".format(perimeter),(cx-30,cy-30),cv2.FONT_HERSHEY_COMPLEX,1,(100,100,200),2,cv2.LINE_8)
    print("perimeter", perimeter)
    cv2.drawContours(img, [max_cnt],-1,(255,0,0),1)
    red=cv2.bitwise_and(img,img,mask=imgproc)
    # Show the image: Hiển thị ảnh
    img =resize_image(img,200)
    cv2.imshow("image", img)
    key = cv2.waitKey(0)
    
    