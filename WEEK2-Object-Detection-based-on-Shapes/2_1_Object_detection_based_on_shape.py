''' 
Author: MsC. Pham The Thinh
Title: OBJECT DETECTION BASED ON SHAPE - Phát hiện đối tượng dựa vào hình dạng
'''
#---------------------------
# Import the nesseary library : Khai báo các thư viện cần thiết để sử dụng trong chương trình
import cv2 
import os
import numpy as np 
import imutils
import argparse
from utils import detect_shape, resize_image, draw_information,detect_color
#-----------------------------      
# MAIN PROGRAM: CHƯƠNG TRÌNH CHÍNH
if __name__ =="__main__":
    parser = argparse.ArgumentParser(description='Code for Thresholding Operations using inRange tutorial.')
    parser.add_argument('--path_image', help='Path of detected image', default="C:\\Users\\ptthi\\OneDrive\\Desktop\\PRACTICE_ON_COMPUTER_VISION\\WEEK2-Object-Detection-based-on-Shapes\\images\\example.png", type=str)
    args = parser.parse_args()
    # Set HSV parameters
    # label_color = {"Red":[0,10,240,255,0,255],"Green":[15,80,250,255,0,255],"Sky_blue":[70,110,240,255,0,255],
    #                "Purple":[100,175,240,255,0,255],"Yellow":[22,44,80,255,0,255],"Orange":[14,23,143,255,0,255]}
    label_color = {"name": ["Red", "Green","Cyan","Purple","Yellow","Orange"],"value":[[0,10,240,255,0,255],[15,80,250,255,0,255],[70,110,240,255,0,255],
                                                                                         [100,175,240,255,0,255],[22,44,80,255,0,255],[14,23,143,255,0,255]]}
    color = {"Red": (0,0,255), "Green":(0,255,0),"Cyan": (2555,255,0),"Purple":(128,0,128), "Yellow": (0,255,255), "Orange": (0, 165, 255)}
    # READ IMAGE: Đọc ảnh
    img = cv2.imread(args.path_image)
    # Detect the color
    red_cnt, red_pts = detect_color(img,3,label_color["value"][0])
    draw_information(img,label_color["name"][0],red_pts,None,None,color["Red"],2)
    green_cnt, green_pts = detect_color(img,3,label_color["value"][1])
    draw_information(img,label_color["name"][1],green_pts,None,None,color["Red"],2)
    sky_cnt, sky_pts = detect_color(img,3,label_color["value"][2])
    draw_information(img,label_color["name"][2],sky_pts,None,None,color["Red"],2)
    purple_cnt, purple_pts = detect_color(img,3,label_color["value"][3])
    draw_information(img,label_color["name"][3],purple_pts,None,None,color["Red"],2)
    yellow_cnt, yellow_pts = detect_color(img,3,label_color["value"][4])
    draw_information(img,label_color["name"][4],yellow_pts,None,None,color["Red"],2)
    orange_cnt, orange_pts = detect_color(img,3,label_color["value"][5])
    draw_information(img,label_color["name"][5],orange_pts,None,None,color["Red"],2)
    # Detect the shape
    out = detect_shape(img,17)
    # Draw the detected objects
    for i in range(len(out["name"])):
        draw_information(img,out["name"][i],out["center_pts"][i],out["center_pts"][i],out["shape"][i],(0,0,0),2)
    cv2.imshow("image", resize_image(img,70))
    key = cv2.waitKey(0)
    
    