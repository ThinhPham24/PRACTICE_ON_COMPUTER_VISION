''' 
Author: Ms. Pham The Thinh
Title: Tìm các giá trị giới hạn của không gian màu HSV
'''

# Import the nesseary library : Khai báo các thư viện cần thiết để sử dụng trong chương trình
import cv2 
import argparse
import numpy as np
import os
import glob
#------------------------------
# Import user-defined function : Gọi thư viện hàm phục vụ chương trình chính 
from utils import resize_image
#---------------------------------
# Khai báo các giá trị cho trước H, S,V
max_value = 255
max_value_H = 360 // 2
low_H = 0
low_S = 0
low_V = 0
high_H = max_value_H
high_S = max_value
high_V = max_value
#-----------------------------
# Create window name: Tạo cửa sổ hiện thị tên là Image Capture và Object Detection
window_image_name = 'Image Capture'
window_detection_name = 'Object Detection'
cv2.namedWindow(window_image_name,cv2.WINDOW_NORMAL)
cv2.namedWindow(window_detection_name,cv2.WINDOW_NORMAL)
#-----------------------------
# Gán các tên biến
low_H_name = 'Low H'
low_S_name = 'Low S'
low_V_name = 'Low V'
high_H_name = 'High H'
high_S_name = 'High S'
high_V_name = 'High V'
#------------SUBFUNCTION----------------------
## Subfunction: Chương trình con phục vụ chương trình chính tìm ngưỡng của H
def on_low_H_thresh_trackbar(val):
    global low_H
    global high_H
    low_H = val
    low_H = min(high_H - 1, low_H)
    cv2.setTrackbarPos(low_H_name, window_detection_name, low_H)
def on_high_H_thresh_trackbar(val):
    global low_H
    global high_H
    high_H = val
    high_H = max(high_H, low_H + 1)
    cv2.setTrackbarPos(high_H_name, window_detection_name, high_H)
## Subfunction: Chương trình con phục vụ chương trình chính tìm ngưỡng của S
def on_low_S_thresh_trackbar(val):
    global low_S
    global high_S
    low_S = val
    low_S = min(high_S - 1, low_S)
    cv2.setTrackbarPos(low_S_name, window_detection_name, low_S)
def on_high_S_thresh_trackbar(val):
    global low_S
    global high_S
    high_S = val
    high_S = max(high_S, low_S + 1)
    cv2.setTrackbarPos(high_S_name, window_detection_name, high_S)
## Subfunction: Chương trình con phục vụ chương trình chính tìm ngưỡng của V
def on_low_V_thresh_trackbar(val):
    global low_V
    global high_V
    low_V = val
    low_V = min(high_V - 1, low_V)
    cv2.setTrackbarPos(low_V_name, window_detection_name, low_V)
def on_high_V_thresh_trackbar(val):
    global low_V
    global high_V
    high_V = val
    high_V = max(high_V, low_V + 1)
    cv2.setTrackbarPos(high_V_name, window_detection_name, high_V)

#---------------------------------------
# Gán các giá trị khi chạy chương trình
'''
We’ve added the add_argument() method, which is what we use to specify which command-line options the program is willing to accept. In this case, I’ve named it echo so that it’s in line with its function.

Calling our program now requires us to specify an option.

The parse_args() method actually returns some data from the options specified, in this case, echo.

The variable is some form of ‘magic’ that argparse performs for free (i.e. no need to specify which variable that value is stored in). You will also notice that its name matches the string argument given to the method, echo.'''

parser = argparse.ArgumentParser(description='Code for Thresholding Operations using inRange tutorial.')
parser.add_argument('--path_image', help='Path of detected image', default="C:\\Users\\ptthi\\OneDrive\\Desktop\\PRACTICE_OF_COMPUTER_VISION\\Object-Detection-based-on-Colors\\images\\ballread.png", type=str)
args = parser.parse_args()
#-------------------------------------------------
# Read image using OpenCV libary: Đọc ảnh sử dụng thư viện của OPENCV
image_or = cv2.imread(args.path_image)
# Resize image: 
image = resize_image(image_or, 100)
#----------------
# Create trackbar: Tạo thanh trượt
cv2.createTrackbar(low_H_name, window_detection_name, low_H, max_value_H, on_low_H_thresh_trackbar)
cv2.createTrackbar(high_H_name, window_detection_name, high_H, max_value_H, on_high_H_thresh_trackbar)
cv2.createTrackbar(low_S_name, window_detection_name, low_S, max_value, on_low_S_thresh_trackbar)
cv2.createTrackbar(high_S_name, window_detection_name, high_S, max_value, on_high_S_thresh_trackbar)
cv2.createTrackbar(low_V_name, window_detection_name, low_V, max_value, on_low_V_thresh_trackbar)
cv2.createTrackbar(high_V_name, window_detection_name, high_V, max_value, on_high_V_thresh_trackbar)
# -----------------
# Chương trình chính vòng lặp
while True:
    # If no image, program breaks
    if image is None:
        break
    # Convert color space from BGR to HSV --> Đổi không gian màu từ BGR thành HSV
    image_HSV = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # Keep the objects that correspond with threshold of H,S,V value: Tìm các đối tượng tương ứng ngưỡng giá trị của H,S,V
    image_threshold = cv2.inRange(image_HSV, (low_H, low_S, low_V), (high_H, high_S, high_V))
    # Convert color space from GRAY to BGR --> Đổi không gian màu từ GRAY thành BGR
    image_cvt = cv2.cvtColor(image_threshold,cv2.COLOR_GRAY2BGR)
    # Merge two images along x: hợp hai ảnh theo phương ngang
    image_merge = np.concatenate((image,image_cvt), axis =1)
    # cv2.imshow(window_capture_name, image)
    # Show the image: Hiển thị kết quả
    cv2.imshow(window_detection_name, image_merge)
    cv2.imshow(window_image_name, image_merge)
    key = cv2.waitKey(30)
    if key == ord('q') or key == 27:
        break