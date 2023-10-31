import cv2
import numpy as np
import os

def resize_image(image, percent):
    x = int((image.shape[1]*percent)/100)
    y = int((image.shape[0]*percent)/100)
    image_resize = cv2.resize(image,(x, y), interpolation = cv2.INTER_LINEAR)
    return image_resize
def img_proc(img, kernel_size, flag, range_HSV):
    # Take color-based threshold
    if flag == 1:
        ### Binary by HSV inrange ###
        # low_H, low_S, low_V, high_H, high_S, high_V = [0, 77, 185, 180, 255, 255]
        blurred = cv2.GaussianBlur(img, (5, 5), 0)
        low_H, high_H, low_S, high_S, low_V, high_V = range_HSV
        image_HSV = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
        image_threshold = cv2.inRange(image_HSV, (low_H, low_S, low_V), (high_H, high_S, high_V))
    #Take gray-based threshold
    else:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # blurred = cv2.GaussianBlur(gray, (7, 7), 0)
        ret, image_threshold = cv2.threshold(gray,240,255,cv2.THRESH_BINARY_INV)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    frame_erode = cv2.erode(image_threshold, kernel)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    frame_dilate = cv2.dilate(frame_erode, kernel)
    closing = cv2.morphologyEx(frame_dilate,cv2.MORPH_CLOSE, kernel)
    return image_threshold, closing
def add_information(label,name,ctpts,shape):
    label['name'].append(name)
    label['center_pts'].append(ctpts)
    label['shape'].append(shape)
    return label
def detect_shape(image,size_kernel):
    _,closing = img_proc(image,size_kernel,0,None)
    cnts, _ = cv2.findContours(closing, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
    label = {"name":[], "center_pts" : [], "shape": []}
    for cnt in cnts:
        epsilon = 0.01*cv2.arcLength(cnt, True)
        approx=cv2.approxPolyDP(cnt,epsilon,True)
        M=cv2.moments(cnt)
        cx=int(M['m10']/ M['m00'])
        cy=int(M['m01']/M['m00'])
        if len(approx) == 3:
            add_information(label,"Triangle",(cx,cy),cnt)
        elif len(approx) == 4:
            x,y,w,h=cv2.boundingRect(cnt)
            ratio = float(h/w)
            if 0.95 < ratio < 1.1:
                add_information(label,"Squard",(int((x+w)/2),int((y+h)/2)),cnt)
            else:
                add_information(label,"Rectangle",(cx,cy),cnt)
        elif len(approx) == 5:
            add_information(label,"Pentagon",(cx,cy),cnt)
        elif 6 < len(approx) <= 15:
            add_information(label,"Ellipse",(cx,cy),cnt)
        else:
            add_information(label,"Circle",(cx,cy),cnt)
    return label
def detect_color(image, size_kernel,color):
    _,out = img_proc(image,size_kernel,1,color)
    cnts,_ = cv2.findContours(out,cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE) 
    cnt = max(cnts, key=cv2.contourArea)
    M=cv2.moments(cnt)
    cx=int(M['m10']/ M['m00'])
    cy=int(M['m01']/M['m00'])
    out_img = cv2.bitwise_and(image,image,mask = out)
    return cnt,(cx,cy+50)

def draw_information(image, name, location, center, shape, color, thick):
    if shape is None:
        pass
    else: 
        cv2.drawContours(image,[shape],-1,color,thick)
    cv2.putText(image,name,location,cv2.FONT_HERSHEY_COMPLEX, 1, color,thick)
    cv2.circle(image,center,5,color,thick)
    return image
def crop_contour(img,contour):
    # _,closing = img_proc(img,size_kernel,0,None)
    # _, contours, _ = cv2.findContours(closing) # Your call to find the contours
    # idx = ... # The index of the contour that surrounds your object
    mask = np.zeros_like(img) # Create mask where white is what we want, black otherwise
    cv2.drawContours(mask, contour, -1, 255, -1) # Draw filled contour in mask
    out = np.zeros_like(img) # Extract out the object and place into output image
    out[mask == 255] = img[mask == 255]
    # Now crop
    (y, x) = np.where(mask == 255)
    (topy, topx) = (np.min(y), np.min(x))
    (bottomy, bottomx) = (np.max(y), np.max(x))
    out = out[topy:bottomy+1, topx:bottomx+1]
    return out

def projectPointsErr(objpoints,imgpoints, rvecs, tvecs, mtx, dist):
    mean_error = []
    proj_error=0
    total_points=0
    for i in range(len(objpoints)):
        reprojected_points, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
        # reprojected_points=reprojected_points.reshape(-1,2)
        proj_error = np.sum(np.abs(imgpoints[i]-reprojected_points)**2)
        total_points = len(objpoints[i])
        
        #print("imgpointsL",imgpointsL)
        mean_error.append([i,round(np.sqrt(proj_error/total_points),2)])
    return mean_error
def mean(data): 
    return sum(data) / len(data) 
 
def stddev(data): 
    squared_data = [x*x for x in data] 
    return (mean(squared_data) - mean(data)**2)**.5 