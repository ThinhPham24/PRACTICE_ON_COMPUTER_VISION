import os
import cv2
import os.path as osp
from tqdm import tqdm
import json
import numpy as np
from scipy.spatial import distance
from imutils import perspective
from imutils import contours
import imutils
from utils import resize_image, pad_dict_list
import pandas as pd
import statistics
current_dir = os.getcwd()
print("path", current_dir)
# # Path 
# path = os.path.join(current_dir, name) 
# Folder Name:
name = "Parameters.json"
file_path = current_dir + '/' + "{}".format(name)
# Load parameters
# Opening JSON file
with open(file_path, 'r') as openfile:
    parameters = json.load(openfile)
mtx = np.array(parameters['Camera matrix'])
dist = np.array(parameters['Distortion'])

# mtx = np.array([[4959.2906/6,0,1903.6512/6],[0,4958.4507/6,1300.6562/6],[0,0,1]])
# dist = np.array([[-0.1183, 0.1711,0,0,0]])

# mtx = np.array([[825.8362,0,322.7181],[0,825.4186,214.3449],[0,0,1]])
# dist = np.array([[-0.1178, 0.1594,0,0,0]])

# mtx = np.array([[841.3157,0,320.1032],[0,840.4380,224.6275],[0,0,1]])
# dist = np.array([[-0.1356,0.4846,0.0026, 0.0006,-2.0132]])
CHECKERBOARD = (10,7)
# specified number of iterations are completed. 
criteria = (cv2.TERM_CRITERIA_EPS + 
            cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001) 
if __name__ == "__main__":
    # image_path = current_dir + "\\Single_Camera\\image1.png"
    image_path = current_dir + "\\Images\\image_25.png"
    image = cv2.imread(image_path)
    
    # Normal image
    grayColor = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    tem_img = image
    # Appling the Calibrated camera
    # h,  w = image.shape[:2]
    # newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))
    # mapx, mapy = cv2.initUndistortRectifyMap(mtx, dist, None, newcameramtx, (w,h), 5)
    # dist_img = cv2.remap(image, mapx, mapy, cv2.INTER_LINEAR)
    # tem_img = dist_img
    # grayColor = cv2.cvtColor(dist_img,cv2.COLOR_BGR2GRAY)
    #---------------------
    # # crop the image
    # x, y, w, h = roi
    # dist_img = dist_img[y:y+h, x:x+w]
    ret, corners = cv2.findChessboardCorners( 
            grayColor, CHECKERBOARD, None) 
    # Draw checker point on image
    image = cv2.drawChessboardCorners(tem_img,  
                                          CHECKERBOARD,  
                                          corners, ret)
    # for given 2d points. 
    corners2 = cv2.cornerSubPix( 
        grayColor, corners, (11, 11), (-1, -1), criteria) 
    corners = corners2.reshape(70,2)
    # print("dist", corners[1][1])
    dA = distance.euclidean((float(corners[28][0]), float(corners[28][1])), (float(corners[29][0]), float(corners[29][1])))
    # print("Distance:", dA)
    # if the pixels per metric has not been initialized, then
	# compute it as the ratio of pixels to supplied metric
	# (in this case, mm)
    units = 10 #(mm)
    pixelsPerMetric = dA /units
    # cv.imwrite('calibresult.png', dst)
    # merge = np.concatenate((image, dist_img), axis = 1)
    dist_point = {"number point": [], "distance": [], "Mean_error":[], "Stdev": []}
    error = []
    for idx in range(70):
        n = 1
        if idx in [9, 19, 29,39,49,59]:
            pts69 = corners[idx-9*n]
            pts70 = corners[idx + 1]
            n +=1
        else: 
            pts69 = corners[idx]
            pts70 = corners[idx + 1]
        dB  = distance.euclidean(pts69,pts70)
        dimB = dB / pixelsPerMetric
        dist_point["number point"].append(idx)
        dist_point["distance"].append(round(dimB,4))
        error.append((round(10-dimB,4)))
        if idx == 68:
           break
    #     print("Distance from point 70 to point 69:", dimB)
    # print(dist_point)
    mean_error = statistics.mean(error)
    stdev = statistics.stdev(error)
    dist_point["Mean_error"].append(round(mean_error,4))
    dist_point["Stdev"].append(round(stdev,4))
    dict_list = pad_dict_list(dist_point, 0)
    df = pd.DataFrame(dict_list)
    print(df)
    # Write DataFrame to Excel file
    df.to_excel(current_dir + '\camera_distortion.xlsx')
    
    cv2.imshow("image", resize_image(image,100))
    cv2.waitKey(0)
    cv2.destroyAllWindows()
