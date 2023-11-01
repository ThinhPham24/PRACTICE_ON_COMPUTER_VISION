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
from utils import resize_image, midpoint
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
units = 10 #(mm)
if __name__ == "__main__":
    # image_path = current_dir + "\\Single_Camera\\image1.png"
    image_path = current_dir + "\\Images\\image_25.png"
    image = cv2.imread(image_path)
    # Appling the Calibrated camera
    h,  w = image.shape[:2]
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))
    mapx, mapy = cv2.initUndistortRectifyMap(mtx, dist, None, newcameramtx, (w,h), 5)
    dist_img = cv2.remap(image, mapx, mapy, cv2.INTER_LINEAR)
    tem_img = dist_img
    grayColor = cv2.cvtColor(dist_img,cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(grayColor, (7, 7), 0)
    # perform edge detection, then perform a dilation + erosion to
    # close gaps in between object edges
    edged = cv2.Canny(gray, 50, 100)
    edged = cv2.dilate(edged, None, iterations=1)
    edged = cv2.erode(edged, None, iterations=1)
    # find contours in the edge map
    cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    # sort the contours from left-to-right and initialize the
    # 'pixels per metric' calibration variable
    (cnts, _) = contours.sort_contours(cnts)
    pixelsPerMetric = None
    # loop over the contours individually
    for c in cnts:
        # if the contour is not sufficiently large, ignore it
        if cv2.contourArea(c) < 100:
            continue
        # compute the rotated bounding box of the contour
        orig = image.copy()
        box = cv2.minAreaRect(c)
        box = cv2.cv.BoxPoints(box) if imutils.is_cv2() else cv2.boxPoints(box)
        box = np.array(box, dtype="int")
        # order the points in the contour such that they appear
        # in top-left, top-right, bottom-right, and bottom-left
        # order, then draw the outline of the rotated bounding
        # box
        box = perspective.order_points(box)
        cv2.drawContours(orig, [box.astype("int")], -1, (0, 255, 0), 2)
        # loop over the original points and draw them
        for (x, y) in box:
            cv2.circle(orig, (int(x), int(y)), 5, (0, 0, 255), -1)
            # unpack the ordered bounding box, then compute the midpoint
            # between the top-left and top-right coordinates, followed by
            # the midpoint between bottom-left and bottom-right coordinates
            (tl, tr, br, bl) = box
            (tltrX, tltrY) = midpoint(tl, tr)
            (blbrX, blbrY) = midpoint(bl, br)
            # compute the midpoint between the top-left and top-right points,
            # followed by the midpoint between the top-righ and bottom-right
            (tlblX, tlblY) = midpoint(tl, bl)
            (trbrX, trbrY) = midpoint(tr, br)
            # draw the midpoints on the image
            cv2.circle(orig, (int(tltrX), int(tltrY)), 5, (255, 0, 0), -1)
            cv2.circle(orig, (int(blbrX), int(blbrY)), 5, (255, 0, 0), -1)
            cv2.circle(orig, (int(tlblX), int(tlblY)), 5, (255, 0, 0), -1)
            cv2.circle(orig, (int(trbrX), int(trbrY)), 5, (255, 0, 0), -1)
            # draw lines between the midpoints
            cv2.line(orig, (int(tltrX), int(tltrY)), (int(blbrX), int(blbrY)),
                (255, 0, 255), 2)
            cv2.line(orig, (int(tlblX), int(tlblY)), (int(trbrX), int(trbrY)),
                (255, 0, 255), 2)
            	# compute the Euclidean distance between the midpoints
            dA = dist.euclidean((tltrX, tltrY), (blbrX, blbrY))
            dB = dist.euclidean((tlblX, tlblY), (trbrX, trbrY))
            # if the pixels per metric has not been initialized, then
            # compute it as the ratio of pixels to supplied metric
            # (in this case, mm)
            if pixelsPerMetric is None:
                pixelsPerMetric = dB / units
            # compute the size of the object
            dimA = dA / pixelsPerMetric
            dimB = dB / pixelsPerMetric
            # draw the object sizes on the image
            cv2.putText(orig, "{:.1f}mm".format(dimA),
                (int(tltrX - 15), int(tltrY - 10)), cv2.FONT_HERSHEY_SIMPLEX,
                0.65, (255, 255, 255), 2)
            cv2.putText(orig, "{:.1f}mm".format(dimB),
                (int(trbrX + 10), int(trbrY)), cv2.FONT_HERSHEY_SIMPLEX,
                0.65, (255, 255, 255), 2)
            # show the output image
            cv2.imshow("Image", orig)
            cv2.waitKey(0)