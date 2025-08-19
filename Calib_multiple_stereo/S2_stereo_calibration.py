import numpy as np
import cv2 as cv
import glob
from stereo_lib import*
import os
#**************
image_dir = "IMG_VIEWS"
Img_name = "degree120"
current_dir = os.getcwd()
image_savepath = os.path.join(current_dir, image_dir)
#*************
calibrated_dir = "Calibrated"
current_dir = os.getcwd()
calib_savepath = os.path.join(current_dir, calibrated_dir)
print("calibration save path", calib_savepath)

if not os.path.isdir(os.path.abspath(calib_savepath)):
    os.mkdir(calib_savepath)
if not os.path.isdir(os.path.abspath(calib_savepath + '/' + str(Img_name))):
    os.mkdir(calib_savepath + '/' + str(Img_name))
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
################ FIND CHESSBOARD CORNERS - OBJECT POINTS AND IMAGE POINTS #############################

chessboardSize = (10,7)
frameSize = (3,5472,3648)

# termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
criteria_stereo= (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((chessboardSize[0] * chessboardSize[1], 3), np.float32)
objp[:,:2] = np.mgrid[0:chessboardSize[0],0:chessboardSize[1]].T.reshape(-1,2)
# #size_of_chessboard_squares_mm = 10.075
size_of_chessboard_squares_mm = 5
objp = objp * size_of_chessboard_squares_mm
# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpointsL = [] # 2d points in image plane.
imgpointsR = [] # 2d points in image plane.

# imagesLeft = glob.glob('images/L-orchid-*.png')
# imagesRight = glob.glob('images/R-orchid-*.png')
imagesLeft = sorted(glob.glob( image_savepath + '/' + str(Img_name) + '/' + 'image_L_*.jpg'))
imagesRight = sorted(glob.glob(image_savepath + '/' + str(Img_name) + '/' + 'image_R_*.jpg'))
for imgLeft, imgRight in zip(imagesLeft, imagesRight):

    imgL_g = cv.imread(imgLeft,0)
    imgR_g = cv.imread(imgRight,0)
    # if True:
    #     temp_L = cv.GaussianBlur(imgL_g, (0, 0), 105)
    #     imgL_g = cv.addWeighted(imgL_g, 1.8, temp_L, -0.8, 0)
    #     temp_R = cv.GaussianBlur(imgR_g, (0, 0), 105)
    #     imgR_g = cv.addWeighted(imgR_g, 1.8, temp_R, -0.8, 0)
    imgL  = imgL_g.copy()
    imgR = imgR_g.copy()
    test_imgL  = imgL_g.copy()
    test_imgR = imgR_g.copy()
    grayL =imgL# cv.cvtColor(imgL, cv.COLOR_RGB2GRAY)
    grayR = imgR#cv.cvtColor(imgR, cv.COLOR_RGB2GRAY)

    # Find the chess board corners
    retL, cornersL = cv.findChessboardCorners(grayL, chessboardSize, cv.CALIB_CB_ADAPTIVE_THRESH | cv.CALIB_CB_FAST_CHECK | cv.CALIB_CB_NORMALIZE_IMAGE)
    retR, cornersR = cv.findChessboardCorners(grayR, chessboardSize, cv.CALIB_CB_ADAPTIVE_THRESH | cv.CALIB_CB_FAST_CHECK | cv.CALIB_CB_NORMALIZE_IMAGE)

    # If found, add object points, image points (after refining them)
    if retL == True and retR == True:
        print(imgLeft)
        print(imgRight)

        objpoints.append(objp)

        cornersL = cv.cornerSubPix(grayL, cornersL, (11,11), (-1,-1), criteria)
        imgpointsL.append(cornersL)

        cornersR = cv.cornerSubPix(grayR, cornersR, (11,11), (-1,-1), criteria)
        imgpointsR.append(cornersR)

        # Draw and display the corners
        cv.drawChessboardCorners(imgL, chessboardSize, cornersL, retL)
        cv.imshow('img left', resized_img(imgL,15))
        cv.drawChessboardCorners(imgR, chessboardSize, cornersR, retR)
        cv.imshow('img right', resized_img(imgR,15))
        cv.waitKey(1)

    else: 
        print("Cannot detection")
        print(imgLeft)
        print(imgRight)

cv.destroyAllWindows()



# Determine the new values for different parameters
#   Right Side
retR, mtxR, distR, rvecsR, tvecsR = cv2.calibrateCamera(objpoints,
                                                        imgpointsR,
                                                        grayR.shape[::-1],None,None)
hR,wR= grayR.shape[:2]
OmtxR, roiR= cv2.getOptimalNewCameraMatrix(mtxR,distR,
                                                   (wR,hR),1,(wR,hR))

#   Left Side
retL, mtxL, distL, rvecsL, tvecsL = cv2.calibrateCamera(objpoints,
                                                        imgpointsL,
                                                        grayL.shape[::-1],None,None)
proj_err_L = projectPointsErr(objpoints,imgpointsL, rvecsL, tvecsL, mtxL, distL)
proj_err_R = projectPointsErr(objpoints,imgpointsR, rvecsR, tvecsR, mtxR, distR)


print("Mean reprojection error left: ", proj_err_L)
print("Mean reprojection error right: ", proj_err_R)
# mean_error=np.sqrt(proj_error/total_points)
# print("mean_error",mean_error)
hL,wL= grayL.shape[:2]
OmtxL, roiL= cv2.getOptimalNewCameraMatrix(mtxL,distL,(wL,hL),1,(wL,hL))
print(mtxL,'\n')


#********************************************
#***** Calibrate the Cameras for Stereo *****
#********************************************

# StereoCalibrate function
flags = 0
flags |= cv2.CALIB_FIX_INTRINSIC
flags |= cv2.CALIB_FIX_PRINCIPAL_POINT
flags |= cv2.CALIB_USE_INTRINSIC_GUESS
flags |= cv2.CALIB_FIX_FOCAL_LENGTH
flags |= cv2.CALIB_FIX_ASPECT_RATIO
flags |= cv2.CALIB_ZERO_TANGENT_DIST
flags |= cv2.CALIB_RATIONAL_MODEL
flags |= cv2.CALIB_SAME_FOCAL_LENGTH
flags |= cv2.CALIB_FIX_K3
flags |= cv2.CALIB_FIX_K4
flags |= cv2.CALIB_FIX_K5
retS, MLS, dLS, MRS, dRS, R, T, E, F= cv2.stereoCalibrate(objpoints,
                                                          imgpointsL,
                                                          imgpointsR,
                                                          mtxL,
                                                          distL,
                                                          mtxR,
                                                          distR,
                                                          grayR.shape[::-1],
                                                          criteria_stereo,
                                                          flags)

# StereoRectify function
rectify_scale= 0 # if 0 image croped, if 1 image nor croped
RL, RR, PL, PR, Q, roiL, roiR= cv2.stereoRectify(MLS, dLS, MRS, dRS,
                                                 grayL.shape[::-1], R, T,
                                                 rectify_scale,(0,0))  # last paramater is alpha, if 0= croped, if 1= not croped
# initUndistortRectifyMap function
stereoMapL= cv2.initUndistortRectifyMap(MLS, dLS, RL, PL,
                                             grayL.shape[::-1], cv2.CV_16SC2)   # cv2.CV_16SC2 this format enables us the programme to work faster
stereoMapR= cv2.initUndistortRectifyMap(MRS, dRS, RR, PR,
                                              grayR.shape[::-1], cv2.CV_16SC2)
print("Q here!",Q)
############## CALIBRATION #######################################################
print("Saving parameters!")

cv_file = cv.FileStorage(calib_savepath + '/'+ str(Img_name) + '/' + 'stereoMap.txt', cv.FILE_STORAGE_WRITE)

cv_file.write('stereoMapL_x',stereoMapL[0])
cv_file.write('stereoMapL_y',stereoMapL[1])
cv_file.write('stereoMapR_x',stereoMapR[0])
cv_file.write('stereoMapR_y',stereoMapR[1])
cv_file.write('q', Q)
cv_file.write('camera_R_mxt', mtxR)
cv_file.write('camera_L_mxt', mtxL)

cv_file.release()


IR, IL = undistortRectify(test_imgR, test_imgL)
grayL = IL#cv.cvtColor(IL, cv.COLOR_RGB2GRAY)
grayR = IR#cv.cvtColor(IR, cv.COLOR_RGB2GRAY)
IR_N = cv.cvtColor(IR, cv.COLOR_GRAY2RGB)
IL_N = cv.cvtColor(IL, cv.COLOR_GRAY2RGB)
test_imgR_N = cv.cvtColor(test_imgR, cv.COLOR_GRAY2RGB)
test_imgL_N = cv.cvtColor(test_imgL, cv.COLOR_GRAY2RGB)
    # Find the chess board corners
retL, cornersL = cv.findChessboardCorners(grayL, chessboardSize, cv.CALIB_CB_ADAPTIVE_THRESH | cv.CALIB_CB_FAST_CHECK | cv.CALIB_CB_NORMALIZE_IMAGE)
retR, cornersR = cv.findChessboardCorners(grayR, chessboardSize, cv.CALIB_CB_ADAPTIVE_THRESH | cv.CALIB_CB_FAST_CHECK | cv.CALIB_CB_NORMALIZE_IMAGE)

    # If found, add object points, image points (after refining them)
if retL == True and retR == True:

    cornersL = cv.cornerSubPix(grayL, cornersL, (11,11), (-1,-1), criteria)

    cornersR = cv.cornerSubPix(grayR, cornersR, (11,11), (-1,-1), criteria)
    error = []
    for i in range(0, 70):
        error.append(cornersL[i][0][1]-cornersR[i][0][1])
    mean_err = mean(error)
    std_err = stddev(error)
    print("mean_err",mean_err)
    print("std_err",std_err)

    # Draw and display the corners
else: 
    print("Cannot detection")
# Draw Red lines
for line in range(0, int(IR_N.shape[0]/40)): # Draw the Lines on the images Then numer of line is defines by the image Size/20
    IL_N[line*40,:]= (0,255,0)
    IR_N[line*40,:]= (0,255,0)
##
for line in range(0, int(test_imgR_N.shape[0]/40)): # Draw the Lines on the images Then numer of line is defines by the image Size/20
    test_imgL_N[line*40,:]= (0,0,255)
    test_imgR_N[line*40,:]= (0,0,255)
# Show the Undistorted images
cv2.imshow('Both Images', resized_img(np.hstack([IL_N, IR_N]),30))
cv2.imshow('Normal', resized_img(np.hstack([test_imgL_N, test_imgR_N]),30))
cv2.imshow('IR',resized_img(IR,30))
cv2.imshow('IL',resized_img(IL,30))
k = cv2.waitKey(0)