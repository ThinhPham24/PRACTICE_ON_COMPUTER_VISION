import os
os.environ["PYLON_CAMEMU"] = "9"
import numpy as np
from pypylon import genicam
from pypylon import pylon
import sys
import cv2
import glob
import json
import open3d as o3d
from crop import ProcForOrchid
import time
import sub_create_pc
def resized_img(img,percent):
    width = int(img.shape[1] * percent / 100)
    height = int(img.shape[0] * percent / 100)
    dim = (width, height)
    # resize image
    resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
    return resized
converter = pylon.ImageFormatConverter()
# converting to opencv bgr format
converter.OutputPixelFormat = pylon.PixelType_BGR8packed
converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned

maxCamerasToUse = 8
cv_file_top = cv2.FileStorage('/home/airlab/Desktop/Calib_multiple_stereo/Calibrated/'  + 'top/stereoMap.txt',cv2.FileStorage_READ)
stereoMapR_x_top = cv_file_top.getNode('stereoMapR_x').mat()
stereoMapR_y_top= cv_file_top.getNode('stereoMapR_y').mat()
stereoMapL_x_top = cv_file_top.getNode('stereoMapL_x').mat()
stereoMapL_y_top = cv_file_top.getNode('stereoMapL_y').mat()
Q_top = cv_file_top.getNode('q').mat()
###############
cv_file_0 = cv2.FileStorage('/home/airlab/Desktop/Calib_multiple_stereo/Calibrated/'  + 'degree0/stereoMap.txt',cv2.FileStorage_READ)
stereoMapR_x_0 = cv_file_0.getNode('stereoMapR_x').mat()
stereoMapR_y_0= cv_file_0.getNode('stereoMapR_y').mat()
stereoMapL_x_0 = cv_file_0.getNode('stereoMapL_x').mat()
stereoMapL_y_0 = cv_file_0.getNode('stereoMapL_y').mat()
Q_0 = cv_file_0.getNode('q').mat()
###############
#######################
cv_file_120= cv2.FileStorage('/home/airlab/Desktop/Calib_multiple_stereo/Calibrated/'  + 'degree120/stereoMap.txt',cv2.FileStorage_READ)
stereoMapR_x_120 = cv_file_120.getNode('stereoMapR_x').mat()
stereoMapR_y_120 = cv_file_120.getNode('stereoMapR_y').mat()
stereoMapL_x_120 = cv_file_120.getNode('stereoMapL_x').mat()
stereoMapL_y_120 = cv_file_120.getNode('stereoMapL_y').mat()
Q_120 = cv_file_120.getNode('q').mat()
######################
cv_file_240 = cv2.FileStorage('/home/airlab/Desktop/Calib_multiple_stereo/Calibrated/'  + 'degree240/stereoMap.txt',cv2.FileStorage_READ)
stereoMapR_x_240 = cv_file_240.getNode('stereoMapR_x').mat()
stereoMapR_y_240= cv_file_240.getNode('stereoMapR_y').mat()
stereoMapL_x_240 = cv_file_240.getNode('stereoMapL_x').mat()
stereoMapL_y_240 = cv_file_240.getNode('stereoMapL_y').mat()
Q_240 = cv_file_240.getNode('q').mat()
def map_image(img,stereoMapL_x,stereoMapL_y,stereoMapR_x,stereoMapR_y):
        imgL = cv2.remap(img[0], stereoMapL_x, stereoMapL_y, cv2.INTER_LANCZOS4, cv2.BORDER_CONSTANT, 0)
        imgR = cv2.remap(img[1], stereoMapR_x, stereoMapR_y, cv2.INTER_LANCZOS4, cv2.BORDER_CONSTANT, 0)
        return imgL,imgR
# The exit code of the sample application.\
def save_image(img_number,imL, imR, crop,idx):
    if idx == 1:
        cv2.imwrite('/home/airlab/Desktop/Calib_multiple_stereo/images/img_crop/' + str(
            img_number) + '_1.jpg', crop)
        cv2.imwrite('/home/airlab/Desktop/Calib_multiple_stereo/images/img_origin/' + str(
            img_number) + '_1_L.jpg',imL)
        cv2.imwrite('/home/airlab/Desktop/Calib_multiple_stereo/images/img_origin/' + str(
            img_number) + '_1_R.jpg',imR)
    elif idx ==2:
        cv2.imwrite('/home/airlab/Desktop/Calib_multiple_stereo/images/img_crop/' + str(
            img_number) + '_2.jpg', crop)
        cv2.imwrite('/home/airlab/Desktop/Calib_multiple_stereo/images/img_origin/' + str(
            img_number) + '_2_L.jpg', imL)
        cv2.imwrite('/home/airlab/Desktop/Calib_multiple_stereo/images/img_origin/' + str(
            img_number) + '_2_R.jpg', imR)
    elif idx ==3:
        cv2.imwrite('/home/airlab/Desktop/Calib_multiple_stereo/images/img_crop/' + str(
            img_number) + '_3.jpg', crop)
        cv2.imwrite('/home/airlab/Desktop/Calib_multiple_stereo/images/img_origin/' + str(
            img_number) + '_3_L.jpg',imL)
        cv2.imwrite('/home/airlab/Desktop/Calib_multiple_stereo/images/img_origin/' + str(
            img_number) + '_3_R.jpg', imR)
    if idx ==4:
        cv2.imwrite('/home/airlab/Desktop/Calib_multiple_stereo/images/img_crop/' + str(
            img_number) + '_4.jpg',crop)
        cv2.imwrite('/home/airlab/Desktop/Calib_multiple_stereo/images/img_origin/' + str(
            img_number) + '_4_L.jpg', imL)
        cv2.imwrite('/home/airlab/Desktop/Calib_multiple_stereo/images/img_origin/' + str(
            img_number) + '_4_R.jpg', imR)
    print('Done save')
    return True

exitCode = 0
id_image = 0
try:
    # Get the transport layer factory.
    tlFactory = pylon.TlFactory.GetInstance()

    # Get all attached devices and exit application if no device is found.
    devices = tlFactory.EnumerateDevices()
    if len(devices) == 0:
        raise pylon.RuntimeException("No camera present.")

    # Create an array of instant cameras for the found devices and avoid exceeding a maximum number of devices.
    cameras = pylon.InstantCameraArray(8)

    l = cameras.GetSize()

    # Create and attach all Pylon Devices.
    for i, cam in enumerate(cameras):
        cam.Attach(tlFactory.CreateDevice(devices[i]))
        #cam.ExposureTime.SetValue(200000)
        #print(cam)
        #cam.StartGrabbing(pylon.GrabStrategy_LatestImageOnly) # PROBLEM
        # Print the model name of the camera.
        print("Using device ", cam.GetDeviceInfo().GetModelName())
    cameras.Open()
    convert = ProcForOrchid()
    for idx, cam in enumerate(cameras):
        camera_serial = cam.DeviceInfo.GetSerialNumber()
        print(f"set context {idx} for camera {camera_serial}")
        cam.SetCameraContext(idx)
    # set the exposure time for each camera
    for idx, cam in enumerate(cameras):
        camera_serial = cam.DeviceInfo.GetSerialNumber()
        print(f"set Exposuretime {idx} for camera {camera_serial}")
        cam.ExposureTimeAbs = 20000
        cam.AcquisitionFrameRateEnable.SetValue(True)
        cam.AcquisitionFrameRateAbs.SetValue(5)
        cam.Width.SetValue(1600)
        cam.Height.SetValue(1466)
########Camera 240
        if idx==0:  #L
            cam.OffsetX.SetValue(800)
            cam.OffsetY.SetValue(122)
        elif idx == 1: #R
            cam.OffsetX.SetValue(0)
            cam.OffsetY.SetValue(0)
##########Camera 120
        elif idx == 2:#R
            cam.OffsetX.SetValue(96)
            cam.OffsetY.SetValue(176)
        elif idx == 4: #L
            cam.OffsetX.SetValue(832)
            cam.OffsetY.SetValue(120)
########Camera Top
        elif idx == 3: #L
            cam.OffsetX.SetValue(864)
            cam.OffsetY.SetValue(440)
        elif idx == 5: #R
            cam.OffsetX.SetValue(224)
            cam.OffsetY.SetValue(440)
##########Camera 0
        elif idx == 6: #R
            cam.OffsetX.SetValue(64)
            cam.OffsetY.SetValue(140)
        elif idx == 7: #L
            cam.OffsetX.SetValue(704)
            cam.OffsetY.SetValue(122)
    cameras.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
    # Grab c_countOfImagesToGrab from the cameras.
    check = [0,0,0,0,0,0,0,0]
    number_image = 0
    print("Wait")
    while cameras.IsGrabbing():
        grabResult = cameras.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
        tar = time.time()
        if grabResult.GrabSucceeded():
            id_c = grabResult.GetCameraContext()
            image = converter.Convert(grabResult)
            img = image.GetArray()
            images = np.ones((2048, 2592, 3), dtype=np.uint8)
            images[:, :] = (0, 0, 0)
            image_RGB = images
            if id_c ==3:
                check[id_c] =1
                image_RGB[440:440 + 1466, 864:864 +1600] = img
                L_top = image_RGB
            elif id_c == 5:
                check[id_c] = 1
                image_RGB[440:440 + 1466, 224:224 + 1600] = img
                R_top = image_RGB
            elif id_c == 2:
                check[id_c] = 1
                image_RGB[176:176 + 1466, 96:96 + 1600] = img #y,x
                R_120 = image_RGB
            elif id_c == 4:
                check[id_c] = 1
                image_RGB[120:120 + 1466, 832:832 + 1600] = img
                L_120 = image_RGB
            elif id_c == 6:
                check[id_c] = 1
                image_RGB[140:140 + 1466, 64:64 + 1600] = img
                R_0 = image_RGB
            elif id_c == 7:
                check[id_c] = 1
                image_RGB[122:122 + 1466, 704:704 + 1600] = img
                L_0 = image_RGB
            elif id_c == 1:
                check[id_c] = 1
                image_RGB[0:0 + 1466, 0:0 + 1600] = img
                R_240 = image_RGB
            elif id_c == 0:
                check[id_c] = 1
                image_RGB[122:122 + 1466, 800:800 + 1600] = img
                L_240 = image_RGB
            ######## Clarity parameter
            # image_set =['07']
            min_disparity = '{"1" : "680", "2" : "680", "3" : "680", "4" :"680"}'
            # min_disparity = '{"1" : "715", "2" : "620", "3" : "760"}' # 3 VIEWS
            min_disparity_dict = json.loads(min_disparity)
            # [ 0degree, 120degree, 240degree,top]
            # print('test',min_disparity_dict["1"])
            #     print(ma.split('_')[1])
            lower = [0, 52, 44]
            upper = [179, 255, 255]
            # lower = [0, 0, 138]
            # upper = [114, 36, 255]
            HSV = np.vstack((lower, upper))
            print("wait opening 8")
            if np.sum(np.asarray(check)) == 8:
                print("here")
                img_L_top, img_R_top = map_image((L_top,R_top),stereoMapL_x_top,stereoMapL_y_top,stereoMapR_x_top,stereoMapR_y_top)
                img_L_0, img_R_0 = map_image((L_0, R_0), stereoMapL_x_0, stereoMapL_y_0, stereoMapR_x_0, stereoMapR_y_0)
                img_L_120, img_R_120 = map_image((L_120, R_120), stereoMapL_x_120, stereoMapL_y_120, stereoMapR_x_120, stereoMapR_y_120)
                img_L_240, img_R_240 = map_image((L_240, R_240), stereoMapL_x_240, stereoMapL_y_240, stereoMapR_x_240, stereoMapR_y_240)
                crop_img_top = convert.crop_for_DL(img_L_top)
                crop_img_0 = convert.crop_for_DL(img_L_0)
                crop_img_120 = convert.crop_for_DL(img_L_120)
                crop_img_240 = convert.crop_for_DL(img_L_240)
                begin = input("To start capturing images, please press {S} or {b} to exit\n")
                begin_time = time.time()
                if begin == "S" or begin == "s":
                    # print('test', min_disparity_dict["1"])
                    cond = False
                    cond1 = save_image(number_image,img_L_top,img_R_top,crop_img_top,4)
                    cond2 = save_image(number_image, img_L_240, img_R_240, crop_img_240, 3)
                    cond3 = save_image(number_image, img_L_120, img_R_120, crop_img_120, 2)
                    cond4 = save_image(number_image, img_L_0, img_R_0, crop_img_0, 1)
                    cond = cond1 and cond2 and cond3 and cond4
                    if cond == True:
                        print("time cost", time.time()-tar)
                        print("TOP")
                        check = [0, 0, 0, 0, 0, 0, 0, 0]
                        points_all_top, colors_all_top, mask_top = sub_create_pc.reconstruct_2d_to_3d(img_L_top,img_R_top,min_disparity_dict["4"],160,HSV,Q_top)
                        sub_create_pc.write_ply(points_all_top,colors_all_top,number_image,4)
                        cv2.imwrite("/home/airlab/Desktop/Calib_multiple_stereo/images/mask/{}_4.jpg".format(number_image),mask_top)
                        colors_all_top = np.asarray(colors_all_top / 255)  # rescale to 0 to 1
                        pcd_top = o3d.geometry.PointCloud()
                        pcd_top.points = o3d.utility.Vector3dVector(points_all_top)
                        pcd_top.colors = o3d.utility.Vector3dVector(colors_all_top)
                        # o3d.visualization.draw_geometries([pcd_top])
                        #######################
                        print("O")
                        points_all_0, colors_all_0, mask_0= sub_create_pc.reconstruct_2d_to_3d(img_L_0, img_R_0, min_disparity_dict["1"], 160,HSV, Q_0)
                        sub_create_pc.write_ply(points_all_0, colors_all_0, number_image, 1)
                        cv2.imwrite("/home/airlab/Desktop/Calib_multiple_stereo/images/mask/{}_1.jpg".format(number_image), mask_0)
                        colors_all_0 = np.asarray(colors_all_0 / 255)  # rescale to 0 to 1
                        pcd_0 = o3d.geometry.PointCloud()
                        pcd_0.points = o3d.utility.Vector3dVector(points_all_0)
                        pcd_0.colors = o3d.utility.Vector3dVector(colors_all_0)
                        # o3d.visualization.draw_geometries([pcd_0])
                        ###################
                        print("12O")
                        points_all_120, colors_all_120, mask_120 = sub_create_pc.reconstruct_2d_to_3d(img_L_120, img_R_120, min_disparity_dict["2"], 160, HSV, Q_120)
                        sub_create_pc.write_ply(points_all_120, colors_all_120, number_image, 2)
                        cv2.imwrite("/home/airlab/Desktop/Calib_multiple_stereo/images/mask/{}_2.jpg".format(number_image), mask_120)
                        colors_all_120 = np.asarray(colors_all_120 / 255)  # rescale to 0 to 1
                        pcd_120 = o3d.geometry.PointCloud()
                        pcd_120.points = o3d.utility.Vector3dVector(points_all_120)
                        pcd_120.colors = o3d.utility.Vector3dVector(colors_all_120)
                        # o3d.visualization.draw_geometries([pcd_120])
                        ######################
                        print("24O")
                        points_all_240, colors_all_240, mask_240 = sub_create_pc.reconstruct_2d_to_3d(img_L_240, img_R_240, min_disparity_dict["3"], 160, HSV, Q_240)
                        sub_create_pc.write_ply(points_all_240, colors_all_240, number_image, 3)
                        cv2.imwrite("/home/airlab/Desktop/Calib_multiple_stereo/images/mask/{}_3.jpg".format(number_image), mask_240)
                        colors_all_240 = np.asarray(colors_all_240 / 255)  # rescale to 0 to 1
                        pcd_240 = o3d.geometry.PointCloud()
                        pcd_240.points = o3d.utility.Vector3dVector(points_all_240)
                        pcd_240.colors = o3d.utility.Vector3dVector(colors_all_240)
                        # o3d.visualization.draw_geometries([pcd_240])
                        number_image += 1
                        print("time of begin", time.time() - begin_time)
                        print("number", number_image)
                        print("DONE!")
                        cond = False
                    begin = "o"
                elif begin =="b" or begin == "B":
                    break
        grabResult.Release()
except genicam.GenericException as e:
    # Error handling
    print("An exception occurred.", e.GetDescription())
    exitCode = 1