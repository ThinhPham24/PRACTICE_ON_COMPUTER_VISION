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
from proforimage import ProcForOrchid
from submain_v1 import*
from subfunction2d import*
# from get_image import*
import time
from sub_leaf_dark_v2 import LEAF_DARK_CUTTING 
from utilities import function_crop
from utility import *
# from Robotlib_auto import DensoRobotControl
# from HiwinElecGripLib import HiwinGrip
# from DeltaTurnTableLib import DeltaServo
# from orchid_operations import*
###################ROBOT FUNCTIONS##########################################################################################################################################
# rHomeH = [400, -340, 280, 180, 0, -90]
# rHomeC = [180, 190, 175, 180, 0, 90]
# speed = 10
# tool =2
# fp =60
# station = DeltaServo("/dev/ttyACM1")
# eGrip = HiwinGrip("/dev/ttyACM0")
# robotHold = DensoRobotControl("/dev/ttyUSB0")
# robotCut = DensoRobotControl("/dev/ttyUSB1")
# robotHold.SetTimeOut(100, 60000)
# robotCut.SetTimeOut(100, 60000)
 
# time.sleep(2) 
# station.ServoOn()

# rInfo = robotInfo(HomeH = rHomeH, HomeC = rHomeC, HoldTool = 2, CutTool =2, Speed =10)
#################################################################################################################################################################################

output = './OUTPUT/IMAGE_PRE6'
outputcut = './OUTPUT/CUT6'
leaf_dark = LEAF_DARK_CUTTING(output, outputcut)
crop_leaf_dark = function_crop()
#------------------------------------

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

############

input_yolact, output_yolact, input_MRCNN, outMRCNN =  "./yolact_edge/image_4","OUTPUT/Mask_processed", "./yolact_edge/image_4","OUTPUT/maskrcnn_angle"
try:
    os.mkdir("OUTPUT")
    os.mkdir(input_yolact)
    os.mkdir(output_yolact)
    os.mkdir(input_MRCNN)
    os.mkdir(outMRCNN)
    os.mkdir("./yolact_edge/original_image")    
except:
    pass
###########

maxCamerasToUse = 8
cv_file_top = cv2.FileStorage('./Calibrated/'  + 'top/stereoMap.txt',cv2.FileStorage_READ)
# cv_file.open('/home/airlab/Desktop/Transformation_matrix/result_image/Calibrated/' + str(address) + '/stereoMap.txt',cv2.FileStorage_READ)
stereoMapR_x_top = cv_file_top.getNode('stereoMapR_x').mat()
stereoMapR_y_top= cv_file_top.getNode('stereoMapR_y').mat()
stereoMapL_x_top = cv_file_top.getNode('stereoMapL_x').mat()
stereoMapL_y_top = cv_file_top.getNode('stereoMapL_y').mat()
Q1_top = cv_file_top.getNode('q').mat()
cv_file_0 = cv2.FileStorage('./Calibrated/'  + 'degree0/stereoMap.txt',cv2.FileStorage_READ)
# cv_file.open('/home/airlab/Desktop/Transformation_matrix/result_image/Calibrated/' + str(address) + '/stereoMap.txt',cv2.FileStorage_READ)
stereoMapR_x_0 = cv_file_0.getNode('stereoMapR_x').mat()
stereoMapR_y_0= cv_file_0.getNode('stereoMapR_y').mat()
stereoMapL_x_0 = cv_file_0.getNode('stereoMapL_x').mat()
stereoMapL_y_0 = cv_file_0.getNode('stereoMapL_y').mat()
Q1_0 = cv_file_0.getNode('q').mat()
cv_file_120= cv2.FileStorage('./Calibrated/'  + 'degree120/stereoMap.txt',cv2.FileStorage_READ)
# cv_file.open('/home/airlab/Desktop/Transformation_matrix/result_image/Calibrated/' + str(address) + '/stereoMap.txt',cv2.FileStorage_READ)
stereoMapR_x_120 = cv_file_120.getNode('stereoMapR_x').mat()
stereoMapR_y_120 = cv_file_120.getNode('stereoMapR_y').mat()
stereoMapL_x_120 = cv_file_120.getNode('stereoMapL_x').mat()
stereoMapL_y_120 = cv_file_120.getNode('stereoMapL_y').mat()
Q1_120 = cv_file_120.getNode('q').mat()
cv_file_240 = cv2.FileStorage('./Calibrated/'  + 'degree240/stereoMap.txt',cv2.FileStorage_READ)
# cv_file.open('/home/airlab/Desktop/Transformation_matrix/result_image/Calibrated/' + str(address) + '/stereoMap.txt',cv2.FileStorage_READ)
stereoMapR_x_240 = cv_file_240.getNode('stereoMapR_x').mat()
stereoMapR_y_240= cv_file_240.getNode('stereoMapR_y').mat()
stereoMapL_x_240 = cv_file_240.getNode('stereoMapL_x').mat()
stereoMapL_y_240 = cv_file_240.getNode('stereoMapL_y').mat()
Q1_240 = cv_file_240.getNode('q').mat()
def map_image(img,stereoMapL_x,stereoMapL_y,stereoMapR_x,stereoMapR_y):
    imgL = cv2.remap(img[0], stereoMapL_x, stereoMapL_y, cv2.INTER_LANCZOS4, cv2.BORDER_CONSTANT, 0)
    imgR = cv2.remap(img[1], stereoMapR_x, stereoMapR_y, cv2.INTER_LANCZOS4, cv2.BORDER_CONSTANT, 0)
    return imgL,imgR
# The exit code of the sample application.\
def save_image(path,img_number,imL, imR, crop,idx):
    if idx == 1:
        cv2.imwrite('./yolact_edge/image_4/' + str(img_number) + '_1.jpg', crop)
        cv2.imwrite( path + str(img_number) + '_1_L.jpg',imL)
        cv2.imwrite( path + str(img_number) + '_1_R.jpg',imR)
    elif idx ==2:
        cv2.imwrite('./yolact_edge/image_4/' + str(img_number) + '_2.jpg', crop)
        cv2.imwrite(path + str(img_number) + '_2_L.jpg', imL)
        cv2.imwrite(path + str(img_number) + '_2_R.jpg', imR)
    elif idx ==3:
        cv2.imwrite('./yolact_edge/image_4/' + str(img_number) + '_3.jpg', crop)
        cv2.imwrite(path + str(img_number) + '_3_L.jpg',imL)
        cv2.imwrite(path + str(img_number) + '_3_R.jpg', imR)
    if idx ==4:
        cv2.imwrite('./yolact_edge/image_4/' + str(img_number) + '_4.jpg',crop)
        cv2.imwrite(path + str(img_number) + '_4_L.jpg', imL)
        cv2.imwrite(path + str(img_number) + '_4_R.jpg', imR)

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
        cam.ExposureTimeAbs = 15000
        cam.AcquisitionFrameRateEnable.SetValue(True)
        cam.AcquisitionFrameRateAbs.SetValue(5)
        cam.Width.SetValue(1600)
        cam.Height.SetValue(1490)
########Camera 240
        if idx==0:  #L
            cam.OffsetX.SetValue(768)
            cam.OffsetY.SetValue(154)
        elif idx == 1: #R
            cam.OffsetX.SetValue(32)
            cam.OffsetY.SetValue(168)
##########Camera 120
        elif idx == 2:#R
            cam.OffsetX.SetValue(32)
            cam.OffsetY.SetValue(160)
        elif idx == 4: #L
            cam.OffsetX.SetValue(896)
            cam.OffsetY.SetValue(130)
########Camera Top
        elif idx == 3: #L
            cam.OffsetX.SetValue(928)
            cam.OffsetY.SetValue(406)
        elif idx == 5: #R
            cam.OffsetX.SetValue(224)
            cam.OffsetY.SetValue(406)
##########Camera 0
        elif idx == 6: #R
            cam.OffsetX.SetValue(32)
            cam.OffsetY.SetValue(168)
        elif idx == 7: #L
            cam.OffsetX.SetValue(736)
            cam.OffsetY.SetValue(186)
    cameras.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
    # Grab c_countOfImagesToGrab from the cameras.
    check = [0,0,0,0,0,0,0,0]
    number_image = 37
    stage =1 #stage 1 tach nhanh, satge 2 cat la
    seperated = False
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
                image_RGB[406:406 + 1490, 928:928 +1600] = img #y,x
                L_top = image_RGB
            elif id_c == 5:
                check[id_c] = 1
                image_RGB[406:406 + 1490, 224:224 + 1600] = img
                R_top = image_RGB
            elif id_c == 2:
                check[id_c] = 1
                image_RGB[160:160 + 1490, 32:32 + 1600] = img #y,x
                R_120 = image_RGB
            elif id_c == 4:
                check[id_c] = 1
                image_RGB[130:130 + 1490, 896:896 + 1600] = img
                L_120 = image_RGB
            elif id_c == 6:
                check[id_c] = 1
                image_RGB[168:168 + 1490, 32:32 + 1600] = img
                R_0 = image_RGB
            elif id_c == 7:
                check[id_c] = 1
                image_RGB[186:186 + 1490, 736:736 + 1600] = img
                L_0 = image_RGB
            elif id_c == 1:
                check[id_c] = 1
                image_RGB[168:168 + 1490, 32:32 + 1600] = img
                R_240 = image_RGB
            elif id_c == 0:
                check[id_c] = 1
                image_RGB[154:154 + 1490, 768:768 + 1600] = img
                L_240 = image_RGB
            ######## Clarity parameter
            address = '{"1": "0degree", "2" : "120degree", "3": "240degree", "4" : "top"}'
            address_dict = json.loads(address)
            # image_set =['07']
            min_disparity = '{"1" : "680", "2" : "680", "3" : "680", "4" :"680"}'
            # min_disparity = '{"1" : "715", "2" : "620", "3" : "760"}' # 3 VIEWS
            min_disparity_dict = json.loads(min_disparity)
            all_imgs_top, all_imgs_0, all_imgs_120,all_imgs_240, crop_all = [],[],[],[],[]
            if np.sum(np.asarray(check)) == 8:
                print("here")
                #Calibrate image
                img_L_top, img_R_top = map_image((L_top,R_top),stereoMapL_x_top,stereoMapL_y_top,stereoMapR_x_top,stereoMapR_y_top)
                img_L_0, img_R_0 = map_image((L_0, R_0), stereoMapL_x_0, stereoMapL_y_0, stereoMapR_x_0, stereoMapR_y_0)
                img_L_120, img_R_120 = map_image((L_120, R_120), stereoMapL_x_120, stereoMapL_y_120, stereoMapR_x_120, stereoMapR_y_120)
                img_L_240, img_R_240 = map_image((L_240, R_240), stereoMapL_x_240, stereoMapL_y_240, stereoMapR_x_240, stereoMapR_y_240)
                #-----------------OLD CROP
                # crop_img_top = convert.crop_for_DL(img_L_top)
                # crop_img_0 = convert.crop_for_DL(img_L_0)
                # crop_img_120 = convert.crop_for_DL(img_L_120)
                # crop_img_240 = convert.crop_for_DL(img_L_240)
                #-----------------NEW
                crop_img_top = convert.crop_detect_orchid(img_L_top)
                crop_img_0 = convert.crop_detect_orchid(img_L_0)
                crop_img_120 = convert.crop_detect_orchid(img_L_120)
                crop_img_240 = convert.crop_detect_orchid(img_L_240)
                #----------
                begin_time = time.time()
                check = [0, 0, 0, 0, 0, 0, 0, 0]
                print("opening camera time", time.time() - begin_time)
                begin = input("To start capturing images, please press {S} or {b} to exit\n")
                if begin == "S" or begin == "s":
                    cond = False
                    if stage ==1:
                        path_save_original = './yolact_edge/original_image/'
                        cond1 = save_image(path_save_original,number_image,img_L_top,img_R_top,crop_img_top,4)
                        cond2 = save_image(path_save_original,number_image, img_L_240, img_R_240, crop_img_240, 3)
                        cond3 = save_image(path_save_original,number_image, img_L_120, img_R_120, crop_img_120, 2)
                        cond4 = save_image(path_save_original,number_image, img_L_0, img_R_0, crop_img_0, 1)
                        cond = cond1 and cond2 and cond3 and cond4
                    else:
                        path_save_original = './yolact_edge/cutting_image/'
                        cond = save_image(path_save_original,number_image,img_L_top,img_R_top,crop_img_top,4)
                    if cond == True:
                        if stage ==1:
                            print("OKE")
                            # check = [0, 0, 0, 0, 0, 0, 0, 0] ### because camera is not enough speed
                            image_set = [str(number_image)]
                            print('num', image_set)
                            m_time = time.time()
                            for i in range(len(image_set)):
                                process_mask_yolact = main_2d(input_yolact,output_yolact, input_MRCNN,outMRCNN) # main2d(image_set(4views)) #activate class
                                # process_mask_yolact = main_2d()  # main2d(image_set(4views)) # this code is old
                                best_view,number_bud = process_mask_yolact.run(image_set[i]) ###Chon ra best view 
                                print("time for chosing best view", time.time()-m_time)
                                print("number bud:", number_bud)
                                if number_bud > 1:
                                    path = './OUTPUT/maskrcnn_angle/TXT/' + '{}.txt'.format(best_view)
                                    path_ImL = './yolact_edge/original_image/' + '{}_L.jpg'.format(
                                        best_view)
                                    path_ImR = './yolact_edge/original_image/' + '{}_R.jpg'.format(
                                        best_view)
                                    path_mask = './OUTPUT/Mask_processed/{}/fullbud_mask.png'.format(best_view)
                                    path_image = np.hstack((path_ImL, path_ImR)) #gop 2 hinh thanh 1 nhom

                                    #-----------OLD CROP
                                    # image_crop = convert.crop_for_DL(cv2.imread(path_ImL))
                                    # txt_arrow = convert.convert_arrow2orinal(path)
                                    # mask_img = convert.convert_image(path_mask)
                                    
                                    #-----------------
                                    
                                    # ----- NEW
                                    image_crop = convert.crop_detect_orchid(cv2.imread(path_ImL))
                                    txt_arrow = convert.convert_arrow_detect_orchid(path)
                                    mask_img = convert.convert_detect_image(path_mask)
                                    #[[1834 1109 1566 1011] [1799 1200 1353 1199]]
                                    #[[1826 1097 1566  998] [1799 1172 1355 1172]]

                                    #------
                                    txt_arrow = np.asarray(txt_arrow)

                                    print("LOCATION OF ARROW:",  txt_arrow)

                                    index = best_view.split('_')[1]
                                    print('address index', address_dict[index])
                                    print('index of min', int(min_disparity_dict[index]))
                                    if address_dict[index] == "0degree":
                                        cut = Operate_cutting_plane(Q1_0, path_image, txt_arrow,
                                                                    int(min_disparity_dict[index]), 160,best_view,mask_img)
                                    elif address_dict[index] == "120degree":
                                        cut = Operate_cutting_plane(Q1_120, path_image, txt_arrow,
                                                                    int(min_disparity_dict[index]), 160, best_view,mask_img)
                                    elif address_dict[index] == "240degree":
                                        cut = Operate_cutting_plane(Q1_240, path_image, txt_arrow,
                                                                    int(min_disparity_dict[index]), 160, best_view,mask_img)
                                    else:
                                        cut = Operate_cutting_plane(Q1_top, path_image, txt_arrow,
                                                                    int(min_disparity_dict[index]), 160, best_view,mask_img)
                                    cutting_pts, angle = cut.main_plane()
                                    print('CUTTING POINT AND ANGLE', (cutting_pts, angle))
                                    if cutting_pts is None or angle is None:
                                        print("NO SEPERATION")
                                        stage =2
                                        seperated = False
                                        continue
                                    print("SEPERATED")
                                    # robot separate#############################################################################
                                    # posCS, posHS, angleS = read_data_saparate("/home/airlab/Desktop/ORCHID_SYSTEM/ROBOT_TXT/target.txt")
                                    # robot_operation(robotCut,robotHold,station, eGrip, posCS, posHS, angleS, 1, False, False,True)
                                    # time.sleep(1) 
                                    #-----------------------------
                                    # tarmm =  input_data_saparate("/home/airlab/Desktop/ORCHID_SYSTEM/ROBOT_TXT/target.txt")
                                    # target_ = robotTransform(station,tarmm)
                                    # dualArmSeparate(robotCut, robotHold, station, eGrip, rInfo, target_, False)
                                    #############################################################################################
                                    stage =2
                                    seperated = False  # True
                                    continue

                                else:
                                    '''Leaf and drak area cutting'''
                                    print("NO SEPERATION")
                                    stage =2
                                    seperated = False
                                    continue
                            cond = False
                            print("Total time of proceduce",time.time()-m_time)
                            continue
                        elif(stage ==2):
                            f_time = time.time()
                            image_crop_top = crop_leaf_dark.crop_detect(L_top)
                            im, data_sent_cut = leaf_dark.operation(image_crop_top,number_image,seperated)
                            crop_leaf_dark.sent_inf(data_sent_cut[0],data_sent_cut[1],data_sent_cut[2],data_sent_cut[3],data_sent_cut[4])
                            print("time for leaf and dark area cutting", time.time()-f_time)
                            # cv2.imshow("image", resized_img(im, 50))
                            # cv2.waitKey(0)
                            # cv2.destroyWindow("image")
                            # robt trim leave###############################################################
                            #  posCC1, posHC1, angle1, numProc1, darkArea1 = read_data("/home/airlab/Desktop/ORCHID_SYSTEM/ROBOT_TXT/target_leaf1.txt")
                            #  robot_operation(robotCut,robotHold,station, eGrip, posCC1, posHC1, angle1, numProc1, darkArea1, False,False)
                            # posCC2, posHC2, angle2, numProc2, darkArea2 = read_data("/home/airlab/Desktop/ORCHID_SYSTEM/ROBOT_TXT/target_leaf2.txt")
                            # robot_operation(robotCut,robotHold,station, eGrip, posCC2, posHC2, angle2, numProc2, darkArea2, True,False)
                            #--------------------------
                            # tarPix = input_data("/home/airlab/Desktop/ORCHID_SYSTEM/ROBOT_TXT/target_leaf1.txt")
                            # target = pixel2world(tarPix)
                            # target_ =  robotTransform(station,target)
                            # dualArmRun(robotCut, robotHold, station, eGrip, rInfo, target_, False)
                            # if(seperated):
                            #     tarPix = input_data("/home/airlab/Desktop/ORCHID_SYSTEM/ROBOT_TXT/target_leaf2.txt")
                            #     target = pixel2world(tarPix)
                            #     target_ =  robotTransform(station,target)
                            #     dualArmRun(robotCut, robotHold, station, eGrip, rInfo, target_, False)
                            ##################################################################################

                            # station.ServoOff()
                            check = [0, 0, 0, 0, 0, 0, 0, 0]
                            cond = False
                            stage =1
                            number_image += 1
                            cv2.imshow("image", resized_img(im, 50))
                            cv2.waitKey(0)
                            cv2.destroyWindow("image")
                            continue
                    begin = "o"   
                elif begin =="b" or begin == "B":
                    break
            grabResult.Release()
except genicam.GenericException as e:
    # Error handling
    print("An exception occurred.", e.GetDescription())
    exitCode = 1