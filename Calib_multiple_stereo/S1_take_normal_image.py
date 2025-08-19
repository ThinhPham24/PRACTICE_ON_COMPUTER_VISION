from pypylon import pylon
import cv2
import os, os.path
import glob, os


image_dir = "IMG_VIEWS"
current_dir = os.getcwd()
image_savepath = os.path.join(current_dir, image_dir)
print("image save path", image_savepath)
if not os.path.isdir(os.path.abspath(image_savepath)):
    os.mkdir(image_savepath)

#*******************************
def gige_camera(serial_number_L, serial_number_R,camera_L_exposureTime,camera_R_exposureTime,address):
    info_L_0 = pylon.DeviceInfo()
    info_L_0.SetSerialNumber(serial_number_L)

    print('imform',info_L_0)
    # 40003776  40118977
    camera_L_0 = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateDevice(info_L_0))
    camera_L_0.Open()
    camera_L_0.ExposureTimeAbs = camera_L_exposureTime


    info_R_0 = pylon.DeviceInfo()
    info_R_0.SetSerialNumber(serial_number_R)
    # 40118981   40070109
    camera_R_0 = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateDevice(info_R_0))
    camera_R_0.Open()
    camera_R_0.ExposureTimeAbs = camera_R_exposureTime
    print('imform',info_L_0)
    # 40003776  40118977
    camera_R_0 = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateDevice(info_R_0))
    camera_R_0.Open()



    # Grabing Continusely (video) with minimal delay
    camera_L_0.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
    camera_R_0.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
    converter = pylon.ImageFormatConverter()

    # converting to opencv bgr format
    converter.OutputPixelFormat = pylon.PixelType_BGR8packed
    converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned
    num = 1

    while camera_L_0.IsGrabbing() and camera_R_0.IsGrabbing():
        grabResult1 = camera_L_0.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
        grabResult2 = camera_R_0.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)

        if grabResult1.GrabSucceeded() and grabResult2.GrabSucceeded():

            # Access the image data
            imageL_0 = converter.Convert(grabResult1)
            imageR_0 = converter.Convert(grabResult2)

            imgL_0 = imageL_0.GetArray()  # 原始影像
            imgR_0 = imageR_0.GetArray()  # 原始影像



            img_resizeL_0 = cv2.resize(imgL_0, (600, 600), interpolation=cv2.INTER_AREA)
            img_resizeR_0 = cv2.resize(imgR_0, (600, 600), interpolation=cv2.INTER_AREA)


            cv2.imshow('img_resizeL_0', img_resizeL_0)
            cv2.imshow('img_resizeR_0', img_resizeR_0)
            k = cv2.waitKey(1)

            if k == ord("s") or k == ord("S"):
                if not os.path.isdir(os.path.abspath(image_savepath + '/' + str(address))):
                    os.mkdir(image_savepath + '/' + str(address)) 
                cv2.imwrite(image_savepath + '/' + str(address) + '/image_L_' + str(num) + '.jpg', imgL_0)
                cv2.imwrite(image_savepath + '/' + str(address) + '/image_R_' + str(num) + '.jpg', imgR_0)
                # print("images saved!")
                print("number of image", num)
                num += 1
                

            if k == 27:
                break


            cv2.waitKey(1)

        grabResult1.Release()
        #grabResult2.Release()


    # Releasing the resource
    camera_L_0.StopGrabbing()
    camera_L_0.Close()
    camera_R_0.StopGrabbing()
    camera_R_0.Close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    val = input('Input 1-4 for taking image for calculation\n' + '(1-240degree  2-Top 3-120degree 4-0degree)\n' + 'Input:')
    if val == '1':
        serial_number_L = "40003776"
        serial_number_R = "40070109"
        camera_L_exposureTime = 15000   #for take the chessboard, can neglect
        camera_R_exposureTime = 15000   #for take the chessboard, can neglect

        address = 'degree240'

        gige_camera(serial_number_L, serial_number_R,camera_L_exposureTime,camera_R_exposureTime,address)

    if val == '2':
        serial_number_L = "40118981"
        serial_number_R = "40156407"
        camera_L_exposureTime = 15000   #for take the chessboard, can neglect
        camera_R_exposureTime = 15000   #for take the chessboard, can neglect

        address = 'top'

        gige_camera(serial_number_L, serial_number_R, camera_L_exposureTime, camera_R_exposureTime, address)


    if val == '3':
        serial_number_L = "40118988"
        serial_number_R = "40118977"
        camera_L_exposureTime = 15000   #for take the chessboard, can neglect
        camera_R_exposureTime = 15000   #for take the chessboard, can neglect

        address = 'degree120'

        gige_camera(serial_number_L, serial_number_R, camera_L_exposureTime, camera_R_exposureTime, address)

    if val == '4':
        serial_number_L = "40156421"
        serial_number_R = "40156409"
        camera_L_exposureTime = 15000   #for take the chessboard, can neglect
        camera_R_exposureTime = 15000   #for take the chessboard, can neglect

        address = 'degree0'

        gige_camera(serial_number_L, serial_number_R, camera_L_exposureTime, camera_R_exposureTime, address)
