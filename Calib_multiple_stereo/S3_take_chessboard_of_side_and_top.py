import os
os.environ["PYLON_CAMEMU"] = "3"
from pypylon import genicam
from pypylon import pylon
import sys
import cv2
import time
from stereo_lib import*
ply_header = '''ply
format ascii 1.0
element vertex %(vert_num)d
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
end_header
'''
image_dir = "IMAGES_CALIB"
current_dir = os.getcwd()
image_savepath = os.path.join(current_dir, image_dir)
print("image save path", image_savepath)
if not os.path.isdir(os.path.abspath(image_savepath)):
    os.mkdir(image_savepath)
def compute_Disparity(imgpointsL, imgpointsR):
    disparity_points = []
    disparity_coors = []
    for i in range(0, 70):
        disparity_coors.append(imgpointsL[i][0])
        disparity_points.append(imgpointsL[i][0][0]-imgpointsR[i][0][0])
    print("disparity_points",disparity_points)
    return disparity_coors, disparity_points
def checker_detect(imgL,imgR):
    chessboardSize = (10,7)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    criteria_stereo= (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    objp = np.zeros((chessboardSize[0] * chessboardSize[1], 3), np.float32)
    objp[:,:2] = np.mgrid[0:chessboardSize[0],0:chessboardSize[1]].T.reshape(-1,2)
    size_of_chessboard_squares_mm = 5
    objp = objp * size_of_chessboard_squares_mm
    objpoints = []
    imgpointsL = [] 
    imgpointsR = []
    grayL = cv2.cvtColor(imgL, cv2.COLOR_RGB2GRAY)
    grayR = cv2.cvtColor(imgR, cv2.COLOR_RGB2GRAY)
    retL, cornersL = cv2.findChessboardCorners(grayL, chessboardSize, None)
    retR, cornersR = cv2.findChessboardCorners(grayR, chessboardSize, None)

        # If found, add object points, image points (after refining them)
    if retL == True and retR == True:
        objpoints.append(objp)
        cornersL = cv2.cornerSubPix(grayL, cornersL, (11,11), (-1,-1), criteria)
        cornersR = cv2.cornerSubPix(grayR, cornersR, (11,11), (-1,-1), criteria)
        cv2.drawChessboardCorners(imgL, chessboardSize, cornersL, retL)
        cv2.drawChessboardCorners(imgR, chessboardSize, cornersR, retR)
        cv2.imshow('img left', resized_img(imgL,15))
        cv2.imshow('img right', resized_img(imgR,15))
        # print("imgpointsL",cornersL)
        # print("imgpointsR",cornersR)
        cv2.waitKey(1)
    return cornersL, cornersR
def write_ply(fn, verts, colors):
    verts = verts.reshape(-1, 3)
    colors = colors.reshape(-1, 3)
    verts = np.hstack([verts, colors])
    with open(fn, 'wb') as f:
        f.write((ply_header % dict(vert_num=len(verts))).encode('utf-8'))
        np.savetxt(f, verts, fmt='%f %f %f %d %d %d ')
def write_ply_nColor(fn, verts, colors):
    verts = verts.reshape(-1, 3)
    colors = colors.reshape(-1, 3)
    verts = np.hstack([verts, colors])
    with open(fn, 'wb') as f:
        f.write((ply_header % dict(vert_num=len(verts))).encode('utf-8'))
        np.savetxt(f, verts, fmt='%f %f %f %d %d %d ')
def depth_map(imgL, imgR):
    """ Depth map calculation. Works with SGBM and WLS. Need rectified images, returns depth map ( left to right disparity ) """
    # SGBM Parameters -----------------
    window_size = 7  # wsize default 3; 5; 7 for SGBM reduced size image; 15 for SGBM full size image (1300px and above); 5 Works nicely

    left_matcher = cv2.StereoSGBM_create(
        minDisparity=28*32,
        numDisparities=10*32,  # max_disp has to be dividable by 16 f. E. HH 192, 256
        blockSize=window_size,
        P1=8 * 3 * window_size,
        # wsize default 3; 5; 7 for SGBM reduced size image; 15 for SGBM full size image (1300px and above); 5 Works nicely
        P2=32 * 3 * window_size,
        disp12MaxDiff=1,
        uniquenessRatio=10,
        speckleWindowSize=400,
        speckleRange=2,
        preFilterCap=63,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
    )
    right_matcher = cv2.ximgproc.createRightMatcher(left_matcher)
    # FILTER Parameters
    lmbda = 80000
    sigma = 1.3
    visual_multiplier = 6

    wls_filter = cv2.ximgproc.createDisparityWLSFilter(matcher_left=left_matcher)
    wls_filter.setLambda(lmbda)

    wls_filter.setSigmaColor(sigma)
    displ = left_matcher.compute(imgL, imgR)  # .astype(np.float32)/16
    dispr = right_matcher.compute(imgR, imgL)  # .astype(np.float32)/16
    displ = np.int16(displ)
    dispr = np.int16(dispr)
    filteredImg = wls_filter.filter(displ, imgL, None, dispr)  # important to put "imgL" here!!!

    filteredImg = cv2.normalize(src=filteredImg, dst=filteredImg, beta=0, alpha=255, norm_type=cv2.NORM_MINMAX)
    filteredImg = np.uint8(filteredImg)

    return filteredImg
def img_cal(img, mode):
    if mode=='UMat':
        img = cv2.UMat(img)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.GaussianBlur(img, (7, 7), 1.5)
        img = cv2.Canny(img, 0, 50)
        if type(img) == 'cv2.UMat': 
            img = cv2.UMat.get(img)
    return img
def compute3D(disparity_coors, disparity_points, Q):
    points3D = []
    color = []
    for i in range(0, np.asarray(disparity_coors).shape[0]):
        x = disparity_coors[i][0]
        y = disparity_coors[i][1]
        X= -round((x - 1296)*(1/Q[3][2])/disparity_points[i],2)
        Y= -round((y - 1024)*(-1/Q[3][2])/disparity_points[i],2)
        Z = -round((1/Q[3][2])*Q[2][3]/disparity_points[i],2)
        points3D.append([X,Y,Z])
        color.append([255,0,0])
    print("points3D",points3D)
    return points3D, color
##############top
def disparity_SGBM(left_image, right_image, minDisparity = 760, numDisparities= 160): #760 80,715
    '''
    TO CALCULATE DISPARITY IMAGE
    :param left_image:
    :param right_image:
    :param minDisparity:
    :param numDisparities:
    :return:
    '''

    # SGBM匹配參數設置
    if left_image.ndim == 2:
        img_channels = 1
    else:
        img_channels = 3
    blockSize = 7
    win_size = 7

    param = {'minDisparity': minDisparity,
             'numDisparities': numDisparities,
             'blockSize': blockSize,
             'P1': 8 * img_channels * win_size ** 2,
             'P2': 32 * img_channels * win_size ** 2,
             'disp12MaxDiff': 1,
             'preFilterCap': 63,
             'uniquenessRatio': 12,
             'speckleWindowSize': 400, #400
             'speckleRange': 2, #2
             'mode': cv2.STEREO_SGBM_MODE_SGBM_3WAY
             }
    # 構建SGBM對象
    sgbm = cv2.StereoSGBM_create(**param)
    # 計算視差圖
    disparity = sgbm.compute(left_image, right_image)
    # norm_image = cv2.normalize(disparity, None, alpha = 0, beta = 1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    disparity = disparity.astype(np.float32)/16 # if we use SGBM, we need disparity to divise to 16 , dtype=cv2.CV_8U
   
    #cv2.imshow('disp', resized_img(norm_image,25))
    
    # disparity_normalized = cv2.normalize(disparity, disparity, alpha=0, beta=255,norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    return disparity
class Orchid:
    def __init__(self,ImL, ImR,path_address, miniDisparity):
     
        self.ImL = ImL
        self.imgs = ImL
        # -----Calculate disparity
        ImL = cv2.cvtColor(ImL, cv2.COLOR_BGR2GRAY)
        ImR = cv2.cvtColor(ImR, cv2.COLOR_BGR2GRAY)
        disparity = disparity_SGBM(ImL, ImR, miniDisparity)
        self.disparity = disparity
        self.img = disparity
        self.X = self.img.shape[1]
        self.Y = self.img.shape[0]
        self.path_address = path_address
    def reconstruct_2d_to_3d(self):
        point1 = []
        color1 = []
        if len(self.imgs.shape) != len(self.img.shape):
            self.imgs= cv2.cvtColor(self.imgs, cv2.COLOR_BGR2GRAY)
        bitwiseAnd = cv2.bitwise_and(self.img, self.img, mask = self.imgs)
        cv_file = cv2.FileStorage()
        cv_file.open('./Calibrated/'+ str(self.path_address) + '/stereoMap.txt', cv2.FileStorage_READ)
        Q1 = cv_file.getNode('q').mat()
        print("Q",Q1)
        Q= np.float32([[1, 0, 0, -self.X / 2.0],
                      [0, -1, 0, self.Y / 2.0],
                      [0, 0, 0, Q1[2, 3]],
                      [0, 0, -Q1[3, 2], Q1[3, 3]]])
        print("Q",Q)
        points = cv2.reprojectImageTo3D(self.img, Q,True)
        self.allpoints = points
        colors = cv2.cvtColor(self.ImL, cv2.COLOR_BGR2RGB)
        mask = bitwiseAnd > bitwiseAnd.min()
        out_points = points[mask]
        out_colors = colors[mask]
        return out_points, out_colors
def generate_chessboard(imgL,imgR,path_address,minDisparity,name,path):
    orchid = Orchid(imgL, imgR,path_address,minDisparity)
    points_all, colors_all = orchid.reconstruct_2d_to_3d()
    write_ply(path + '/' +f'de_{name}.ply',points_all,colors_all)
    cv_file = cv2.FileStorage()
    cv_file.open('./Calibrated/'+str(path_address) + '/stereoMap.txt', cv2.FileStorage_READ)
    Q = cv_file.getNode('q').mat()
    imgpointsL, imgpointsR = checker_detect(imgL,imgR)
    disparity_coors, disparity_points =compute_Disparity(imgpointsL, imgpointsR)
    points, colors = compute3D(disparity_coors, disparity_points, Q)
    write_ply(path +'/'+f'checker_{name}.ply',np.asarray(points), np.asarray(colors))

def gige_camera(serial_number_L, serial_number_R,camera_L_exposureTime,camera_R_exposureTime,address,name1):
    # Camera parameters to undistort and rectify images
    calibrated_dir = "Calibrated"
    current_dir = os.getcwd()
    calib_savepath = os.path.join(current_dir, calibrated_dir)
    Img_name = address
    cv_file = cv2.FileStorage()
    cv_file.open(calib_savepath + '/'+ str(Img_name) + '/' + 'stereoMap.txt', cv2.FileStorage_READ)
    stereoMapL_x = cv_file.getNode('stereoMapL_x').mat()
    stereoMapL_y = cv_file.getNode('stereoMapL_y').mat()
    stereoMapR_x = cv_file.getNode('stereoMapR_x').mat()
    stereoMapR_y = cv_file.getNode('stereoMapR_y').mat()
    def undistortRectify(frameR, frameL):
        # Undistort and rectify images
        undistortedL= cv2.remap(frameL, stereoMapL_x, stereoMapL_y, cv2.INTER_LANCZOS4, cv2.BORDER_CONSTANT, 0)
        undistortedR= cv2.remap(frameR, stereoMapR_x, stereoMapR_y, cv2.INTER_LANCZOS4, cv2.BORDER_CONSTANT, 0)
        return undistortedR, undistortedL
    Check = False
    info_L_0 = pylon.DeviceInfo()
    info_L_0.SetSerialNumber(serial_number_L)

    print('imform',info_L_0)
    # 40003776  40118977
    camera_L_0 = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateDevice(info_L_0))
    camera_L_0.Open()
    camera_L_0.ExposureTimeAbs = camera_L_exposureTime
    camera_L_0.Width.SetValue(2592)
    camera_L_0.Height.SetValue(2048)

    info_R_0 = pylon.DeviceInfo()
    info_R_0.SetSerialNumber(serial_number_R)
    # 40118981   40070109
    camera_R_0 = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateDevice(info_R_0))
    camera_R_0.Open()
    camera_R_0.ExposureTimeAbs = camera_R_exposureTime
    camera_R_0.Width.SetValue(2592)
    camera_R_0.Height.SetValue(2048)
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

            imgR1, imgL1 = undistortRectify(imgR_0, imgL_0)
            # cv2.imshow('img_resizeL_0', img_resizeL_0)
            # cv2.imshow('img_resizeR_0', img_resizeR_0)
            # k = cv2.waitKey(1)

            if not os.path.isdir(os.path.abspath(image_savepath + '/' + str(address))):
                os.mkdir(image_savepath + '/' + str(address))
            path_images_final =  image_savepath + '/' + str(address)
            cv2.imwrite(image_savepath + '/' + str(address) + '/image_L_' + str(name1) + '.png', imgL1)
            cv2.imwrite(image_savepath + '/' + str(address) + '/image_R_' + str(name1) + '.png', imgR1)
            generate_chessboard(imgL=imgL1,imgR=imgR1, path_address=address,minDisparity=715,name=name1,path = path_images_final)
            print("images saved!")
            Check = True
            if Check == True:
                break
        grabResult1.Release()
        #grabResult2.Release()
    # Releasing the resource
    camera_L_0.StopGrabbing()
    camera_L_0.Close()
    camera_R_0.StopGrabbing()
    camera_R_0.Close()
    cv2.destroyAllWindows()
if __name__ == '__main__':
    val = input('Input 1-3 for taking image for calculation\n' + '(1-TOP_240degree  2-Top_0degree 3-TOP_120degree)\n' + 'Input:')
    if val == '1':
        serial_number_L = "40003776"
        serial_number_R = "40070109"
        camera_L_exposureTime = 20000   #for take the chessboard, can neglect
        camera_R_exposureTime = 20000   #for take the chessboard, can neglect
        address = '240degree'
        name = '240_degree'
        gige_camera(serial_number_L, serial_number_R,camera_L_exposureTime,camera_R_exposureTime,address,name)
        serial_number_L = "40118981"
        serial_number_R = "40156407"
        camera_L_exposureTime = 20000   #for take the chessboard, can neglect
        camera_R_exposureTime = 20000   #for take the chessboard, can neglect
        address = 'top'
        name = 'top_240'
        gige_camera(serial_number_L, serial_number_R, camera_L_exposureTime, camera_R_exposureTime, address,name)
        val = 0


    if val == '2':
        serial_number_L = "40118981"
        serial_number_R = "40156407"
        camera_L_exposureTime = 20000   #for take the chessboard, can neglect
        camera_R_exposureTime = 20000   #for take the chessboard, can neglect
        address = 'top'
        name = 'top_0'
        gige_camera(serial_number_L, serial_number_R, camera_L_exposureTime, camera_R_exposureTime, address,name)

        serial_number_L = "40156421"
        serial_number_R = "40156409"
        camera_L_exposureTime = 20000   #for take the chessboard, can neglect
        camera_R_exposureTime = 20000   #for take the chessboard, can neglect
        address = '0degree'
        name = '0_degree'
        gige_camera(serial_number_L, serial_number_R, camera_L_exposureTime, camera_R_exposureTime, address,name)
        val = 0

    if val == '3':
        serial_number_L = "40118981"
        serial_number_R = "40156407"
        camera_L_exposureTime = 20000   #for take the chessboard, can neglect
        camera_R_exposureTime = 20000   #for take the chessboard, can neglect
        address = 'top'
        name = 'top_120'
        gige_camera(serial_number_L, serial_number_R, camera_L_exposureTime, camera_R_exposureTime, address,name)

        serial_number_L = "40118988"
        serial_number_R = "40118977"
        camera_L_exposureTime = 20000   #for take the chessboard, can neglect
        camera_R_exposureTime = 20000   #for take the chessboard, can neglect
        address = '120degree'
        name = '120_degree'
        gige_camera(serial_number_L, serial_number_R, camera_L_exposureTime, camera_R_exposureTime, address,name)
        val = 0