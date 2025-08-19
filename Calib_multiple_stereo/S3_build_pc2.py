import os
os.environ["PYLON_CAMEMU"] = "3"
from pypylon import genicam
from pypylon import pylon
import sys
import cv2
import time
from stereo_lib import*
import open3d as o3d

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
# TOP: 680, 240: 680, 0: 680, 

def compute_Disparity(imgpointsL, imgpointsR):
    disparity_points = []
    disparity_coors = []
    for i in range(0, 70):
        disparity_coors.append(imgpointsL[i][0])
        disparity_points.append(imgpointsL[i][0][0]-imgpointsR[i][0][0])
    # print("disparity_points",disparity_points)
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
    retL, cornersL = cv2.findChessboardCorners(resized_img(grayL,50), chessboardSize, None)
    retR, cornersR = cv2.findChessboardCorners(resized_img(grayR,50), chessboardSize, None)
    cornersL = cornersL*2
    cornersR = cornersR*2

        # If found, add object points, image points (after refining them)
    if retL == True and retR == True:
        objpoints.append(objp)
        cornersL = cv2.cornerSubPix(grayL, cornersL, (11,11), (-1,-1), criteria)
        cornersR = cv2.cornerSubPix(grayR, cornersR, (11,11), (-1,-1), criteria)
        cv2.drawChessboardCorners(imgL, chessboardSize, cornersL, retL)
        cv2.drawChessboardCorners(imgR, chessboardSize, cornersR, retR)
        # cv2.imshow('img left', resized_img(imgL,15))
        # cv2.imshow('img right', resized_img(imgR,15))
        # # print("imgpointsL",cornersL)
        # # print("imgpointsR",cornersR)
        # cv2.waitKey(1)
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
        minDisparity= 28*32, 
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
    # print("points3D",points3D)
    return points3D, color
##############top

def disparity_SGBM(left_image, right_image, minDisparity= 680, numDisparities= 160):
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
    blockSize = 7 #5
    win_size = 7 #5

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
    def __init__(self,ImL, ImR,path_address):
     
        self.ImL = ImL
        self.imgs = ImL
        # -----Calculate disparity
        ImL = cv2.cvtColor(ImL, cv2.COLOR_BGR2GRAY)
        ImR = cv2.cvtColor(ImR, cv2.COLOR_BGR2GRAY)
        disparity = disparity_SGBM(ImL, ImR)
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
        cv_file.open('/home/airlab/Desktop/Calib_multiple_stereo/Calibrated/top/stereoMap.txt', cv2.FileStorage_READ)
        Q1 = cv_file.getNode('q').mat()
        # print("Q",Q1)
        Q= np.float32([[1, 0, 0, -self.X / 2.0],
                      [0, -1, 0, self.Y / 2.0],
                      [0, 0, 0, Q1[2, 3]],
                      [0, 0, -Q1[3, 2], Q1[3, 3]]])
        # print("Q",Q)
        points = cv2.reprojectImageTo3D(self.img, Q,True)
        self.allpoints = points
        colors = cv2.cvtColor(self.ImL, cv2.COLOR_BGR2RGB)
        mask = bitwiseAnd > bitwiseAnd.min()
        out_points = points[mask]
        out_colors = colors[mask]
        return out_points, out_colors
if __name__ == '__main__':
    #*************
    # path_ply = "/home/airlab/Desktop/Calib_multiple_stereo/PLY/combine.ply"
    # pcd = o3d.io.read_point_cloud(path_ply)
    # o3d.visualization.draw_geometries([pcd])
    #*************

    converter = pylon.ImageFormatConverter()
    converter.OutputPixelFormat = pylon.PixelType_BGR8packed
    converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned
    maxCamerasToUse = 2
    exitCode = 0
    id_image = 0
    try:
        tlFactory = pylon.TlFactory.GetInstance()
        devices = tlFactory.EnumerateDevices()
        if len(devices) == 0:
            raise pylon.RuntimeException("No camera present.")
        cameras = pylon.InstantCameraArray(2)
        l = cameras.GetSize()
        for i, cam in enumerate(cameras):
            cam.Attach(tlFactory.CreateDevice(devices[i])) 
            print("Using device ", cam.GetDeviceInfo().GetModelName())
        cameras.Open()
        for idx, cam in enumerate(cameras):
            camera_serial = cam.DeviceInfo.GetSerialNumber()
            #print(f"set context {idx} for camera {camera_serial}")
            cam.SetCameraContext(idx)
        for idx, cam in enumerate(cameras):
            camera_serial = cam.DeviceInfo.GetSerialNumber()
            print(f"set Exposuretime {idx} for camera {camera_serial}")
            # cam.ExposureTimeAbs = 20000
            # cam.Width.SetValue(2596)
            # cam.Height.SetValue(2048)
            cam.ExposureTimeAbs = 20000
            cam.AcquisitionFrameRateEnable.SetValue(True)
            cam.AcquisitionFrameRateAbs.SetValue(5)
            cam.Width.SetValue(2592)
            cam.Height.SetValue(2048)
            #cam.AutoFunctionROIUseWhiteBalance.SetValue(True)
            #cam.BalanceWhiteAuto.SetValue("BalanceWhiteAuto_Once")
            # cam.CenterX.SetValue(True)
            # cam.CenterY.SetValue(True)
        cameras.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
        while cameras.IsGrabbing():
            grabResult = cameras.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
            if grabResult.GrabSucceeded():

                cameraContextValue = grabResult.GetCameraContext()
                image = converter.Convert(grabResult)
                img = image.GetArray()

                window = 'Camera-{}'.format(cameraContextValue)
                cv2.namedWindow(window, cv2.WINDOW_AUTOSIZE)
                cv2.imshow(window, resized_img(img,15))
                print("id camera",cameraContextValue )
                if cameraContextValue==0:
                    #if 0,120,240 change IR
                    frameL = img
                    # frameR = img
                    # cv2.imshow("frameL", resized_img(frameR,15))

                else:
                    # frameL = img
                    frameR = img
                    # cv2.imshow("frameL", resized_img(frameL,15))
                    cv2.imwrite('images/R-test.png'.format(id_image),frameR)
                    cv2.imwrite('images/L-test.png'.format(id_image),frameL)
                    cv2.destroyAllWindows()
                    cameras.Close()
                    break
                k = cv2.waitKey(1)
    except genicam.GenericException as e:
        # print("An exception occurred.", e.GetDescription())
        exitCode = 1
    path_address = 'images/'
    
    ImL = cv2.imread("images/L-test.png",1)
    ImR = cv2.imread("images/R-test.png",1)
    imgR, imgL = undistortRectify(ImR, ImL)
    cv2.imwrite('images/R-test_calib.png',imgR)
    cv2.imwrite('images/L-test_calib.png',imgL)
    orchid = Orchid(imgL, imgR, path_address)
    #cv2.imshow('disp', resized_img(orchid,25))
    #cv2.waitKey(0)
    points_all, colors_all = orchid.reconstruct_2d_to_3d()
    # print(points_all)
    verts = points_all.reshape(-1, 3)
    colors = colors_all.reshape(-1, 3) 
    colors = np.asarray(colors/255) # rescale to 0 to 1
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(verts)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    o3d.visualization.draw_geometries([pcd])
     
    write_ply('PLY/chessboard_top.ply',points_all,colors_all)


    cv_file = cv2.FileStorage()
    cv_file.open('/home/airlab/Desktop/Calib_multiple_stereo/Calibrated/top/stereoMap.txt', cv2.FileStorage_READ)
    Q = cv_file.getNode('q').mat()
    imgpointsL, imgpointsR = checker_detect(imgL,imgR)
    disparity_coors, disparity_points =compute_Disparity(imgpointsL, imgpointsR)

    points, colors = compute3D(disparity_coors, disparity_points, Q)
    write_ply('PLY/chessboard_checker_top.ply',np.asarray(points), np.asarray(colors))
    cv2.destroyAllWindows()
