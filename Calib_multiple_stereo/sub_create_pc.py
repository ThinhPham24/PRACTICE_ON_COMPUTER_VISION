import numpy as np
import cv2
from matplotlib import pyplot as plt
import os
import json
import open3d as o3d
import math
from mpl_toolkits.mplot3d import Axes3D
import skspatial.objects as Line
import skspatial.objects as Points
import skspatial.plotting as plot_3d
from skspatial.objects import Line
from skspatial.objects import Plane
from skspatial.objects import Point
from skspatial.plotting import plot_3d
from scipy.spatial import distance
import time
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
def write_ply(verts, colors,number,idx):
    verts = verts.reshape(-1, 3)
    colors = colors.reshape(-1, 3)
    verts = np.hstack([verts, colors])
    with open("/home/airlab/Desktop/Calib_multiple_stereo/images/ply/{}_{}.ply".format(number,idx), 'wb') as f:
        f.write((ply_header % dict(vert_num=len(verts))).encode('utf-8'))
        np.savetxt(f, verts, fmt='%f %f %f %d %d %d ')
    # pds = o3d.io.read_point_cloud("{}".format(fn))
    # return pds
class Display:
    def __init__(seft):
        print('Display function')
    def draw_geometries(seft, pcds):
        """
        Draw Geometries
        Args:
            - pcds (): [pcd1,pcd2,...]
        """
        o3d.visualization.draw_geometries(pcds)

    def draw_plane(seft, origin=None, vec1=None, vec2=None, color=[1, 1, 0.6]):
        '''
        draw plane from 2 vectors
        :param origin:
        :param vec1:
        :param vec2:
        :return: plane
        '''
        scale = 10
        points = []
        vec1 = np.asarray(vec1)
        vec2 = np.asarray(vec2)
        unit_vector1 = vec1/np.linalg.norm(vec1)
        unit_vector2 = vec2/np.linalg.norm(vec2)
        origin = np.asarray(origin)
        points.append(origin + scale*unit_vector1 + scale*unit_vector2)
        points.append(origin + scale*unit_vector2 - scale*unit_vector1)
        points.append(origin + scale*unit_vector1 - scale*unit_vector2)
        points.append(origin - scale*unit_vector1 - scale*unit_vector2)
        triangles = np.array([
            [0, 1, 2],
            [0, 2, 3],
            [1, 3, 2],
            [1, 0, 2],
            [3, 2, 1],
            [3, 1, 0]
        ])
        plane_mesh = o3d.geometry.TriangleMesh()
        plane_mesh.vertices = o3d.utility.Vector3dVector(points)
        plane_mesh.triangles = o3d.utility.Vector3iVector(triangles)
        plane_mesh.compute_vertex_normals()
        plane_mesh.paint_uniform_color(color)
        return plane_mesh

    def draw_point(seft, sphere, color=[0, 1, 0]):
        mesh_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.02, resolution=10)
        mesh_sphere.translate(sphere)
        mesh_sphere.paint_uniform_color(color)
        mesh_sphere.compute_vertex_normals()
        return mesh_sphere

    def get_arrow(seft, origin=None, end=None, vec=None, cond=None):
        """
        Creates an arrow from an origin point to an end point,
        or create an arrow from a vector vec starting from origin.
        Args:
            - end (): End point. [x,y,z]
            - vec (): Vector. [i,j,k]
        """
        def vector_magnitude(vec):
            """
            Calculates a vector's magnitude.
            Args:
                - vec ():
            """
            magnitude = np.sqrt(np.sum(vec ** 2))
            return (magnitude)
        def caculate_align_mat(pVec_Arr):
            def get_cross_prod_mat(pVec_Arr):
                # pVec_Arr shape (3)
                qCross_prod_mat = np.array([
                    [0, -pVec_Arr[2], pVec_Arr[1]],
                    [pVec_Arr[2], 0, -pVec_Arr[0]],
                    [-pVec_Arr[1], pVec_Arr[0], 0],
                ])
                return qCross_prod_mat

            scale = np.linalg.norm(pVec_Arr)
            pVec_Arr = pVec_Arr / scale
            # must ensure pVec_Arr is also a unit vec.
            z_unit_Arr = np.array([0, 0, 1])
            z_mat = get_cross_prod_mat(z_unit_Arr)

            z_c_vec = np.matmul(z_mat, pVec_Arr)
            z_c_vec_mat = get_cross_prod_mat(z_c_vec)

            if np.dot(z_unit_Arr, pVec_Arr) == -1:
                qTrans_Mat = -np.eye(3, 3)
            elif np.dot(z_unit_Arr, pVec_Arr) == 1:
                qTrans_Mat = np.eye(3, 3)
            else:
                qTrans_Mat = np.eye(3, 3) + z_c_vec_mat + np.matmul(z_c_vec_mat,
                                                                    z_c_vec_mat) / (1 + np.dot(z_unit_Arr, pVec_Arr))

            # qTrans_Mat *= scale
            qTrans_Mat
            return qTrans_Mat

        def create_arrow(scale=10, color=[1, 0, 1]):
            """
            Create an arrow in for Open3D
            """
            cone_height = scale * 0.2
            cylinder_height = scale * 0.8
            cone_radius = scale / 10
            cylinder_radius = scale / 20
            mesh_frame = o3d.geometry.TriangleMesh.create_arrow(cone_radius=0.2,
                                                                cone_height=cone_height,
                                                                cylinder_radius=0.05,
                                                                cylinder_height=cylinder_height)
            mesh_frame.paint_uniform_color(color)
            mesh_frame.compute_vertex_normals()
            return (mesh_frame)
        scale = 50
        Ry = Rz = np.eye(3)
        T = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
        T[:3, -1] = origin
        if end is not None:
            vec = np.array(end) - np.array(origin)
        elif vec is not None:
            vec = np.array(vec)
        if end is not None or vec is not None:
            scale = vector_magnitude(vec)
            R = caculate_align_mat(vec)
        if cond[0] == 1:
            mesh = create_arrow(scale, [0, 0, 0]) # black Tiny bud
        elif cond[0] == 0 and cond[1]==0:
            mesh = create_arrow(scale, [1, 0, 1]) #vector of cutting plane
        elif cond[0] == 2 and cond[1] == 0:
            mesh = create_arrow(scale, [0, 0, 1]) # blue small bud
        elif cond[0] == 2 and cond[1] == 1:
            mesh = create_arrow(scale,[0.82, 0.57, 0.46])  # big length in small buds Green Yellow
        elif cond[0] ==3 and cond[1] == 0:
            mesh = create_arrow(scale, [1, 0, 0])  # red big bud
        elif cond[0]==3 and cond[1] ==1:
            mesh = create_arrow(scale, [1, 0.6, 0])  #big length in big buds Dark organge
        # Create the arrow
        mesh.rotate(R, center=np.array([0, 0, 0]))
        mesh.translate(origin)
        return (mesh)

    
def disparity_SGBM(left_image, right_image, minDisparity, numDisparities):
    # SGBM匹配參數設置
    if left_image.ndim == 2:
        img_channels = 1
    else:
        img_channels = 3
    blockSize = 7
    param = {'minDisparity': minDisparity,
             'numDisparities': numDisparities,
             'blockSize': blockSize,
             'P1': 8 * img_channels * blockSize ** 2,
             'P2': 32 * img_channels * blockSize ** 2,
             'disp12MaxDiff': 1, #1
             'preFilterCap': 63,
             'uniquenessRatio': 12, #20
             'speckleWindowSize': 400, #100
             'speckleRange': 2, #2
             'mode': cv2.STEREO_SGBM_MODE_SGBM_3WAY
             }
    # 構建SGBM對象
    sgbm = cv2.StereoSGBM_create(**param)
    # 計算視差圖
    disparity = sgbm.compute(left_image, right_image)
    disparity = disparity.astype(np.float32) / 16
    return disparity
def resized_img(img,percent):
    width = int(img.shape[1] * percent / 100)
    height = int(img.shape[0] * percent / 100)
    dim = (width, height)
    # resize image
    resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
    return resized
def reconstruct_2d_to_3d(ImL,ImR,minDis,numberDis,HSV,Q1):
    # -----Calculate disparity
    minDis = int(minDis)
    ImL1 = cv2.cvtColor(ImL,cv2.COLOR_BGR2GRAY)
    ImR1 = cv2.cvtColor(ImR,cv2.COLOR_BGR2GRAY)
    disparity = disparity_SGBM(ImL1, ImR1, minDis, numberDis)
    image_blur = cv2.GaussianBlur(ImL, (5, 5), 0) #5
    lower = np.asarray(HSV[0])
    upper = np.asarray(HSV[1])
    # Convert to HSV format and color threshold
    image_hsv = cv2.cvtColor(image_blur, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(image_hsv, lower, upper)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))#5
    mask_closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask_clean = cv2.morphologyEx(mask_closed, cv2.MORPH_OPEN, kernel)
    # th, mask_clean = cv2.threshold(mask_clean, 127, 255, cv2.THRESH_BINARY_INV)
    # cv2.imshow("mask", resized_img(mask_clean, 30))
    contours, hierarchy = cv2.findContours(mask_clean, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # find largest area contour
    max_area1 = -1
    # max_area2 = -1
    for i in range(len(contours)):
        area = cv2.contourArea(contours[i])
        if area > max_area1:
            cnt = contours[i]
            max_area1 = area
            contour = contours[i]
    # for i in range(len(contours)):
    #     area = cv2.contourArea(contours[i])
    #     if area > max_area2 and area<max_area1:
    #         cnt = contours[i]
    #         max_area2 = area
    #         contour = contours[i]
    img_new = np.zeros((2048, 2592), dtype=np.uint8)
    cv2.drawContours(img_new, [contour], -1, 255, cv2.FILLED)
    mask_clean = cv2.bitwise_and(mask_clean,img_new)
    # IN,im_th = cv2.threshold(img_new, 10, 255, cv2.THRESH_OTSU)
    # img_new_fill = im_th.copy()
    # h, w = im_th.shape[:2]
    # mask_1 = np.zeros((h + 2, w + 2), np.uint8)
    # cv2.floodFill(img_new_fill, mask_1, (0, 0), 255)
    # img_new_fill_inv = cv2.bitwise_not(img_new_fill)
    # im_th = img_new | img_new_fill_inv
    # cv2.imshow("palt",resized_img(img_new,30))
    # kernel = np.ones((5, 5), np.uint8)
    # mask = cv2.erode(mask,kernel, iterations=3)
    # mask = cv2.dilate(mask,kernel, iterations=3)
    # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    # mask_closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    # mask_clean = cv2.morphologyEx(mask_closed, cv2.MORPH_OPEN, kernel)
    plat = np.zeros((2048, 2592), dtype=np.uint8)
    plat[550:1800, 1200:2300] = 0
    imgs = cv2.bitwise_or(mask_clean, plat)
    # cv2.imshow("oaa1", resized_img(mask_clean, 30))
    # cv2.waitKey(0)
    # cv2.destroyWindow("palt")
    # ---------------------------------
    img = disparity
    X = img.shape[1]
    Y = img.shape[0]
    point1 = []
    color1 = []
    if len(imgs.shape) != len(img.shape):
        imgs= cv2.cvtColor(imgs, cv2.COLOR_BGR2GRAY)
    bitwiseAnd = cv2.bitwise_and(img, img, mask = imgs)
    Q = np.float32([[1, 0, 0, -X / 2.0],
                    [0, -1, 0, Y / 2.0],
                    [0, 0, 0, Q1[2, 3]],
                    [0, 0, -Q1[3, 2], Q1[3, 3]]])
    # print('QUASNSD', Q)
    points = cv2.reprojectImageTo3D(bitwiseAnd, Q)
    colors = cv2.cvtColor(ImL, cv2.COLOR_BGR2RGB)
    mask = bitwiseAnd > bitwiseAnd.min()
    out1_points = points[mask]
    out_colors = colors[mask]
    mean = np.mean(out1_points[:, 2])
    for i in range(len(out1_points)):
        if out1_points[i, 2] > mean-10: #-5
            point1.append(out1_points[i, :])
            color1.append(out_colors[i, :])
    out_points = np.asarray(point1)
    out_colors = np.asarray(color1)
    return out_points, out_colors, bitwiseAnd