# Name: Pham The Thinh
# Student's ID: M10903821
# Course: Computer Vision and Applications
import cv2
import numpy as np
import os
import PIL
import glob
import time
our_data = 'CalibrationData.txt'
intrinsics_fundament = np.genfromtxt(our_data, skip_header=1,skip_footer=5)
extrinsics = np.genfromtxt(our_data, skip_header=10,skip_footer=2)
K_l = intrinsics_fundament[0:3]
K_r = intrinsics_fundament[3:6]
Rt_l = extrinsics[0:3]
Rt_r = extrinsics[3:6]
#############
P_L = K_l.dot(Rt_l)
print("P matrix of left camera",P_L)
P_R = K_r.dot(Rt_r)
print("P matrix of right camera",P_R)
F = intrinsics_fundament[6:9]
print("Fundamental matrix of both camera", F)
############
# Load all files of both folders that contain the images of two cameras
files_L = np.sort(glob.glob("./L/*.JPG"))
files_R = np.sort(glob.glob('./R/*.JPG'))
#####
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
def nothing(x):
    pass
def resize_image(image,percent):
    if image.ndim == 3:
        x, y,_ = image.shape
        resized_result = cv2.resize(image, (int(y * percent / 100), int(x * percent / 100)),
                                    interpolation=cv2.INTER_AREA)
    else:
        x, y = image.shape
        resized_result = cv2.resize(image, (int(y * percent / 100), int(x * percent / 100)),
                                    interpolation=cv2.INTER_AREA)
    return resized_result
def filter_red(image,thre_l,thre_h):
    image = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
    kernel = np.ones((9, 9), np.float32) / 25
    img = cv2.filter2D(image, -1, kernel)
    bin_ret, binary = cv2.threshold(img, thre_l, thre_h, cv2.THRESH_OTSU)
    thinned = cv2.ximgproc.thinning(binary, cv2.ximgproc.THINNING_ZHANGSUEN)
    return thinned
def get_pixel(image):
    x_y = []
    [y, x] = image.shape[:2]
    for m in range(y):#y
        for s in range(x): #x
            if image[m,s] == 255:
                # y_n, x_n = np.where(np.all(image == color, axis=2))
                x_y.append([s,m,1])
    x_y = np.asarray(x_y,dtype = np.float64)
    return x_y
def get_pixel_right(image_R,F,x_y_L,x_y_R_o):
    lines1 = cv2.computeCorrespondEpilines(np.asarray(x_y_L[:, :2], dtype=np.float64).reshape(-1, 1, 2), 1, F)
    lines1 = lines1.reshape(-1, 3)
    img3 = drawlines(image_R,lines1, x_y_R_o)
    con_im = mapping_img(image_R, img3) #can change here
    x_y_R = get_pixel((con_im))
    return x_y_R
def drawlines(img1,lines,pts1):
    ''' we draw the epilines for the points in img1
        lines - corresponding epilines '''
    if img1.ndim !=3:
        r, c= img1.shape
    else:
        r,c,_= img1.shape
    plat_img = np.ones((1280,720),np.uint8)
    for r,pt1 in zip(lines,pts1):
        color = tuple(np.array([255,255,255]).tolist())
        color1 = tuple(np.array([0, 0, 0]).tolist())
        x0,y0 = map(int, [0, -r[2]/r[1] ])
        x1,y1 = map(int, [c, -(r[2]+r[0]*c)/r[1] ])
        img_line = cv2.line(plat_img, (x0,y0), (x1,y1), color,1)
    return img_line
def mapping_img(img1,img2):
    if img1.shape != img2.shape:
        img1 = cv2.cvtColor(img1,cv2.COLOR_RGB2GRAY)
    con_img = cv2.bitwise_and(img1,img2)
    return con_img
def find_point_pair(xyL,xyR,F):
    mp1= []
    mp2 =[]
    for m in range(len(xyR)):
        e = []
        for i in range(len(xyL)):
            am = np.transpose(xyL[i])
            # x'T * F * x : right image *fundamental matrix* left image
            error = xyR[m].dot(F.dot(am))
            e.append(error)
        c = np.min(abs(np.asarray(e)))
        index_e = abs(np.asarray(e)).tolist().index(c)
        # print("index", index_e)
        x_y_left = xyL[index_e]
        mp1.append([x_y_left[0],x_y_left[1]])
        mp2.append([xyR[m][0],xyR[m][1]])
    return mp1, mp2
def convert2d_3d(mpl1,mpr2,P_l,P_r):
    points = []
    for i in range(len(mpl1)):
        A = np.array(
        [[mpr2[i][0] * P_r[2][0] - P_r[0][0], mpr2[i][0] * P_r[2][1] - P_r[0][1], mpr2[i][0] * P_r[2][2] - P_r[0][2],mpr2[i][0] * P_r[2][3] - P_r[0][3]],
        [mpr2[i][1] * P_r[2][0] - P_r[1][0], mpr2[i][1] * P_r[2][1] - P_r[1][1], mpr2[i][1] * P_r[2][2] - P_r[1][2], mpr2[i][1] * P_r[2][3] - P_r[1][3]],
        [mpl1[i][0] * P_l[2][0] - P_l[0][0], mpl1[i][0] * P_l[2][1] - P_l[0][1], mpl1[i][0] * P_l[2][2] - P_l[0][2], mpl1[i][0] * P_l[2][3] - P_l[0][3]],
        [mpl1[i][1] * P_l[2][0] - P_l[1][0], mpl1[i][1] * P_l[2][1] - P_l[1][1], mpl1[i][1] * P_l[2][2] - P_l[1][2], mpl1[i][1] * P_l[2][3] - P_l[1][3]]])
        USV = np.linalg.svd(A)
        V = np.transpose(USV[2])
        X = np.array([[V[0][3]], [V[1][3]], [V[2][3]], [V[3][3]]])
        X = X / X[3][0]
        points.append([np.round(X[0][0], 0), np.round(X[1][0], 0), np.round(X[2][0])])
    return points
def get_pointcloud(points):
    all =[]
    point_xyz_1 = []
    for i in range(len(points)):
        for j in range(len(points[i])):
            subpoint = points[i]
            all.append([subpoint[j][0],subpoint[j][1],subpoint[j][2]])
            point_xyz_1.append([subpoint[j][0],subpoint[j][1],subpoint[j][2],1])
    return all,point_xyz_1
def write_xyzcolor(fn, verts, colors,cond):
    verts = np.asarray(verts)
    colors = np.asarray(colors)
    verts = verts.reshape(-1, 3)
    colors = colors.reshape(-1, 3)
    verts = np.hstack([verts, colors])
    if cond == "txt":
        with open(fn, 'w') as f:
            np.savetxt(f, verts, fmt='%f %f %f %d %d %d ')
    else:
        with open(fn, 'wb') as f:
            f.write((ply_header % dict(vert_num=len(verts))).encode('utf-8'))
            np.savetxt(f, verts, fmt='%f %f %f %d %d %d ')
if __name__ == "__main__":
    start = time.time()
    points = []
    colors = []
    number_image = 0
    for i in range(len(files_L)):
        print("file name", files_L[i])
        # Read image
        image_L_c = cv2.imread(files_L[i])
        image_R_c = cv2.imread(files_R[i])
        # We will filter noise and remain only red color
        image_L = filter_red(image_L_c,50,255)
        image_R = filter_red(image_R_c,50,255)
        mix_image= cv2.hconcat([image_L,image_R]) 
        # Next step, we are going to pick out each pixel that contains red color
        x_y_L_all = get_pixel(image_L)
        x_y_R_all = get_pixel(image_R)
        # This is an option, we get x y coordinates of the right image according to x y coordinates of the left image
        # via the fundamental matrix
        x_y_R = get_pixel_right(image_R,F,x_y_L_all,x_y_R_all)
        # We will find point pairs of both images
        mpL,mpR = find_point_pair(x_y_L_all,x_y_R_all,F) # if we put x_y_R_all directly is more density than x_y_R.
        # From the above result, we will compute 3D coordinate
        xyz_3d = convert2d_3d(mpL,mpR,P_L,P_R)
        points.append(xyz_3d)
        print("number_image", number_image)
        number_image +=1
    # Save point cloud into xyz format
    all_points,all_xyz1 = get_pointcloud(points) # return 2 types of format of all points
    ########
    '''# np.savetxt('M10903821_points.xyz', all_xyz1, fmt='%0.10g', delimiter=' ')
    # np.savetxt('M10903821_1.xyz', all_points, fmt='%0.10g', delimiter=' ')
    # all_points = np.genfromtxt('M10903821_1.xyz')
    # all_xyz1 = all_xyz1 = np.genfromtxt("M10903821_points.xyz")''' # Test code for color
    # # Then, we will find the color of object
    # Read image texture
    texture = cv2.imread('TextureImage.JPG')
    for j in range(len(all_xyz1)):
        xyz_T = np.transpose(np.array(all_xyz1[j])).reshape(4,1)
        # This is projective matrix of 3D on 2D image, I calculated it by matlab
        # Convert 3D point onto 2D image and get information uv on image of 3D points
        P = np.array([[21.9034, -0.1663, 1.2928, 829.0357],
                      [1.2204, 19.5062, -2.1049, 1592.4],
                      [0.0012, 0.00005, -0.00009, 1]])
        uv = P.dot(xyz_T)
        uv = uv/uv[2]
        # We will find color responding to position of each 3D points on image
        color = texture[int(np.round(uv[1],0)),int(np.round(uv[0],0))] # (B,G.R) and [y,x]==[v,u]       
        colors.append([color[2],color[1],color[0]]) # R,G,B
    # Save points and colors
    write_xyzcolor("M10903821.ply",all_points,colors,"ply")
    write_xyzcolor("M10903821.xyz", all_points, colors,"txt")
    print("Done!")
    print("Total time:", (time.time() - start))