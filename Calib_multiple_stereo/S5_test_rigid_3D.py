#!/usr/bin/env python3

import numpy as np
from rigid_transform_3D import rigid_transform_3D
import open3d as o3d
import copy
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
def write_ply(fn, verts, colors):
    verts = verts.reshape(-1, 3)
    colors = colors.reshape(-1, 3)
    verts = np.hstack([verts, colors])
    with open(fn, 'wb') as f:
        f.write((ply_header % dict(vert_num=len(verts))).encode('utf-8'))
        np.savetxt(f, verts, fmt='%f %f %f %d %d %d ')
mesh_top = o3d.io.read_point_cloud("/home/airlab/Desktop/Calib_multiple_stereo/PLY/checker_top_0.ply")
B1= []
B1.append([np.asarray(mesh_top.points)[3][0],np.asarray(mesh_top.points)[60][0], np.asarray(mesh_top.points)[69][0]])
B1.append([np.asarray(mesh_top.points)[3][1],np.asarray(mesh_top.points)[60][1], np.asarray(mesh_top.points)[69][1]])
B1.append([np.asarray(mesh_top.points)[3][2],np.asarray(mesh_top.points)[60][2], np.asarray(mesh_top.points)[69][2]])
B1 = np.asarray(B1)
#########
mesh = o3d.io.read_point_cloud("/home/airlab/Desktop/Calib_multiple_stereo/PLY/checker_0.ply")
A2= []
A2.append([np.asarray(mesh.points)[3][0],np.asarray(mesh.points)[60][0], np.asarray(mesh.points)[69][0]])
A2.append([np.asarray(mesh.points)[3][1],np.asarray(mesh.points)[60][1], np.asarray(mesh.points)[69][1]])
A2.append([np.asarray(mesh.points)[3][2],np.asarray(mesh.points)[60][2], np.asarray(mesh.points)[69][2]])
A2 = np.asarray(A2)
print("A2",A2)
# A = np.array(([2.37,9.04,-28.91],[-18.56,-1.09,11.79],[-495.9,-479.2,-469.1]))
# A1 = np.array(([-13.74,-38.28,-29.91],[18.98,14.63,-8.04],[-460.3,-467.6,-499.1]))
# A2 = np.array(([-37.59,-19.1,10.37],[-20.5,-31.73,-9.02],[-477.6,-492.5,-478]))
# B1 = np.array(([-7.15,-0.71,-38.89],[-13.9,10.87,23.72],[-452.9,-453.7,-460.1])) #top
# Recover R and t
ret_R, ret_t = rigid_transform_3D(A2, B1)

# Compare the recovered R and t with the original
A2_2 = np.asarray(mesh.points).T
print("A2",A2_2)

B2 = (ret_R@A2) + ret_t
uni_matrix = [0,0,0,1]
T = np.hstack((ret_R,ret_t))
trans= np.vstack((T,uni_matrix))
# trans_2 = np.asarray([[1,-0.003,0.0046,2.0268],[0.003,1,0.004,1.8282],[-0.0046,-0.004,1,-0.1081],[0,0,0,1]]) #120
trans_2 = np.asarray([[1,0.0004,0.0012,0.4701],[-0.0004,1,0.0057,2.5371],[-0.0012,-0.0057,1,0.0016],[0,0,0,1]])#0
# trans_2 = np.asarray([[1,-0.0002,-0.005,-2.1938],[0.0002,1,0.0028,1.1661],[0.005,-0.0028,1,0.2027],[0,0,0,1]])#240
T_all = np.dot(trans_2,trans)
mat = T_all
print("tall",T_all)
with open("translation0_matrix.txt","wb") as f:
    for line in mat:
        np.savetxt(f, mat)
# print("T",trans)
mesh_trans = copy.deepcopy(mesh).transform(trans)
# mesh_120_trans.paint_uniform_color([0, 0, 1])
mesh_all = copy.deepcopy(mesh_trans).transform(trans_2)
# mesh_all.paint_uniform_color([0, 1, 0])
# write_ply("aline_240.ply",np.asarray(mesh_trans.points),np.asarray(mesh_trans.colors))
# all = np.hstack((mesh_top,mesh_all))
# o3d.visualization.draw_geometries(all)
# Find the root mean squared error
mesh_0 = o3d.io.read_point_cloud("/home/airlab/Desktop/Calib_multiple_stereo/images/ply/0_1.ply")
mesh_120 = o3d.io.read_point_cloud("/home/airlab/Desktop/Calib_multiple_stereo/images/ply/0_2.ply")
mesh_240 = o3d.io.read_point_cloud("/home/airlab/Desktop/Calib_multiple_stereo/images/ply/0_3.ply")
mesh_top = o3d.io.read_point_cloud("/home/airlab/Desktop/Calib_multiple_stereo/images/ply/0_4.ply")
# mesh_120_trans = copy.deepcopy(mesh_0).transform(trans)
# # mesh_120_trans.paint_uniform_color([0, 0, 1])
# mesh_all = copy.deepcopy(mesh_120_trans).transform(trans_2)
a = copy.deepcopy(mesh_0).transform(T_all)
# mesh_120_trans.paint_uniform_color([0, 0, 1])
# mesh_all.paint_uniform_color([1, 0, 0])
a.paint_uniform_color([1, 1, 0.5])
# # mesh_all.paint_uniform_color([0, 1, 0])
# write_ply("aline_120.ply",np.asarray(mesh_120_trans.points),np.asarray(mesh_120_trans.colors))
all = np.hstack((mesh_top,a))
o3d.visualization.draw_geometries(all)
err = B2 - B1
err = err * err
err = np.sum(err)
rmse = np.sqrt(err/3)

print("Points B2")
print(B2)
print("")

print("Points B")
print(B1)
print("")

# print("Ground truth rotation")
# print(R)

print("Recovered rotation")
print(ret_R)
print("")

# print("Ground truth translation")
# print(t)

print("Recovered translation")
print(ret_t)
print("")

print("RMSE:", rmse)

if rmse < 1e-5:
    print("Everything looks good!")
else:
    print("Hmm something doesn't look right ...")