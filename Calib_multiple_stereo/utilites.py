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
    pds = o3d.io.read_point_cloud("{}".format(fn))
    return pds
def draw_geometries(pcds):
    """
    Draw Geometries
    Args:
        - pcds (): [pcd1,pcd2,...]
    """
    o3d.visualization.draw_geometries(pcds)

def draw_plane(origin=None, vec1=None, vec2=None, color=[1, 0, 1]):
    '''
    draw plane from 2 vectors
    :param origin:
    :param vec1:
    :param vec2:
    :return: plane
    '''
    scale_h = 8
    scale_v = 4
    points = []
    vec1 = np.asarray(vec1)
    vec2 = np.asarray(vec2)
    unit_vector1 = vec1/np.linalg.norm(vec1)
    unit_vector2 = vec2/np.linalg.norm(vec2)
    origin = np.asarray(origin)
    points.append(origin + scale_h*unit_vector1 + scale_v*unit_vector2)
    points.append(origin + scale_v*unit_vector2 - scale_h*unit_vector1)
    points.append(origin + scale_h*unit_vector1 - scale_v*unit_vector2)
    points.append(origin - scale_h*unit_vector1 - scale_v*unit_vector2)
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

def draw_point(sphere, color=[0, 1, 0]):
    mesh_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.02, resolution=10)
    mesh_sphere.translate(sphere)
    mesh_sphere.paint_uniform_color(color)
    mesh_sphere.compute_vertex_normals()
    return mesh_sphere

def get_arrow(origin=None, end=None, vec=None, cond=None):
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
        cone_height = scale * 0.1
        cylinder_height = scale * 0.9
        cone_radius = scale / 10
        cylinder_radius = scale / 20
        mesh_frame = o3d.geometry.TriangleMesh.create_arrow(cone_radius=0.5,
                                                            cone_height=cone_height,
                                                            cylinder_radius=0.2,
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