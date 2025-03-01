from cal_ico import cal_icosahedron
import os
from pathlib import Path
import numpy as np
import potpourri3d as pp3d
import open3d as o3d
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from itertools import permutations
import os


def compute_vertex_normals(vertices, faces):
    """
    Computes the vertex normals of a mesh given its vertices and faces.
    vertices: a tensor of shape (num_vertices, 3) containing the 3D positions of the vertices
    faces: a tensor of shape (num_faces, 3) containing the vertex indices of each face
    returns: a tensor of shape (num_vertices, 3) containing the 3D normals of each vertex
    """
    # Compute the face normals
    p0 = vertices[faces[:, 0], :]
    p1 = vertices[faces[:, 1], :]
    p2 = vertices[faces[:, 2], :]
    face_normals = torch.cross(p1 - p0, p2 - p0)
    face_normals = face_normals / torch.norm(face_normals, dim=1, keepdim=True)    # Accumulate the normals for each vertex
    vertex_normals = torch.zeros_like(vertices)
    vertex_normals.index_add_(0, faces[:, 0], face_normals)
    vertex_normals.index_add_(0, faces[:, 1], face_normals)
    vertex_normals.index_add_(0, faces[:, 2], face_normals)    # Normalize the accumulated normals
    vertex_normals = vertex_normals / torch.norm(vertex_normals, dim=1, keepdim=True)
    
def filter_rows(A, B):
    # 初始化一个空列表，用来收集满足条件的行
    filtered_rows = []

    for row in A:
        # 检查每行的元素是否都在B中
        if np.all(np.in1d(row, B)):
            # 如果是，那就保留这一行
            filtered_rows.append(row)

    # 将满足条件的行转化为一个NumPy数组
    filtered_A = np.array(filtered_rows)

    return filtered_A

save_path='result/data'
save_name='shrec19_r'

for root, dirs, files in os.walk('../../data/shrec19_r/shapes_test'):
    for file in files:
        pth = os.path.join(root, file)
        verts, faces = pp3d.read_mesh(pth)  # off ob
        shape_name=os.path.splitext(file)[0]
        verts = torch.tensor(np.ascontiguousarray(verts)).float()
        faces = torch.tensor(np.ascontiguousarray(faces))
        # verts = pc_normalize(verts)
        target_normal = compute_vertex_normals(verts,faces)

        rotation_matrix = cal_icosahedron()
        for i in range(12):
            print(i)
            rotated_normal = torch.matmul(target_normal, torch.tensor(rotation_matrix[i]).float())
            idx_partial = torch.from_numpy(np.asarray(np.where(rotated_normal[:,2]>0))).long().squeeze()
            verts_pc = verts[idx_partial]
            idx = idx_partial.detach().cpu().squeeze(0).numpy()
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(verts_pc)
            filtered_face = filter_rows(faces.numpy(), idx)
            for pair in filtered_face:
                pair[0] = np.where(idx == pair[0])[0][0]
                pair[1] = np.where(idx == pair[1])[0][0]
                pair[2] = np.where(idx == pair[2])[0][0]

            filtered_face = torch.tensor(filtered_face)
            save_path_t = os.path.join(save_path ,save_name, f'index_partial')
            if not os.path.exists(save_path_t):
                os.makedirs(save_path_t)
            filename_index = f'index_{shape_name}_view_{i+1}.txt'
            idx1p = idx_partial.detach().cpu().squeeze(0).numpy()
            np.savetxt(os.path.join(save_path_t, filename_index), idx1p, fmt='%i')

            save_path_t = os.path.join(save_path ,save_name, f'points')
            if not os.path.exists(save_path_t):
                os.makedirs(save_path_t)
            filename_pcd = f'pcd_{shape_name}_view_{i+1}.ply'
            o3d.io.write_point_cloud(os.path.join(save_path_t, filename_pcd),pcd)

            save_path_mesh = os.path.join(save_path ,save_name, f'mesh')
            if not os.path.exists(save_path_mesh):
                os.makedirs(save_path_mesh)
            save_mesh = o3d.geometry.TriangleMesh()
            save_mesh.vertices =  o3d.utility.Vector3dVector(verts_pc.squeeze().detach().cpu().numpy())
            save_mesh.triangles = o3d.utility.Vector3iVector(filtered_face)
            o3d.io.write_triangle_mesh(os.path.join(save_path_mesh,f'{shape_name}_view_{i+1}.off'), save_mesh)