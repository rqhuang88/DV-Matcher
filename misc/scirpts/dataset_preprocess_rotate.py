import os
from pathlib import Path
import numpy as np
import potpourri3d as pp3d
import open3d as o3d
import torch
import os
from tools import get_sampled_rotation_matrices_by_axisAngle

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


device = torch.device("cuda:0")


save_path='results/shape-rot'
save_name='scape'
angle = 1/6 # 30'
for root, dirs, files in os.walk('data/scape/shapes_test/'):
    for file in files:
        pth = os.path.join(root, file)
        verts, faces = pp3d.read_mesh(pth)  # off ob
        shape_name=os.path.splitext(file)[0]
        verts = torch.tensor(np.ascontiguousarray(verts)).float().to(device) 
        faces = torch.tensor(np.ascontiguousarray(faces))
        rot  = get_sampled_rotation_matrices_by_axisAngle(1,angle)
        rotated_vertices = torch.matmul(verts, rot[0])
        save_path_mesh = os.path.join(save_path ,save_name)
        if not os.path.exists(save_path_mesh):
            os.makedirs(save_path_mesh)
        save_mesh = o3d.geometry.TriangleMesh()
        save_mesh.vertices =  o3d.utility.Vector3dVector(rotated_vertices.squeeze().detach().cpu().numpy())
        save_mesh.triangles = o3d.utility.Vector3iVector(faces)
        o3d.io.write_triangle_mesh(os.path.join(save_path_mesh,f'{shape_name}.off'), save_mesh)