# Adapted from https://github.com/nintendops/DynamicFusion_Body/blob/b6b31d890974cc870d9b439fd9647d5934eeedaa/core/fusion_dm.py

import sys
import numpy as np
import torch
import torch.nn as nn
from scipy.spatial import KDTree
import torch.nn.functional as F
from psbody.mesh import Mesh

from .mesh_sampling import generate_transform_matrices

from .utils import col, batch_rodrigues

eps = sys.float_info.epsilon
import matplotlib.pyplot as plt

def farthest_point_sample(xyz, npoint):
    # xyz = xyz.unsqueeze(0)
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N,dtype=torch.float32).to(device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    return centroids


def index_points_idx(points, idx):
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points

def index_points(points, idx):
    """
    :param points: points.shape == (B, N, C)
    :param idx: idx.shape == (B, N, K)
    :return:indexed_points.shape == (B, N, K, C)
    """
    raw_shape = idx.shape
    idx = idx.reshape(raw_shape[0], -1)
    res = torch.gather(points, 1, idx[..., None].expand(-1, -1, points.shape[-1]))
    return res.view(*raw_shape, -1)

class PyTorchKDTree:
    def __init__(self, nodes):
        self.nodes = nodes

    def query(self, point, k=1):
        # 计算点与所有节点之间的距离
        distances = torch.cdist(self.nodes.float(),point.float())
        # 获取最近的k个点的索引
        _, indices = torch.topk(distances, k, dim=2)
        dist_idx = index_points(distances,indices)
        # 返回索引和对应的距离
        return indices, dist_idx
    
class DeformationGraph_geod(nn.Module):
    def __init__(self, radius=0.1, k=3, sampling_strategy='qslim'):
        super().__init__()
        
        self.radius = radius
        self.k = k
        self.max_neigh_num = 18
        self.sampling_strategy = sampling_strategy
        self.one_ring_neigh = []
        self.nodes_idx = None
        self.weights = None
        self.influence_nodes_idx = []
        self.dists = []
        
        # self.pre_idx = np.zeros()
        
    # def construct_graph_euclidean(self, vertices=None,geod=None,device=None):
    #         if self.sampling_strategy == 'qslim':
    #             self.nodes_idx = farthest_point_sample(torch.tensor(vertices),vertices.shape[1]//2)
    #             self.nodes = index_points_idx(vertices,self.nodes_idx)
    #             node_tree = PyTorchKDTree(self.nodes)
    #             self.max_neigh_num = 9
    #             _, self.one_ring_neigh = node_tree.query(self.nodes, 9)
    #         print('one_ring_neigh',self.one_ring_neigh.shape,geod.shape)
    #         # exit()
    #         geod_mat = -index_points(geod,self.nodes_idx).transpose(2,1) # (B,N,K)
    #         self.dists,self.influence_nodes_idx = geod_mat.topk(self.k,dim=-1) #
    #         self.dists = -self.dists
    #         _, self.pre_idx = geod_mat.topk(1,dim=-1)
    #         node_tree_all = PyTorchKDTree(vertices)
    #         dist,_ = node_tree_all.query(vertices,2)
    #         self.sigma =   20  * torch.mean(dist[:,:,1].float(),dim=1)
    #         # exit()
    #         self.weights = torch.exp(
    #             -((self.dists**2)) / (2 * self.sigma[:, None, None] * self.sigma[:, None, None])
    #         )
    #         # self.weights = torch.tensor(self.weights/col(self.weights.sum(1))).to(device)
    #         sums = self.weights.sum(dim=2, keepdim=True)
    #         self.weights = self.weights / sums
    #         print('weights',self.weights.shape)
    #         self.influence_nodes_idx = self.influence_nodes_idx.to(device)
    #         print('influence_nodes_idx',self.influence_nodes_idx.shape)
        
    # def construct_graph(self, vertices=None, faces=None,geod=None,device=None):
    #     self.faces = faces
    #     if self.sampling_strategy == 'qslim':
    #         m = Mesh(v=vertices, f=faces)
    #         M, A, D = generate_transform_matrices(m, [2])
    #         self.graph = M[1]
    #         # nodes_v = M[1].v
    #         self.nodes_idx = D[0].nonzero()[1]
    #         adj_mat = A[1].toarray()
    #         for i in range(adj_mat.shape[0]):
    #             self.one_ring_neigh.append(adj_mat[i].nonzero()[0].tolist() + [i]*(self.max_neigh_num-len(adj_mat[i].nonzero()[0])))
    #         self.one_ring_neigh = torch.tensor(self.one_ring_neigh).to(device) 
    #     print('one_ring_neigh',self.one_ring_neigh.shape)
    #     geod_mat = torch.from_numpy(-geod[self.nodes_idx]).transpose(1,0)
    
    #     self.dists,self.influence_nodes_idx = geod_mat.topk(self.k,dim=-1)
    #     self.dists = -self.dists

    #     # geod_mat_pre = geod_mat.transpose(1,0)
    #     _, self.pre_idx = geod_mat.topk(1,dim=-1)
    #     self.sigma =   20  * torch.mean(compute_edge_lengths(torch.tensor(self.graph.v).to(vertices.device),torch.tensor(self.graph.f.astype(np.int64)).to(vertices.device)))
    #     # self.weights = torch.tensor(self.weights/col(self.weights.sum(1))).to(device)
    #     self.weights = torch.exp(
    #         -((self.dists**2)) / (2 * self.sigma * self.sigma)
    #     )
    #     self.weights = torch.tensor(self.weights/col(self.weights.sum(1))).to(device)
    #     print('weights',self.weights.shape)
    #     self.influence_nodes_idx = self.influence_nodes_idx.to(device)
    #     print('influence_nodes_idx',self.influence_nodes_idx.shape)
        
    # def forward(self, vertices, opt_d_rotations, opt_d_translations):
    #     batch = vertices.shape[0]
    #     opt_d_rotmat = torch.tensor([]).to(opt_d_rotations.device)
    #     for i in range(batch):
    #         opt_d_rot = batch_rodrigues(opt_d_rotations[i,:,:]).unsqueeze(0) # 1 * N_c * 3 * 3
    #         opt_d_rotmat = torch.cat([opt_d_rotmat,opt_d_rot],dim=0)
    #     nodes =  index_points_idx(vertices,self.nodes_idx)
    #     print(self.influence_nodes_idx.shape)
    #     print(self.nodes.shape)
    #     exit()
    #     influence_nodes_v = nodes[self.influence_nodes_idx.reshape((-1,))]# .reshape((6890,3,3))
    #     opt_d_r = opt_d_rotmat[0, self.influence_nodes_idx.reshape((-1,)), ...]# .reshape((6890,3,3,3)) 
    #     opt_d_t = opt_d_translations[0, self.influence_nodes_idx.reshape((-1,)), ...]# .reshape((6890,3,3))

    #     warpped_vertices = (torch.einsum('bij, bkj->bki', opt_d_r, (vertices.repeat_interleave(3, dim=0) - influence_nodes_v).unsqueeze(1)).squeeze(1) \
    #                         + influence_nodes_v + opt_d_t).reshape((vertices.shape[0],3,3)) * (self.weights.unsqueeze(-1))

    #     warpped_vertices = warpped_vertices.sum(axis=1).float()
        

        
        # diff_term = (nodes + opt_d_translations[0]).repeat_interleave(self.max_neigh_num, dim=0) - \
        #             (nodes[self.one_ring_neigh.reshape((-1,))] + opt_d_translations[0][self.one_ring_neigh.reshape((-1,))]) - \
        #              torch.einsum('bij, bkj->bki', opt_d_rotmat[0].repeat_interleave(self.max_neigh_num, dim=0), \
        #             (nodes.repeat_interleave(self.max_neigh_num, dim=0) - nodes[self.one_ring_neigh.reshape((-1,))]).unsqueeze(1)).squeeze(1)

        # sr_term = opt_d_rotmat[0].repeat_interleave(self.max_neigh_num, dim=0) - opt_d_rotmat[0][self.one_ring_neigh.reshape((-1,))]
        
        # sr_loss = torch.mean(sr_term**2) 
        # arap_loss = torch.sum(diff_term ** 2) / self.nodes_idx.shape[0]
        

        # return warpped_vertices.unsqueeze(0), arap_loss ,sr_loss
    def construct_graph_euclidean(self, vertices=None,geod=None,device=None):
            if self.sampling_strategy == 'qslim':
                self.nodes_idx = farthest_point_sample(torch.tensor(vertices).unsqueeze(0),vertices.shape[0]//2).squeeze().numpy()
                self.nodes = vertices[self.nodes_idx]
                node_tree = KDTree(self.nodes)
                self.max_neigh_num = 9
                _, self.one_ring_neigh = node_tree.query(self.nodes, 9)
            # print('one_ring_neigh',self.one_ring_neigh.shape,geod.shape)
            # exit()
            geod_mat = torch.from_numpy(-geod[self.nodes_idx]).transpose(1,0) # (N,K)
            self.dists,self.influence_nodes_idx = geod_mat.topk(self.k,dim=-1) #
            self.dists = -self.dists
            _, self.pre_idx = geod_mat.topk(1,dim=-1)
            node_tree_all = KDTree(vertices)
            dist,_ = node_tree_all.query(vertices,2)
            self.sigma =   20  * torch.mean(torch.tensor(dist[:,1]))
            # print(dist.shape)
            # exit()
            self.weights = torch.exp(
                -((self.dists**2)) / (2 * self.sigma * self.sigma)
            )
            self.weights = torch.tensor(self.weights/col(self.weights.sum(1))).to(device)
            # print('weights',self.weights.shape)
            self.influence_nodes_idx = self.influence_nodes_idx.to(device)
            # print('influence_nodes_idx',self.influence_nodes_idx.shape)
        
    def construct_graph(self, vertices=None, faces=None,geod=None,device=None):
        self.faces = faces
        if self.sampling_strategy == 'qslim':
            m = Mesh(v=vertices, f=faces)
            M, A, D = generate_transform_matrices(m, [2])
            self.graph = M[1]
            # nodes_v = M[1].v
            self.nodes_idx = D[0].nonzero()[1]
            adj_mat = A[1].toarray()
            for i in range(adj_mat.shape[0]):
                self.one_ring_neigh.append(adj_mat[i].nonzero()[0].tolist() + [i]*(self.max_neigh_num-len(adj_mat[i].nonzero()[0])))
            self.one_ring_neigh = torch.tensor(self.one_ring_neigh).to(device) 
        print('one_ring_neigh',self.one_ring_neigh.shape)
        geod_mat = torch.from_numpy(-geod[self.nodes_idx]).transpose(1,0)
    
        self.dists,self.influence_nodes_idx = geod_mat.topk(self.k,dim=-1)
        self.dists = -self.dists

        # geod_mat_pre = geod_mat.transpose(1,0)
        _, self.pre_idx = geod_mat.topk(1,dim=-1)
        self.sigma =   20  * torch.mean(compute_edge_lengths(torch.tensor(self.graph.v).to(vertices.device),torch.tensor(self.graph.f.astype(np.int64)).to(vertices.device)))
        # self.weights = torch.tensor(self.weights/col(self.weights.sum(1))).to(device)
        self.weights = torch.exp(
            -((self.dists**2)) / (2 * self.sigma * self.sigma)
        )
        self.weights = torch.tensor(self.weights/col(self.weights.sum(1))).to(device)
        print('weights',self.weights.shape)
        self.influence_nodes_idx = self.influence_nodes_idx.to(device)
        print('influence_nodes_idx',self.influence_nodes_idx.shape)
        
    def forward(self, vertices, opt_d_rotations, opt_d_translations):
        
        # opt_d_rotmat = batch_rodrigues(opt_d_rotations[0]).unsqueeze(0) # 1 * N_c * 3 * 3
        opt_d_rotmat = opt_d_rotations
        nodes = vertices[self.nodes_idx, ...]

        influence_nodes_v = nodes[self.influence_nodes_idx.reshape((-1,))]# .reshape((6890,3,3))
        opt_d_r = opt_d_rotmat[0, self.influence_nodes_idx.reshape((-1,)), ...]# .reshape((6890,3,3,3)) 
        opt_d_t = opt_d_translations[0, self.influence_nodes_idx.reshape((-1,)), ...]# .reshape((6890,3,3))

        warpped_vertices = (torch.einsum('bij, bkj->bki', opt_d_r, (vertices.repeat_interleave(3, dim=0) - influence_nodes_v).unsqueeze(1)).squeeze(1) \
                            + influence_nodes_v + opt_d_t).reshape((vertices.shape[0],3,3)) * (self.weights.unsqueeze(-1))

        warpped_vertices = warpped_vertices.sum(axis=1).float()
        

        
        diff_term = (nodes + opt_d_translations[0]).repeat_interleave(self.max_neigh_num, dim=0) - \
                    (nodes[self.one_ring_neigh.reshape((-1,))] + opt_d_translations[0][self.one_ring_neigh.reshape((-1,))]) - \
                     torch.einsum('bij, bkj->bki', opt_d_rotmat[0].repeat_interleave(self.max_neigh_num, dim=0), \
                    (nodes.repeat_interleave(self.max_neigh_num, dim=0) - nodes[self.one_ring_neigh.reshape((-1,))]).unsqueeze(1)).squeeze(1)

        sr_term = opt_d_rotmat[0].repeat_interleave(self.max_neigh_num, dim=0) - opt_d_rotmat[0][self.one_ring_neigh.reshape((-1,))]
        
        sr_loss = torch.mean(sr_term**2) 
        arap_loss = torch.sum(diff_term ** 2) / self.nodes_idx.shape[0]
        

        return warpped_vertices.unsqueeze(0), arap_loss ,sr_loss
    

def generate_color_map(num_colors, cmap_name='inferno'):
    cmap = plt.get_cmap(cmap_name)
    colors = [cmap(i) for i in np.linspace(0, 1, num_colors)]
    # Convert RGBA to RGB and scale from [0, 1] to [0, 255]
    colors = [[r , g, b ] for r, g, b, _ in colors]
    return np.array(colors)


def color_map_from_positions(mesh, cmap_name='viridis'):
    cmap = plt.get_cmap(cmap_name)
    
    # Get the X, Y, and Z coordinates of the vertices and normalize them to [0, 1]
    vertices = np.asarray(mesh.vertices)
    normalized_coordinates = (vertices - np.min(vertices, axis=0)) / (np.max(vertices, axis=0) - np.min(vertices, axis=0))
    
    # Compute a value between 0 and 1 for each vertex based on its normalized coordinates
    vertex_values = normalized_coordinates[:, 0] * 0.3 + normalized_coordinates[:, 1] * 0.59 + normalized_coordinates[:, 2] * 0.11
    
    # Generate the color map based on the vertex values
    vertex_colors = np.array([cmap(v)[:3] for v in vertex_values])
    
    # Convert the color map from [0, 1] to [0, 255]
    vertex_colors = vertex_colors 
    
    return vertex_colors

def color_map_from_positions_pointcloud(target_pcd, cmap_name='viridis'):
    cmap = plt.get_cmap(cmap_name)
    
    # Get the X, Y, and Z coordinates of the vertices and normalize them to [0, 1]
    vertices = np.asarray(target_pcd.points)
    normalized_coordinates = (vertices - np.min(vertices, axis=0)) / (np.max(vertices, axis=0) - np.min(vertices, axis=0))
    
    # Compute a value between 0 and 1 for each vertex based on its normalized coordinates
    vertex_values = normalized_coordinates[:, 0] * 0.3 + normalized_coordinates[:, 1] * 0.59 + normalized_coordinates[:, 2] * 0.11
    
    # Generate the color map based on the vertex values
    vertex_colors = np.array([cmap(v)[:3] for v in vertex_values])
    
    # Convert the color map from [0, 1] to [0, 255]
    vertex_colors = vertex_colors 
    
    return vertex_colors


def color_map_from_positions_pcd(mesh, cmap_name='viridis'):
    cmap = plt.get_cmap(cmap_name)
    
    # Get the Z-coordinates of the vertices and normalize them to [0, 1]
    z_coordinates = np.asarray(mesh.points)[:, 2]
    normalized_z_coordinates = (z_coordinates - np.min(z_coordinates)) / (np.max(z_coordinates) - np.min(z_coordinates))
    
    # Generate the color map based on the normalized Z-coordinates
    vertex_colors = np.array([cmap(z)[:3] for z in normalized_z_coordinates])
    
    # Convert the color map from [0, 1] to [0, 255]
    vertex_colors = vertex_colors 
    
    return vertex_colors


def compute_edge_lengths(vertices, faces):
    """
    Compute the edge lengths for a given mesh.
    
    Args:
        vertices (torch.Tensor): The vertices of the mesh, shape (num_vertices, 3)
        faces (torch.Tensor): The faces of the mesh, shape (num_faces, 3)

    Returns:
        edge_lengths (torch.Tensor): The edge lengths for each face, shape (num_faces, 3)
    """
    face_vertices = vertices[faces]  # shape (num_faces, 3, 3)
    edge_vectors = torch.roll(face_vertices, -1, dims=1) - face_vertices
    edge_lengths = torch.norm(edge_vectors, dim=2)

    return edge_lengths


def compute_face_normals(vertices, faces):
    """
    Compute the face normals for a given mesh.

    Args:
        vertices (torch.Tensor): The vertices of the mesh, shape (num_vertices, 3)
        faces (torch.Tensor): The faces of the mesh, shape (num_faces, 3)

    Returns:
        normals (torch.Tensor): The face normals, shape (num_faces, 3)
    """
    face_vertices = vertices[faces]  # shape (num_faces, 3, 3)
    edge_vectors = torch.roll(face_vertices, -1, dims=1) - face_vertices
    normals = torch.cross(edge_vectors[:, 0], edge_vectors[:, 1], dim=1)
    normals = normals / torch.norm(normals, dim=1, keepdim=True)

    return normals
