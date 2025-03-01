import random
import numpy as np
import torch
import torch.nn as nn
import random
from typing import Union
import torch
from torch_scatter import scatter
import torch.nn.functional as F
from pytorch3d.ops.knn import knn_gather, knn_points
from pytorch3d.structures.pointclouds import Pointclouds
import matplotlib.pyplot as plt
import potpourri3d as pp3d
from ChamferDistancePytorch.chamfer3D import dist_chamfer_3D
from misc.gaussian_kl import *
from misc.utils import auto_WKS, farthest_point_sample, square_distance, farthest_point_sample_new
from pytorch3d.ops import iterative_closest_point
from pytorch3d.ops import corresponding_points_alignment
import math
from lib.deformation_graph_point import  DeformationGraph_geod
from torchmetrics import StructuralSimilarityIndexMeasure
import pytorch_lightning as pl
import torch_geometric
from models.model import Deformer
from torch.autograd import Variable
import os

def matrix_to_rotation_6d(R):
    # R is a tensor of shape (B, N, 3, 3)
    # Extract the first two columns of each 3x3 rotation matrix
    d6_1 = R[..., :, 0]  # Shape: (B, N, 3)
    d6_2 = R[..., :, 1]  # Shape: (B, N, 3)
    
    # Concatenate the two columns along the last dimension
    d6 = torch.cat((d6_1, d6_2), dim=-1)  # Shape: (B, N, 6)
    
    return d6

def rotation_6d_to_matrix(d6):
    a1, a2 = d6[..., :3], d6[..., 3:]
    b1 = F.normalize(a1, dim=-1)
    b2 = a2 - (b1 * a2).sum(-1, keepdim=True) * b1
    b2 = F.normalize(b2, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack((b1, b2, b3), dim=-2)
        
def _apply_similarity_transform(
    X: torch.Tensor, R: torch.Tensor, T: torch.Tensor, s: torch.Tensor
) -> torch.Tensor:
    """
    Applies a similarity transformation parametrized with a batch of orthonormal
    matrices `R` of shape `(minibatch, d, d)`, a batch of translations `T`
    of shape `(minibatch, d)` and a batch of scaling factors `s`
    of shape `(minibatch,)` to a given `d`-dimensional cloud `X`
    of shape `(minibatch, num_points, d)`
    """
    X = s[:, None, None] * torch.bmm(X, R) + T[:, None, :]
    return X

def compute_rigid_transform(src_points, tgt_points):
    """
    计算两个点云之间的刚性变换
    :param src_points: 源点云，形状为 [1200, 10, 3]
    :param tgt_points: 目标点云，形状为 [1200, 10, 3]
    :return: 旋转矩阵，形状为 [1200, 3, 3] 和平移矩阵，形状为 [1200, 3]
    """
    # 使用 PyTorch3D 的 corresponding_points_alignment 函数
    R, T, _ = corresponding_points_alignment(src_points, tgt_points, weights=None, estimate_scale=False)
    return R, T
        
def pca(tensor, C):
    # 标准化数据
    mean = torch.mean(tensor, dim=-1, keepdim=True)
    std = torch.std(tensor, dim=-1, keepdim=True)
    normalized_tensor = (tensor - mean) / std

    # 计算协方差矩阵
    cov_matrix = torch.matmul(normalized_tensor.transpose(-1, -2), normalized_tensor) / (normalized_tensor.size(-1) - 1)

    # 计算特征值和特征向量
    eigenvalues, eigenvectors = torch.linalg.eigh(cov_matrix)

    # 选择前C个主成分
    top_eigenvectors = eigenvectors[..., -C:]

    # 投影数据
    projected_tensor = torch.matmul(normalized_tensor, top_eigenvectors)

    return projected_tensor

def knnsearch_t(x, y):
    # distance = torch.cdist(x.float(), y.float())
    distance = torch.cdist(x.float(), y.float(), compute_mode='donot_use_mm_for_euclid_dist')
    _, idx = distance.topk(k=1, dim=-1, largest=False)
    return idx

def knn_grad(x, y, k):
    # distance = torch.cdist(x.float(), y.float())
    distance = torch.cdist(x.float(), y.float())
    _, idx = distance.topk(k=k, dim=-1, largest=False)
    return idx

def save_off_file(filename, points):
    with open(filename, 'w') as f:
        f.write('OFF\n')
        f.write(f'{points.shape[0]} 0 0\n')  # 点数 面数 边数
        for point in points:
            f.write(f'{point[0]} {point[1]} {point[2]}\n')
            
def knnsearch_t_grad(x, y,alpha=100):
    distance = torch.cdist(x.float(), y.float())
    Pi_xy = F.softmax(-alpha*distance, dim=-1)
    # print(Pi_xy.max(dim=2).values)
    return Pi_xy

def knnsearch_t_similarity(x, y):
    distance = torch.cdist(x.float(), y.float())
    Pi_xy = F.softmax(-2*distance, dim=-1)
    return Pi_xy

def search_t(A1, A2):
    T12 = knnsearch_t(A1, A2)
    # T21 = knnsearch_t(A2, A1)
    return T12

def cal_geo(V):
    dist = torch.tensor([])
    solver = pp3d.PointCloudHeatSolver(V)
    for i in range(V.shape[0]):
        dist = torch.cat([dist,torch.tensor(solver.compute_distance(i)).unsqueeze(1)],dim=-1)
    return dist

def _validate_chamfer_reduction_inputs(
        batch_reduction: Union[str, None], point_reduction: str
):
    """Check the requested reductions are valid.
    Args:
        batch_reduction: Reduction operation to apply for the loss across the
            batch, can be one of ["mean", "sum"] or None.
        point_reduction: Reduction operation to apply for the loss across the
            points, can be one of ["mean", "sum"].
    """
    if batch_reduction is not None and batch_reduction not in ["mean", "sum"]:
        raise ValueError('batch_reduction must be one of ["mean", "sum"] or None')
    if point_reduction not in ["mean", "sum"]:
        raise ValueError('point_reduction must be one of ["mean", "sum"]')

def _handle_pointcloud_input(
        points: Union[torch.Tensor, Pointclouds],
        lengths: Union[torch.Tensor, None],
        normals: Union[torch.Tensor, None],
):
    """
    If points is an instance of Pointclouds, retrieve the padded points tensor
    along with the number of points per batch and the padded normals.
    Otherwise, return the input points (and normals) with the number of points per cloud
    set to the size of the second dimension of `points`.
    """
    if isinstance(points, Pointclouds):
        X = points.points_padded()
        lengths = points.num_points_per_cloud()
        normals = points.normals_padded()  # either a tensor or None
    elif torch.is_tensor(points):
        if points.ndim != 3:
            raise ValueError("Expected points to be of shape (N, P, D)")
        X = points
        if lengths is not None and (
                lengths.ndim != 1 or lengths.shape[0] != X.shape[0]
        ):
            raise ValueError("Expected lengths to be of shape (N,)")
        if lengths is None:
            lengths = torch.full(
                (X.shape[0],), X.shape[1], dtype=torch.int64, device=points.device
            )
        if normals is not None and normals.ndim != 3:
            raise ValueError("Expected normals to be of shape (N, P, 3")
    else:
        raise ValueError(
            "The input pointclouds should be either "
            + "Pointclouds objects or torch.Tensor of shape "
            + "(minibatch, num_points, 3)."
        )
    return X, lengths, normals

def compute_truncated_chamfer_distance(
        x,
        y,
        x_lengths=None,
        y_lengths=None,
        x_normals=None,
        y_normals=None,
        weights=None,
        trunc=0.2,
        batch_reduction: Union[str, None] = "mean",
        point_reduction: str = "mean",
):
    """
    Chamfer distance between two pointclouds x and y.

    Args:
        x: FloatTensor of shape (N, P1, D) or a Pointclouds object representing
            a batch of point clouds with at most P1 points in each batch element,
            batch size N and feature dimension D.
        y: FloatTensor of shape (N, P2, D) or a Pointclouds object representing
            a batch of point clouds with at most P2 points in each batch element,
            batch size N and feature dimension D.
        x_lengths: Optional LongTensor of shape (N,) giving the number of points in each
            cloud in x.
        y_lengths: Optional LongTensor of shape (N,) giving the number of points in each
            cloud in y.
        x_normals: Optional FloatTensor of shape (N, P1, D).
        y_normals: Optional FloatTensor of shape (N, P2, D).
        weights: Optional FloatTensor of shape (N,) giving weights for
            batch elements for reduction operation.
        batch_reduction: Reduction operation to apply for the loss across the
            batch, can be one of ["mean", "sum"] or None.
        point_reduction: Reduction operation to apply for the loss across the
            points, can be one of ["mean", "sum"].

    Returns:
        2-element tuple containing

        - **loss**: Tensor giving the reduced distance between the pointclouds
          in x and the pointclouds in y.
        - **loss_normals**: Tensor giving the reduced cosine distance of normals
          between pointclouds in x and pointclouds in y. Returns None if
          x_normals and y_normals are None.
    """
    _validate_chamfer_reduction_inputs(batch_reduction, point_reduction)
    x, x_lengths, x_normals = _handle_pointcloud_input(x, x_lengths, x_normals)
    y, y_lengths, y_normals = _handle_pointcloud_input(y, y_lengths, y_normals)

    return_normals = x_normals is not None and y_normals is not None

    N, P1, D = x.shape
    P2 = y.shape[1]

    # Check if inputs are heterogeneous and create a lengths mask.
    is_x_heterogeneous = (x_lengths != P1).any()
    is_y_heterogeneous = (y_lengths != P2).any()
    x_mask = (
            torch.arange(P1, device=x.device)[None] >= x_lengths[:, None]
    )  # shape [N, P1]
    y_mask = (
            torch.arange(P2, device=y.device)[None] >= y_lengths[:, None]
    )  # shape [N, P2]

    if y.shape[0] != N or y.shape[2] != D:
        raise ValueError("y does not have the correct shape.")
    if weights is not None:
        if weights.size(0) != N:
            raise ValueError("weights must be of shape (N,).")
        if not (weights >= 0).all():
            raise ValueError("weights cannot be negative.")
        if weights.sum() == 0.0:
            weights = weights.view(N, 1)
            if batch_reduction in ["mean", "sum"]:
                return (
                    (x.sum((1, 2)) * weights).sum() * 0.0,
                    (x.sum((1, 2)) * weights).sum() * 0.0,
                )
            return ((x.sum((1, 2)) * weights) * 0.0, (x.sum((1, 2)) * weights) * 0.0)

    cham_norm_x = x.new_zeros(())
    cham_norm_y = x.new_zeros(())

    x_nn = knn_points(x, y, lengths1=x_lengths, lengths2=y_lengths, K=1)
    y_nn = knn_points(y, x, lengths1=y_lengths, lengths2=x_lengths, K=1)

    cham_x = x_nn.dists[..., 0]  # (N, P1)
    cham_y = y_nn.dists[..., 0]  # (N, P2)


    # truncation
    x_mask[cham_x >= trunc] = True
    y_mask[cham_y >= trunc] = True
    # print(x_mask.shape,x_mask.sum(), y_mask.sum())
    cham_x[x_mask] = 0.0
    cham_y[y_mask] = 0.0


    if is_x_heterogeneous:
        cham_x[x_mask] = 0.0
    if is_y_heterogeneous:
        cham_y[y_mask] = 0.0

    if weights is not None:
        cham_x *= weights.view(N, 1)
        cham_y *= weights.view(N, 1)

    if return_normals:
        # Gather the normals using the indices and keep only value for k=0
        x_normals_near = knn_gather(y_normals, x_nn.idx, y_lengths)[..., 0, :]
        y_normals_near = knn_gather(x_normals, y_nn.idx, x_lengths)[..., 0, :]

        cham_norm_x = 1 - torch.abs(
            F.cosine_similarity(x_normals, x_normals_near, dim=2, eps=1e-6)
        )
        cham_norm_y = 1 - torch.abs(
            F.cosine_similarity(y_normals, y_normals_near, dim=2, eps=1e-6)
        )

        if is_x_heterogeneous:
            cham_norm_x[x_mask] = 0.0
        if is_y_heterogeneous:
            cham_norm_y[y_mask] = 0.0

        if weights is not None:
            cham_norm_x *= weights.view(N, 1)
            cham_norm_y *= weights.view(N, 1)

    # Apply point reduction
    cham_x = cham_x.sum(1)  # (N,)
    cham_y = cham_y.sum(1)  # (N,)
    if return_normals:
        cham_norm_x = cham_norm_x.sum(1)  # (N,)
        cham_norm_y = cham_norm_y.sum(1)  # (N,)
    if point_reduction == "mean":
        cham_x /= x_lengths
        cham_y /= y_lengths
        if return_normals:
            cham_norm_x /= x_lengths
            cham_norm_y /= y_lengths

    if batch_reduction is not None:
        # batch_reduction == "sum"
        cham_x = cham_x.sum()
        cham_y = cham_y.sum()
        if return_normals:
            cham_norm_x = cham_norm_x.sum()
            cham_norm_y = cham_norm_y.sum()
        if batch_reduction == "mean":
            div = weights.sum() if weights is not None else N
            cham_x /= div
            cham_y /= div
            if return_normals:
                cham_norm_x /= div
                cham_norm_y /= div

    cham_dist = 1 * cham_x + 1* cham_y
    # single
    # cham_dist = 1 * cham_y
    # cham_normals = cham_norm_x + cham_norm_y if return_normals else None

    return cham_dist


def arap_cost (R, t, g, e, w, lietorch=True):
    '''
    :param R:
    :param t:
    :param g:
    :param e:
    :param w:
    :return:
    '''

    R_i = R [:, None]
    g_i = g [:, None]
    t_i = t [:, None]

    g_j = g [e]
    t_j = t [e]

    if lietorch :
        e_ij = ((R_i * (g_j - g_i) + g_i + t_i  - g_j - t_j )**2).sum(dim=-1)
    else :
        e_ij = (((R_i @ (g_j - g_i)[...,None]).squeeze() + g_i + t_i  - g_j - t_j )**2).sum(dim=-1)
    
    o = (w * e_ij ).mean()

    return o


def projective_depth_cost(dx, dy):

    x_mask = dx> 0
    y_mask = dy> 0
    depth_error = (dx - dy) ** 2
    depth_error = depth_error[y_mask * x_mask]
    silh_loss = torch.mean(depth_error)

    return silh_loss

def silhouette_cost(x, y):

    x_mask = x[..., 0] > 0
    y_mask = y[..., 0] > 0
    silh_error = (x - y) ** 2
    silh_error = silh_error[~y_mask]
    silh_loss = torch.mean(silh_error)

    return silh_loss

def landmark_cost(x, y, landmarks):
    x = x [ landmarks[0] ]
    y = y [ landmarks[1] ]
    loss = torch.mean(
        torch.sum( (x-y)**2, dim=-1 ))
    return loss

def chamfer_dist(src_pcd,   tgt_pcd):
    '''
    :param src_pcd: warpped_pcd
    :param R: node_rotations
    :param t: node_translations
    :param data:
    :return:
    '''

    """chamfer distance"""
    samples = 3000
    src=torch.randperm(src_pcd.shape[0])
    tgt=torch.randperm(tgt_pcd.shape[0])
    s_sample = src_pcd[ src]
    t_sample = tgt_pcd[ tgt]
    cham_dist = compute_truncated_chamfer_distance(s_sample, t_sample, trunc=0.01)

    return cham_dist

def chamfer_dist_show(src_pcd,   tgt_pcd , trun):
    '''
    :param src_pcd: warpped_pcd
    :param R: node_rotations
    :param t: node_translations
    :param data:
    :return:
    '''

    """chamfer distance"""
    samples = 3000
    src=torch.randperm(src_pcd.shape[0])
    tgt=torch.randperm(tgt_pcd.shape[0])
    s_sample = src_pcd[ src]
    t_sample = tgt_pcd[ tgt]
    cham_dist = compute_truncated_chamfer_distance(s_sample[None], t_sample[None], trunc=trun)

    return cham_dist

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

def knn(a, b, k):
    """
    :param a: a.shape == (B, N, C)
    :param b: b.shape == (B, M, C)
    :param k: int
    """
    inner = -2 * torch.matmul(a, b.transpose(2, 1))  # inner.shape == (B, N, M)
    aa = torch.sum(a**2, dim=2, keepdim=True)  # aa.shape == (B, N, 1)
    bb = torch.sum(b**2, dim=2, keepdim=True)  # bb.shape == (B, M, 1)
    pairwise_distance = -aa - inner - bb.transpose(2, 1)  # pairwise_distance.shape == (B, N, M)
    idx = pairwise_distance.topk(k=k, dim=-1)[1]  # idx.shape == (B, N, K)
    return idx

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

    
class FrobeniusLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, a, b):
        loss = torch.sum(torch.abs(a - b) ** 2, axis=(1, 2))
        return torch.mean(loss)
    
class FrobeniusLoss_decay(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, a, b, a_ori, b_ori):
        decay = 1
        cal_L = torch.exp((-torch.abs(a_ori - b_ori)** 2)/decay)*(torch.abs(a - b) ** 2)
        loss = torch.sum(cal_L,axis=(1, 2, 3))
        return torch.mean(loss)

class LGLoss(nn.Module):
    def __init__(self, w_dist =1,w_map=1,k_dist=1000,k_map=10,N_dist=1000,w_cross_construct=1,w_cross_construct_sc=1,w_neighbor=1,partial=False):
        super().__init__()
        self.w_dist = w_dist
        self.w_map = w_map
        self.w_cross_construct = w_cross_construct
        self.w_cross_construct_sc = w_cross_construct_sc
        self.w_neighbor = w_neighbor
        self.k_dist = k_dist
        self.k_map = k_map
        self.N_dist = N_dist
        self.dist_loss = 0
        self.map_loss = 0
        self.construct_loss = 0
        self.neighbor_loss = 0
        self.partial = partial
        self.frob_loss = FrobeniusLoss()
        self.chamfer_dist_3d = dist_chamfer_3D.chamfer_3DDist()
        
        # self.partial_loss = 0

    def chamfer_loss(self, pos1, pos2):
        if not pos1.is_cuda:
            pos1 = pos1.cuda()

        if not pos2.is_cuda:
            pos2 = pos2.cuda()

        dist1, dist2, idx1, idx2 = self.chamfer_dist_3d(pos1, pos2)
        loss = torch.mean(dist1) + torch.mean(dist2)

        return loss

    def get_neighbor_loss(self,source, source_neigh_idxs, target_cross_recon, k):
        # source.shape[1] is the number of points

        if k < source_neigh_idxs.shape[2]:
            neigh_index_for_loss = source_neigh_idxs[:, :, :k]
        else:
            neigh_index_for_loss = source_neigh_idxs

        source_grouped = index_points(source, neigh_index_for_loss)
        source_diff = source_grouped[:, :, 1:, :] - torch.unsqueeze(source, 2)  # remove fist grouped element, as it is the seed point itself
        source_square = torch.sum(source_diff ** 2, dim=-1)

        target_cr_grouped = index_points(target_cross_recon, neigh_index_for_loss)
        target_cr_diff = target_cr_grouped[:, :, 1:, :] - torch.unsqueeze(target_cross_recon, 2)  # remove fist grouped element, as it is the seed point itself
        target_cr_square = torch.sum(target_cr_diff ** 2, dim=-1)

        GAUSSIAN_HEAT_KERNEL_T = 8.0
        gaussian_heat_kernel = torch.exp(-source_square/GAUSSIAN_HEAT_KERNEL_T)
        neighbor_loss_per_neigh = torch.mul(gaussian_heat_kernel, target_cr_square)

        neighbor_loss = torch.mean(neighbor_loss_per_neigh)

        return neighbor_loss    
    
    def forward(self,cfeats1,cfeats2, feat1,feat2, dist1, dist2, verts1, verts2,verts1_corr,verts2_corr,verts1_corr_v1,verts2_corr_v2):
        loss = 0
            
        if self.partial == True:
            print('Partial Setting')

            if self.w_dist > 0:
                k = self.k_dist
                N = self.N_dist
                batch_size = dist1.shape[0]
            
                num1 = dist1.shape[1]
                num2 = dist2.shape[1]
                random_numbers1 = []
                random_numbers2 = []
                
                numbers1 = random.sample(range(num1), N)
                random_numbers1 = torch.tensor(numbers1) # 1000
                numbers2 = random.sample(range(num2), N)
                random_numbers2 = torch.tensor(numbers2) # 1000
                # shape1
                f1 = feat1[:,random_numbers1] # 2*1000*128
                idx = knn(f1,feat1,k) #2*1000*16
                f2 = index_points(feat1, idx)  # neighbors.shape == 2*1000*16*128
                dist_result_f1 = torch.norm(f2 - f1[:, :, None, :], dim=-1)
                idx_num = random_numbers1.repeat(batch_size, 1).unsqueeze(2).expand(-1, -1, k)
                idx = idx.reshape(batch_size,-1)
                idx_num = idx_num.reshape(batch_size,-1)
                dist_f1 = torch.zeros_like(idx, dtype=torch.float)
                
                for i in range(batch_size):
                    dist_f1[i] = dist1[i, idx[i], idx_num[i]]
                dist_f1 = dist_f1.reshape(batch_size, N, k)
                # # shape2
                f1 = feat2[:,random_numbers2] # 2*1000*128
                idx = knn(f1,feat2,k) #2*1000*16
                f2 = index_points(feat2, idx)  # neighbors.shape == 2*1000*16*128
                dist_result_f2 = torch.norm(f2 - f1[:, :, None, :], dim=-1)
                
                idx_num = random_numbers2.repeat(batch_size, 1).unsqueeze(2).expand(-1, -1, k)
                idx = idx.reshape(batch_size,-1)
                idx_num = idx_num.reshape(batch_size,-1)
                dist_f2 = torch.zeros_like(idx, dtype=torch.float)
                
                for i in range(batch_size):
                    dist_f2[i] = dist2[i, idx[i], idx_num[i]]
                dist_f2 = dist_f2.reshape(batch_size, N, k)
                similarity1 = 1 - torch.abs(torch.nn.functional.cosine_similarity(dist_result_f1, dist_f1, dim=2))
                similarity2 = 1 - torch.abs(torch.nn.functional.cosine_similarity(dist_result_f2, dist_f2, dim=2))
                self.dist_loss = (torch.sum(similarity1)+torch.sum(similarity2))* self.w_dist
                print('dist_loss:',self.dist_loss)
                loss += self.dist_loss
                
            if self.w_cross_construct > 0:
                self.construct_loss = self.chamfer_loss(verts1,verts1_corr)*self.w_cross_construct + (self.chamfer_loss(verts1,verts1_corr_v1) + self.chamfer_loss(verts2,verts2_corr_v2))*self.w_cross_construct_sc
                print('construct_loss:',self.construct_loss)
                loss += self.construct_loss
        else:
            if self.w_dist > 0:
                k = self.k_dist
                N = self.N_dist
                batch_size = dist1.shape[0]
            
                num1 = dist1.shape[1]
                num2 = dist2.shape[1]
                random_numbers1 = []
                random_numbers2 = []
                
                numbers1 = random.sample(range(num1), N)
                random_numbers1 = torch.tensor(numbers1) # 1000
                numbers2 = random.sample(range(num2), N)
                random_numbers2 = torch.tensor(numbers2) # 1000
                # shape1
                f1 = feat1[:,random_numbers1] # 2*1000*128
                idx = knn(f1,feat1,k) #2*1000*16
                f2 = index_points(feat1, idx)  # neighbors.shape == 2*1000*16*128
                dist_result_f1 = torch.norm(f2 - f1[:, :, None, :], dim=-1)
                idx_num = random_numbers1.repeat(batch_size, 1).unsqueeze(2).expand(-1, -1, k)
                idx = idx.reshape(batch_size,-1)
                idx_num = idx_num.reshape(batch_size,-1)
                dist_f1 = torch.zeros_like(idx, dtype=torch.float)
                
                for i in range(batch_size):
                    dist_f1[i] = dist1[i, idx[i], idx_num[i]]
                dist_f1 = dist_f1.reshape(batch_size, N, k)
                # # shape2
                f1 = feat2[:,random_numbers2] # 2*1000*128
                idx = knn(f1,feat2,k) #2*1000*16
                f2 = index_points(feat2, idx)  # neighbors.shape == 2*1000*16*128
                dist_result_f2 = torch.norm(f2 - f1[:, :, None, :], dim=-1)
                
                idx_num = random_numbers2.repeat(batch_size, 1).unsqueeze(2).expand(-1, -1, k)
                idx = idx.reshape(batch_size,-1)
                idx_num = idx_num.reshape(batch_size,-1)
                dist_f2 = torch.zeros_like(idx, dtype=torch.float)
                
                for i in range(batch_size):
                    dist_f2[i] = dist2[i, idx[i], idx_num[i]]
                dist_f2 = dist_f2.reshape(batch_size, N, k)
                similarity1 = 1 - torch.abs(torch.nn.functional.cosine_similarity(dist_result_f1, dist_f1, dim=2))
                similarity2 = 1 - torch.abs(torch.nn.functional.cosine_similarity(dist_result_f2, dist_f2, dim=2))
                self.dist_loss = (torch.sum(similarity1)+torch.sum(similarity2))* self.w_dist
                print('dist_loss:',self.dist_loss)
                loss += self.dist_loss
                
            if self.w_neighbor > 0:   
                k = self.k_map 
                v1_neigh_idxs = knn(verts1, verts1, k=k)
                v2_neigh_idxs = knn(verts2, verts2, k=k)
                neighbor_loss_1to2 = self.get_neighbor_loss(verts1, v1_neigh_idxs, verts2_corr, k)
                neighbor_loss_2to1 = self.get_neighbor_loss(verts2, v2_neigh_idxs, verts1_corr, k)
                self.neighbor_loss = (neighbor_loss_1to2 + neighbor_loss_2to1)*self.w_neighbor
                print('neighbor_loss:',self.neighbor_loss)
                loss += self.neighbor_loss
                
            if self.w_map > 0:
                feat1 = cfeats1
                feat2 = cfeats2
                k = self.k_map
                T12_pred,T21_pred= search_t(feat1, feat2), search_t(feat2, feat1)
                verts1_corr = index_points(verts1, T21_pred).squeeze(2).requires_grad_(True) #6*4995*128
                idx1 = knn(verts2,verts2,k)
                verts1_corr_neighbor = index_points(verts1_corr, idx1).requires_grad_(True) # 6*4995*10*128, T(knn(f1))
                idx2 = knn(verts1,verts1,k)
                verts1_neighbor = index_points(verts1, idx2).requires_grad_(True) # 6*4995*10*128
                verts1_neighbor_corr =  index_points(verts1_neighbor, T21_pred).squeeze(2).requires_grad_(True)
                
                verts2_corr = index_points(verts2, T12_pred).squeeze(2).requires_grad_(True) #6*4995*128
                idx1 = knn(verts1,verts1,k)
                verts2_corr_neighbor = index_points(verts2_corr, idx1).requires_grad_(True) # 6*4995*10*128, T(knn(f1))
                idx2 = knn(verts2,verts2,k)
                verts2_neighbor = index_points(verts2, idx2).requires_grad_(True) # 6*4995*10*128
                verts2_neighbor_corr =  index_points(verts2_neighbor, T12_pred).squeeze(2).requires_grad_(True)

                self.map_loss = (self.frob_loss(verts1_corr_neighbor, verts1_neighbor_corr) + self.frob_loss(verts2_corr_neighbor, verts2_neighbor_corr) )* self.w_map
                print('map_loss:',self.map_loss)
                loss += self.map_loss
                
            if self.w_cross_construct > 0:
                self.construct_loss = (self.chamfer_loss(verts1,verts1_corr) + self.chamfer_loss(verts2,verts2_corr))*self.w_cross_construct + (self.chamfer_loss(verts1,verts1_corr_v1) + self.chamfer_loss(verts2,verts2_corr_v2))*self.w_cross_construct_sc
                print('construct_loss:',self.construct_loss)
                loss += self.construct_loss
            
        return loss,self.dist_loss,self.map_loss,self.construct_loss,self.neighbor_loss

class norm(pl.LightningModule):
    def __init__(self, axis=2):
        super().__init__()
        self.axis = axis

    def forward(self, x):
        mean = torch.mean(x, self.axis,keepdim=True) 
        std = torch.std(x, self.axis,keepdim=True)   
        x = (x-mean)/(std+1e-6) 
        return x
    
class Gradient(torch.autograd.Function):                                                                                                       
    @staticmethod
    def forward(ctx, input):
        return input*8
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output
    
class Modified_softmax(pl.LightningModule):
    def __init__(self, axis=1):
        super(Modified_softmax, self).__init__()
        self.axis = axis
        self.norm = norm(axis = axis)
    def forward(self, x):
        x = self.norm(x)
        x = Gradient.apply(x)
        x = F.softmax(x, dim=self.axis)
        return x
 
class GraphDeformLoss_Neural_Partial(nn.Module):
    def __init__(self, k_deform=10, w_dist =1,w_map=1,k_dist=1000,N_dist=1000,partial=False,w_deform=1,w_img=1,w_rank=1,w_self_rec=1,w_cd=1,w_arap=1,save_name=None):
        super().__init__()
        self.device = 'cuda:0'
        self.input_pts = 4995
        self.w_dist = w_dist
        self.w_map = w_map
        self.w_deform = w_deform
        self.w_self_rec = w_self_rec
        self.w_cd = w_cd
        self.w_arap = w_arap
        self.w_rank = w_rank
        self.w_img = w_img
        self.k_dist = k_dist
        self.N_dist = N_dist
        self.k_deform = k_deform
        self.dist_loss = 0
        self.deform_loss = 0
        self.self_rec_loss = 0
        self.img_loss = 0
        self.rank_loss = 0
        self.map_loss = 0
        self.partial = partial
        self.frob_loss = FrobeniusLoss()
        self.chamfer_dist_3d = dist_chamfer_3D.chamfer_3DDist()
        self.img_size = 224
        self.img_offset = torch.Tensor([[-2, -2], [-2, -1], [-2, 0], [-2, 1], [-2, 2], 
                                        [-1, -2], [-1, -1], [-1, 0], [-1, 1], [-1, 2], 
                                        [0, -2], [0, -1], [0, 0], [0, 1], [0, 2],
                                        [1, -2], [1, -1], [1, 0], [1, 1], [1, 2],
                                        [2, -2], [2, -1], [2, 0], [2, 1], [2, 2]])
        self.img_mean = torch.Tensor([0.485, 0.456, 0.406])
        self.img_std = torch.Tensor([0.229, 0.224, 0.225])
        self.proj_reduction = 'sum'
        self.ssim = StructuralSimilarityIndexMeasure()    
        self.save_name = save_name      


    def chamfer_loss(self, pos1, pos2):
        if not pos1.is_cuda:
            pos1 = pos1.cuda()

        if not pos2.is_cuda:
            pos2 = pos2.cuda()

        dist1, dist2, idx1, idx2 = self.chamfer_dist_3d(pos1, pos2)
        loss = torch.mean(dist1) + torch.mean(dist2)

        return loss   

    def _batch_frobenius_norm(self, matrix1, matrix2):
        loss_F = torch.norm((matrix1-matrix2),dim=(1,2))
        return loss_F
    
    def get_colored_depth_maps(self,raw_depths,H,W):
        import matplotlib
        import matplotlib.cm as cm
        cmap = cm.get_cmap('PiYG')
        norm = matplotlib.colors.Normalize(vmin=0.0, vmax=1.0, clip=True)
        mapper = cm.ScalarMappable(norm=norm, cmap=cmap)
        depth_images = []
        for i in range(raw_depths.size()[0]):
            d = raw_depths[i]
            dmax = torch.max(d) ; dmin = torch.min(d)
            d = (d-dmin)/(dmax-dmin)
            flat_d = d.view(1,-1).cpu().detach().numpy()
            flat_colors = mapper.to_rgba(flat_d)
            depth_colors = np.reshape(flat_colors,(H,W,4))[:,:,:3]
            np_image = depth_colors
            np_image = np_image.astype('float32')
            depth_images.append(np_image)

        return depth_images
    
    def proj2img(self, pc):
        B, N, _ = pc.shape
        
        # calculate range
        pc_range = pc.max(dim=1)[0] - pc.min(dim=1)[0]  # B 3
        grid_size = pc_range[:, :2].max(dim=-1)[0] / (self.img_size - 3)  # B,

        # Point Index
        pc_min = pc.min(dim=1)[0][:, :2].unsqueeze(dim=1)
        grid_size = grid_size.unsqueeze(dim=1).unsqueeze(dim=2)
        idx_xy = torch.floor((pc[:, :, :2] - pc_min) / grid_size)  # B N 2
        
        # print(idx_xy.shape)
        # Point Densify
        idx_xy_dense = (idx_xy.unsqueeze(dim=2) + self.img_offset.unsqueeze(dim=0).unsqueeze(dim=0).to(pc.device)).view(idx_xy.size(0), N*25, 2) + 1
        # B N 1 2 + 1 1 25 2 -> B N 25 2 -> B 25N 2
        
        # Object to Image Center
        idx_xy_dense_center = torch.floor((idx_xy_dense.max(dim=1)[0] + idx_xy_dense.min(dim=1)[0]) / 2).int()
        offset_x = self.img_size / 2 - idx_xy_dense_center[:, 0: 1] - 1
        offset_y = self.img_size / 2 - idx_xy_dense_center[:, 1: 2] - 1
        idx_xy_dense_offset = idx_xy_dense + torch.cat([offset_x, offset_y], dim=1).unsqueeze(dim=1)

        # Expand Point Features
        # f_dense = torch.tensor([])
        f_dense = pc.unsqueeze(dim=2).expand(-1, -1, 25, -1).contiguous().view(pc.size(0), N * 25, 3)[..., 2: 3].repeat(1, 1, 3)
        # f_dense1 = pc.unsqueeze(dim=2).expand(-1, -1, 25, -1).contiguous().view(pc.size(0), N * 25, 3)[..., 0: 1]
        # f_dense2 = pc.unsqueeze(dim=2).expand(-1, -1, 25, -1).contiguous().view(pc.size(0), N * 25, 3)[..., 1: 2]
        # f_dense3 = pc.unsqueeze(dim=2).expand(-1, -1, 25, -1).contiguous().view(pc.size(0), N * 25, 3)[..., 2: 3]
        # f_dense = torch.cat([f_dense1,f_dense2,f_dense3],dim=-1)
        
        idx_zero = idx_xy_dense_offset < 0
        idx_obj = idx_xy_dense_offset > self.img_size - 1
        idx_xy_dense_offset = idx_xy_dense_offset + idx_zero.to(torch.int32)
        idx_xy_dense_offset = idx_xy_dense_offset - idx_obj.to(torch.int32)

        # B N 9 C -> B 9N C
        assert idx_xy_dense_offset.min() >= 0 and idx_xy_dense_offset.max() <= (self.img_size-1), str(idx_xy_dense_offset.min()) + '-' + str(idx_xy_dense_offset.max())
        
        # change idx to 1-dim
        new_idx_xy_dense = idx_xy_dense_offset[:, :, 0] * self.img_size + idx_xy_dense_offset[:, :, 1]
        
        # Get Image Features
        out = scatter(f_dense, new_idx_xy_dense.long(), dim=1, reduce=self.proj_reduction) 

        # need to pad 
        if out.size(1) < self.img_size * self.img_size: 
            delta = self.img_size * self.img_size - out.size(1) 
            zero_pad = torch.zeros(out.size(0), delta, out.size(2)).to(out.device) 
            # zero_pad = -10000*torch.ones(out.size(0), delta, out.size(2)).to(out.device) 
            res = torch.cat([out, zero_pad], dim=1).reshape((out.size(0), self.img_size, self.img_size, out.size(2))) 
        else: 
            res = out.reshape((out.size(0), self.img_size, self.img_size, out.size(2))) 
        
        # B 224 224 C
        img = res.permute(0, 3, 1, 2).contiguous()
        # zero_mask = (img == 0)
        # mean_vec = self.img_mean.unsqueeze(dim=0).unsqueeze(dim=2).unsqueeze(dim=3).cuda()  # 1 3 1 1
        # std_vec = self.img_std.unsqueeze(dim=0).unsqueeze(dim=2).unsqueeze(dim=3).cuda()   # 1 3 1 1    
        img_norm = nn.Sigmoid()(img)
        # img_norm = img.sub(mean_vec).div(std_vec) # 1 3 224 224
        # img_norm = img_norm[:,0,:,:].unsqueeze(1) # 1 1 224 224
        #img_norm = self.get_colored_depth_maps(img_norm,self.img_size,self.img_size)
        # img_norm = torch.tensor(img_norm).permute(0,3,1,2).to(device)
        # img_norm[zero_mask] = 0
        return img_norm, pc_min, grid_size, (offset_x, offset_y)
      
    def chamfer_loss(self, pos1, pos2):
        if not pos1.is_cuda:
            pos1 = pos1.cuda()

        if not pos2.is_cuda:
            pos2 = pos2.cuda()

        dist1, dist2, idx1, idx2 = self.chamfer_dist_3d(pos1, pos2)
        if dist1.shape[1] <= dist2.shape[1]:
            dist = dist1
        else:
            dist = dist2
        # 单边loss
        loss = torch.mean(dist)

        return loss  
     
    def deform(self, verts12,verts1,Pi_12,verts2,k,fps1,dg_list1,feat1,feat2,deformer):
        batch,num,_ = verts1.shape
        idx11 = knn_grad(verts1,verts1,k)
        idx22 = knn_grad(verts2,verts2,k)
        feat2_conv = index_points(feat2,idx22)
        feat1_conv = index_points(feat1,idx11)
        deformations = deformer(feat1_conv,feat2_conv,verts1,verts12,Pi_12,fps1)
        rotations = deformations[:,:,3:]
        iden = Variable(torch.from_numpy(np.array([1,0,0,0,1,0]).astype(np.float32))).view(1,6).repeat(batch,rotations.shape[1],1)
        if rotations.is_cuda:
            iden = iden.cuda()
        rotations = rotations + iden
        T1 = deformations[:,:,:3]
        R1 = rotation_6d_to_matrix(rotations)
        
        i = 0
        deformed_points1 = torch.tensor([]).to(verts1.device)
        arap_all_loss12 = 0
        for dg in dg_list1:
            deformed_pt1, arap, sr_loss= dg(verts1[i],R1[i].unsqueeze(0), T1[i].unsqueeze(0))
            arap_all_loss12 = arap_all_loss12 + arap
            i = i + 1
            deformed_points1 = torch.cat([deformed_points1,deformed_pt1],dim=0)
            
        # print('self-loss:',chamfer_dist(verts12,verts2))
        # print('cross-loss:',chamfer_dist(deformed_points1,verts2))
        # print('arap-loss:',arap_all_loss)
        # deform_loss12 = chamfer_dist(deformed_points1,verts2) + arap_all_loss*0.01 + chamfer_dist(verts12,verts2) 
        cross_deform_loss12 = self.chamfer_loss(deformed_points1,verts2)*self.w_cd + arap_all_loss12*self.w_arap
        self_rec_loss12 =  self.chamfer_loss(verts12,verts2)
        n = str(random.randint(0, 10))
        print(f"Rand:{n}, Deform_Result: cd_loss:{self.chamfer_loss(deformed_points1,verts2)*self.w_cd}, arap_loss:{arap_all_loss12*self.w_arap}")
        
        save_path_t = 'visual_result/' + self.save_name
        if not os.path.exists(save_path_t):
            os.makedirs(save_path_t)
                
        PC1 = deformed_points1[0].detach().cpu().squeeze().numpy()
        save_off_file(save_path_t + '/deform_'+ n +'.off', PC1)
        PC2 = verts2[0].detach().cpu().squeeze().numpy()
        save_off_file(save_path_t + '/target_'+ n +'.off', PC2)
        PC3 = verts1[0].detach().cpu().squeeze().numpy()
        save_off_file(save_path_t + '/source_'+ n +'.off', PC3)
        PC12 = verts12[0].detach().cpu().squeeze().numpy()
        save_off_file(save_path_t + '/pi_verts2_'+ n +'.off', PC12)
        return cross_deform_loss12, self_rec_loss12
    
    def pcd2img(self,x):
        pts_1 = x
        pts_2 = torch.cat((pts_1[..., 2: 3], pts_1[..., 0: 2]), dim=-1)
        pts_3 = torch.cat((pts_1[..., 1: 3], pts_1[..., 0: 1]), dim=-1)
        imgs_1, pc_min_1, grid_size_1, offsets_1 = self.proj2img(pts_1)  # b, 3, 224, 224
        imgs_2, pc_min_2, grid_size_2, offsets_2 = self.proj2img(pts_2)  # b, 3, 224, 224
        imgs_3, pc_min_3, grid_size_3, offsets_3 = self.proj2img(pts_3)  # b, 3, 224, 224 
        img_tensor = torch.cat((imgs_1, imgs_2, imgs_3), dim=0)
        return img_tensor

    def _KFNN(self, x, y, k=10):
        def batched_pairwise_dist(a, b):
            x, y = a.float(), b.float()
            bs, num_points_x, points_dim = x.size()
            bs, num_points_y, points_dim = y.size()
            xx = torch.pow(x, 2).sum(2)
            yy = torch.pow(y, 2).sum(2)
            zz = torch.bmm(x, y.transpose(2, 1))
            rx = xx.unsqueeze(1).expand(bs, num_points_y, num_points_x) 
            ry = yy.unsqueeze(1).expand(bs, num_points_x, num_points_y) 
            P = rx.transpose(2, 1) + ry - 2 * zz
            return P
        pairwise_distance = batched_pairwise_dist(x.permute(0,2,1), y.permute(0,2,1))
        similarity=-pairwise_distance
        idx = similarity.topk(k=k, dim=-1)[1]
        return pairwise_distance, idx
    
    def deformation_graph_node(self,verts1):
        B,N,_ = verts1.shape
        device = verts1.device 
        dg_list = []       
        # new_verts1 = torch.tensor([]).to(device)
        num_nodes_all = torch.tensor([]).to(device)
        for i in range(B):
            dg = DeformationGraph_geod()
            geod = torch.cdist(verts1[i], verts1[i], p=2.0).cpu().numpy()
            dg.construct_graph_euclidean(verts1[i].cpu(),geod,device)
            num_nodes_all = torch.cat([num_nodes_all,torch.tensor(dg.nodes_idx).to(device).unsqueeze(0)],dim=0)
            dg_list.append(dg)
            # opt_d_rotations = torch.zeros((1, num_nodes, 3)).to(device) # axis angle 
            # opt_d_translations = torch.zeros((1, num_nodes, 3)).to(device)
            # new_pcd_verts1,arap,sr_loss= dg(verts1[i], opt_d_rotations, opt_d_translations)
            # new_verts1 = torch.cat([new_verts1,new_pcd_verts1],dim=0)   
        return num_nodes_all,dg_list

    def topk_pi(self,A):
        k = 10  # 例如，选择每个维度上的前10个最大值
        # 获取topk的值和索引
        topk_values, topk_indices = torch.topk(A, k, dim=-1)
        # 创建一个与A形状相同的tensor，初始化为0
        result = torch.zeros_like(A)
        # 使用scatter_将topk的值放回原位置
        result.scatter_(-1, topk_indices, topk_values)
        return result

    def forward(self,feat1,feat2, dist1, dist2, verts1, verts2,alpha_i,deformer):
        loss = 0
        if self.w_dist > 0:
            k = self.k_dist
            N = self.N_dist
            batch_size = dist1.shape[0]
        
            num1 = dist1.shape[1]
            num2 = dist2.shape[1]
            random_numbers1 = []
            random_numbers2 = []
            
            numbers1 = random.sample(range(num1), N)
            random_numbers1 = torch.tensor(numbers1) # 1000
            numbers2 = random.sample(range(num2), N)
            random_numbers2 = torch.tensor(numbers2) # 1000
            # shape1
            f1 = feat1[:,random_numbers1] # 2*1000*128
            idx = knn(f1,feat1,k) #2*1000*16
            f2 = index_points(feat1, idx)  # neighbors.shape == 2*1000*16*128
            dist_result_f1 = torch.norm(f2 - f1[:, :, None, :], dim=-1)
            idx_num = random_numbers1.repeat(batch_size, 1).unsqueeze(2).expand(-1, -1, k)
            idx = idx.reshape(batch_size,-1)
            idx_num = idx_num.reshape(batch_size,-1)
            dist_f1 = torch.zeros_like(idx, dtype=torch.float)
            
            for i in range(batch_size):
                dist_f1[i] = dist1[i, idx[i], idx_num[i]]
            dist_f1 = dist_f1.reshape(batch_size, N, k)
            # # shape2
            f1 = feat2[:,random_numbers2] # 2*1000*128
            idx = knn(f1,feat2,k) #2*1000*16
            f2 = index_points(feat2, idx)  # neighbors.shape == 2*1000*16*128
            dist_result_f2 = torch.norm(f2 - f1[:, :, None, :], dim=-1)
            
            idx_num = random_numbers2.repeat(batch_size, 1).unsqueeze(2).expand(-1, -1, k)
            idx = idx.reshape(batch_size,-1)
            idx_num = idx_num.reshape(batch_size,-1)
            dist_f2 = torch.zeros_like(idx, dtype=torch.float)
            
            for i in range(batch_size):
                dist_f2[i] = dist2[i, idx[i], idx_num[i]]
            dist_f2 = dist_f2.reshape(batch_size, N, k)
            similarity1 = 1 - torch.abs(torch.nn.functional.cosine_similarity(dist_result_f1, dist_f1, dim=2))
            similarity2 = 1 - torch.abs(torch.nn.functional.cosine_similarity(dist_result_f2, dist_f2, dim=2))
            self.dist_loss = (torch.sum(similarity1)+torch.sum(similarity2))* self.w_dist
            # print('dist_loss:',self.dist_loss)
            loss += self.dist_loss
                
        if self.w_deform > 0:
            B,N,_ = verts1.shape
            k = self.k_deform
            device = verts1.device
            num_nodes_all1,dg_list1 = self.deformation_graph_node(verts1)
            num_nodes_all2,dg_list2 = self.deformation_graph_node(verts2)
                
            Pi_12 = knnsearch_t_grad(feat1, feat2, alpha=alpha_i) # 2,4995,4995 
            Pi_21 = knnsearch_t_grad(feat2, feat1, alpha=alpha_i)  # 2,4995,3
            Pi_12 = self.topk_pi(Pi_12)
            Pi_21 = self.topk_pi(Pi_21)
            verts12 = torch.matmul(Pi_12,verts2)       
            verts21 = torch.matmul(Pi_21,verts1)
            fps1 = num_nodes_all2.long()
            cross_deform_loss12, self_rec_loss12 = self.deform(verts12,verts1,Pi_12,verts2,k,num_nodes_all1.long(),dg_list1,feat1,feat2,deformer)
            cross_deform_loss21, self_rec_loss21 = self.deform(verts21,verts2,Pi_21,verts1,k,num_nodes_all2.long(),dg_list2,feat2,feat1,deformer)
            self.deform_loss = (cross_deform_loss12 + cross_deform_loss21)*self.w_deform/2
            # print('deform_loss:',self.deform_loss)
            loss += self.deform_loss
        
        # if self.w_map > 0:
        #     self.map_loss = self.w_map*(map_loss12 + map_loss21)/2
        #     # print('map_loss:',self.map_loss)
        #     loss += self.map_loss
        
        if self.w_self_rec > 0:
            self.self_rec_loss = (self_rec_loss12 + self_rec_loss21)*self.w_self_rec/2
            # print('self_rec_loss:',self.self_rec_loss)
            loss += self.self_rec_loss
                     
        if self.w_rank > 0:
            B,N,_ = verts1.shape
            I_N = torch.eye(n=N, device=self.device)
            I_N = I_N.unsqueeze(0).repeat(B,1,1)
            self.rank_loss = (torch.mean(self._batch_frobenius_norm(torch.bmm(Pi_12,Pi_12.transpose(2, 1).contiguous()), I_N.float())) + torch.mean(self._batch_frobenius_norm(torch.bmm(Pi_21,Pi_21.transpose(2, 1).contiguous()), I_N.float())))*self.w_rank/2
            # print('rank_loss:',self.rank_loss)
            loss += self.rank_loss
        
        return loss,self.dist_loss,self.deform_loss,self.map_loss,self.self_rec_loss
    
class GraphDeformLoss_Neural(nn.Module):
    def __init__(self, k_deform=10, w_dist =1,w_map=1,k_dist=1000,N_dist=1000,partial=False,w_deform=1,w_img=1,w_rank=1,w_self_rec=1,w_cd=1,w_arap=1,save_name=None):
        super().__init__()
        self.device = 'cuda:0'
        self.input_pts = 4995
        self.w_dist = w_dist
        self.w_map = w_map
        self.w_deform = w_deform
        self.w_self_rec = w_self_rec
        self.w_cd = w_cd
        self.w_arap = w_arap
        self.w_rank = w_rank
        self.w_img = w_img
        self.k_dist = k_dist
        self.N_dist = N_dist
        self.k_deform = k_deform
        self.dist_loss = 0
        self.deform_loss = 0
        self.self_rec_loss = 0
        self.img_loss = 0
        self.rank_loss = 0
        self.map_loss = 0
        self.partial = partial
        self.frob_loss = FrobeniusLoss()
        self.chamfer_dist_3d = dist_chamfer_3D.chamfer_3DDist()
        self.img_size = 224
        self.img_offset = torch.Tensor([[-2, -2], [-2, -1], [-2, 0], [-2, 1], [-2, 2], 
                                        [-1, -2], [-1, -1], [-1, 0], [-1, 1], [-1, 2], 
                                        [0, -2], [0, -1], [0, 0], [0, 1], [0, 2],
                                        [1, -2], [1, -1], [1, 0], [1, 1], [1, 2],
                                        [2, -2], [2, -1], [2, 0], [2, 1], [2, 2]])
        self.img_mean = torch.Tensor([0.485, 0.456, 0.406])
        self.img_std = torch.Tensor([0.229, 0.224, 0.225])
        self.proj_reduction = 'sum'
        self.ssim = StructuralSimilarityIndexMeasure()    
        self.save_name = save_name      


    def chamfer_loss(self, pos1, pos2):
        if not pos1.is_cuda:
            pos1 = pos1.cuda()

        if not pos2.is_cuda:
            pos2 = pos2.cuda()

        dist1, dist2, idx1, idx2 = self.chamfer_dist_3d(pos1, pos2)
        loss = torch.mean(dist1) + torch.mean(dist2)

        return loss   

    def _batch_frobenius_norm(self, matrix1, matrix2):
        loss_F = torch.norm((matrix1-matrix2),dim=(1,2))
        return loss_F
    
    def get_colored_depth_maps(self,raw_depths,H,W):
        import matplotlib
        import matplotlib.cm as cm
        cmap = cm.get_cmap('PiYG')
        norm = matplotlib.colors.Normalize(vmin=0.0, vmax=1.0, clip=True)
        mapper = cm.ScalarMappable(norm=norm, cmap=cmap)
        depth_images = []
        for i in range(raw_depths.size()[0]):
            d = raw_depths[i]
            dmax = torch.max(d) ; dmin = torch.min(d)
            d = (d-dmin)/(dmax-dmin)
            flat_d = d.view(1,-1).cpu().detach().numpy()
            flat_colors = mapper.to_rgba(flat_d)
            depth_colors = np.reshape(flat_colors,(H,W,4))[:,:,:3]
            np_image = depth_colors
            np_image = np_image.astype('float32')
            depth_images.append(np_image)

        return depth_images
    
    def proj2img(self, pc):
        B, N, _ = pc.shape
        
        # calculate range
        pc_range = pc.max(dim=1)[0] - pc.min(dim=1)[0]  # B 3
        grid_size = pc_range[:, :2].max(dim=-1)[0] / (self.img_size - 3)  # B,

        # Point Index
        pc_min = pc.min(dim=1)[0][:, :2].unsqueeze(dim=1)
        grid_size = grid_size.unsqueeze(dim=1).unsqueeze(dim=2)
        idx_xy = torch.floor((pc[:, :, :2] - pc_min) / grid_size)  # B N 2
        
        # print(idx_xy.shape)
        # Point Densify
        idx_xy_dense = (idx_xy.unsqueeze(dim=2) + self.img_offset.unsqueeze(dim=0).unsqueeze(dim=0).to(pc.device)).view(idx_xy.size(0), N*25, 2) + 1
        # B N 1 2 + 1 1 25 2 -> B N 25 2 -> B 25N 2
        
        # Object to Image Center
        idx_xy_dense_center = torch.floor((idx_xy_dense.max(dim=1)[0] + idx_xy_dense.min(dim=1)[0]) / 2).int()
        offset_x = self.img_size / 2 - idx_xy_dense_center[:, 0: 1] - 1
        offset_y = self.img_size / 2 - idx_xy_dense_center[:, 1: 2] - 1
        idx_xy_dense_offset = idx_xy_dense + torch.cat([offset_x, offset_y], dim=1).unsqueeze(dim=1)

        # Expand Point Features
        # f_dense = torch.tensor([])
        f_dense = pc.unsqueeze(dim=2).expand(-1, -1, 25, -1).contiguous().view(pc.size(0), N * 25, 3)[..., 2: 3].repeat(1, 1, 3)
        # f_dense1 = pc.unsqueeze(dim=2).expand(-1, -1, 25, -1).contiguous().view(pc.size(0), N * 25, 3)[..., 0: 1]
        # f_dense2 = pc.unsqueeze(dim=2).expand(-1, -1, 25, -1).contiguous().view(pc.size(0), N * 25, 3)[..., 1: 2]
        # f_dense3 = pc.unsqueeze(dim=2).expand(-1, -1, 25, -1).contiguous().view(pc.size(0), N * 25, 3)[..., 2: 3]
        # f_dense = torch.cat([f_dense1,f_dense2,f_dense3],dim=-1)
        
        idx_zero = idx_xy_dense_offset < 0
        idx_obj = idx_xy_dense_offset > self.img_size - 1
        idx_xy_dense_offset = idx_xy_dense_offset + idx_zero.to(torch.int32)
        idx_xy_dense_offset = idx_xy_dense_offset - idx_obj.to(torch.int32)

        # B N 9 C -> B 9N C
        assert idx_xy_dense_offset.min() >= 0 and idx_xy_dense_offset.max() <= (self.img_size-1), str(idx_xy_dense_offset.min()) + '-' + str(idx_xy_dense_offset.max())
        
        # change idx to 1-dim
        new_idx_xy_dense = idx_xy_dense_offset[:, :, 0] * self.img_size + idx_xy_dense_offset[:, :, 1]
        
        # Get Image Features
        out = scatter(f_dense, new_idx_xy_dense.long(), dim=1, reduce=self.proj_reduction) 

        # need to pad 
        if out.size(1) < self.img_size * self.img_size: 
            delta = self.img_size * self.img_size - out.size(1) 
            zero_pad = torch.zeros(out.size(0), delta, out.size(2)).to(out.device) 
            # zero_pad = -10000*torch.ones(out.size(0), delta, out.size(2)).to(out.device) 
            res = torch.cat([out, zero_pad], dim=1).reshape((out.size(0), self.img_size, self.img_size, out.size(2))) 
        else: 
            res = out.reshape((out.size(0), self.img_size, self.img_size, out.size(2))) 
        
        # B 224 224 C
        img = res.permute(0, 3, 1, 2).contiguous()
        # zero_mask = (img == 0)
        # mean_vec = self.img_mean.unsqueeze(dim=0).unsqueeze(dim=2).unsqueeze(dim=3).cuda()  # 1 3 1 1
        # std_vec = self.img_std.unsqueeze(dim=0).unsqueeze(dim=2).unsqueeze(dim=3).cuda()   # 1 3 1 1    
        img_norm = nn.Sigmoid()(img)
        # img_norm = img.sub(mean_vec).div(std_vec) # 1 3 224 224
        # img_norm = img_norm[:,0,:,:].unsqueeze(1) # 1 1 224 224
        #img_norm = self.get_colored_depth_maps(img_norm,self.img_size,self.img_size)
        # img_norm = torch.tensor(img_norm).permute(0,3,1,2).to(device)
        # img_norm[zero_mask] = 0
        return img_norm, pc_min, grid_size, (offset_x, offset_y)
      
    def chamfer_loss(self, pos1, pos2):
        if not pos1.is_cuda:
            pos1 = pos1.cuda()

        if not pos2.is_cuda:
            pos2 = pos2.cuda()

        dist1, dist2, idx1, idx2 = self.chamfer_dist_3d(pos1, pos2)
        loss = torch.mean(dist1) + torch.mean(dist2)

        return loss  
     
    def deform(self, verts12,verts1,Pi_12,verts2,k,fps1,dg_list1,feat1,feat2,deformer):
        idx11 = knn_grad(verts1,verts1,k)
        idx22 = knn_grad(verts2,verts2,k)
        batch,num,_ = verts1.shape
        if self.w_map > 0:
            #verts12 = torch.matmul(Pi_12,verts2)
            verts2_corr = verts12
            verts2_corr_neighbor = index_points(verts2_corr, idx11)# 2*4995*10*128, T(knn(f1))
            verts2_neighbor = index_points(verts2, idx22) # 2*4995*10*3
            verts2_neighbor_corr = torch.einsum('bij, bjkm->bikm',Pi_12,verts2_neighbor)
            map_loss12 = self.frob_loss(verts2_corr_neighbor, verts2_neighbor_corr)
        else:
            map_loss12 = 0
        
        # verts1_neighborhood, verts12_neighborhood = compute_neighborhood(verts1, verts12, k) # (B,N,K,3)
        # fps1_expanded = fps1.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, k, 3)
        # extracted_points1_ori = torch.gather(verts1_neighborhood, 1, fps1_expanded) # torch.Size([2, 600, 10, 3])
        # extracted_points12 = torch.gather(verts12_neighborhood, 1, fps1_expanded) # torch.Size([2, 600, 10, 3])             
        # batch,num,_,_ = extracted_points1_ori.shape
        # extracted_points1 = extracted_points1_ori.reshape(batch*num,k,3)
        # extracted_points12 = extracted_points12.reshape(batch*num,k,3)
        # R, T, s = corresponding_points_alignment(extracted_points1, extracted_points12, weights=None, estimate_scale=False)
        # R = R.reshape(batch,num,3,3)
        # T = T.reshape(batch,num,3)
        # R = matrix_to_rotation_6d(R)
        # rt = torch.cat([T,R],dim=-1)
        feat2_conv = index_points(feat2,idx22)
        feat1_conv = index_points(feat1,idx11)
        
        deformations = deformer(feat1_conv,feat2_conv,verts1,verts12,Pi_12,fps1)
        rotations = deformations[:,:,3:]
        iden = Variable(torch.from_numpy(np.array([1,0,0,0,1,0]).astype(np.float32))).view(1,6).repeat(batch,rotations.shape[1],1)
        if rotations.is_cuda:
            iden = iden.cuda()
        rotations = rotations + iden
        T1 = deformations[:,:,:3]
        R1 = rotation_6d_to_matrix(rotations)
        
        i = 0
        deformed_points1 = torch.tensor([]).to(verts1.device)
        arap_all_loss12 = 0
        for dg in dg_list1:
            deformed_pt1, arap, sr_loss= dg(verts1[i],R1[i].unsqueeze(0), T1[i].unsqueeze(0))
            arap_all_loss12 = arap_all_loss12 + arap
            i = i + 1
            deformed_points1 = torch.cat([deformed_points1,deformed_pt1],dim=0)
            
        # print('self-loss:',chamfer_dist(verts12,verts2))
        # print('cross-loss:',chamfer_dist(deformed_points1,verts2))
        # print('arap-loss:',arap_all_loss)
        # deform_loss12 = chamfer_dist(deformed_points1,verts2) + arap_all_loss*0.01 + chamfer_dist(verts12,verts2) 
        cross_deform_loss12 = self.chamfer_loss(deformed_points1,verts2)*self.w_cd + arap_all_loss12*self.w_arap
        self_rec_loss12 =  self.chamfer_loss(verts12,verts2)
        n = str(random.randint(0, 10))
        print(f"Rand:{n}, Deform_Result: cd_loss:{self.chamfer_loss(deformed_points1,verts2)*self.w_cd}, arap_loss:{arap_all_loss12*self.w_arap}")
        
        save_path_t = 'visual_result/' + self.save_name
        if not os.path.exists(save_path_t):
            os.makedirs(save_path_t)
                
        PC1 = deformed_points1[0].detach().cpu().squeeze().numpy()
        save_off_file(save_path_t + '/deform_'+ n +'.off', PC1)
        PC2 = verts2[0].detach().cpu().squeeze().numpy()
        save_off_file(save_path_t + '/target_'+ n +'.off', PC2)
        PC3 = verts1[0].detach().cpu().squeeze().numpy()
        save_off_file(save_path_t + '/source_'+ n +'.off', PC3)
        PC12 = verts12[0].detach().cpu().squeeze().numpy()
        save_off_file(save_path_t + '/pi_verts2_'+ n +'.off', PC12)
        return map_loss12, cross_deform_loss12, self_rec_loss12
    
    def pcd2img(self,x):
        pts_1 = x
        pts_2 = torch.cat((pts_1[..., 2: 3], pts_1[..., 0: 2]), dim=-1)
        pts_3 = torch.cat((pts_1[..., 1: 3], pts_1[..., 0: 1]), dim=-1)
        imgs_1, pc_min_1, grid_size_1, offsets_1 = self.proj2img(pts_1)  # b, 3, 224, 224
        imgs_2, pc_min_2, grid_size_2, offsets_2 = self.proj2img(pts_2)  # b, 3, 224, 224
        imgs_3, pc_min_3, grid_size_3, offsets_3 = self.proj2img(pts_3)  # b, 3, 224, 224 
        img_tensor = torch.cat((imgs_1, imgs_2, imgs_3), dim=0)
        return img_tensor

    def _KFNN(self, x, y, k=10):
        def batched_pairwise_dist(a, b):
            x, y = a.float(), b.float()
            bs, num_points_x, points_dim = x.size()
            bs, num_points_y, points_dim = y.size()
            xx = torch.pow(x, 2).sum(2)
            yy = torch.pow(y, 2).sum(2)
            zz = torch.bmm(x, y.transpose(2, 1))
            rx = xx.unsqueeze(1).expand(bs, num_points_y, num_points_x) 
            ry = yy.unsqueeze(1).expand(bs, num_points_x, num_points_y) 
            P = rx.transpose(2, 1) + ry - 2 * zz
            return P
        pairwise_distance = batched_pairwise_dist(x.permute(0,2,1), y.permute(0,2,1))
        similarity=-pairwise_distance
        idx = similarity.topk(k=k, dim=-1)[1]
        return pairwise_distance, idx
    
    def deformation_graph_node(self,verts1):
        B,N,_ = verts1.shape
        device = verts1.device 
        dg_list = []       
        # new_verts1 = torch.tensor([]).to(device)
        num_nodes_all = torch.tensor([]).to(device)
        for i in range(B):
            dg = DeformationGraph_geod()
            geod = torch.cdist(verts1[i], verts1[i], p=2.0).cpu().numpy()
            dg.construct_graph_euclidean(verts1[i].cpu(),geod,device)
            num_nodes_all = torch.cat([num_nodes_all,torch.tensor(dg.nodes_idx).to(device).unsqueeze(0)],dim=0)
            dg_list.append(dg)
        return num_nodes_all,dg_list

    def topk_pi(self,A):
        k = 10  # 例如，选择每个维度上的前10个最大值
        # 获取topk的值和索引
        topk_values, topk_indices = torch.topk(A, k, dim=-1)
        # 创建一个与A形状相同的tensor，初始化为0
        result = torch.zeros_like(A)
        # 使用scatter_将topk的值放回原位置
        result.scatter_(-1, topk_indices, topk_values)
        return result

    def forward(self,feat1,feat2, dist1, dist2, verts1, verts2,alpha_i,deformer):
        loss = 0
        if self.w_dist > 0:
            k = self.k_dist
            N = self.N_dist
            batch_size = dist1.shape[0]
        
            num1 = dist1.shape[1]
            num2 = dist2.shape[1]
            random_numbers1 = []
            random_numbers2 = []
            
            numbers1 = random.sample(range(num1), N)
            random_numbers1 = torch.tensor(numbers1) # 1000
            numbers2 = random.sample(range(num2), N)
            random_numbers2 = torch.tensor(numbers2) # 1000
            # shape1
            f1 = feat1[:,random_numbers1] # 2*1000*128
            idx = knn(f1,feat1,k) #2*1000*16
            f2 = index_points(feat1, idx)  # neighbors.shape == 2*1000*16*128
            dist_result_f1 = torch.norm(f2 - f1[:, :, None, :], dim=-1)
            idx_num = random_numbers1.repeat(batch_size, 1).unsqueeze(2).expand(-1, -1, k)
            idx = idx.reshape(batch_size,-1)
            idx_num = idx_num.reshape(batch_size,-1)
            dist_f1 = torch.zeros_like(idx, dtype=torch.float)
            
            for i in range(batch_size):
                dist_f1[i] = dist1[i, idx[i], idx_num[i]]
            dist_f1 = dist_f1.reshape(batch_size, N, k)
            # # shape2
            f1 = feat2[:,random_numbers2] # 2*1000*128
            idx = knn(f1,feat2,k) #2*1000*16
            f2 = index_points(feat2, idx)  # neighbors.shape == 2*1000*16*128
            dist_result_f2 = torch.norm(f2 - f1[:, :, None, :], dim=-1)
            
            idx_num = random_numbers2.repeat(batch_size, 1).unsqueeze(2).expand(-1, -1, k)
            idx = idx.reshape(batch_size,-1)
            idx_num = idx_num.reshape(batch_size,-1)
            dist_f2 = torch.zeros_like(idx, dtype=torch.float)
            
            for i in range(batch_size):
                dist_f2[i] = dist2[i, idx[i], idx_num[i]]
            dist_f2 = dist_f2.reshape(batch_size, N, k)
            similarity1 = 1 - torch.abs(torch.nn.functional.cosine_similarity(dist_result_f1, dist_f1, dim=2))
            similarity2 = 1 - torch.abs(torch.nn.functional.cosine_similarity(dist_result_f2, dist_f2, dim=2))
            self.dist_loss = (torch.sum(similarity1)+torch.sum(similarity2))* self.w_dist
            # print('dist_loss:',self.dist_loss)
            loss += self.dist_loss
            
        B,N,_ = verts1.shape
        k = self.k_deform
        device = verts1.device
        num_nodes_all1,dg_list1 = self.deformation_graph_node(verts1)
        num_nodes_all2,dg_list2 = self.deformation_graph_node(verts2)
            
        Pi_12 = knnsearch_t_grad(feat1, feat2, alpha=alpha_i) # 2,4995,4995 
        Pi_21 = knnsearch_t_grad(feat2, feat1, alpha=alpha_i)  # 2,4995,3
        Pi_12 = self.topk_pi(Pi_12)
        Pi_21 = self.topk_pi(Pi_21)
        verts12 = torch.matmul(Pi_12,verts2)       
        verts21 = torch.matmul(Pi_21,verts1)            
        map_loss12, cross_deform_loss12, self_rec_loss12 = self.deform(verts12,verts1,Pi_12,verts2,k,num_nodes_all1.long(),dg_list1,feat1,feat2,deformer)
        map_loss21, cross_deform_loss21, self_rec_loss21 = self.deform(verts21,verts2,Pi_21,verts1,k,num_nodes_all2.long(),dg_list2,feat2,feat1,deformer)
            
        self.deform_loss = (cross_deform_loss12 + cross_deform_loss21)*N*self.w_deform/2
        # print('deform_loss:',self.deform_loss)
        loss += self.deform_loss
    
        if self.w_map > 0:
            self.map_loss = self.w_map*(map_loss12 + map_loss21)/2
            # print('map_loss:',self.map_loss)
            loss += self.map_loss
        
        if self.w_self_rec > 0:
            self.self_rec_loss = (self_rec_loss12 + self_rec_loss21)*N*self.w_self_rec/2
            # print('self_rec_loss:',self.self_rec_loss)
            loss += self.self_rec_loss
                     
        if self.w_rank > 0:
            B,N,_ = verts1.shape
            I_N = torch.eye(n=N, device=self.device)
            I_N = I_N.unsqueeze(0).repeat(B,1,1)
            self.rank_loss = (torch.mean(self._batch_frobenius_norm(torch.bmm(Pi_12,Pi_12.transpose(2, 1).contiguous()), I_N.float())) + torch.mean(self._batch_frobenius_norm(torch.bmm(Pi_21,Pi_21.transpose(2, 1).contiguous()), I_N.float())))*self.w_rank/2
            print('rank_loss:',self.rank_loss)
            loss += self.rank_loss
        
        return loss,self.dist_loss,self.deform_loss,self.map_loss,self.self_rec_loss

def get_mask(evals1, evals2, gamma=0.5, device="cpu"):
    scaling_factor = max(torch.max(evals1), torch.max(evals2))
    evals1, evals2 = evals1.to(device) / scaling_factor, evals2.to(device) / scaling_factor
    evals_gamma1, evals_gamma2 = (evals1 ** gamma)[None, :], (evals2 ** gamma)[:, None]

    M_re = evals_gamma2 / (evals_gamma2.square() + 1) - evals_gamma1 / (evals_gamma1.square() + 1)
    M_im = 1 / (evals_gamma2.square() + 1) - 1 / (evals_gamma1.square() + 1)
    return M_re.square() + M_im.square()


def square_distance(src, dst):
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist


def nn_interpolate(desc, xyz, dists, idx, idf):
    xyz = xyz.unsqueeze(0)
    B, N, _ = xyz.shape
    mask = torch.from_numpy(np.isin(idx.numpy(), idf.numpy())).int()
    mask = torch.argsort(mask, dim=-1, descending=True)[:, :, :3]
    dists, idx = torch.gather(dists, 2, mask), torch.gather(idx, 2, mask)
    transl = torch.arange(dists.size(1))
    transl[idf.flatten()] = torch.arange(idf.flatten().size(0))
    shape = idx.shape
    idx = transl[idx.flatten()].reshape(shape)
    dists, idx = dists.to(desc.device), idx.to(desc.device)

    dist_recip = 1.0 / (dists + 1e-8)
    norm = torch.sum(dist_recip, dim=2, keepdim=True)
    weight = dist_recip / norm
    interpolated_points = torch.sum(index_points(desc, idx) * weight.view(B, N, 3, 1), dim=2)

    return interpolated_points


def euler_angles_to_rotation_matrix(theta):
    R_x = torch.tensor([[1, 0, 0], [0, torch.cos(theta[0]), -torch.sin(theta[0])], [0, torch.sin(theta[0]), torch.cos(theta[0])]])
    R_y = torch.tensor([[torch.cos(theta[1]), 0, torch.sin(theta[1])], [0, 1, 0], [-torch.sin(theta[1]), 0, torch.cos(theta[1])]])
    R_z = torch.tensor([[torch.cos(theta[2]), -torch.sin(theta[2]), 0], [torch.sin(theta[2]), torch.cos(theta[2]), 0], [0, 0, 1]])

    matrices = [R_x, R_y, R_z]

    R = torch.mm(matrices[2], torch.mm(matrices[1], matrices[0]))
    return R


def get_random_rotation(x, y, z):
    thetas = torch.zeros(3, dtype=torch.float)
    degree_angles = [x, y, z]
    for axis_ind, deg_angle in enumerate(degree_angles):
        rand_deg_angle = random.random() * 2 * deg_angle - deg_angle
        rand_radian_angle = float(rand_deg_angle * np.pi) / 180.0
        thetas[axis_ind] = rand_radian_angle

    return euler_angles_to_rotation_matrix(thetas)


def data_augmentation(verts, rot_x=0, rot_y=90, rot_z=0, std=0.01, noise_clip=0.05, scale_min=0.9, scale_max=1.1):
    # random rotation
    rotation_matrix = get_random_rotation(rot_x, rot_y, rot_z).to(verts.device)
    verts = verts @ rotation_matrix.T

    # random noise
    noise = std * torch.randn(verts.shape).to(verts.device)
    noise = noise.clamp(-noise_clip, noise_clip)
    verts += noise

    # random scaling
    scales = [scale_min, scale_max]
    scale = scales[0] + torch.rand((3,)) * (scales[1] - scales[0])
    verts = verts * scale.to(verts.device)

    return verts


def augment_batch(data, rot_x=0, rot_y=90, rot_z=0, std=0.01, noise_clip=0.05, scale_min=0.9, scale_max=1.1):
    data["shape1"]["xyz"] = data_augmentation(data["shape1"]["xyz"], rot_x, rot_y, rot_z, std, noise_clip, scale_min, scale_max)
    data["shape2"]["xyz"] = data_augmentation(data["shape2"]["xyz"], rot_x, rot_y, rot_z, std, noise_clip, scale_min, scale_max)

    return data


def data_augmentation_sym(shape):
    """
    we symmetrise the shape which results in conjugation of complex info
    """
    shape["gradY"] = -shape["gradY"]  # gradients get conjugated

    # so should complex data (to double check)
    shape["cevecs"] = torch.conj(shape["cevecs"])
    shape["spec_grad"] = torch.conj(shape["spec_grad"])
    if "vts_sym" in shape:
        shape["vts"] = shape["vts_sym"]


def augment_batch_sym(data, rand=True):
    """
    if rand = False : (test time with sym only) we symmetrize the shape
    if rand = True  : with a probability of 0.5 we symmetrize the shape
    """
    #print(data["shape1"]["gradY"][0,0])
    if not rand or random.randint(0, 1) == 1:
        # print("sym")
        data_augmentation_sym(data["shape1"])
    #print(data["shape1"]["gradY"][0,0], data["shape2"]["gradY"][0,0])
    return data


def auto_WKS(evals, evects, num_E, scaled=True):
    """
    Compute WKS with an automatic choice of scale and energy

    Parameters
    ------------------------
    evals       : (K,) array of  K eigenvalues
    evects      : (N,K) array with K eigenvectors
    landmarks   : (p,) If not None, indices of landmarks to compute.
    num_E       : (int) number values of e to use
    Output
    ------------------------
    WKS or lm_WKS : (N,num_E) or (N,p*num_E)  array where each column is the WKS for a given e
                    and possibly for some landmarks
    """
    abs_ev = sorted(np.abs(evals))

    e_min, e_max = np.log(abs_ev[1]), np.log(abs_ev[-1])
    sigma = 7*(e_max-e_min)/num_E

    e_min += 2*sigma
    e_max -= 2*sigma

    energy_list = np.linspace(e_min, e_max, num_E)

    return WKS(abs_ev, evects, energy_list, sigma, scaled=scaled)


def WKS(evals, evects, energy_list, sigma, scaled=False):
    """
    Returns the Wave Kernel Signature for some energy values.

    Parameters
    ------------------------
    evects      : (N,K) array with the K eigenvectors of the Laplace Beltrami operator
    evals       : (K,) array of the K corresponding eigenvalues
    energy_list : (num_E,) values of e to use
    sigma       : (float) [positive] standard deviation to use
    scaled      : (bool) Whether to scale each energy level

    Output
    ------------------------
    WKS : (N,num_E) array where each column is the WKS for a given e
    """
    assert sigma > 0, f"Sigma should be positive ! Given value : {sigma}"

    evals = np.asarray(evals).flatten()
    indices = np.where(evals > 1e-5)[0].flatten()
    evals = evals[indices]
    evects = evects[:, indices]

    e_list = np.asarray(energy_list)
    coefs = np.exp(-np.square(e_list[:, None] - np.log(np.abs(evals))[None, :])/(2*sigma**2))  # (num_E,K)

    weighted_evects = evects[None, :, :] * coefs[:, None, :]  # (num_E,N,K)

    natural_WKS = np.einsum('tnk,nk->nt', weighted_evects, evects)  # (N,num_E)

    if scaled:
        inv_scaling = coefs.sum(1)  # (num_E)
        return (1/inv_scaling)[None, :] * natural_WKS

    else:
        return natural_WKS


def read_geodist(mat):
    # get geodist matrix
    if 'Gamma' in mat:
        G_s = mat['Gamma']
    elif 'G' in mat:
        G_s = mat['G']
    else:
        raise NotImplementedError('no geodist file found or not under name "G" or "Gamma"')

    # get square of mesh area
    if 'SQRarea' in mat:
        SQ_s = mat['SQRarea'][0]
        # print("from mat:", SQ_s)
    else:
        SQ_s = 1

    return G_s, SQ_s