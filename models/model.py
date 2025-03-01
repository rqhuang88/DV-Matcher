import os
import sys
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch_scatter import scatter
from einops import rearrange, repeat
import math
import random
import numpy as np
import matplotlib.pyplot as plt
from misc.modules import *
import torchvision.transforms as T
from PIL import Image
from featup.util import pca, remove_axes
from featup.util import norm, unnorm
import open3d as o3d
import torchvision.transforms as transforms
from misc.render_point_cloud import batch_render
from torchvision import transforms as tfs
from pytorch3d.ops import ball_query

device='cuda:0'

def figure_show(imgs):
    for i in range(imgs.shape[0]):
        img = imgs[i,:,:,:]
        img = img.cpu().numpy()
        plt.figure()
        plt.imshow(np.moveaxis(img, 0, -1))  # 多通道RGB图像
        plt.savefig('image/image_'+str(i)+'.png')
        plt.close()

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

def farthest_point_sample(xyz, npoint):
    xyz = xyz.unsqueeze(0)
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

def rotate_point_cloud_batch_torch(cloud, angle, axis='z'):
    # Ensure cloud is on the correct device
    # cloud = cloud.to(device)    # 定义不同轴的旋转矩阵
    if axis == 'z':
        rotation_matrix = torch.tensor([
            [np.cos(angle), -np.sin(angle), 0],
            [np.sin(angle), np.cos(angle), 0],
            [0, 0, 1]
        ], device=device)
    elif axis == 'y':
        rotation_matrix = torch.tensor([
            [np.cos(angle), 0, np.sin(angle)],
            [0, 1, 0],
            [-np.sin(angle), 0, np.cos(angle)]
        ], device=device)
    elif axis == 'x':
        rotation_matrix = torch.tensor([
            [1, 0, 0],
            [0, np.cos(angle), -np.sin(angle)],
            [0, np.sin(angle), np.cos(angle)]
        ], device=device)
    else:
        raise ValueError("Axis must be 'x', 'y', or 'z'")    
    rotation_matrix = rotation_matrix.float()    # b代表batch size，p代表点的数量
    b, p, _ = cloud.shape    # 扩增旋转矩阵至批量维度
    rotation_matrix = rotation_matrix[None, :].repeat(b, 1, 1)    # 使用旋转矩阵旋转每批点云
    # print(cloud.shape)
    # print(rotation_matrix.shape)
    rotated_cloud = torch.bmm(cloud.permute(0,2,1), rotation_matrix)    
    return rotated_cloud


class SA_Layer(nn.Module):
    def __init__(self, channels):
        super(SA_Layer, self).__init__()
        self.bn1 = nn.BatchNorm1d(64)
        self.conv1 = nn.Sequential(nn.Conv1d(128, 64, kernel_size=1, bias=False),
                                   self.bn1,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.q_conv = nn.Conv1d(channels, channels // 4, 1, bias=False)
        self.k_conv = nn.Conv1d(channels, channels // 4, 1, bias=False)
        self.q_conv.weight = self.k_conv.weight 
        self.v_conv = nn.Conv1d(channels, channels, 1)
        self.trans_conv = nn.Conv1d(channels, channels, 1)
        self.after_norm = nn.BatchNorm1d(channels)
        self.act = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        x_q = self.q_conv(x).permute(0, 2, 1) # b, n, c 
        x_k = self.k_conv(x)# b, c, n        
        x_v = self.v_conv(x)
        energy = torch.bmm(x_q, x_k) # b, n, n 
        attention = self.softmax(energy)
        attention = attention / (1e-9 + attention.sum(dim=1, keepdims=True))
        x_r = torch.bmm(x_v, attention) # b, c, n 
        x_r = self.act(self.after_norm(self.trans_conv(x - x_r)))
        x = x + x_r
        return x

class GlobalDownSample(nn.Module):
    def __init__(self, npts_ds,dim_v):
        super(GlobalDownSample, self).__init__()
        self.npts_ds = npts_ds
        self.q_conv = nn.Conv1d(dim_v, dim_v, 1, bias=False)
        self.k_conv = nn.Conv1d(dim_v, dim_v, 1, bias=False)
        self.v_conv = nn.Conv1d(dim_v, dim_v, 1, bias=False)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        q = self.q_conv(x)  # (B, C, N) -> (B, C, N)
        k = self.k_conv(x)  # (B, C, N) -> (B, C, N)
        v = self.v_conv(x)  # (B, C, N) -> (B, C, N)
        energy = rearrange(q, 'B C N -> B N C').contiguous() @ k  # (B, N, C) @ (B, C, N) -> (B, N, N)
        scale_factor = math.sqrt(q.shape[-2])
        attention = self.softmax(energy / scale_factor)  # (B, N, N) -> (B, N, N)
        selection = torch.sum(attention, dim=-2)  # (B, N, N) -> (B, N)
        self.idx = selection.topk(self.npts_ds, dim=-1)[1]  # (B, N) -> (B, M)
        scores = torch.gather(attention, dim=1, index=repeat(self.idx, 'B M -> B M N', N=attention.shape[-1]))  # (B, N, N) -> (B, M, N)
        v = scores @ rearrange(v, 'B C N -> B N C').contiguous()  # (B, M, N) @ (B, N, C) -> (B, M, C)
        out = rearrange(v, 'B M C -> B C M').contiguous()  # (B, M, C) -> (B, C, M)
        return out


class LocalDownSample(nn.Module):
    def __init__(self, npts_ds):
        super(LocalDownSample, self).__init__()
        self.npts_ds = npts_ds  # number of downsampled points
        self.K = 32  # number of neighbors
        self.group_type = 'diff'
        self.q_conv = nn.Conv2d(128, 128, 1, bias=False)
        self.k_conv = nn.Conv2d(128, 128, 1, bias=False)
        self.v_conv = nn.Conv2d(128, 128, 1, bias=False)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        neighbors = group(x, self.K, self.group_type)  # (B, C, N) -> (B, C, N, K)
        q = self.q_conv(rearrange(x, 'B C N -> B C N 1')).contiguous()  # (B, C, N) -> (B, C, N, 1)
        q = rearrange(q, 'B C N 1 -> B N 1 C').contiguous()  # (B, C, N, 1) -> (B, N, 1, C)
        k = self.k_conv(neighbors)  # (B, C, N, K) -> (B, C, N, K)
        k = rearrange(k, 'B C N K -> B N C K').contiguous()  # (B, C, N, K) -> (B, N, C, K)
        v = self.v_conv(neighbors)  # (B, C, N, K) -> (B, C, N, K)
        v = rearrange(v, 'B C N K -> B N K C').contiguous()  # (B, C, N, K) -> (B, N, K, C)
        energy = q @ k  # (B, N, 1, C) @ (B, N, C, K) -> (B, N, 1, K)
        scale_factor = math.sqrt(q.shape[-1])
        attention = self.softmax(energy / scale_factor)  # (B, N, 1, K) -> (B, N, 1, K)
        selection = rearrange(torch.std(attention, dim=-1, unbiased=False), 'B N 1 -> B N').contiguous()  # (B, N, 1, K) -> (B, N, 1) -> (B, N)
        self.idx = selection.topk(self.npts_ds, dim=-1)[1]  # (B, N) -> (B, M)
        scores = torch.gather(attention, dim=1, index=repeat(self.idx, 'B M -> B M 1 K', K=attention.shape[-1]))  # (B, N, 1, K) -> (B, M, 1, K)
        v = torch.gather(v, dim=1, index=repeat(self.idx, 'B M -> B M K C', K=v.shape[-2], C=v.shape[-1]))  # (B, N, K, C) -> (B, M, K, C)
        out = rearrange(scores@v, 'B M 1 C -> B C M').contiguous()  # (B, M, 1, K) @ (B, M, K, C) -> (B, M, 1, C) -> (B, C, M)
        return out


class UpSample(nn.Module):
    def __init__(self,dim_v):
        super(UpSample, self).__init__()
        self.q_conv = nn.Conv1d(dim_v, dim_v, 1, bias=False)
        self.k_conv = nn.Conv1d(dim_v, dim_v, 1, bias=False)
        self.v_conv = nn.Conv1d(dim_v, dim_v, 1, bias=False)
        self.skip_link = nn.Conv1d(dim_v, dim_v, 1, bias=False)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, pcd_up, pcd_down):
        q = self.q_conv(pcd_up)  # (B, C, N) -> (B, C, N)
        k = self.k_conv(pcd_down)  # (B, C, M) -> (B, C, M)
        v = self.v_conv(pcd_down)  # (B, C, M) -> (B, C, M)
        energy = rearrange(q, 'B C N -> B N C').contiguous() @ k  # (B, N, C) @ (B, C, M) -> (B, N, M)
        scale_factor = math.sqrt(q.shape[-2])
        attention = self.softmax(energy / scale_factor)  # (B, N, M) -> (B, N, M)
        x = attention @ rearrange(v, 'B C M -> B M C').contiguous()  # (B, N, M) @ (B, M, C) -> (B, N, C)
        x = rearrange(x, 'B N C -> B C N').contiguous()  # (B, N, C) -> (B, C, N)
        x = self.skip_link(pcd_up) + x  # (B, C, N) + (B, C, N) -> (B, C, N)
        return x

class Embedding(nn.Module):
    def __init__(self,k=32):
        super(Embedding, self).__init__()
        self.device = 'cuda:0'
        self.K = k
        self.group_type = 'center_diff'
        self.conv1 = nn.Sequential(nn.Conv2d(6, 128, 1, bias=False), nn.BatchNorm2d(128), nn.LeakyReLU(0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(128, 64, 1, bias=False), nn.BatchNorm2d(64), nn.LeakyReLU(0.2))
        self.conv3 = nn.Sequential(nn.Conv2d(128, 128, 1, bias=False), nn.BatchNorm2d(128), nn.LeakyReLU(0.2))
        self.conv4 = nn.Sequential(nn.Conv2d(128, 64, 1, bias=False), nn.BatchNorm2d(64), nn.LeakyReLU(0.2))
        self.conv5 = nn.Sequential(nn.Conv1d(128, 384, 1, bias=False), nn.BatchNorm1d(384), nn.LeakyReLU(0.2))
        self.conv6 = nn.Sequential(nn.Conv1d(384, 64, 1, bias=False), nn.BatchNorm1d(64), nn.LeakyReLU(0.2))

    def pos_encoding_sin_wave(self, coor):
        # ref to https://arxiv.org/pdf/2003.08934v2.pdf
        D = 64 #
        # normal the coor into [-1, 1], batch wise
        normal_coor = 2 * ((coor - coor.min()) / (coor.max() - coor.min())) - 1 

        # define sin wave freq
        freqs = torch.arange(D, dtype=torch.float).cuda() 
        freqs = np.pi * (2**freqs)       

        freqs = freqs.view(*[1]*len(normal_coor.shape), -1) # 1 x 1 x 1 x D
        normal_coor = normal_coor.unsqueeze(-1) # B x 3 x N x 1
        k = normal_coor * freqs # B x 3 x N x D
        s = torch.sin(k) # B x 3 x N x D
        c = torch.cos(k) # B x 3 x N x D
        x = torch.cat([s,c], -1) # B x 3 x N x 2D
        pos = x.transpose(-1,-2).reshape(coor.shape[0], -1, coor.shape[-1]) # B 6D N
        # zero_pad = torch.zeros(x.size(0), 2, x.size(-1)).cuda()
        # pos = torch.cat([x, zero_pad], dim = 1)
        # pos = self.pos_embed_wave(x)
        return pos
    
    def forward(self, x):
        batch_size = x.size(0)
        pos = self.pos_encoding_sin_wave(x)
        x_list = []
        x = group(x,self.K, self.group_type)  # (B, C=3, N) -> (B, C=6, N, K)
        x = self.conv1(x)  # (B, C=6, N, K) -> (B, C=128, N, K)
        x = self.conv2(x)  # (B, C=128, N, K) -> (B, C=64, N, K)
        x = x.max(dim=-1, keepdim=False)[0]  # (B, C=64, N, K) -> (B, C=64, N)
        x_list.append(x)
        x = group(x,self.K, self.group_type)  # (B, C=64, N) -> (B, C=128, N, K)
        x = self.conv3(x)  # (B, C=128, N, K) -> (B, C=128, N, K)
        x = self.conv4(x)  # (B, C=128, N, K) -> (B, C=64, N, K)
        x = x.max(dim=-1, keepdim=False)[0]  # (B, C=64, N, K) -> (B, C=64, N)
        x_list.append(x)
        x = torch.cat(x_list, dim=1)  # (B, C=128, N)
        x = self.conv5(x)
        x = x + pos
        x = self.conv6(x)
        return x

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


def knn_new(a, b, k):
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

def select_neighbors(pcd, K, neighbor_type):
    #batch_size = pcd.size(0)
    pcd = pcd.permute(0, 2, 1)  # pcd.shape == (B, N, C)
    if neighbor_type == 'neighbor':
        idx = knn_new(pcd, pcd, K)  # idx.shape == (B, N, K)
        neighbors = index_points(pcd, idx)  # neighbors.shape == (B, N, K, C)
        neighbors = neighbors.permute(0, 3, 1, 2)  # output.shape == (B, C, N, K)
    elif neighbor_type == 'diff':
        idx = knn_new(pcd, pcd, K)  # idx.shape == (B, N, K)
        neighbors = index_points(pcd, idx)  # neighbors.shape == (B, N, K, C)
        diff = neighbors - pcd[:, :, None, :]  # diff.shape == (B, N, K, C)
        neighbors = diff.permute(0, 3, 1, 2)  # output.shape == (B, C, N, K)            
    elif neighbor_type == 'grobal':
        b = pcd.shape[0]
        n = pcd.shape[1]
        k = 512
        arr = torch.zeros(b,n,k)
        indices = torch.arange(k).reshape(1,1,k)
        idx = indices.expand_as(arr).to('cuda:0')
        neighbors = index_points(pcd, idx)  # neighbors.shape == (B, N, K, C)
        diff = neighbors - pcd[:, :, None, :]  # diff.shape == (B, N, K, C)
        neighbors = diff.permute(0, 3, 1, 2)  # output.shape == (B, C, N, K)
    else:
        raise ValueError(f'neighbor_type should be "neighbor" or "diff", but got {neighbor_type}')
    return neighbors

def group(pcd, K, group_type):
    if group_type == 'neighbor':
        neighbors = select_neighbors(pcd, K, 'neighbor')  # neighbors.shape == (B, C, N, K)
    elif group_type == 'diff':
        diff = select_neighbors(pcd, K, 'diff')  # diff.shape == (B, C, N, K)
        output = diff  # output.shape == (B, C, N, K)
    elif group_type == 'grobal':
        diff = select_neighbors(pcd, K, 'grobal')  # diff.shape == (B, C, N, K)
        output = diff  # output.shape == (B, C, N, K)
    elif group_type == 'center_neighbor':
        neighbors = select_neighbors(pcd, K, 'neighbor')   # neighbors.shape == (B, C, N, K)
        output = torch.cat([pcd[:, :, :, None].repeat(1, 1, 1, K), neighbors], dim=1)  # output.shape == (B, 2C, N, K)
    elif group_type == 'center_diff':
        diff = select_neighbors(pcd, K, 'diff')  # diff.shape == (B, C, N, K)
        output = torch.cat([pcd[:, :, :, None].repeat(1, 1, 1, K), diff], dim=1)  # output.shape == (B, 2C, N, K)
    else:
        raise ValueError(f'group_type should be neighbor, diff, center_neighbor or center_diff, but got {group_type}')
    return output.contiguous()

class N2PAttention(nn.Module):
    def __init__(self,k):
        super(N2PAttention, self).__init__()
        self.heads = 4
        self.K = k
        self.group_type = 'diff'
        self.q_conv = nn.Conv2d(64, 64, 1, bias=False)
        self.k_conv = nn.Conv2d(64, 64, 1, bias=False)
        self.v_conv = nn.Conv2d(64, 64, 1, bias=False)
        self.softmax = nn.Softmax(dim=-1)
        self.ff = nn.Sequential(nn.Conv1d(64, 256, 1, bias=False), nn.LeakyReLU(0.2), nn.Conv1d(256, 64, 1, bias=False))
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)

    def forward(self, x):
        neighbors = group(x,self.K, self.group_type)  # (B, C, N) -> (B, C, N, K)
        q = self.q_conv(rearrange(x, 'B C N -> B C N 1')).contiguous()  # (B, C, N) -> (B, C, N, 1)
        q = self.split_heads(q, self.heads)  # (B, C, N, 1) -> (B, H, N, 1, D)
        k = self.k_conv(neighbors)  # (B, C, N, K) -> (B, C, N, K)
        k = self.split_heads(k, self.heads)  # (B, C, N, K) -> (B, H, N, K, D)
        v = self.v_conv(neighbors)  # (B, C, N, K) -> (B, C, N, K)
        v = self.split_heads(v, self.heads)  # (B, C, N, K) -> (B, H, N, K, D)
        energy = q @ rearrange(k, 'B H N K D -> B H N D K').contiguous()  # (B, H, N, 1, D) @ (B, H, N, D, K) -> (B, H, N, 1, K)
        scale_factor = math.sqrt(q.shape[-1])
        attention = self.softmax(energy / scale_factor)  # (B, H, N, 1, K) -> (B, H, N, 1, K)
        tmp = rearrange(attention@v, 'B H N 1 D -> B (H D) N').contiguous()  # (B, H, N, 1, K) @ (B, H, N, K, D) -> (B, H, N, 1, D) -> (B, C=H*D, N)
        x = self.bn1(x + tmp)  # (B, C, N) + (B, C, N) -> (B, C, N)
        tmp = self.ff(x)  # (B, C, N) -> (B, C, N)
        x = self.bn2(x + tmp)  # (B, C, N) + (B, C, N) -> (B, C, N)
        return x

    @staticmethod
    def split_heads(x, heads):
        x = rearrange(x, 'B (H D) N K -> B H N K D', H=heads).contiguous()  # (B, C, N, K) -> (B, H, N, K, D)
        return x
    
class N2PAttention_DIM(nn.Module):
    def __init__(self,k):
        super(N2PAttention_DIM, self).__init__()
        self.heads = 4
        self.K = k
        self.group_type = 'diff'
        self.q_conv = nn.Conv2d(128, 128, 1, bias=False)
        self.k_conv = nn.Conv2d(128, 128, 1, bias=False)
        self.v_conv = nn.Conv2d(128, 128, 1, bias=False)
        self.softmax = nn.Softmax(dim=-1)
        self.ff = nn.Sequential(nn.Conv1d(128, 512, 1, bias=False), nn.LeakyReLU(0.2), nn.Conv1d(512, 128, 1, bias=False))
        self.bn1 = nn.BatchNorm1d(128)
        self.bn2 = nn.BatchNorm1d(128)

    def forward(self, x):
        neighbors = group(x,self.K, self.group_type)  # (B, C, N) -> (B, C, N, K)
        q = self.q_conv(rearrange(x, 'B C N -> B C N 1')).contiguous()  # (B, C, N) -> (B, C, N, 1)
        q = self.split_heads(q, self.heads)  # (B, C, N, 1) -> (B, H, N, 1, D)
        k = self.k_conv(neighbors)  # (B, C, N, K) -> (B, C, N, K)
        k = self.split_heads(k, self.heads)  # (B, C, N, K) -> (B, H, N, K, D)
        v = self.v_conv(neighbors)  # (B, C, N, K) -> (B, C, N, K)
        v = self.split_heads(v, self.heads)  # (B, C, N, K) -> (B, H, N, K, D)
        energy = q @ rearrange(k, 'B H N K D -> B H N D K').contiguous()  # (B, H, N, 1, D) @ (B, H, N, D, K) -> (B, H, N, 1, K)
        scale_factor = math.sqrt(q.shape[-1])
        attention = self.softmax(energy / scale_factor)  # (B, H, N, 1, K) -> (B, H, N, 1, K)
        tmp = rearrange(attention@v, 'B H N 1 D -> B (H D) N').contiguous()  # (B, H, N, 1, K) @ (B, H, N, K, D) -> (B, H, N, 1, D) -> (B, C=H*D, N)
        x = self.bn1(x + tmp)  # (B, C, N) + (B, C, N) -> (B, C, N)
        tmp = self.ff(x)  # (B, C, N) -> (B, C, N)
        x = self.bn2(x + tmp)  # (B, C, N) + (B, C, N) -> (B, C, N)
        return x

    @staticmethod
    def split_heads(x, heads):
        x = rearrange(x, 'B (H D) N K -> B H N K D', H=heads).contiguous()  # (B, C, N, K) -> (B, H, N, K, D)
        return x
        
class P2PAttention(nn.Module):
    def __init__(self):
        super(P2PAttention, self).__init__()
        self.heads = 4
        self.K = 512
        self.group_type = 'grobal'
        self.q_conv = nn.Conv2d(128, 128, 1, bias=False)
        self.k_conv = nn.Conv2d(128, 128, 1, bias=False)
        self.v_conv = nn.Conv2d(128, 128, 1, bias=False)
        self.softmax = nn.Softmax(dim=-1)
        self.ff = nn.Sequential(nn.Conv1d(128, 512, 1, bias=False), nn.LeakyReLU(0.2), nn.Conv1d(512, 128, 1, bias=False))
        self.bn1 = nn.BatchNorm1d(128)
        self.bn2 = nn.BatchNorm1d(128)

    def forward(self, x):
        neighbors = group(x, self.K, self.group_type)  # (B, C, N) -> (B, C, N, K)
        q = self.q_conv(rearrange(x, 'B C N -> B C N 1')).contiguous()  # (B, C, N) -> (B, C, N, 1)
        q = self.split_heads(q, self.heads)  # (B, C, N, 1) -> (B, H, N, 1, D)
        k = self.k_conv(neighbors)  # (B, C, N, K) -> (B, C, N, K)
        k = self.split_heads(k, self.heads)  # (B, C, N, K) -> (B, H, N, K, D)
        v = self.v_conv(neighbors)  # (B, C, N, K) -> (B, C, N, K)
        v = self.split_heads(v, self.heads)  # (B, C, N, K) -> (B, H, N, K, D)
        energy = q @ rearrange(k, 'B H N K D -> B H N D K').contiguous()  # (B, H, N, 1, D) @ (B, H, N, D, K) -> (B, H, N, 1, K)
        scale_factor = math.sqrt(q.shape[-1])
        attention = self.softmax(energy / scale_factor)  # (B, H, N, 1, K) -> (B, H, N, 1, K)
        tmp = rearrange(attention@v, 'B H N 1 D -> B (H D) N').contiguous()  # (B, H, N, 1, K) @ (B, H, N, K, D) -> (B, H, N, 1, D) -> (B, C=H*D, N)
        x = self.bn1(x + tmp)  # (B, C, N) + (B, C, N) -> (B, C, N)
        tmp = self.ff(x)  # (B, C, N) -> (B, C, N)
        x = self.bn2(x + tmp)  # (B, C, N) + (B, C, N) -> (B, C, N)
        return x
    
    @staticmethod
    def split_heads(x, heads):
        x = rearrange(x, 'B (H D) N K -> B H N K D', H=heads).contiguous()  # (B, C, N, K) -> (B, H, N, K, D)
        return x
    
class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dims = [], bias = True, act = nn.ELU()):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dims = hidden_dims
        self.act = act
        if len(hidden_dims)>0:
            fc = [nn.Linear(input_dim, self.hidden_dims[0], bias = bias)]
            fc.append(act)
            for i in range(len(self.hidden_dims)-1):
                fc.append(nn.Linear(self.hidden_dims[i], self.hidden_dims[i+1], bias = bias))
                fc.append(act)
            fc.append(nn.Linear(self.hidden_dims[-1], output_dim, bias = bias))
        else:
            fc = [nn.Linear(input_dim, output_dim, bias = bias), act]
        self.linear = nn.Sequential(*fc)
    
    def forward(self, x):
        return self.linear(x)
     
class Deformer(nn.Module):
    def __init__(self,k):
        super(Deformer, self).__init__()
        act = nn.ELU()
        # self.bn = nn.BatchNorm1d(128)
        # self.predeform = nn.Sequential(nn.Conv1d(128*2+3*2, 128, kernel_size=1, bias=False),
        #                            self.bn,
        #                            nn.LeakyReLU(negative_slope=0.2))        
        self.conv_layer = nn.Conv2d(in_channels=k, out_channels=1, kernel_size=(1, 1))
        self.deformation_decoder_layer = MLP(input_dim = 128*2+3*2, output_dim = 3 + 6, hidden_dims = [512,256,128], bias = True, act = act) 
    def forward(self,feat1_conv,feat2_conv,verts1,verts12,Pi_12,fps1):
        # feat1 = torch.mean(feat1_conv, dim=2)
        # feat2 = torch.mean(feat2_conv, dim=2)
        # print(feat1.shape)
        feat1 = self.conv_layer(feat1_conv.permute(0,2,1,3)).squeeze(1)
        feat2 = self.conv_layer(feat2_conv.permute(0,2,1,3)).squeeze(1) 
        # feat2 = index_points_idx(feat2,T12_pred)
        feat2 = torch.matmul(Pi_12,feat2)
        st_projected_vts2 = index_points_idx(verts12,fps1)
        st_projected_feat2 = index_points_idx(feat2,fps1)        
        st_vts1 = index_points_idx(verts1,fps1)
        st_feat1 = index_points_idx(feat1,fps1)
        input_vec = torch.cat([st_vts1,st_feat1,st_projected_vts2,st_projected_feat2],dim=-1)
        deformations = self.deformation_decoder_layer(input_vec)
        return deformations
 
class Uni3FC(nn.Module):
    def __init__(self, k=40):
        super(Uni3FC, self).__init__()
        self.device = 'cuda:0'
        self.k = k
        self.emb_dims = 512
        self.img_size = 224
        self.img_offset = torch.Tensor([[-2, -2], [-2, -1], [-2, 0], [-2, 1], [-2, 2], 
                                        [-1, -2], [-1, -1], [-1, 0], [-1, 1], [-1, 2], 
                                        [0, -2], [0, -1], [0, 0], [0, 1], [0, 2],
                                        [1, -2], [1, -1], [1, 0], [1, 1], [1, 2],
                                        [2, -2], [2, -1], [2, 0], [2, 1], [2, 2]])
        self.img_mean = torch.Tensor([0.485, 0.456, 0.406])
        self.img_std = torch.Tensor([0.229, 0.224, 0.225])
        self.proj_reduction = 'sum'

        self.bn = nn.BatchNorm1d(384)
        self.bn0 = nn.BatchNorm1d(64) 
        self.bn1 = nn.BatchNorm1d(self.emb_dims)
        self.bn2 = nn.BatchNorm1d(self.emb_dims)
        self.bn3 = nn.BatchNorm1d(128)
        self.bn4 = nn.BatchNorm1d(128)
        self.bn5 = nn.BatchNorm1d(128)
        self.bn6 = nn.BatchNorm1d(128)
        self.out = 128
       
        self.conv = nn.Sequential(nn.Conv1d(1152, 384, kernel_size=1, bias=False),
                                   self.bn,
                                   nn.LeakyReLU(negative_slope=0.2)) 
        self.conv0 = nn.Sequential(nn.Conv1d(384, 64, kernel_size=1, bias=False),
                                   self.bn0,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv1 = nn.Sequential(nn.Conv1d(256, self.emb_dims, kernel_size=1, bias=False),
                                   self.bn1,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv1d(256, self.emb_dims, kernel_size=1, bias=False),
                                   self.bn2,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(nn.Conv1d(256+self.emb_dims, 128, kernel_size=1, bias=False),
                                   self.bn3,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv4 = nn.Sequential(nn.Conv1d(256+self.emb_dims, 128, kernel_size=1, bias=False),
                                   self.bn4,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv5 = nn.Sequential(nn.Conv1d(256, 128, kernel_size=1, bias=False),
                                        self.bn5,
                                        nn.LeakyReLU(negative_slope=0.2))
        self.conv6 = nn.Sequential(nn.Conv1d(512, 128, kernel_size=1, bias=False),
                                        self.bn6,
                                        nn.LeakyReLU(negative_slope=0.2))

        self.n2p_attention1 = N2PAttention(self.k)
        self.n2p_attention2 = N2PAttention(self.k)
        self.n2p_attention3 = N2PAttention(self.k)
        self.n2p_attention4 = N2PAttention(self.k)
        self.n2p_attention5 = N2PAttention_DIM(self.k)
        self.n2p_attention6 = N2PAttention_DIM(self.k)
        self.n2p_attention7 = N2PAttention_DIM(self.k)

        self.sa1 = SA_Layer(64)
        self.sa2 = SA_Layer(64)
        self.sa3 = SA_Layer(64)
        self.sa4 = SA_Layer(64)

    def pos_encoding_sin_wave(self, coor):
        # ref to https://arxiv.org/pdf/2003.08934v2.pdf
        D = 64 #
        # normal the coor into [-1, 1], batch wise
        normal_coor = 2 * ((coor - coor.min()) / (coor.max() - coor.min())) - 1 

        # define sin wave freq
        freqs = torch.arange(D, dtype=torch.float).cuda() 
        freqs = np.pi * (2**freqs)       

        freqs = freqs.view(*[1]*len(normal_coor.shape), -1) # 1 x 1 x 1 x D
        normal_coor = normal_coor.unsqueeze(-1) # B x 3 x N x 1
        k = normal_coor * freqs # B x 3 x N x D
        s = torch.sin(k) # B x 3 x N x D
        c = torch.cos(k) # B x 3 x N x D
        x = torch.cat([s,c], -1) # B x 3 x N x 2D
        pos = x.transpose(-1,-2).reshape(coor.shape[0], -1, coor.shape[-1]) # B 6D N
        return pos

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

    ''' Efficient Projection '''
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
        zero_mask = (img == 0)
        # zero_mask = (img == -10000)
        mean_vec = self.img_mean.unsqueeze(dim=0).unsqueeze(dim=2).unsqueeze(dim=3).cuda()  # 1 3 1 1
        std_vec = self.img_std.unsqueeze(dim=0).unsqueeze(dim=2).unsqueeze(dim=3).cuda()   # 1 3 1 1    
        img = nn.Sigmoid()(img)
        img_norm = img.sub(mean_vec).div(std_vec) # 1 3 224 224
        img_norm = img_norm[:,0,:,:].unsqueeze(1) # 1 1 224 224
        img_norm = self.get_colored_depth_maps(img_norm,self.img_size,self.img_size)
        img_norm = torch.tensor(img_norm).permute(0,3,1,2).to(device)
        img_norm[zero_mask] = -1
        return img_norm, pc_min, grid_size, (offset_x, offset_y)

    ''' Image-to-Point Back-projection '''
    def I2P(self, pc, f, pc_min, grid_size, offsets):
        B, N, _ = pc.shape

        # Point Index
        idx_xy = torch.floor((pc[:, :, :2] - pc_min) / grid_size)  # B N 2
        
        # Point Densify
        idx_xy_dense = idx_xy + 1 # 5,64,2
        # B N 1 2 + 1 1 9 2 -> B N 9 2 -> B 9N 2
        
        # Object to Image Center
        idx_xy_dense_offset = idx_xy_dense + torch.cat(offsets, dim=1).unsqueeze(dim=1)  # B, N, 2

        # Expand Point Features
        B, C, H, W = f.shape
        f_dense = F.interpolate(f, size=(self.img_size, self.img_size), mode='bicubic').reshape(B, C, -1).permute(0, 2, 1)  # B, N, C

        # B N 9 C -> B 9N C
        assert idx_xy_dense_offset.min() >= 0 and idx_xy_dense_offset.max() <= (self.img_size - 1), str(idx_xy_dense_offset.min()) + '-' + str(idx_xy_dense_offset.max())
        
        # change idx to 1-dim
        new_idx_xy_dense = idx_xy_dense_offset[:, :, 0] * self.img_size + idx_xy_dense_offset[:, :, 1]  # B, N
        
        # Get Image Features
        out = torch.gather(f_dense, 1, new_idx_xy_dense.to(dtype=torch.int64).unsqueeze(-1).repeat(1, 1, C))
        return out    
    
    def forward(self, x, dino_feat,upsampler):
        batch_size = x.size(0)
        num_points = x.size(2)
        if dino_feat == None:
            pts_1 = rotate_point_cloud_batch_torch(x, -math.pi/2, axis='z')
            pts_2 = torch.cat((pts_1[..., 2: 3], pts_1[..., 0: 2]), dim=-1)
            pts_3 = torch.cat((pts_1[..., 1: 3], pts_1[..., 0: 1]), dim=-1)
            imgs_1, pc_min_1, grid_size_1, offsets_1 = self.proj2img(pts_1)  # b, 3, 224, 224
            imgs_2, pc_min_2, grid_size_2, offsets_2 = self.proj2img(pts_2)  # b, 3, 224, 224
            imgs_3, pc_min_3, grid_size_3, offsets_3 = self.proj2img(pts_3)  # b, 3, 224, 224     
            
            with torch.no_grad():
                image_tensor = torch.cat((imgs_1, imgs_2, imgs_3), dim=0) # b, c, h, w
                img_feats = upsampler(image_tensor)
                
                ''' 2D Visual Features '''
                B, _, _, _ = img_feats.shape
                img_feats_1 = img_feats[0: B // 3]
                img_feats_2 = img_feats[B // 3: B // 3 * 2]
                img_feats_3 = img_feats[B // 3 * 2: ]

                clip_feats_1 = self.I2P(pts_1, img_feats_1, pc_min_1, grid_size_1, offsets_1)
                clip_feats_1 = torch.nn.functional.normalize(clip_feats_1,dim=-1)
                clip_feats_2 = self.I2P(pts_2, img_feats_2, pc_min_2, grid_size_2, offsets_2)
                clip_feats_2 = torch.nn.functional.normalize(clip_feats_2,dim=-1)
                clip_feats_3= self.I2P(pts_3, img_feats_3, pc_min_3, grid_size_3, offsets_3)
                clip_feats_3 = torch.nn.functional.normalize(clip_feats_3,dim=-1)
            
            clip_feats = torch.cat((clip_feats_1,clip_feats_2,clip_feats_3),dim=-1)   
        else:
            clip_feats = dino_feat
        
        clip_feats = clip_feats.permute(0,2,1)
        clip_feats = self.conv(clip_feats)     
        pos = self.pos_encoding_sin_wave(x)
        clip_feats_new = clip_feats + pos
        #clip_feats_new = pos
        tmp = self.conv0(clip_feats_new)
        # x1:feature
        x1 = self.n2p_attention1(tmp)
        x1_g = self.sa1(tmp)

        x2 = self.n2p_attention2(x1)
        x2_g = self.sa2(x1_g)

        x3 = self.n2p_attention3(x2)
        x3_g = self.sa3(x2_g)

        x4 = self.n2p_attention4(x3)
        x4_g = self.sa4(x3_g)
        
        x = torch.cat((x1, x2, x3, x4), dim=1)
        x_g = torch.cat((x1_g, x2_g, x3_g, x4_g), dim=1)

        x = self.conv1(x)    # (batch_size, 64*4, num_points) -> (batch_size, emb_dims, num_points)
        x_g = self.conv2(x_g)    # (batch_size, 64*4, num_points) -> (batch_size, emb_dims, num_points)

        x = x.max(dim=-1, keepdim=True)[0]      # (batch_size, emb_dims, num_points) -> (batch_size, emb_dims, 1)    
        x_g = x_g.max(dim=-1, keepdim=True)[0]      # (batch_size, emb_dims, num_points) -> (batch_size, emb_dims, 1)

        x = x.repeat(1, 1, num_points)          # (batch_size, emb_dims, num_points) 
        x_g = x_g.repeat(1, 1, num_points)          # (batch_size, emb_dims, num_points) 

        x = torch.cat((x, x1, x2, x3,x4), dim=1)   # (batch_size, 1024+64*4, num_points)
        x_g = torch.cat((x_g, x1_g, x2_g, x3_g,x4_g), dim=1)   # (batch_size, 1024+64*4, num_points)

        x = self.conv3(x)                       # (batch_size, 1024+64*4, num_points) -> (batch_size, 128, num_points)
        x_g = self.conv4(x_g)                   # (batch_size, 1024+64*4, num_points) -> (batch_size, 128, num_points)

        x = torch.cat((x, x_g), dim=1)   # (batch_size, 128*2, num_points)
        x_1 = self.conv5(x)
        x_2 = self.n2p_attention5(x_1)
        x_3 = self.n2p_attention6(x_2)
        x_4 = self.n2p_attention7(x_3)

        x = torch.cat((x_1, x_2, x_3, x_4), dim=1) 
        x = self.conv6(x)

        x = x.transpose(2,1).contiguous()
        x = x.view(batch_size, num_points, self.out)
        cfeats = tmp.permute(0,2,1)
        return x, cfeats

class cross_transformer(nn.Module):

    def __init__(self, d_model=256, d_model_out=256, nhead=4, dim_feedforward=1024, dropout=0.0):
        super().__init__()
        self.multihead_attn1 = nn.MultiheadAttention(d_model_out, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear11 = nn.Linear(d_model_out, dim_feedforward)
        self.dropout1 = nn.Dropout(dropout)
        self.linear12 = nn.Linear(dim_feedforward, d_model_out)

        self.norm12 = nn.LayerNorm(d_model_out)
        self.norm13 = nn.LayerNorm(d_model_out)

        self.dropout12 = nn.Dropout(dropout)
        self.dropout13 = nn.Dropout(dropout)

        self.activation1 = torch.nn.GELU()

        self.input_proj = nn.Conv1d(d_model, d_model_out, kernel_size=1)

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    # 原始的transformer
    def forward(self, src1, src2, if_act=False):
        src1 = self.input_proj(src1)
        src2 = self.input_proj(src2)

        b, c, _ = src1.shape

        src1 = src1.reshape(b, c, -1).permute(2, 0, 1)
        src2 = src2.reshape(b, c, -1).permute(2, 0, 1)

        src1 = self.norm13(src1)
        src2 = self.norm13(src2)

        src12 = self.multihead_attn1(query=src1,
                                     key=src2,
                                     value=src2)[0]


        src1 = src1 + self.dropout12(src12)
        src1 = self.norm12(src1)

        src12 = self.linear12(self.dropout1(self.activation1(self.linear11(src1))))
        src1 = src1 + self.dropout13(src12)


        src1 = src1.permute(1, 2, 0)

        return src1
    
class Uni3FC_DINO_proj(nn.Module):
    def __init__(self):
        super(Uni3FC_DINO_proj, self).__init__()
        self.device = 'cuda:0'
        self.img_size = 224
        self.img_offset = torch.Tensor([[-2, -2], [-2, -1], [-2, 0], [-2, 1], [-2, 2], 
                                        [-1, -2], [-1, -1], [-1, 0], [-1, 1], [-1, 2], 
                                        [0, -2], [0, -1], [0, 0], [0, 1], [0, 2],
                                        [1, -2], [1, -1], [1, 0], [1, 1], [1, 2],
                                        [2, -2], [2, -1], [2, 0], [2, 1], [2, 2]])
        self.img_mean = torch.Tensor([0.485, 0.456, 0.406])
        self.img_std = torch.Tensor([0.229, 0.224, 0.225])
        self.proj_reduction = 'sum'
        
    ''' Efficient Projection '''
    def proj2img(self, pc):
        B, N, _ = pc.shape
        
        # calculate range
        pc_range = pc.max(dim=1)[0] - pc.min(dim=1)[0]  # B 3
        grid_size = pc_range[:, :2].max(dim=-1)[0] / (self.img_size - 3)  # B,

        # Point Index
        pc_min = pc.min(dim=1)[0][:, :2].unsqueeze(dim=1)
        grid_size = grid_size.unsqueeze(dim=1).unsqueeze(dim=2)
        idx_xy = torch.floor((pc[:, :, :2] - pc_min) / grid_size)  # B N 2
        # Point Densify
        idx_xy_dense = (idx_xy.unsqueeze(dim=2) + self.img_offset.unsqueeze(dim=0).unsqueeze(dim=0).to(pc.device)).view(idx_xy.size(0), N*25, 2) + 1
        # B N 1 2 + 1 1 25 2 -> B N 25 2 -> B 25N 2
        
        # Object to Image Center
        idx_xy_dense_center = torch.floor((idx_xy_dense.max(dim=1)[0] + idx_xy_dense.min(dim=1)[0]) / 2).int()
        offset_x = self.img_size / 2 - idx_xy_dense_center[:, 0: 1] - 1
        offset_y = self.img_size / 2 - idx_xy_dense_center[:, 1: 2] - 1
        idx_xy_dense_offset = idx_xy_dense + torch.cat([offset_x, offset_y], dim=1).unsqueeze(dim=1)

        f_dense = pc.unsqueeze(dim=2).expand(-1, -1, 25, -1).contiguous().view(pc.size(0), N * 25, 3)[..., 2: 3].repeat(1, 1, 3)
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
            #zero_pad = -10000*torch.ones(out.size(0), delta, out.size(2)).to(out.device)
            res = torch.cat([out, zero_pad], dim=1).reshape((out.size(0), self.img_size, self.img_size, out.size(2))) 
        else: 
            res = out.reshape((out.size(0), self.img_size, self.img_size, out.size(2))) 
        
        # B 224 224 C
        img = res.permute(0, 3, 1, 2).contiguous()
        zero_mask = (img == 0)
        mean_vec = self.img_mean.unsqueeze(dim=0).unsqueeze(dim=2).unsqueeze(dim=3).cuda()  # 1 3 1 1
        std_vec = self.img_std.unsqueeze(dim=0).unsqueeze(dim=2).unsqueeze(dim=3).cuda()   # 1 3 1 1    
        img = nn.Sigmoid()(img)
        img_norm = img.sub(mean_vec).div(std_vec) # 1 3 224 224
        img_norm = img_norm[:,0,:,:].unsqueeze(1) # 1 1 224 224
        img_norm = self.get_colored_depth_maps(img_norm,self.img_size,self.img_size)
        img_norm = torch.tensor(img_norm).permute(0,3,1,2).to(device)
        img_norm[zero_mask] = -1
        return img_norm, pc_min, grid_size, (offset_x, offset_y)

    ''' Image-to-Point Back-projection '''
    def I2P(self, pc, f, pc_min, grid_size, offsets):
        B, N, _ = pc.shape

        # Point Index
        idx_xy = torch.floor((pc[:, :, :2] - pc_min) / grid_size)  # B N 2
        
        # Point Densify
        idx_xy_dense = idx_xy + 1 # 5,64,2
        # B N 1 2 + 1 1 9 2 -> B N 9 2 -> B 9N 2
        
        # Object to Image Center
        idx_xy_dense_offset = idx_xy_dense + torch.cat(offsets, dim=1).unsqueeze(dim=1)  # B, N, 2

        # Expand Point Features
        B, C, H, W = f.shape
        f_dense = F.interpolate(f, size=(self.img_size, self.img_size), mode='bicubic').reshape(B, C, -1).permute(0, 2, 1)  # B, N, C
        # B N 9 C -> B 9N C
        assert idx_xy_dense_offset.min() >= 0 and idx_xy_dense_offset.max() <= (self.img_size - 1), str(idx_xy_dense_offset.min()) + '-' + str(idx_xy_dense_offset.max())
        
        # change idx to 1-dim
        new_idx_xy_dense = idx_xy_dense_offset[:, :, 0] * self.img_size + idx_xy_dense_offset[:, :, 1]  # B, N
        
        # Get Image Features
        out = torch.gather(f_dense, 1, new_idx_xy_dense.to(dtype=torch.int64).unsqueeze(-1).repeat(1, 1, C))
        return out,f_dense    
    
    def plot_feats(self,image, lr, hr,fr):
        assert len(image.shape) == len(lr.shape) == len(hr.shape) == len(fr.shape) == 3
        [lr_feats_pca, hr_feats_pca,fr_feats_pca], _ = pca([lr.unsqueeze(0), hr.unsqueeze(0),fr.unsqueeze(0)])
        fig, ax = plt.subplots(1, 4, figsize=(20, 5))
        ax[0].imshow(image.permute(1, 2, 0).detach().cpu())
        ax[0].set_title("Image")
        ax[1].imshow(lr_feats_pca[0].permute(1, 2, 0).detach().cpu())
        ax[1].set_title("Original Features")
        ax[2].imshow(hr_feats_pca[0].permute(1, 2, 0).detach().cpu())
        ax[2].set_title("Upsampled Features")
        ax[3].imshow(fr_feats_pca[0].permute(1, 2, 0).detach().cpu())
        ax[3].set_title("Up-downsampled Features")
        remove_axes(ax)
        plt.savefig('image/feat_img0.png')
        print('save successfully!')
        plt.close(fig)  # 关闭图形，释放内存

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

    def forward(self, x, upsampler):
        batch_size = x.size(0)
        pts_1 = rotate_point_cloud_batch_torch(x, -math.pi/2, axis='z')
        pts_2 = torch.cat((pts_1[..., 2: 3], pts_1[..., 0: 2]), dim=-1)
        pts_3 = torch.cat((pts_1[..., 1: 3], pts_1[..., 0: 1]), dim=-1)
        imgs_1, pc_min_1, grid_size_1, offsets_1 = self.proj2img(pts_1)  # b, 3, 224, 224
        imgs_2, pc_min_2, grid_size_2, offsets_2 = self.proj2img(pts_2)  # b, 3, 224, 224
        imgs_3, pc_min_3, grid_size_3, offsets_3 = self.proj2img(pts_3)  # b, 3, 224, 224     
        
        with torch.no_grad():
            image_tensor = torch.cat((imgs_1, imgs_2, imgs_3), dim=0) # b, c, h, w
            # figure_show(image_tensor)
            img_feats = upsampler(image_tensor)
            lr_feats = upsampler.model(image_tensor)
            
            ''' 2D Visual Features '''
            B, _, _, _ = img_feats.shape
            img_feats_1 = img_feats[0: B // 3]
            img_feats_2 = img_feats[B // 3: B // 3 * 2]
            img_feats_3 = img_feats[B // 3 * 2: ]

            clip_feats_1,f_dense1 = self.I2P(pts_1, img_feats_1, pc_min_1, grid_size_1, offsets_1)
            clip_feats_1 = torch.nn.functional.normalize(clip_feats_1,dim=-1)
            f_dense1 = torch.nn.functional.normalize(f_dense1,dim=-1)
            clip_feats_2,f_dense2 = self.I2P(pts_2, img_feats_2, pc_min_2, grid_size_2, offsets_2)
            clip_feats_2 = torch.nn.functional.normalize(clip_feats_2,dim=-1)
            clip_feats_3,f_dense3 = self.I2P(pts_3, img_feats_3, pc_min_3, grid_size_3, offsets_3)
            clip_feats_3 = torch.nn.functional.normalize(clip_feats_3,dim=-1)
         
        f_dense1 = f_dense1.view(1*batch_size,224,224,384).permute(0,3,1,2)
        
        # self.plot_feats(image_tensor[0,:,:,:].to(device), lr_feats[0], img_feats[0],f_dense1[0])
        
        clip_feats = torch.cat((clip_feats_1,clip_feats_2,clip_feats_3),dim=-1)
        return clip_feats