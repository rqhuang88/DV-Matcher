import torch
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

def compute_graph(neighborhood):
    B, M, k, _ = neighborhood.shape
    # 计算均值和协方差矩阵
    mu = torch.mean(neighborhood, dim=2)
    sigma = torch.stack([torch.cov(neigh.T) for neigh in neighborhood.view(B * M, k, 3)]).view(B, M, 3, 3)
    
    return mu, sigma

def compute_neighborhood_statistics(point_cloud, k):
    """
    计算点云中给定点的邻域的均值和协方差矩阵
    :param point_cloud: 点云数据，形状为 (B, N, 3)
    :param point_indices: 目标点的索引列表，形状为 (B, M)
    :param k: 邻域点的数量
    :return: 均值向量和协方差矩阵
    """
    device = point_cloud.device
    B, N, _ = point_cloud.shape  # B: batch size, N: number of points per batch
    
    # 计算所有点的距离矩阵
    distances = torch.cdist(point_cloud.float(), point_cloud.float())
    dist, indices = torch.topk(distances, k+1, largest=False, dim=2)  # 包括自身, (B, N, K)
    # 获取邻域点
    # weight = F.softmax(dist, dim=2)
    neighborhood = index_points(point_cloud,indices[:,:,1:])
    neighborhood = neighborhood - point_cloud.unsqueeze(2).expand(-1,-1,k,-1)
    # neighborhood = neighborhood_dir * weight.unsqueeze(dim=3)
    # 计算均值和协方差矩阵
    mu = torch.mean(neighborhood, dim=2)
    sigma = torch.stack([torch.cov(neigh.T) for neigh in neighborhood.view(B * N, k, 3)]).view(B, N, 3, 3)
    return mu, sigma

def compute_neighborhood(pointcloud, pointtarget, k):
    distances = torch.cdist(pointcloud.float(), pointcloud.float())
    dist, indices = torch.topk(distances, k, largest=False, dim=2)  # 包括自身, (B, N, K)
    pointcloud_neighborhood = index_points(pointcloud,indices)
    pointtarget_neighborhood = index_points(pointtarget,indices)
    return pointcloud_neighborhood, pointtarget_neighborhood


def kl_divergence(mean1, cov1, mean2, cov2):
    k = mean1.size(-1)
    
    # 计算协方差矩阵的逆
    cov2_inv = torch.inverse(cov2)
    
    # 计算 tr(Σ2^-1 * Σ1)
    # B,N,3,3
    tr_cov2_inv_cov1 = torch.einsum('...ij,...ij->...', cov2_inv, cov1)
    
    # 计算 (μ2 - μ1)^T * Σ2^-1 * (μ2 - μ1)
    diff = mean2 - mean1
    quad_form = torch.einsum('...i,...ij,...j->...', diff, cov2_inv, diff)
    
    # 计算 log(det(Σ2) / det(Σ1))
    log_det_cov2 = torch.logdet(cov2)
    log_det_cov1 = torch.logdet(cov1)
    log_det_ratio = log_det_cov2 - log_det_cov1
    
    # 计算 KL 散度
    kl = 0.5 * (tr_cov2_inv_cov1 + quad_form - k + log_det_ratio)
    
    return kl

def gaussian_kl_divergence(mu_p, sigma_p, mu_q, sigma_q):
    """
    计算两个高斯分布之间的KL散度
    :param mu_p: 点 p 的均值向量
    :param sigma_p: 点 p 的协方差矩阵
    :param mu_q: 点 q 的均值向量
    :param sigma_q: 点 q 的协方差矩阵
    :return: KL散度值
    """
    k = mu_p.shape[-1]  # 维度数
    mu_q = mu_q.unsqueeze(1).expand(-1, mu_p.shape[1], -1, -1)
    sigma_q = sigma_q.unsqueeze(1).expand(-1, mu_p.shape[1], -1, -1,-1)
    sigma_q_inv = torch.inverse(sigma_q)
    
    #torch.Size([6, 4995(P点数), *4995**（Q点数）, 3, 3])
    term1 = torch.einsum('blijj->bli', torch.matmul(sigma_q_inv, sigma_p))
    term2 = torch.einsum('blij,blijk,blik->bli', mu_q - mu_p, sigma_q_inv, mu_q - mu_p)
    term3 = -k
    term4 = torch.logdet(sigma_q) - torch.logdet(sigma_p)
    kl_divergence = 0.5 * (term1 + term2 + term3 + term4)

    return kl_divergence

def compute_kl_divergences(mu_p, sigma_p, point_cloud_Q, k, device):
    """
    计算点 p 到点云 Q 中每个点的KL散度
    :param mu_p: 点 p 的均值向量
    :param sigma_p: 点 p 的协方差矩阵
    :param point_cloud_Q: 点云 Q 的数据，形状为 (B, M, 3)
    :param k: 邻域点的数量
    :return: KL散度列表
    """
    B, M, _ = point_cloud_Q.shape  # B: batch size, M: number of points per batch
    
    point_cloud_Q = point_cloud_Q.to(device)
    mu_q, sigma_q = compute_neighborhood_statistics(point_cloud_Q, torch.arange(M)[None, :].expand(B, M), k, device) # torch.Size([6, 4995, 3]),torch.Size([6, 4995, 3, 3])
    
    # 扩展 mu_p 和 sigma_p 以匹配 mu_q 和 sigma_q 的形状
    mu_p_expanded = mu_p.unsqueeze(2).expand(-1, -1, M, -1) # torch.Size([6, 4995, **4995**, 3])
    sigma_p_expanded = sigma_p.unsqueeze(2).expand(-1, -1, M, -1,-1) #torch.Size([6, 4995, *4995**, 3, 3])
    
    # 计算 KL 散度
    kl_divergences = gaussian_kl_divergence(mu_p_expanded, sigma_p_expanded, mu_q, sigma_q)
    return kl_divergences