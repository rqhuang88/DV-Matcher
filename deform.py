import argparse
import yaml
import os
import torch
from models.dataset import testDataset, shape_to_device
from misc.utils import DQFMLoss, augment_batch, augment_batch_sym
from misc.utils_geod import compute_geodesic_distmat
from sklearn.neighbors import NearestNeighbors
from pathlib import Path
import numpy as np
from tqdm import tqdm
import torch.optim as optim
import matplotlib.pyplot as plt
import torch.nn.functional as F
from tensorboardX import SummaryWriter
import scipy
import scipy.io as sio
from transformers import AutoModel, AutoImageProcessor, CLIPImageProcessor
from models.model import Uni3FC,Deformer,Uni3FC_DINO_proj
from lib.deformation_graph_point import  DeformationGraph_geod
import potpourri3d as pp3d
import open3d as o3d

def knn_grad(x, y, k):
    # distance = torch.cdist(x.float(), y.float())
    distance = torch.cdist(x.float(), y.float())
    _, idx = distance.topk(k=k, dim=-1, largest=False)
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

def deformation_graph_node(verts1):
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

def rotation_6d_to_matrix(d6):
    a1, a2 = d6[..., :3], d6[..., 3:]
    b1 = F.normalize(a1, dim=-1)
    b2 = a2 - (b1 * a2).sum(-1, keepdim=True) * b1
    b2 = F.normalize(b2, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack((b1, b2, b3), dim=-2)

def knnsearch_t_grad(x, y,alpha=100):
    distance = torch.cdist(x.float(), y.float())
    Pi_xy = F.softmax(-alpha*distance, dim=-1)
    # print(Pi_xy.max(dim=2).values)
    return Pi_xy

def topk_pi(A):
    k = 10  # 例如，选择每个维度上的前10个最大值
    # 获取topk的值和索引
    topk_values, topk_indices = torch.topk(A, k, dim=-1)
    # 创建一个与A形状相同的tensor，初始化为0
    result = torch.zeros_like(A)
    # 使用scatter_将topk的值放回原位置
    result.scatter_(-1, topk_indices, topk_values)
    return result
    
def save_off_file(filename, points):
    with open(filename, 'w') as f:
        f.write('OFF\n')
        f.write(f'{points.shape[0]} 0 0\n')  # 点数 面数 边数
        for point in points:
            f.write(f'{point[0]} {point[1]} {point[2]}\n')
            
def knnsearch_t(x, y):
    # distance = torch.cdist(x.float(), y.float())
    distance = torch.cdist(x.float(), y.float(), compute_mode='donot_use_mm_for_euclid_dist')
    _, idx = distance.topk(k=1, dim=-1, largest=False)
    return idx+1

def search_t(A1, A2):
    T12 = knnsearch_t(A1, A2)
    # T21 = knnsearch_t(A2, A1)
    return T12

def euclidean_dist(x, y):
    """
    Args:
      x: pytorch Variable, with shape [m, d]
      y: pytorch Variable, with shape [n, d]
    Returns:
      dist: pytorch Variable, with shape [m, n]
    """
    bs, m, n = x.size(0), x.size(1), y.size(1)
    xx = torch.pow(x, 2).sum(2, keepdim=True).expand(bs, m, n)
    yy = torch.pow(y, 2).sum(2, keepdim=True).expand(bs, n, m).transpose(1, 2)
    dist = xx + yy - 2 * torch.bmm(x, y.transpose(1, 2))
    dist = dist.clamp(min=1e-12).sqrt() 
    return dist


# It is equal to Tij = knnsearch(j, i) in Matlab
def knnsearch(x, y, alpha):
    distance = euclidean_dist(x, y)
    output = F.softmax(-alpha*distance, dim=-1)
    # _, idx = distance.topk(k=k, dim=-1)
    return output


def convert_C(Phi1, Phi2, A1, A2, alpha):
    Phi1, Phi2 = Phi1[:,:, :50], Phi2[:,:, :50]
    D1 = torch.bmm(Phi1, A1)
    D2 = torch.bmm(Phi2, A2)
    T12 = knnsearch(D1, D2, alpha)
    T21 = knnsearch(D2, D1, alpha)
    C12_new = torch.bmm(torch.pinverse(Phi2), torch.bmm(T21, Phi1))
    C21_new = torch.bmm(torch.pinverse(Phi1), torch.bmm(T12, Phi2))

    return C12_new, C21_new

def eval_net(cfg):
    if torch.cuda.is_available() and cfg["misc"]["cuda"]:
        device = torch.device(f'cuda:{cfg["misc"]["device"]}')
    else:
        device = torch.device("cpu")
    
    if cfg["isPartial"]:
        save_path = 'result/deform_partial_' + cfg["partialexpname"]
        model_path = 'ckpt/' + cfg["partialexpname"]+ '/ep_deformer_val_best.pth' 
        pointbackbone_path = 'ckpt/' + cfg["partialexpname"]+ '/ep_val_best.pth' 
    else:
        save_path = 'result/deform_' + cfg["expname"]
        model_path = 'ckpt/' + cfg["expname"]+ '/ep_deformer_val_best.pth' 
        pointbackbone_path = 'ckpt/' + cfg["expname"]+ '/ep_val_best.pth' 
        
    deformer = Deformer(k=cfg["loss"]["k_deform"]).to(device)
    deformer.load_state_dict(torch.load(model_path, map_location=device)) 
    point_backbone = Uni3FC(k=40).to(device)
    #point_backbone = Uni3FC_wo_DINO(k=40).to(device)
    point_backbone.load_state_dict(torch.load(pointbackbone_path, map_location=device)) 
    point_backbone_dino = Uni3FC_DINO_proj().to(device)
    point_backbone_dino.eval() 
    point_backbone.eval()  
    deformer.eval()
    use_norm = True
    upsampler = torch.hub.load("mhamilton723/FeatUp", 'dinov2', use_norm=use_norm).to(device)
    
    shape1_pth = "data/scape_r/shapes_train/mesh000.off"
    shape2_pth = "data/scape_r/shapes_test/mesh053.off"
    name1 = "mesh000"
    name2 = "mesh053"         
    with torch.no_grad():
        point_backbone_dino.eval() 
        point_backbone.eval() 
        deformer.eval()
        if cfg["deform_mesh"]:
            verts1, faces1 = pp3d.read_mesh(shape1_pth) 
            verts2, faces2 = pp3d.read_mesh(shape2_pth) 
            verts1 = torch.tensor(verts1).to(device).unsqueeze(0).float()
            verts2 = torch.tensor(verts2).to(device).unsqueeze(0).float()
            faces1 = torch.tensor(faces1).to(device).unsqueeze(0)
            faces2 = torch.tensor(faces2).to(device).unsqueeze(0)
            dino_feat1 = point_backbone_dino(verts1.permute(0,2,1),upsampler)
            dino_feat2 = point_backbone_dino(verts2.permute(0,2,1),upsampler)

            feat1, cfeats1 = point_backbone(verts1.permute(0,2,1), dino_feat1,upsampler)
            feat2, cfeats2 = point_backbone(verts2.permute(0,2,1), dino_feat2,upsampler)
            
            # feat1, cfeats1 = point_backbone(verts1.permute(0,2,1))
            # feat2, cfeats2 = point_backbone(verts2.permute(0,2,1))
            dg = DeformationGraph_geod()
            geod = compute_geodesic_distmat(verts1.squeeze(0).cpu().numpy(),faces1.squeeze(0).cpu().numpy())
            dg.construct_graph(verts1.squeeze(0).cpu(),faces1.squeeze(0).cpu(),geod,device)
            num_nodes_all1 = torch.tensor(dg.nodes_idx).to(device).unsqueeze(0)
            
            Pi_12 = knnsearch_t_grad(feat1, feat2, alpha=100) # 2,4995,4995 
            Pi_12 = topk_pi(Pi_12)
            
            idx11 = knn_grad(verts1,verts1,cfg["loss"]["k_deform"])
            idx22 = knn_grad(verts2,verts2,cfg["loss"]["k_deform"])
            
            feat2_conv = index_points(feat2,idx22)
            feat1_conv = index_points(feat1,idx11)
            
            verts12 = torch.matmul(Pi_12,verts2) 
            deformations = deformer(feat1_conv,feat2_conv,verts1,verts12,Pi_12,num_nodes_all1.long())
            rotations = deformations[:,:,3:]
            iden = torch.from_numpy(np.array([1,0,0,0,1,0]).astype(np.float32)).view(1,6).repeat(1,rotations.shape[1],1)
            if rotations.is_cuda:
                iden = iden.cuda()
            rotations = rotations + iden
            T1 = deformations[:,:,:3]
            R1 = rotation_6d_to_matrix(rotations)
            
            deformed_pt1, arap, sr_loss= dg(verts1[0], R1, T1)
            pcd = deformed_pt1[0].detach().cpu().squeeze().numpy()
            
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            filename_deform12 = f'{save_path}/deform_{name1}_{name2}.off'
            
            save_mesh = o3d.geometry.TriangleMesh()
            save_mesh.vertices =  o3d.utility.Vector3dVector(pcd)
            save_mesh.triangles = o3d.utility.Vector3iVector(faces1[0].detach().cpu().squeeze().numpy())
            o3d.io.write_triangle_mesh(filename_deform12, save_mesh)   
            
            # save_off_file(filename_deform12, PC1)
        else:
            verts1, faces1 = pp3d.read_mesh(shape1_pth) 
            verts2, faces2 = pp3d.read_mesh(shape2_pth) 
            verts1 = torch.tensor(verts1).to(device).unsqueeze(0).float()
            verts2 = torch.tensor(verts2).to(device).unsqueeze(0).float()
            faces1 = torch.tensor(faces1).to(device).unsqueeze(0)
            faces2 = torch.tensor(faces2).to(device).unsqueeze(0)
            dino_feat1 = point_backbone_dino(verts1.permute(0,2,1),upsampler)
            dino_feat2 = point_backbone_dino(verts2.permute(0,2,1),upsampler)
                
            feat1, cfeats1 = point_backbone(verts1.permute(0,2,1), dino_feat1,upsampler)
            feat2, cfeats2 = point_backbone(verts2.permute(0,2,1), dino_feat2,upsampler)
            
            num_nodes_all1,dg_list1 = deformation_graph_node(verts1)
            Pi_12 = knnsearch_t_grad(feat1, feat2, alpha=100) # 2,4995,4995 
            Pi_12 = topk_pi(Pi_12)
            
            idx11 = knn_grad(verts1,verts1,cfg["loss"]["k_deform"])
            idx22 = knn_grad(verts2,verts2,cfg["loss"]["k_deform"])
            
            feat2_conv = index_points(feat2,idx22)
            feat1_conv = index_points(feat1,idx11)
            
            verts12 = torch.matmul(Pi_12,verts2) 
            deformations = deformer(feat1_conv,feat2_conv,verts1,verts12,Pi_12,num_nodes_all1.long())
            rotations = deformations[:,:,3:]
            iden = torch.from_numpy(np.array([1,0,0,0,1,0]).astype(np.float32)).view(1,6).repeat(1,rotations.shape[1],1)
            if rotations.is_cuda:
                iden = iden.cuda()
            rotations = rotations + iden
            T1 = deformations[:,:,:3]
            R1 = rotation_6d_to_matrix(rotations)
            
            i = 0
            deformed_points1 = torch.tensor([]).to(verts1.device)
            for dg in dg_list1:
                deformed_pt1, arap, sr_loss= dg(verts1[i],R1[i].unsqueeze(0), T1[i].unsqueeze(0))
                i = i + 1
                deformed_points1 = torch.cat([deformed_points1,deformed_pt1],dim=0)
            PC1 = deformed_points1[0].detach().cpu().squeeze().numpy()
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            filename_deform12 = f'{save_path}/deform_{name1}_{name2}.off'
            save_off_file(filename_deform12, PC1)




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Launch the training of LGAttention model.")
    parser.add_argument('--savedir', required=False, default="./data", help='root directory of the dataset')
    parser.add_argument("--config", type=str, default="deform", help="Config file name")

    args = parser.parse_args()
    cfg = yaml.safe_load(open(f"./config/{args.config}.yaml", "r"))
    eval_net(cfg)
