import argparse
import yaml
import os
import torch
from models.dataset_partial import testDataset, shape_to_device
from sklearn.neighbors import NearestNeighbors
import numpy as np
from tqdm import tqdm
import torch.optim as optim
import matplotlib.pyplot as plt
import torch.nn.functional as F
from tensorboardX import SummaryWriter
import random
from misc.utils import auto_WKS, farthest_point_sample, square_distance
from transformers import AutoModel, AutoImageProcessor, CLIPImageProcessor
from models.model import Uni3FC
from misc import switch_functions
import scipy
from misc.correspondence_utils import get_s_t_neighbors
# from torch_cluster import knn_new

def knn(x, k):
    inner = -2 * torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x ** 2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)
    idx = pairwise_distance.topk(k=k, dim=-1)[1]  # (batch_size, num_points, k)
    return idx

def get_graph_feature(x, k, idx=None, only_intrinsic=False, permute_feature=True):
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        idx = knn(x, k=k)  # (batch_size, num_points, k)
    else:
        if(len(idx.shape)==2):
            idx = idx.unsqueeze(0).repeat(batch_size,1,1)
        idx = idx[:, :, :k]
        k = min(k,idx.shape[-1])

    num_idx = idx.shape[1]

    idx_base = torch.arange(0, batch_size, device=idx.device).view(-1, 1, 1) * num_points

    idx = idx + idx_base

    idx = idx.contiguous()
    idx = idx.view(-1)

    _, num_dims, _ = x.size()

    x = x.transpose(
        2, 1
    ).contiguous()  # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
    feature = x.view(batch_size * num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_idx, k, num_dims)
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)

    if only_intrinsic is True:
        feature = feature - x
    elif only_intrinsic == 'neighs':
        feature = feature
    elif only_intrinsic == 'concat':
        feature = torch.cat((feature, x), dim=3)
    else:
        feature = torch.cat((feature - x, x), dim=3)

    if permute_feature:
        feature = feature.permute(0, 3, 1, 2).contiguous()

    return feature

def reconstruction(pos, nn_idx, nn_weight, k):
    nn_pos = get_graph_feature(pos.transpose(1, 2), k=k, idx=nn_idx, only_intrinsic='neighs', permute_feature=False)
    nn_weighted = nn_pos * nn_weight.unsqueeze(dim=3)
    recon = torch.sum(nn_weighted, dim=2)

    recon_hard = nn_pos[:, :, 0, :]

    return recon, recon_hard

def forward_source_target(feat_source, feat_target,vert_source,vert_target):
    # measure cross similarity
    P_non_normalized = switch_functions.measure_similarity("cosine", feat_source, feat_target)
    
    temperature = None
    P_normalized = P_non_normalized

    # cross nearest neighbors and weights
    source_cross_nn_weight, source_cross_nn_sim, source_cross_nn_idx, target_cross_nn_weight, target_cross_nn_sim, target_cross_nn_idx = get_s_t_neighbors(40, P_normalized, sim_normalization="softmax")

    # cross reconstruction
    source_cross_recon, source_cross_recon_hard = reconstruction(vert_source, target_cross_nn_idx, target_cross_nn_weight, 40)
    target_cross_recon, target_cross_recon_hard = reconstruction(vert_target, source_cross_nn_idx, source_cross_nn_weight, 40)

    return source_cross_recon, target_cross_recon

def forward_shape(feat,verts):
    P_self = switch_functions.measure_similarity("cosine", feat, feat)

    # measure self similarity
    nn_idx = None
    self_nn_weight, _, self_nn_idx, _, _, _ = get_s_t_neighbors(41, P_self, sim_normalization="softmax", s_only=True, ignore_first=True,nn_idx=nn_idx)

    # self reconstruction
    recon, _ = reconstruction(verts, self_nn_idx, self_nn_weight, 40)

    return recon

def knn_dual(a, b, k):
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

def cross_construct(x, y, verts2, k_num):
    distance = torch.cdist(x.float(), y.float(), compute_mode='donot_use_mm_for_euclid_dist')            
    _, idx = distance.topk(k=k_num, dim=-1, largest=False)
    verts2_corr = index_points(verts2, idx).squeeze(2) #6*4995*10*3
    feat2_corr = index_points(y.float(), idx).squeeze(2) #6*4995*10*128
    feat1 = x.float().unsqueeze(2).repeat(1, 1, k_num, 1) #6*4995*10*128
    similarity = F.cosine_similarity(feat1, feat2_corr, dim=3)
    weight = F.softmax(similarity, dim=2).unsqueeze(-1).repeat(1, 1, 1, 3) #6*4995*10
    # print(weight[1,1,:])
    verts2_corr = torch.sum(verts2_corr * weight,dim=2)
    return verts2_corr

def partial_ponit(verts):
    device = 'cuda:0'
    batch_size = verts.shape[0]
    idx = torch.tensor([],device=device)
    num = random.randint(0, 5)
    for i in range(batch_size):
        if num == 0:
            idx_partial = torch.nonzero(verts[i,:,0] > 0).squeeze() 
        elif num == 1:
            idx_partial = torch.nonzero(verts[i,:, 1] > 0).squeeze() 
        elif num == 2:
            idx_partial = torch.nonzero(verts[i,:, 2] > 0).squeeze() 
        elif num == 3:
            idx_partial = torch.nonzero(verts[i,:, 0] < 0).squeeze() 
        elif num == 4:
            idx_partial = torch.nonzero(verts[i, :,1] < 0).squeeze()   
        else:
            idx_partial = torch.nonzero(verts[i, :,2] < 0).squeeze()
        verts_pc = verts[i,idx_partial,:]
        fps = farthest_point_sample(verts_pc,verts_pc.shape[0]).squeeze()
        idx_partial_new = idx_partial[fps[:100]]
        idx = torch.cat([idx,idx_partial_new.unsqueeze(0)],dim=0)
    return idx

def knnsearch_t(x, y):
    # distance = torch.cdist(x.float(), y.float())
    distance = torch.cdist(x.float(), y.float(), compute_mode='donot_use_mm_for_euclid_dist')
    _, idx = distance.topk(k=1, dim=-1, largest=False)
    return idx

def search_t(A1, A2):
    T12 = knnsearch_t(A1, A2)
    # T21 = knnsearch_t(A2, A1)
    return T12 + 1

def read_file(file_path):
    lines = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            line = line.strip()  # 去除行末尾的换行符和空格
            try:
                number = int(line)  # 将行内容转换为整数
                lines.append(number)
            except ValueError:
                print(f"Invalid line: {line}")  # 非数字行则打印错误提示
    return lines

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
    base_path = os.path.dirname(__file__)
    dataset_path_test = os.path.join(cfg["dataset"]["root_dataset"], cfg["dataset"]["root_test"])

    save_dir_name = cfg["expname"]
    if not os.path.exists(os.path.join(base_path, f"ckpt/{save_dir_name}/")):
        os.makedirs(os.path.join(base_path, f"ckpt/{save_dir_name}/"))

    # standard structured (source <> target) vts dataset
    if cfg["dataset"]["type"] == "vts":
        test_dataset = testDataset(dataset_path_test, name=cfg["dataset"]["name"], use_cache=True,train=False)
    # else not implemented
    else:
        raise NotImplementedError("dataset not implemented!")
    
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)

    # define model
    model_path = 'ckpt/' + cfg["expname"]+ '/ep_val_best.pth'
    point_backbone = Uni3FC(k=40).to(device)
    point_backbone.load_state_dict(torch.load(model_path, map_location=device)) 
    point_backbone.eval()   
    use_norm = True
    upsampler = torch.hub.load("mhamilton723/FeatUp", 'dinov2', use_norm=use_norm).to(device)
    
    
    save_path = 'result/' + cfg["expname"] + '_' + cfg["dataset"]["name"]
    with torch.no_grad():
        point_backbone.eval()
        for i, data in tqdm(enumerate(test_loader)):
            data = shape_to_device(data, device)
            verts1, verts2 = data["shape1"]["xyz"],data["shape2"]["xyz"]
            name1,name2 = data["shape1"]["name"][0],data["shape2"]["name"][0]
            print(name1)
            print(name2)
            feat1, cfeats1 = point_backbone(verts1.permute(0,2,1), None,upsampler)
            feat2, cfeats2 = point_backbone(verts2.permute(0,2,1), None,upsampler)

            T12_pred,T21_pred= search_t(feat1, feat2), search_t(feat2, feat1)
            #save
            save_path_t = os.path.join(save_path, f'T')
            
            if not os.path.exists(save_path_t):
                os.makedirs(save_path_t)
            filename_t12 = f'T_{name1}_{name2}.txt'
            t12 = T12_pred.detach().cpu().squeeze(0).numpy()
            np.savetxt(os.path.join(save_path_t, filename_t12), t12, fmt='%i')
            filename_t21 = f'T_{name2}_{name1}.txt'
            t21 = T21_pred.detach().cpu().squeeze(0).numpy()
            np.savetxt(os.path.join(save_path_t, filename_t21), t21, fmt='%i')

            save_path_t = os.path.join(save_path, f'feature')
            if not os.path.exists(save_path_t):
                os.makedirs(save_path_t)
            filename_index = f'usefeature_{name1}.mat'
            u_feat1 = feat1.detach().cpu().squeeze(0).numpy()
            u_feat1 = {'uphi': u_feat1}
            scipy.io.savemat(os.path.join(save_path_t, filename_index), u_feat1)
            filename_index = f'usefeature_{name2}.mat'
            u_feat2 = feat2.detach().cpu().squeeze(0).numpy()
            u_feat2 = {'uphi': u_feat2}
            scipy.io.savemat(os.path.join(save_path_t, filename_index), u_feat2)    




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Launch the training of LGAttention model.")
    parser.add_argument('--savedir', required=False, default="./data", help='root directory of the dataset')
    parser.add_argument("--config", type=str, default="scape_partial", help="Config file name")

    args = parser.parse_args()
    cfg = yaml.safe_load(open(f"./config/{args.config}.yaml", "r"))
    eval_net(cfg)