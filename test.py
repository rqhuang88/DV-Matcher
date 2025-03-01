import argparse
import yaml
import os
import torch
from models.dataset import testDataset, shape_to_device
from misc.utils import DQFMLoss, augment_batch, augment_batch_sym
from sklearn.neighbors import NearestNeighbors
import numpy as np
from tqdm import tqdm
import torch.optim as optim
import matplotlib.pyplot as plt
import torch.nn.functional as F
from tensorboardX import SummaryWriter
import scipy
import scipy.io as sio
from transformers import AutoModel, AutoImageProcessor, CLIPImageProcessor
from models.model import Uni3FC

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

    # important paths
    base_path = os.path.dirname(__file__)
    op_cache_dir = os.path.join(base_path, cfg["dataset"]["cache_dir"])
    dataset_path_test = os.path.join(cfg["dataset"]["root_dataset"], cfg["dataset"]["root_test"])
    
    test_dataset = testDataset(dataset_path_test, name=cfg["dataset"]["name"], use_cache=True,train=False,with_dino=cfg["with_dino"],feat_mat=cfg["feat_mat"])
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)
    
    # define model
    model_path = 'ckpt/' + cfg["expname"]+ '/ep_val_best.pth'
    point_backbone = Uni3FC(k=40).to(device)
    # point_backbone = Uni3FC_wo_DINO(k=40).to(device)
    point_backbone.load_state_dict(torch.load(model_path, map_location=device)) 
    point_backbone.eval()   
    use_norm = True
    upsampler = torch.hub.load("mhamilton723/FeatUp", 'dinov2', use_norm=use_norm).to(device)
    
    if cfg["cache"]:
        save_path = 'result/' + cfg["expname"] + '_' + cfg["dataset"]["name"] + '_cache'
    else:      
        save_path = 'result/' + cfg["expname"] + '_' + cfg["dataset"]["name"]
    with torch.no_grad():
        point_backbone.eval()
        for i, data in tqdm(enumerate(test_loader)):
            data = shape_to_device(data, device)
            verts1, verts2 = data["shape1"]["xyz"],data["shape2"]["xyz"]
            if cfg["with_dino"]:
                dino_feat1, dino_feat2 = data["shape1"]["feat"],data["shape2"]["feat"]
            else:
                dino_feat1, dino_feat2 = None,None
            
            feat1, cfeats1 = point_backbone(verts1.permute(0,2,1), dino_feat1,upsampler)
            feat2, cfeats2 = point_backbone(verts2.permute(0,2,1), dino_feat2,upsampler)
            # feat1, cfeats1 = point_backbone(verts1.permute(0,2,1))
            # feat2, cfeats2 = point_backbone(verts2.permute(0,2,1))
            name1 = data["shape1"]["name"][0]
            name2 = data["shape2"]["name"][0]

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

            save_path_t = save_path + '/feature/'
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
    parser.add_argument("--config", type=str, default="scape_r", help="Config file name")

    args = parser.parse_args()
    cfg = yaml.safe_load(open(f"./config/{args.config}.yaml", "r"))
    eval_net(cfg)
