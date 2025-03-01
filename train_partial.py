import argparse
import yaml
import os
import torch
from models.dataset_partial import Dataset, shape_to_device
from models.loss import LGLoss, GraphDeformLoss_Neural_Partial
from sklearn.neighbors import NearestNeighbors
from models.model import Deformer,Uni3FC
import numpy as np
from tqdm import tqdm
import torch.optim as optim
import matplotlib.pyplot as plt
import torch.nn.functional as F
from tensorboardX import SummaryWriter
from misc.utils import auto_WKS, farthest_point_sample, square_distance
from transformers import AutoModel, AutoImageProcessor, CLIPImageProcessor
from misc import switch_functions
from misc.correspondence_utils import get_s_t_neighbors,get_s_t_neighbors_new
from lib.deformation_graph_point import  DeformationGraph_geod

def save_off_file(filename, points):
    with open(filename, 'w') as f:
        f.write('OFF\n')
        f.write(f'{points.shape[0]} 0 0\n')  # 点数 面数 边数
        for point in points:
            f.write(f'{point[0]} {point[1]} {point[2]}\n')

def train_net(cfg):
    if torch.cuda.is_available() and cfg["misc"]["cuda"]:
        device = torch.device(f'cuda:{cfg["misc"]["device"]}')
    else:
        device = torch.device("cpu")
    base_path = os.path.dirname(__file__)
    op_cache_dir = os.path.join(base_path, cfg["dataset"]["cache_dir"])
    dataset_path_train = os.path.join(cfg["dataset"]["root_dataset"], cfg["dataset"]["root_train"])
    dataset_path_test = os.path.join(cfg["dataset"]["root_dataset"], cfg["dataset"]["root_test"])

    save_dir_name = cfg["expname"]
    model_save_path = os.path.join(base_path, f"ckpt/{save_dir_name}/ep" + "_{}.pth")
    if not os.path.exists(os.path.join(base_path, f"ckpt/{save_dir_name}/")):
        os.makedirs(os.path.join(base_path, f"ckpt/{save_dir_name}/"))


    train_writer = SummaryWriter(os.path.join(base_path, 'tensorboard', cfg["expname"]))
    # create dataset
    # standard structured (source <> target) vts dataset
    if cfg["dataset"]["type"] == "vts":
        train_dataset = Dataset(dataset_path_train, name=cfg["dataset"]["name"],use_cache=True, train=True)
        test_dataset = Dataset(dataset_path_test, name=cfg["dataset"]["name"], use_cache=True,train=False)    # else not implemented
    else:
        raise NotImplementedError("dataset not implemented!")

    # data loader
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=cfg["training"]["batch_size"], shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=cfg["training"]["batch_size"], shuffle=False)
    # point_backbone_dino = Uni3FC_DINO_proj_semantic().to(device)
    # # point_backbone_dino.eval() 
    # define model
    # 40
    point_backbone = Uni3FC(k=40).to(device)
    deformer = Deformer(k=cfg["loss"]["k_deform"]).to(device)
    params1 = point_backbone.parameters()
    params2 = deformer.parameters()
    all_params = list(params1) + list(params2)
    lr = float(cfg["optimizer"]["lr"])
    optimizer = torch.optim.Adam(all_params, lr=lr, betas=(cfg["optimizer"]["b1"], cfg["optimizer"]["b2"]))
    criterion = GraphDeformLoss_Neural_Partial(k_deform=cfg["loss"]["k_deform"],w_dist=cfg["loss"]["w_dist"],w_map=cfg["loss"]["w_map"],k_dist=cfg["loss"]["k_dist"],N_dist=cfg["loss"]["N_dist"],partial=cfg["loss"]["partial"],w_deform=cfg["loss"]["w_deform"],w_img = cfg["loss"]["w_img"],w_rank = cfg["loss"]["w_rank"],w_self_rec = cfg["loss"]["w_self_rec"],w_cd = cfg["loss"]["deform"]["w_cd"],w_arap = cfg["loss"]["deform"]["w_arap"],save_name = cfg["expname"]).to(device)
    
    use_norm = True
    upsampler = torch.hub.load("mhamilton723/FeatUp", 'dinov2', use_norm=use_norm).to(device)
    # Training loop
    print("start training")
    alpha_list = np.linspace(cfg["loss"]["min_alpha"], cfg["loss"]["max_alpha"]+1, cfg["training"]["epochs"])
    eval_best_loss = 1e10
    for epoch in range(1, cfg["training"]["epochs"] + 1):
        if epoch % cfg["optimizer"]["decay_iter"] == 0:
            lr *= cfg["optimizer"]["decay_factor"]
            print(f"Decaying learning rate, new one: {lr}")
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
        loss_sum= 0
        dist_loss_sum = 0
        deform_loss_sum = 0
        self_rec_loss_sum = 0

        iterations = 0
        alpha_i = alpha_list[epoch-1]
        point_backbone.train()
        deformer.train()
        for i, data in tqdm(enumerate(train_loader)):

            data = shape_to_device(data, device)
            # data = augment_batch(data,rot_x=0,rot_y=180,rot_z=0,
            #                      std=0.01,noise_clip=0.05,
            #                      scale_min=0.9,scale_max=1.1)
            verts1, verts2 = data["shape1"]["xyz"],data["shape2"]["xyz"]
            
            if cfg["with_dino"]:
                dino_feat1, dino_feat2 = data["shape1"]["feat"],data["shape2"]["feat"]
            else:
                dino_feat1, dino_feat2 = None,None
            
            feat1, cfeats1 = point_backbone(verts1.permute(0,2,1), dino_feat1,upsampler)
            feat2, cfeats2 = point_backbone(verts2.permute(0,2,1), dino_feat2,upsampler)
            
            loss,dist_loss,deform_loss,map_loss,self_rec_loss = criterion(feat1,feat2,data["shape1"]["dist"],data["shape2"]["dist"],verts1,verts2,alpha_i,deformer)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            # log
            iterations += 1
            loss_sum += loss
            dist_loss_sum += dist_loss
            deform_loss_sum += deform_loss
            self_rec_loss_sum += self_rec_loss
                       
            log_batch = (i + 1) % cfg["misc"]["log_interval"] == 0
            print(f"epoch:{epoch}, loss:{loss_sum/iterations}, dist_loss:{dist_loss_sum/iterations}, deform_loss:{deform_loss_sum/iterations}, self_rec_loss:{self_rec_loss_sum/iterations}")
            if log_batch:
                torch.save(point_backbone.state_dict(), model_save_path.format('train_best'))
                torch.save(deformer.state_dict(), model_save_path.format('deformer_train_best'))
        
        if train_writer is not None:
            train_writer.add_scalar('Train_Loss', (loss_sum/iterations).item(), epoch)
            train_writer.add_scalar('Dist_Loss', (dist_loss_sum/iterations).item(), epoch)
            train_writer.add_scalar('Deform_Loss', (deform_loss_sum/iterations).item(), epoch)
            train_writer.add_scalar('Self_Rec_Loss', (self_rec_loss_sum/iterations).item(), epoch)
            
        with torch.no_grad():
            eval_loss = 0  
            val_iters = 0
            point_backbone.eval()
            deformer.eval()
            for i, data in tqdm(enumerate(test_loader)):
                data = shape_to_device(data, device)
                optimizer.zero_grad()
                verts1, verts2 = data["shape1"]["xyz"],data["shape2"]["xyz"]
                if cfg["with_dino"]:
                    dino_feat1, dino_feat2 = data["shape1"]["feat"],data["shape2"]["feat"]
                else:
                    dino_feat1, dino_feat2 = None,None
                feat1, cfeats1 = point_backbone(verts1.permute(0,2,1), dino_feat1,upsampler)
                feat2, cfeats2 = point_backbone(verts2.permute(0,2,1), dino_feat2,upsampler)
                
                loss,dist_loss,deform_loss,map_loss,self_rec_loss = criterion(feat1,feat2,data["shape1"]["dist"],data["shape2"]["dist"],verts1,verts2,alpha_i,deformer)

                val_iters += 1
                eval_loss += loss

            print(f"epoch:{epoch}, val_loss:{eval_loss/val_iters}")

        if train_writer is not None:
            train_writer.add_scalar('Val_Loss', (eval_loss/val_iters).item(), epoch)

        # save model
        if (epoch + 1) % cfg["misc"]["checkpoint_interval"] == 0:
            torch.save(point_backbone.state_dict(), model_save_path.format(epoch))
            torch.save(deformer.state_dict(), model_save_path.format('deformer' + str(epoch)))
            
        if eval_loss <= eval_best_loss:
            eval_best_loss = eval_loss
            torch.save(point_backbone.state_dict(), model_save_path.format('val_best'))
            torch.save(deformer.state_dict(), model_save_path.format('deformer_val_best'))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Launch the training of DINO_LGAttn model.")
    parser.add_argument('--savedir', required=False, default="./data", help='root directory of the dataset')
    parser.add_argument("--config", type=str, default="scape_partial", help="Config file name")

    args = parser.parse_args()
    cfg = yaml.safe_load(open(f"./config/{args.config}.yaml", "r"))
    train_net(cfg)
