import os
from pathlib import Path
import numpy as np
import potpourri3d as pp3d
import torch
from torch.utils.data import Dataset
from misc.utils import auto_WKS, farthest_point_sample, square_distance
from models.model import Uni3FC_DINO_proj
from tqdm import tqdm
from itertools import permutations
import Tools.mesh as qm
from Tools.utils import op_cpl
import scipy.io as sio
import numpy as np

def load_off_point_cloud(file_path):
    points = []
    with open(file_path, 'r') as f:
        # 跳过文件头部
        f.readline()
        # 读取点的数量和面的数量
        num_points, num_faces, _ = map(int, f.readline().strip().split())
        # 读取点的坐标
        for _ in range(num_points):
            x, y, z = map(float, f.readline().strip().split())
            points.append([x, y, z])
    return points

def find_max_distances(evals_list, k):
    index_set = set()
    
    for i in range(len(evals_list)):
        distances = []
        
        for j in range(len(evals_list)):
            if i != j:
                distance = np.linalg.norm(np.array(evals_list[i]) - np.array(evals_list[j]))
                distances.append((j, distance))
        
        distances.sort(key=lambda x: x[1], reverse=True)
        max_distances = distances[:k]
        
        for index, _ in max_distances:
            if (i, index) not in index_set and (index, i) not in index_set:
                index_set.add((i, index))
    
    return list(index_set)

def cal_geo(V):
    dist = torch.tensor([])
    solver = pp3d.PointCloudHeatSolver(V)
    for i in range(V.shape[0]):
        dist = torch.cat([dist,torch.tensor(solver.compute_distance(i)).unsqueeze(1)],dim=-1)
    return dist

class Dataset(Dataset):
    """
    Implementation of shape matching Dataset !WITH VTS! correspondence files (to compute ground-truth).
    This type of dataset is loaded if config['type'] = 'vts'
    It is called Scape Dataset because historically, SCAPE was re-meshed with vts files to track correspondence
    to the original SCAPE dataset. Any dataset using vts (re-meshed as in
    [Continuous and orientation-preserving correspondences via functional maps, Ren et al. 2018 TOG])
    falls into this category and can therefore be utilized via this class.

    ---Parameters:
    @ root_dir: root folder containing shapes_train and shapes_test folder
    @ name: name of the dataset. ex: scape-remeshed, or scape-anisotropic
    @ k_eig: number of Laplace-Beltrami eigenvectors loaded
    @ n_fmap: number of eigenvectors used for fmap computation
    @ n_cfmap: number of complex eigenvectors used for complex fmap computation
    @ with_wks: None if no WKS (C_in <= 3), else the number of WKS descriptors
    @ use_cache: cache for storing dataset (True by default)
    @ op_cache_dir: cache for diffusion net operators (from config['dataset']['cache_dir'])
    @ train: for train or test set

    ---At initialisation, loads:
    1) verts, faces and vts
    2) geometric operators (Laplacian, Gradient)
    3) (optional if C_in = 3) WKS descriptors (for best setting)
    4) (optional if n_cfmap = 0) complex operators (for orientation-aware unsupervised learning)

    ---When delivering an element of the dataset, yields a dictionary with:
    1) shape1 containing all necessary info for source shape
    2) shape2 containing all necessary info for target shape
    3) ground-truth functional map Cgt (obtained with vts files)
    """

    def __init__(self, root_dir, name="scape-remeshed",
                 k_eig=128, n_fmap=30, n_cfmap=20,
                 with_wks=None, with_sym=False,
                 use_cache=True, op_cache_dir=None,
                 train=True,with_dino=False,feat_mat=False):

        self.with_dino = with_dino
        self.feat_mat = feat_mat
        self.k_eig = k_eig
        self.n_fmap = n_fmap
        self.n_cfmap = n_cfmap
        self.root_dir = root_dir
        self.cache_dir = root_dir
        self.op_cache_dir = op_cache_dir
        self.with_sym = with_sym
        self.device = 'cuda:0'
        self.name = name
        # check the cache
        split = "train" if train else "test"
        wks_suf = "" if with_wks is None else "wks_"
        sym_suf = "" if not with_sym else "sym_"
        if use_cache:
            load_cache = os.path.join(self.cache_dir, f"cache_{name}_{sym_suf}{wks_suf}{split}.pt")
            print("using dataset cache path: " + str(load_cache))
            if os.path.exists(load_cache):
                print("  --> loading dataset from cache")
                (
                    # main
                    self.verts_list,
                    self.used_shapes,
                    self.fps_list,
                    self.dist_list
                ) = torch.load(load_cache)
                if self.name == 'amass_ssft':
                    animal_indices = [index for index, item in enumerate(self.used_shapes) if any(animal in item for animal in ['cat', 'centaur', 'dog', 'gorilla', 'horse'])]
                    animal_combinations = list(permutations(animal_indices, 2))
                    non_animal_indices = [index for index, item in enumerate(self.used_shapes) if not any(animal in item for animal in ['cat', 'centaur', 'dog', 'gorilla', 'horse'])]
                    non_animal_combinations = list(permutations(non_animal_indices, 2))
                    self.combinations = animal_combinations + non_animal_combinations
                else:
                    self.combinations = list(permutations(range(len(self.verts_list)), 2))
                    return
            print("  --> dataset not in cache, repopulating")

        # Load the meshes
        # define files and order
        shapes_split = "shapes_" + split
        self.used_shapes = sorted([x.stem for x in (Path(root_dir) / shapes_split).iterdir() if 'DS_' not in x.stem])
        if self.name == 'amass_ssft':
            animal_indices = [index for index, item in enumerate(self.used_shapes) if any(animal in item for animal in ['cat', 'centaur', 'dog', 'gorilla', 'horse'])]
            animal_combinations = list(permutations(animal_indices, 2))
            non_animal_indices = [index for index, item in enumerate(self.used_shapes) if not any(animal in item for animal in ['cat', 'centaur', 'dog', 'gorilla', 'horse'])]
            non_animal_combinations = list(permutations(non_animal_indices, 2))
            self.combinations = animal_combinations + non_animal_combinations
        else:
            self.combinations = list(permutations(range(len(self.used_shapes)), 2))

        mesh_dirpath = Path(root_dir) / shapes_split
        vts_dirpath = Path(root_dir) / "corres"

        # Get all the files
        ext = '.off'
        self.verts_list = []
        self.fps_list = []
        self.vts_list = []
        self.dist_list = []
        if with_sym:
            vts_sym_list = []

        
        if self.with_dino:
            point_backbone = Uni3FC_DINO_proj().to(self.device)
            point_backbone.eval() 
            use_norm = True
            upsampler = torch.hub.load("mhamilton723/FeatUp", 'dinov2', use_norm=use_norm).to(self.device)
            for shape_name in tqdm(self.used_shapes):
                if self.name == 'spleen' or self.name == 'spleen_test' or self.name == 'step2_posed_templates_5k_off' or self.name == 'clothes_align_5k_off' or self.name == 'clothes_heavy_off' or ('clothes_shape' in self.name):
                    verts = load_off_point_cloud(str(mesh_dirpath / f"{shape_name}{ext}"))
                    verts = torch.tensor(verts)
                else:
                    verts, faces = pp3d.read_mesh(str(mesh_dirpath / f"{shape_name}{ext}"))  # off obj
                #verts = pp3d.read_point_cloud(str(mesh_dirpath / f"{shape_name}{ext}"))  # off obj
                print('Cal Geo..')
                dist = cal_geo(verts)
                self.dist_list.append(dist)
                verts = torch.tensor(np.ascontiguousarray(verts)).float()
                fps = farthest_point_sample(verts,verts.shape[0]).squeeze()
                if self.name == 'spleen' or self.name == 'spleen_test' :
                    fps = fps[:1024]
                elif self.name == 'fourleg':
                    fps = fps[:5000]
                else:                   
                    fps = fps[:4995]
                self.verts_list.append(verts)
                self.fps_list.append(fps)
                vts = verts.unsqueeze(0).to(self.device)
                feat = point_backbone(vts.permute(0,2,1),upsampler).squeeze(0).cpu().numpy()
                ## 保存.mat文件
                if self.feat_mat:
                    save_path = Path(root_dir) / "feat"
                    if not os.path.exists(save_path):
                        os.makedirs(os.path.join(save_path))
                    feat = {'feat': feat}
                    sio.savemat(self.root_dir + "/feat/" + shape_name + ".mat", feat)
                ## append 
                else:
                    self.feat_list.append(feat)
        else: 
            for shape_name in tqdm(self.used_shapes):
                if self.name == 'spleen' or self.name == 'spleen_test' or self.name == 'se-ornet-tosca' or self.name == 'step2_posed_templates_5k_off' or self.name == 'clothes_align_5k_off' or self.name == 'clothes_heavy_off' or ('clothes_shape' in self.name):
                    verts = load_off_point_cloud(str(mesh_dirpath / f"{shape_name}{ext}"))
                    verts = torch.tensor(verts)
                else:
                    verts, faces = pp3d.read_mesh(str(mesh_dirpath / f"{shape_name}{ext}"))  # off obj
                print('Cal Geo..')
                dist = cal_geo(verts)
                self.dist_list.append(dist)
                
                verts = torch.tensor(np.ascontiguousarray(verts)).float()
                fps = farthest_point_sample(verts,verts.shape[0]).squeeze()
                if self.name == 'spleen' or self.name == 'spleen_test' or self.name == 'se-ornet-tosca':
                    fps = fps[:1024]
                elif self.name == 'fourleg':
                    fps = fps[:5000]
                else:                   
                    fps = fps[:4995]
                self.verts_list.append(verts)
                self.fps_list.append(fps)
                
        # save to cache
        if use_cache:
            torch.save(
                (
                    self.verts_list,
                    self.used_shapes,
                    self.fps_list,
                    self.dist_list
                    # self.vts_list
                ),
                load_cache,
            )

    def __len__(self):
        return len(self.combinations)

    def __getitem__(self, idx):

        # get indexes
        idx1, idx2 = self.combinations[idx]
        fps1= self.fps_list[idx1]
        fps2= self.fps_list[idx2]
        
        if self.name == "step2_posed_templates_5k_off" or self.name == 'clothes_align_5k_off' or self.name == 'clothes_heavy_off' or ('clothes_shape' in self.name):
            if self.with_dino:
                if self.feat_mat:
                    feat_pth1 = self.root_dir + "/feat/" + self.used_shapes[idx1] + ".mat"
                    mat1 = sio.loadmat(feat_pth1)
                    data1 = mat1['feat']
                    feat1 = torch.tensor(data1, dtype=torch.float32)
                    
                    feat_pth2 = self.root_dir + "/feat/" + self.used_shapes[idx2] + ".mat"
                    mat2 = sio.loadmat(feat_pth2)
                    data2 = mat2['feat']
                    feat2 = torch.tensor(data2, dtype=torch.float32)
                else:
                    feat1 = self.feat_list[idx1]
                    feat2 = self.feat_list[idx2]
                
                shape1 = {
                    # test
                    "xyz": self.verts_list[idx1],
                    "feat":feat1,
                    "name": self.used_shapes[idx1],
                    "dist":self.dist_list[idx1]
                }

                shape2 = {
                    "xyz": self.verts_list[idx2],
                    "feat": feat2,
                    "name": self.used_shapes[idx2],
                    "dist":self.dist_list[idx2]
                }
            else:
                feat1 = torch.tensor([])
                feat2 = torch.tensor([])
                shape1 = {
                    "xyz": self.verts_list[idx1][fps1],
                    "feat":feat1,
                    "name": self.used_shapes[idx1],
                    "dist":self.dist_list[idx1][fps1][:,fps1]
                }

                shape2 = {
                    "xyz": self.verts_list[idx2][fps2],
                    "feat": feat2,
                    "name": self.used_shapes[idx2],
                    "dist":self.dist_list[idx2][fps2][:,fps2]
                }          
        else:
            if self.with_dino:
                if self.feat_mat:
                    feat_pth1 = self.root_dir + "/feat/" + self.used_shapes[idx1] + ".mat"
                    mat1 = sio.loadmat(feat_pth1)
                    data1 = mat1['feat']
                    feat1 = torch.tensor(data1, dtype=torch.float32)
                    feat1 = feat1[fps1]
                    
                    feat_pth2 = self.root_dir + "/feat/" + self.used_shapes[idx2] + ".mat"
                    mat2 = sio.loadmat(feat_pth2)
                    data2 = mat2['feat']
                    feat2 = torch.tensor(data2, dtype=torch.float32)
                    feat2 = feat2[fps2]
                else:
                    feat1 = self.feat_list[idx1][fps1]
                    feat2 = self.feat_list[idx2][fps2]  
                
                shape1 = {
                    # test
                    "xyz": self.verts_list[idx1][fps1],
                    "feat":feat1,
                    "name": self.used_shapes[idx1],
                    "dist":self.dist_list[idx1][fps1][:,fps1]
                }

                shape2 = {
                    "xyz": self.verts_list[idx2][fps2],
                    "feat": feat2,
                    "name": self.used_shapes[idx2],
                    "dist":self.dist_list[idx2][fps2][:,fps2]
                }

            else:
                feat1 = torch.tensor([])
                feat2 = torch.tensor([])
                shape1 = {
                    "xyz": self.verts_list[idx1][fps1],
                    "feat":feat1,
                    "name": self.used_shapes[idx1],
                    "dist":self.dist_list[idx1][fps1][:,fps1]
                }

                shape2 = {
                    "xyz": self.verts_list[idx2][fps2],
                    "feat": feat2,
                    "name": self.used_shapes[idx2],
                    "dist":self.dist_list[idx2][fps2][:,fps2]
                }

        # add sym vts if necessary
        if self.with_sym:
            shape1["vts_sym"], shape2["vts_sym"] = self.vts_sym_list[idx1], self.vts_sym_list[idx2]

        return {"shape1": shape1, "shape2": shape2}


class testDataset(Dataset):
    """
    Implementation of shape matching Dataset !WITH VTS! correspondence files (to compute ground-truth).
    This type of dataset is loaded if config['type'] = 'vts'
    It is called Scape Dataset because historically, SCAPE was re-meshed with vts files to track correspondence
    to the original SCAPE dataset. Any dataset using vts (re-meshed as in
    [Continuous and orientation-preserving correspondences via functional maps, Ren et al. 2018 TOG])
    falls into this category and can therefore be utilized via this class.

    ---Parameters:
    @ root_dir: root folder containing shapes_train and shapes_test folder
    @ name: name of the dataset. ex: scape-remeshed, or scape-anisotropic
    @ k_eig: number of Laplace-Beltrami eigenvectors loaded
    @ n_fmap: number of eigenvectors used for fmap computation
    @ n_cfmap: number of complex eigenvectors used for complex fmap computation
    @ with_wks: None if no WKS (C_in <= 3), else the number of WKS descriptors
    @ use_cache: cache for storing dataset (True by default)
    @ op_cache_dir: cache for diffusion net operators (from config['dataset']['cache_dir'])
    @ train: for train or test set

    ---At initialisation, loads:
    1) verts, faces and vts
    2) geometric operators (Laplacian, Gradient)
    3) (optional if C_in = 3) WKS descriptors (for best setting)
    4) (optional if n_cfmap = 0) complex operators (for orientation-aware unsupervised learning)

    ---When delivering an element of the dataset, yields a dictionary with:
    1) shape1 containing all necessary info for source shape
    2) shape2 containing all necessary info for target shape
    3) ground-truth functional map Cgt (obtained with vts files)
    """

    def __init__(self, root_dir, name="scape-remeshed",
                 k_eig=128, n_fmap=30, n_cfmap=20,
                 with_wks=None, with_sym=False,
                 use_cache=True, op_cache_dir=None,
                 train=True,with_dino=False,feat_mat=False):

        self.with_dino = with_dino
        self.feat_mat = feat_mat
        self.k_eig = k_eig
        self.n_fmap = n_fmap
        self.n_cfmap = n_cfmap
        self.root_dir = root_dir
        self.cache_dir = root_dir
        self.op_cache_dir = op_cache_dir
        self.with_sym = with_sym
        self.device = 'cuda:0'
        self.name = name
        # check the cache
        split = "train" if train else "test"
        wks_suf = "" if with_wks is None else "wks_"
        sym_suf = "" if not with_sym else "sym_"
        if use_cache:
            load_cache = os.path.join(self.cache_dir, f"cache_{name}_{sym_suf}{wks_suf}{split}_test.pt")
            print("using dataset cache path: " + str(load_cache))
            if os.path.exists(load_cache):
                print("  --> loading dataset from cache")
                (
                    # main
                    self.verts_list,
                    self.used_shapes,
                    self.fps_list
                ) = torch.load(load_cache)
                if self.name == 'tosca':
                    cat = list(permutations(range(11), 2))
                    centaur = list(permutations(range(11,17), 2))
                    dog = list(permutations(range(17,26), 2))
                    gorilla = list(permutations(range(26,30), 2))
                    horse = list(permutations(range(30,38), 2))
                    wolf = list(permutations(range(38,41), 2))
                    self.combinations = cat + centaur + dog + gorilla + horse + wolf
                    print(self.used_shapes)
                    print(self.combinations)
                else:      
                    self.combinations = list(permutations(range(len(self.verts_list)), 2))
                return
            print("  --> dataset not in cache, repopulating")

        # Load the meshes
        # define files and order
        shapes_split = "shapes_" + split
        self.used_shapes = sorted([x.stem for x in (Path(root_dir) / shapes_split).iterdir() if 'DS_' not in x.stem])
        self.combinations = list(permutations(range(len(self.used_shapes)), 2))
        #
        mesh_dirpath = Path(root_dir) / shapes_split
        vts_dirpath = Path(root_dir) / "corres"

        # Get all the files
        ext = '.off'
        self.verts_list = []
        self.fps_list = []
        self.vts_list = []
        if with_sym:
            vts_sym_list = []

        
        if self.with_dino:
            point_backbone = Uni3FC_DINO_proj().to(self.device)
            point_backbone.eval() 
            use_norm = True
            upsampler = torch.hub.load("mhamilton723/FeatUp", 'dinov2', use_norm=use_norm).to(self.device)
            for shape_name in tqdm(self.used_shapes):
                if self.name == 'spleen' or self.name == 'spleen_test' or self.name == 'se-ornet-tosca' or self.name == 'step2_posed_templates_5k_off' or self.name == 'clothes_align_5k_off' or self.name == 'clothes_heavy_off' or ('clothes_shape' in self.name):
                    verts = load_off_point_cloud(str(mesh_dirpath / f"{shape_name}{ext}"))
                    verts = torch.tensor(verts)
                else:
                    verts, faces = pp3d.read_mesh(str(mesh_dirpath / f"{shape_name}{ext}"))  # off obj
                verts = torch.tensor(np.ascontiguousarray(verts)).float()
                fps = farthest_point_sample(verts,verts.shape[0]).squeeze()
                if self.name == 'spleen' or self.name == 'spleen_test' or self.name == 'se-ornet-tosca':
                    fps = fps[:1024]
                elif self.name == 'fourleg':
                    fps = fps[:5000]
                else:                   
                    fps = fps[:4995]
                self.verts_list.append(verts)
                self.fps_list.append(fps)
                vts = verts.unsqueeze(0).to(self.device)
                feat = point_backbone(vts.permute(0,2,1),upsampler).squeeze(0).cpu().numpy()
                ## 保存.mat文件
                if self.feat_mat:
                    save_path = Path(root_dir) / "feat"
                    if not os.path.exists(save_path):
                        os.makedirs(os.path.join(save_path))
                    feat = {'feat': feat}
                    sio.savemat(self.root_dir + "/feat/" + shape_name + ".mat", feat)
                ## append 
                else:
                    self.feat_list.append(feat)
        else: 
            for shape_name in tqdm(self.used_shapes):
                if self.name == 'spleen' or self.name == 'spleen_test' or self.name == 'se-ornet-tosca' or self.name == 'step2_posed_templates_5k_off' or self.name == 'clothes_align_5k_off' or self.name == 'clothes_heavy_off' or ('clothes_shape' in self.name):
                    verts = load_off_point_cloud(str(mesh_dirpath / f"{shape_name}{ext}"))
                    verts = torch.tensor(verts)
                else:
                    verts, faces = pp3d.read_mesh(str(mesh_dirpath / f"{shape_name}{ext}"))  # off obj
                verts = torch.tensor(np.ascontiguousarray(verts)).float()
                fps = farthest_point_sample(verts,verts.shape[0]).squeeze()
                if self.name == 'spleen' or self.name == 'spleen_test' or self.name == 'se-ornet-tosca':
                    fps = fps[:1024]
                elif self.name == 'fourleg':
                    fps = fps[:5000]
                else:                   
                    fps = fps[:4995]
                self.verts_list.append(verts)
                self.fps_list.append(fps)
                
        # save to cache
        if use_cache:
            torch.save(
                (
                    self.verts_list,
                    self.used_shapes,
                    self.fps_list
                    # self.vts_list
                ),
                load_cache,
            )

    def __len__(self):
        return len(self.combinations)

    def __getitem__(self, idx):

        # get indexes
        idx1, idx2 = self.combinations[idx]
        print(idx1)
        print(idx2)
        print(self.used_shapes[idx1])
        print(self.used_shapes[idx2])
        dist1 = torch.tensor([])
        dist2 = torch.tensor([])
        if self.with_dino:
            if self.feat_mat:
                feat_pth1 = self.root_dir + "/feat/" + self.used_shapes[idx1] + ".mat"
                mat1 = sio.loadmat(feat_pth1)
                data1 = mat1['feat']
                feat1 = torch.tensor(data1, dtype=torch.float32)
                feat1 = feat1
                
                feat_pth2 = self.root_dir + "/feat/" + self.used_shapes[idx2] + ".mat"
                mat2 = sio.loadmat(feat_pth2)
                data2 = mat2['feat']
                feat2 = torch.tensor(data2, dtype=torch.float32)
                feat2 = feat2
            else:
                feat1 = self.feat_list[idx1]
                feat2 = self.feat_list[idx2]  
                 
            shape1 = {
                # test
                "xyz": self.verts_list[idx1],
                "feat":feat1,
                "name": self.used_shapes[idx1],
                "dist": dist1
            }

            shape2 = {
                "xyz": self.verts_list[idx2],
                "feat": feat2,
                "name": self.used_shapes[idx2],
                "dist": dist2
            }

        else:
            feat1 = torch.tensor([])
            feat2 = torch.tensor([])
            shape1 = {
                "xyz": self.verts_list[idx1],
                "feat":feat1,
                "name": self.used_shapes[idx1],
                "dist": dist1
            }

            shape2 = {
                "xyz": self.verts_list[idx2],
                "feat": feat2,
                "name": self.used_shapes[idx2],
                "dist": dist2
            }

        # add sym vts if necessary
        if self.with_sym:
            shape1["vts_sym"], shape2["vts_sym"] = self.vts_sym_list[idx1], self.vts_sym_list[idx2]

        return {"shape1": shape1, "shape2": shape2}
    

def shape_to_device(dict_shape, device):
    names_to_device = ["xyz","feat","dist"]
    for k, v in dict_shape.items():
        if "shape" in k:
            for name in names_to_device:
                if v[name] is not None:
                    v[name] = v[name].to(device)  # .float()
            dict_shape[k] = v
        else:
            dict_shape[k] = v.to(device)

    return dict_shape
