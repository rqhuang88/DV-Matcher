import os
from pathlib import Path
import numpy as np
import potpourri3d as pp3d
import torch
from torch.utils.data import Dataset
from misc.utils import auto_WKS, farthest_point_sample, square_distance
#
from tqdm import tqdm
from itertools import permutations
import Tools.mesh as qm
from Tools.utils import op_cpl
import random
import numpy as np

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
                 train=True):

        self.k_eig = k_eig
        self.n_fmap = n_fmap
        self.n_cfmap = n_cfmap
        self.root_dir = root_dir
        self.cache_dir = root_dir
        self.op_cache_dir = op_cache_dir
        self.with_sym = with_sym
        self.name = name
            # check the cache
        split = "train" if train else "test"
        wks_suf = "" if with_wks is None else "wks_"
        sym_suf = "" if not with_sym else "sym_"
        if use_cache:
            if self.name == "shrec16_cuts" or self.name == "shrec16_holes":
                load_cache = os.path.join(self.cache_dir, f"cache_{name}_{sym_suf}{wks_suf}train.pt")
            else:
                load_cache = os.path.join(self.cache_dir, f"cache_{name}_{sym_suf}{wks_suf}{split}.pt")
            print("using dataset cache path: " + str(load_cache))
            if os.path.exists(load_cache):
                print("  --> loading dataset from cache")
                (
                    # main
                    self.verts_list,
                    self.used_shapes,
                    self.fps_list,
                    self.dist_list,
                    # self.vts_list
                ) = torch.load(load_cache)
                if self.name == "shrec16_cuts":
                    if split == "train":
                        self.combinations = [(0,2),(0,3),(0,4),(0,5),(0,6),(0,7),(0,8),(0,9),(0,10),(0,11),(0,12),(0,13),(0,14),
                                            (1,17),(1,18),(0,19),(1,20),(1,21),(1,22),(1,23),(1,24),(1,25),(1,26),(1,27),(1,28),(1,29),
                                            (122,32),(122,33),(122,34),(122,35),(122,36),(122,37),(122,38),(122,39),(122,40),(122,41),(122,42),(122,43),(122,44),
                                            (123,47),(123,48),(123,49),(123,50),(123,51),(123,52),(123,53),(123,54),(123,55),(123,56),(123,57),(123,58),(123,59),
                                            (124,62),(124,63),(124,64),(124,65),(124,66),(124,67),(124,68),(124,69),(124,70),(124,71),(124,72),(124,73),(124,74),
                                            (125,77),(125,78),(125,79),(125,80),(125,81),(125,82),(125,83),(125,84),(125,85),(125,86),(125,87),(125,88),(125,89),
                                            (126,92),(126,93),(126,94),(126,95),(126,96),(126,97),(126,98),(126,99),(126,100),(126,101),(126,102),(126,103),(126,104)]
                                            #(127,107),(127,108),(127,109),(127,110),(127,111),(127,112),(127,112),(127,113),(127,114),(127,115),(127,116),(127,117),(127,118),(127,119)]
                    else:
                        self.combinations = [(0,15),(0,16),
                                            (1,30),(1,31),
                                            (122,45),(122,46),
                                            (123,60),(123,61),
                                            (124,75),(124,76),
                                            (125,90),(125,91),
                                            (126,105),(126,106)]
                                           #(127,120),(127,121)]
                    print(self.used_shapes)
                elif self.name == "shrec16_holes":
                    if split == "train":
                        self.combinations = [(0,4),(0,5),(0,6),(0,7),(0,8),(0,9),(0,10),(0,11),(0,12),
                                            (1,14),(1,15),(0,16),(1,17),(1,18),(1,19),(1,20),(1,21),(1,22),
                                            (2,24),(2,25),(2,26),(2,27),(2,28),(2,29),(2,30),(2,31),(2,32),
                                            (3,34),(3,35),(3,36),(3,37),(3,38),(3,39),(3,40),(3,41),(3,42),
                                            (83,44),(83,45),(83,46),(83,47),(83,48),(83,49),(83,50),(83,51),(83,52),
                                            (84,54),(84,55),(84,56),(84,57),(84,58),(84,59),(84,60),(84,61),(84,62),
                                            (85,64),(85,65),(85,66),(85,67),(85,68),(85,69),(85,70),(85,71),(85,72)]
                                            # (86,74),(86,75),(86,76),(86,77),(86,78),(86,79),(86,80),(86,81)]
                    else:
                        self.combinations = [(0,13),
                                            (1,23),
                                            (2,33),
                                            (3,43),
                                            (83,53),
                                            (84,63),
                                            (85,73)]
                                             #(86,82)]
                    print(self.used_shapes)   
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
        self.dist_list = []
        if with_sym:
            vts_sym_list = []

        # Load the actual files
        for shape_name in tqdm(self.used_shapes):
            verts, faces = pp3d.read_mesh(str(mesh_dirpath / f"{shape_name}{ext}"))  # off obj

            if self.name == 'faust_partial':
                dist = torch.tensor([])
            else:
                print('Cal Geo..')
                dist = cal_geo(verts)
            # to torch
            verts = torch.tensor(np.ascontiguousarray(verts)).float()
            fps = farthest_point_sample(verts,verts.shape[0]).squeeze()
            if self.name == 'shrec16_cuts':
                fps = fps
            elif self.name == 'shrec16_holes':
                fps = fps
            else:
                fps = fps[:4995]

            self.verts_list.append(verts)
            self.fps_list.append(fps)
            self.dist_list.append(dist)
            # # vts
            # vts = np.loadtxt(str(vts_dirpath / f"{shape_name}.vts"), dtype=int) - 1
            # self.vts_list.append(vts)
            if with_sym:
                vts_sym = np.loadtxt(str(vts_dirpath / f"{shape_name}.sym.vts"), dtype=int) - 1
                vts_sym_list.append(vts_sym)
                
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
        
        if self.name == "shrec16_cuts" or self.name == "shrec16_holes":
            fps1= self.fps_list[idx1][:1024]
            fps2= self.fps_list[idx2][:1024]
        else:
            while True:
                random_integer = random.randint(0, 11)
                # rotation_matrix = cal_icosahedron()
                # rotated_normal = torch.matmul(normal1, torch.tensor(rotation_matrix[random_integer]).float())
                # idx_partial = torch.from_numpy(np.asarray(np.where(rotated_normal[:,2]>0))).long().squeeze()
                partial_path = os.path.join(self.root_dir,'index_partial')
                pth = partial_path + "/index_" + self.used_shapes[idx2] +"_view_"+ str(random_integer+1)+ ".txt"  
                #print(pth)
                idx_partial = torch.tensor(read_file(pth)).long().squeeze()
                if idx_partial.shape[0] > 2200:
                    break
                
            verts2 = self.verts_list[idx2][idx_partial]
            fps2 = farthest_point_sample(verts2, verts2.shape[0]).squeeze()
            fps2 = fps2[:2200]
            verts2 = verts2[fps2]
            dist2 = self.dist_list[idx2][idx_partial][:,idx_partial]
        
        if self.name == 'shrec16_cuts' or self.name == 'shrec16_holes':
            shape1 = {
                "xyz": self.verts_list[idx1],
                "dist":self.dist_list[idx1],
                "name": self.used_shapes[idx1]
            }

            shape2 = {
                "xyz": self.verts_list[idx2][fps2],
                "dist":self.dist_list[idx2][fps2][:,fps2],
                "name": self.used_shapes[idx2]
            }          
        else:
            shape1 = {
                # test
                "xyz": self.verts_list[idx1][fps1],
                "dist":self.dist_list[idx1][fps1][:,fps1],
                "name": self.used_shapes[idx1]
            }

            shape2 = {
                "xyz": verts2,
                "dist":dist2[fps2][:,fps2],
                "name": self.used_shapes[idx2]
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
                 train=True):

        self.k_eig = k_eig
        self.n_fmap = n_fmap
        self.n_cfmap = n_cfmap
        self.root_dir = root_dir
        self.cache_dir = root_dir
        self.op_cache_dir = op_cache_dir
        self.with_sym = with_sym
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
                    self.dist_list,
                    # self.vts_list
                ) = torch.load(load_cache)
                if split  == "train":
                    if self.name == 'faust_partial':
                        self.combinations = [(70,70)]
                        print(self.used_shapes)
                    else:
                        self.combinations = [(0,0)]
                else:
                    if self.name == "shrec16_cuts_test":
                        print(self.used_shapes)
                        cat = [(0, i) for i in range(2, 32)]
                        centaur = [(1, i) for i in range(32, 44)]
                        david = [(202, i) for i in range(44, 64)]
                        dog = [(203, i) for i in range(64, 84)]
                        horse = [(204, i) for i in range(84, 104)]
                        michael = [(205, i) for i in range(104, 163)]
                        victoria = [(206, i) for i in range(163, 195)]
                        wolf = [(207, i) for i in range(195, 202)]
                        self.combinations = cat + centaur + david + dog + horse + michael + victoria + wolf
                        print(self.combinations)
                    elif self.name == "shrec16_holes_test":
                        print(self.used_shapes)
                        cat = [(0, i) for i in range(4, 29)]
                        centaur = [(1, i) for i in range(29, 46)]
                        david = [(2, i) for i in range(46, 66)]
                        dog = [(3, i) for i in range(66, 92)]
                        horse = [(204, i) for i in range(92, 113)]
                        michael = [(205, i) for i in range(113, 167)]
                        victoria = [(206, i) for i in range(167, 194)]
                        wolf = [(207, i) for i in range(194, 204)]
                        self.combinations = cat + centaur + david + dog + horse + michael + victoria + wolf
                        print(self.combinations)
                    else:
                        self.combinations =  []
                        for i in range(len(self.verts_list)):
                            self.combinations.append((0,i))
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
        self.dist_list = []
        if with_sym:
            vts_sym_list = []

        # Load the actual files
        for shape_name in tqdm(self.used_shapes):
            verts, faces = pp3d.read_mesh(str(mesh_dirpath / f"{shape_name}{ext}"))  # off obj

            if self.name == 'faust_partial' or self.name =='shrec16_cuts_test' or self.name == 'shrec16_holes_test':
                dist = torch.tensor([])
            else:
                print('Cal Geo..')
                dist = cal_geo(verts)
            # to torch
            verts = torch.tensor(np.ascontiguousarray(verts)).float()
            fps = farthest_point_sample(verts,verts.shape[0]).squeeze()
            if self.name == 'shrec16_cuts':
                fps = fps
            elif self.name == 'shrec16_holes':
                fps = fps
            else:
                fps = fps[:4995]

            self.verts_list.append(verts)
            self.fps_list.append(fps)
            self.dist_list.append(dist)
            # # vts
            # vts = np.loadtxt(str(vts_dirpath / f"{shape_name}.vts"), dtype=int) - 1
            # self.vts_list.append(vts)
            if with_sym:
                vts_sym = np.loadtxt(str(vts_dirpath / f"{shape_name}.sym.vts"), dtype=int) - 1
                vts_sym_list.append(vts_sym)
                
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
        idx1, idx2 = self.combinations[idx]
        
        print(self.used_shapes[idx1])
        print(self.used_shapes[idx2])
        shape1 = {
            "xyz": self.verts_list[idx1],
            "dist":self.dist_list[idx1],
            "name": self.used_shapes[idx1]
        }

        shape2 = {
            "xyz": self.verts_list[idx2],
            "dist":self.dist_list[idx2],
            "name": self.used_shapes[idx2]
        }

        # add sym vts if necessary
        if self.with_sym:
            shape1["vts_sym"], shape2["vts_sym"] = self.vts_sym_list[idx1], self.vts_sym_list[idx2]

        return {"shape1": shape1, "shape2": shape2}

def shape_to_device(dict_shape, device):
    names_to_device = ["xyz", "dist"]
    for k, v in dict_shape.items():
        if "shape" in k:
            for name in names_to_device:
                if v[name] is not None:
                    v[name] = v[name].to(device)  # .float()
            dict_shape[k] = v
        else:
            dict_shape[k] = v.to(device)

    return dict_shape
