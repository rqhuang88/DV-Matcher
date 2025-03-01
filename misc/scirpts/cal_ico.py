import numpy as np
from itertools import product
import matplotlib.pyplot as plt
from itertools import combinations

def getVertex(G):
    pt2 =  [(a,b) for a,b in product([1,-1], [G, -G])]
    pts =  [(a,b,0) for a,b in pt2]
    pts += [(0,a,b) for a,b in pt2]
    pts += [(b,0,a) for a,b in pt2]
    return np.array(pts)

def rotation_matrix(a, b):
    a = a/np.linalg.norm(a)  # 计算向量a的单位向量
    b = b/np.linalg.norm(b)  # 计算向量b的单位向量

    v = np.cross(a, b)  # 计算旋转轴向量v

    theta = np.arccos(np.dot(a, b))  # 计算夹角θ

    vx = v[0]
    vy = v[1]
    vz = v[2]

    R = np.array([[0, -vz, vy],
                  [vz, 0, -vx],
                  [-vy, vx, 0]])

    I = np.eye(3)  # 3x3的单位矩阵

    # 使用罗德里格斯公式计算旋转矩阵
    rotation_matrix = np.cos(theta) * I + (1 - np.cos(theta)) * np.outer(v, v) + np.sin(theta) * R

    return rotation_matrix

def getDisMat(pts):
    N = len(pts)
    dMat = np.ones([N,N])*np.inf
    for i in range(N):
        for j in range(i):
            dMat[i,j] = np.linalg.norm([pts[i]-pts[j]])
    return dMat

def isFace(e1, e2, e3):
    pts = np.vstack([e1, e2, e3])
    pts = np.unique(pts, axis=0)
    return len(pts)==3

def cal_icosahedron():
    # 正二十面体
    G = (np.sqrt(5)-1)/2
    pts = getVertex(G)
    dMat = getDisMat(pts)
    ix, jx = np.where((dMat-np.min(dMat))<0.01)

    edges = [pts[[i,j]] for i,j in zip(ix, jx)]
    faces = [es for es in combinations(edges, 3) 
        if isFace(*es)]

    face_centers = []

    for pair in faces:
        x,y,z = 0,0,0
        for zp in pair:
            x += (zp[0][0]+zp[1][0])
            y += (zp[0][1]+zp[1][1])
            z += (zp[0][2]+zp[1][2])
        center = (x/6, y/6, z/6)
        face_centers.append(center)
        
    vertex = []
    xs,ys,zs = getVertex(G).T
    for i in range(len(xs)):
        vertex.append((xs[i],ys[i],zs[i]))
    
    R = []
    for i in range(len(xs)):
        a = vertex[i]
        b = [0,0,1]
        # 计算向量a和向量b的夹角，单位为弧度
        R.append(rotation_matrix(a, b))
    return R  
        
# ax = plt.subplot(projection='3d')
# for f in faces:
#     pt = np.unique(np.vstack(f), axis=0)
#     try:
#         ax.plot_trisurf(*pt.T)
#     except:
#         pass

# plt.show()