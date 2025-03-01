from Tools.mesh import *

################
# show p2p map #
################
# import meshplot as mp


#def show_p2p(X, Y, T12, axis=0, axis_col=0, withuv=False, T12_gt=None):
#   left = X
#    right = Y
#    offset = np.zeros(3)
#    offset[axis] = np.max([-np.min(right.v[:, axis]), np.max(left.v[:, axis])])
#    all_v = np.concatenate([left.v-offset, right.v+offset])
#    all_f = np.concatenate([left.f, right.f+left.v.shape[0]])
#
#    c = Y.v[:, axis_col]
#    all_c = np.concatenate([c[T12], c])
#    # err #
#    if T12_gt is not None:
#        err = euc_err(Y, T12, T12_gt)
#        print('euc err:', err)
#    # uv #
#    uv = None
#    if withuv:
#        uv = Y.v[:,:2]
#        uv = np.concatenate([uv[T12], uv])
#    p = mp.plot(all_v, all_f, all_c, uv=uv, return_plot=True)
#    i_spl = 10
#    X.fps_3d(i_spl)
#    spl = X.samples
#    p.add_lines((left.v-offset)[spl], (right.v+offset)[T12[spl]],
#                shading={'line_color': 'red'})
#    return

# Jacobian tools
def to_rot(J):
    u, _, v = np.linalg.svd(J)
    #R = (u @ v).transpose([0,2,1])
    R = u@v
    dets = (np.linalg.det(R) < 0)
    R[dets, :, 0] *= -1
    return R

######################
# bij ZO from random #
######################


def to_rl(A):
    a, b = A.realb, A.imag
    c = np.stack([a, b], -1)
    #print(c.shape)
    c = c.reshape(A.shape[0], 2*A.shape[1])
    return c


def op_cpl(op):
    idv = np.arange(op.shape[1]//2)
    a = op[:, 2 * idv]
    b = op[:, 2 * idv + 1]
    return a + 1j * b

#random T12
def initialize_pMap(nX, nY):
    T12 = np.random.randint(nY, size=nX)
    return T12

def fMap2pMap(L1, L2, fmap):
    dim1 = fmap.shape[1]
    dim2 = fmap.shape[0]

    B1 = L1[:,:dim1]
    B2 = L2[:,:dim2]
    C12 = fmap
    _, T21 = spatial.cKDTree(B1@C12.T).query(B2, n_jobs=-1)
    return T21

def pMap2fMap(L1, L2, pmap):
    C21 = np.linalg.pinv(L1) @ L2[pmap]
    return C21


#### bijective zoomout from map tree
def bij_fMap2pMap(B1, B2, C12, C21):
    _, T12 = spatial.cKDTree(B2@C21.T).query(B1, n_jobs=-1)
    _, T21 = spatial.cKDTree(B1@C12.T).query(B2, n_jobs=-1)

    #bijective modification for C12, C21
    B2_aug = np.concatenate([B2, B2[T12]], axis=0)
    B1_aug = np.concatenate([B1[T21], B1], axis=0)
    C12 = np.linalg.pinv(B2_aug) @ B1_aug
    #
    B2_aug = np.concatenate([B2[T12], B2], axis=0)
    B1_aug = np.concatenate([B1, B1[T21]], axis=0)
    C21 = np.linalg.pinv(B1_aug) @ B2_aug
    
    #use both maps to get point to point
    B2_aug = np.concatenate([B2@C21.T, B2@C12], axis=1)
    B1_aug = np.concatenate([B1, B1], axis=1)
    _, T12 = spatial.cKDTree(B2_aug).query(B1_aug, n_jobs=-1)
    #
    B2_aug = np.concatenate([B1@C12.T, B1@C21], axis=1)
    B1_aug = np.concatenate([B2, B2], axis=1)
    _, T21 = spatial.cKDTree(B2_aug).query(B1_aug, n_jobs=-1)
    return T12, T21
####


# This is our Q-step (regular)
def CMap2QMap(X, Y, C, k2, verbose=0):
    k1 = C.shape[0]
    # we can consider only a subspace of the eigenspace (crop)
    if not hasattr(X, 'Df') or X.Df is None:
        if verbose > 0:
            print('recomputing ops')
        X.fun_scal_op_basis(k1, k2)
        Y.fun_scal_op_basis(k1, k2)
    # print(X.Df.shape, Y.Df.shape, C.shape)
    #
    k1_ = X.Df.shape[1]
    k2_ = X.Df.shape[2]//2
    if k1 > k1_ or k2 > k2_:
        X.fun_scal_op_basis(k1, k2)
        Y.fun_scal_op_basis(k1, k2)
    #
    CD = np.einsum('ij,bjk->bik', C, X.Df[:k1, :k1, :2*k2])  # size b x k1 x k2
    DC = np.einsum('bi,bjk->ijk', C, Y.Df[:k1, :k1, :2*k2])  # size b x k1 x k2
    #
    CD = op_cpl(np.concatenate(list(CD), axis=0))
    DC = op_cpl(np.concatenate(list(DC), axis=0))
    #
    Q = np.linalg.lstsq(DC,CD,rcond=None)[0]
    return np.conjugate(Q)


# This is our Q-step (with orthogonal constraint)
def CMap2QMap_procrustes(X, Y, C, k2, verbose=0):
    k1 = C.shape[0]
    # we can consider only a subspace of the eigenspace (crop)
    if not hasattr(X, 'Df') or X.Df is None:
        if verbose > 0:
            print('recomputing ops')
        X.fun_scal_op_basis(k1, k2)
        Y.fun_scal_op_basis(k1, k2)
    # print(X.Df.shape, Y.Df.shape, C.shape)
    #
    k1_ = X.Df.shape[1]
    k2_ = X.Df.shape[2]//2
    if k1>k1_ or k2>k2_:
        X.fun_scal_op_basis(k1,k2)
        Y.fun_scal_op_basis(k1,k2)
    #
    CD = np.einsum('ij,bjk->bik', C, X.Df[:k1,:k1,:2*k2])  # size b x k1 x k2
    DC = np.einsum('bi,bjk->ijk', C, Y.Df[:k1,:k1,:2*k2])  # size b x k1 x k2
    #
    CD = op_cpl(np.concatenate(list(CD), axis=0))
    DC = op_cpl(np.concatenate(list(DC), axis=0))
    #
    M = np.conjugate(DC).T @ CD
    u, _, v = np.linalg.svd(M)
    R = u@v
    return np.conjugate(R)


def QMap2pMap(B1, B2, Q12):
    k = Q12.shape[0]
    #
    B1_Q12 = to_rl(B1[:, :k]@np.conjugate(Q12.T))
    B2_ = to_rl(B2[:, :k])
    #
    _, T21 = spatial.cKDTree(B1_Q12).query(B2_, n_jobs=-1)
    return T21


def bij3_fMap2pMap_withQ(X, Y, B1, B2, C12, C21, L1, L2,
                         w1=1, w2=1, w3=1, wQ=1):
    
    ### first convert to Q to "get rid" of symmetries
    if wQ:
        k = C12.shape[0]
        Q12 = CMap2QMap_procrustes(X,Y,C12,k)
        Q21 = CMap2QMap_procrustes(Y,X,C21,k)

        ### convert Q back to p2p
        T21 = QMap2pMap(X.gpdir[X.samples], Y.gpdir[Y.samples], Q12)
        T12 = QMap2pMap(Y.gpdir[Y.samples], X.gpdir[X.samples], Q21)

        #bijective modification for C12, C21
        B2_aug = np.concatenate([B2, B2[T12]], axis=0)
        B1_aug = np.concatenate([B1[T21], B1], axis=0)
        C12 = np.linalg.pinv(B2_aug) @ B1_aug
        #
        B2_aug = np.concatenate([B2[T12], B2], axis=0)
        B1_aug = np.concatenate([B1, B1[T21]], axis=0)
        C21 = np.linalg.pinv(B1_aug) @ B2_aug
    
    #use triple energy to get T12, T21
    B2_aug = [w1 * B2@C21.T, w2 * B2@C12]
    #if w3>0: B2_aug += [w3 * B2@L2@C21.T]
    if w3>0: B2_aug += [w3 * B2@L2@C12]
    B2_aug = np.concatenate(B2_aug, axis=1)
    #
    B1_aug = [w1 * B1, w2 * B1]
    #if w3>0: B1_aug += [w3 * B1@L1]
    if w3>0: B1_aug += [w3 * B1@L1]
    B1_aug = np.concatenate(B1_aug, axis=1)
    #
    _, T12 = spatial.cKDTree(B2_aug).query(B1_aug, n_jobs=-1)
    #####
    B2_aug = [w1 * B1@C12.T, w2 * B1@C21]
    #if w3>0: B2_aug += [w3 * B1@L1@C12.T]
    if w3>0: B2_aug += [w3 * B1@L1@C21]
    B2_aug = np.concatenate(B2_aug, axis=1)
    #
    B1_aug = [w1 * B2, w2 * B2]
    #if w3>0: B1_aug += [w3 * B2@L2]
    if w3>0: B1_aug += [w3 * B2@L2]
    B1_aug = np.concatenate(B1_aug, axis=1)
    #
    _, T21 = spatial.cKDTree(B2_aug).query(B1_aug, n_jobs=-1)
    return T12, T21


def func_bijective_zm_fmap(X, Y, C12_ini, C21_ini, k_init=10,k_step=1, k_final=30,
                           N_inter=5, N_fps=500,
                           w1=1, w2=2, w3=1, wQ=1,
                           verbose=-1):
    
    X.fps_3d(N_fps)
    Y.fps_3d(N_fps)

    B1_all = X.eig[X.samples]
    B2_all = Y.eig[Y.samples]

    T12 = fMap2pMap(B2_all, B1_all, C21_ini)
    T21 = fMap2pMap(B1_all, B2_all, C12_ini)

    for k in range(k_init, k_final, k_step):
        
        if verbose>=0: print("step:", k)
        #print("step:", k)
        for n in range(N_inter):
        
            B1 = B1_all[:, :k]
            B2 = B2_all[:, :k]
            #classic ZO step
            #print(B2.shape, B1[T21].shape)
            C12 = np.linalg.pinv(B2) @ B1[T21]
            C21 = np.linalg.pinv(B1) @ B2[T12]
            ##
            L1 = np.diag(X.vals[:k]); L2 = np.diag(Y.vals[:k]);
            
            #
            T12, T21 = bij3_fMap2pMap_withQ(X,Y,B1, B2, C12, C21, L1, L2,
                                        w1=w1, w2=w2, w3=w3, wQ=wQ)
            
            
                

    B1 = B1_all[:, :k_final]
    B2 = B2_all[:, :k_final]
    C21 = np.linalg.pinv(B1) @ B2[T12]
    C12 = np.linalg.pinv(B2) @ B1[T21]
    return C12, C21


def zo_fmap(B1, B2, C21_ini, k_init=30, k_final=50, k_step=1, N_inter=1):
    #X.fps_3d(500)
    #Y.fps_3d(500)

    B1_all = B1
    B2_all = B2

    T12 = fMap2pMap(B2_all, B1_all, C21_ini)

    for k in range(k_init, k_final, k_step):

        # if verbose >= 0: print("step:", k)
        # print("step:", k)
        for n in range(N_inter):

            B1 = B1_all[:, :k]
            B2 = B2_all[:, :k]
            # classic ZO step
            C21 = np.linalg.pinv(B1) @ B2[T12]
            ##
            #L1 = np.diag(X.vals[:k]);
            #L2 = np.diag(Y.vals[:k]);
            # print(L1.shape, L2.shape)
            #if not wQ:
            T12 = fMap2pMap(B2, B1, C21)
            C21 = pMap2fMap(B1, B2, T12)

            # else:
            #    raise NameError('wQ must be 0 or 1')

    B1 = B1_all[:, :k_final]
    B2 = B2_all[:, :k_final]
    C21 = np.linalg.pinv(B1) @ B2[T12]
    # C12 = np.linalg.pinv(B2) @ B1[T21]
    return C21

def func_zm_fmap(X, Y, C21_ini, k_init=10,k_step=1, k_final=30,
                   N_inter=5, wQ=0,
                   verbose=-1):
    
    X.fps_3d(500)
    Y.fps_3d(500)

    B1_all = X.eig[X.samples]
    B2_all = Y.eig[Y.samples]

    T12 = fMap2pMap(B2_all, B1_all, C21_ini)

    for k in range(k_init, k_final, k_step):
        
        if verbose>=0: print("step:", k)
        #print("step:", k)
        for n in range(N_inter):
        
            B1 = B1_all[:, :k]
            B2 = B2_all[:, :k]
            #classic ZO step
            C21 = np.linalg.pinv(B1) @ B2[T12]
            ##
            L1 = np.diag(X.vals[:k]); L2 = np.diag(Y.vals[:k]);
            #print(L1.shape, L2.shape)
            if not wQ:
                T12 = fMap2pMap(B2,B1,C21)
                C21 = pMap2fMap(B1,B2,T12)
            elif wQ:
                Q21 = CMap2QMap_procrustes(Y,X,C21,k)
                T12 = QMap2pMap(Y.gpdir[Y.samples], X.gpdir[X.samples], Q21)
                C21 = pMap2fMap(B1,B2,T12)
                    
            else: raise NameError('wQ must be 0 or 1')
                

    B1 = B1_all[:, :k_final]
    B2 = B2_all[:, :k_final]
    C21 = np.linalg.pinv(B1) @ B2[T12]
    #C12 = np.linalg.pinv(B2) @ B1[T21]
    return C21


### for euclidean error
def euc_err(Y, T, Tgt):
    area = np.sum(igl.doublearea(Y.v, Y.f)/2)
    #print(np.sqrt(area))
    d = Y.v[T] - Y.v[Tgt]
    err = np.mean(np.linalg.norm(d, axis=1)) / np.sqrt(area)
    return err


#########################################
# to write ply files with vertex colors #
#########################################

def write_ply(filename, verts, faces, color=['0', '0', '0']):
    nv = verts.shape[0]
    nf = faces.shape[0]
    file = open(filename, 'w')
    file.write("ply\n")
    file.write("format ascii 1.0\n")
    file.write("comment made by Nicolas Donati\n")
    file.write("comment this file is a FAUST shape with mapping colormap\n")
    file.write("element vertex " + str(nv) + "\n")
    file.write("property float x\n")
    file.write("property float y\n")
    file.write("property float z\n")
    file.write("property uchar red\n")
    file.write("property uchar green\n")
    file.write("property uchar blue\n")
    file.write("element face " + str(nf) + "\n")
    file.write("property list uchar int vertex_index\n")
    file.write("end_header\n")

    for i, item in enumerate(verts):
        if type(color[0]) is list or type(color[0]) is np.ndarray:
            #print(color, type(color[0]))
            col1, col2, col3 = color[i, 0], color[i, 1], color[i, 2]
        else:
            #print("wesh")
            col1, col2, col3 = color[0], color[1], color[2]
        #
        file.write("{0} {1} {2} {3} {4} {5}\n".format(item[0], item[1], item[2],
                                                      col1, col2, col3))

    # for item in normals:
    # file.write("vn {0} {1} {2}\n".format(item[0],item[1],item[2]))
    for item in faces:
        file.write("3 {0} {1} {2}\n".format(item[0], item[1], item[2]))
    file.close()
    return