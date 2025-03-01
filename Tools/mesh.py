#import igl
##
import numpy as np
from scipy import spatial
import scipy as sp
##
import os
from os.path import join, exists

#################################
#################################
#   Nicolas Donati on 01/2021   #
#################################
#################################


class mesh:

    # main fields
    v = None
    f = None
    e = None

    # normals
    n = None
    nv = None

    # laplacian and eigenvalues (real and complex)
    m = None  # mass matrix
    l = None  # laplacian
    cl = None  # connection laplacian
    eig = None  # eigen basis
    vals = None  # eigen values
    ceig = None
    cvals = None

    # halfedge structure
    op = None  # opposite halfedge
    nex = None  # next halfedge
    he_start = None  # local basis of each tangent plane (on vertices)
    # additional data for halfedge structure
    K = None  # Gaussian curvature
    he_angles_norm = None  # normalized angles for each vertex neighbor
    rho = None  # parallel transport shift along halfedge

    # vts and samples
    vts = None
    samples = None

    # differential operators
    degv = None  # degree (=valence) of vertex
    eig_trans = None  # invert of eig basis (needed to put in operators in spectral basis)
    ceig_trans = None
    ceig_real = None  # complex eig flattened in real
    ceig_trans_real = None
    #
    gradv = None  # vertex gradient
    spec_gradv = None  # spectral projection of gradient
    Df = None  # spectral tensor for D_f(X) = <X, grad.f>, with f_i in eig space

    def __init__(self, path,
                 normalized=False, normals=False,
                 spectral=30, complex_spectral=30, spectral_folder=None,
                 verbose=0):

        # extract info from path
        self.path = path
        self.name = path.split('/')[-1]
        self.folder = self.path[:-len(self.name)-1]
        self.verbose = verbose
        
        if verbose > 0:
            print("loading ", self.name, "with init spectral =", spectral)

        # define vertices and faces
        self.v, self.f = igl.read_triangle_mesh(path)

        # do some pre-processing if needed
        if normalized:
            self.center_and_scale(scale=True)
        if normals:
            self.normals_on_faces()
            self.normals_on_vertices()

        # get spectral
        if spectral > 0:
            self.spectral(k=spectral, spectral_folder=spectral_folder)
        if complex_spectral > 0:
            self.complex_spectral(k=complex_spectral, spectral_folder=spectral_folder)

    def get_vts(self, cor_folder='cor'):
        """
        fetches vts files from self.folder. These files store ground truth correspondence to template
        They were initially computed in matlab, so one needs to do vts -= 1
        """
        if cor_folder is None:
            self.vts = np.arange(self.v.shape[0])
            return
        vts_file = join(self.folder, cor_folder, self.name[:-4] + '.vts')
        self.vts = np.loadtxt(vts_file, dtype=np.int32) - 1
    
    def center_and_scale(self, scale=False):
        self.v -= np.mean(self.v, axis=0)
        if scale:
            area = np.sum(igl.doublearea(self.v, self.f))/2
            print('area was', area)
            self.v /= np.sqrt(area)

    def halfedge(self):
        """
        This is used to reproduce the halfedge structure where each undirected edge (i,j) is split
        between  i -> j and j->i. One then needs to keep track of the next and opposite halfedge.
        next (self.nex)    : in face (i,j,k) nex(i->j) = j->k
        opposite (self.op) : op(i->j) = j-> i
        nex and op are indices pointing in the self.e array
        """
        self.e, ue, emap, ue2e = igl.unique_edge_map(self.f)
        e2ue2e = np.array(ue2e)[emap]
        ee = np.tile(np.arange(self.e.shape[0]), [2, 1]).T
        op_mask = (e2ue2e != ee)
        self.op = e2ue2e[op_mask]
        
        e2f = np.remainder(np.arange(self.e.shape[0]), self.f.shape[0])
        e2f_in = np.arange(self.e.shape[0]) // self.f.shape[0]
        nex_in = (e2f_in + 1) % 3
        self.nex = nex_in * self.f.shape[0] + e2f

    def normals_on_faces(self):
        self.n = igl.per_face_normals(self.v, self.f, 0*self.v[0])
    
    def normals_on_vertices(self):
        self.nv = igl.per_vertex_normals(self.v, self.f)

    def spectral(self, k=30, save=True, spectral_folder=None):
        """
        compute spectral operators, and save them if needed
        if already computed, load them
        """

        if spectral_folder is not None:
            spec_folder = join(spectral_folder, 'spectral')
        else:
            spec_folder = join(self.folder, 'spectral')
        ###
        if not os.path.exists(spec_folder):
            os.makedirs(spec_folder)
        save_spectral = join(spec_folder, self.name[:-4]+'.npz')
        load = exists(save_spectral)
        if load:
            F = np.load(save_spectral)
            if np.isin('eig', F.files) and F['eig'].shape[1] >= k:
                save = False
                if self.verbose > 0:
                    print('spectral from file')
                self.eig = F['eig'][:, :k]
                self.vals = F['val'][:k]

        # compute Laplacian
        self.l = -igl.cotmatrix(self.v, self.f)
        self.m = igl.massmatrix(self.v, self.f, igl.MASSMATRIX_TYPE_VORONOI)

        # only compute eig if necessary
        if not save:
            return

        self.vals, self.eig = sp.sparse.linalg.eigsh(self.l, k, self.m, sigma=0, which="LM")  # eig ~ n*k
        # self.eig_trans = (m @ self.eig).T

        # save if needed
        if save:
            print('saving spectral')
            if load and np.isin('ceig', F.files):
                np.savez(save_spectral, eig=self.eig, val=self.vals, ceig=F['ceig'], cval=F['cval'])
                return
            np.savez(save_spectral, eig=self.eig, val=self.vals)

    def complex_spectral(self, k=30, save=True, spectral_folder=None):
        """
        compute complex spectral operators, and save them if needed
        if already computed, load them
        """

        if self.verbose > 0:
            print("loading complex spectral =", k)

        ###
        if spectral_folder is not None:
            spec_folder = join(spectral_folder, 'spectral')
        else:
            spec_folder = join(self.folder, 'spectral')
        ###
        if not os.path.exists(spec_folder):
            os.makedirs(spec_folder)
        save_spectral = join(spec_folder, self.name[:-4]+'.npz')
        load = exists(save_spectral)
        if load:
            F = np.load(save_spectral)
            if np.isin('ceig', F.files) and F['ceig'].shape[1] >= k:
                save = False
                if self.verbose > 0:
                    print('complex spectral from file')
                self.ceig = F['ceig'][:, :k]
                self.cvals = F['cval'][:k]

        # compute connection Laplacian
        self.cl = self.complex_laplacian()

        # only compute ceig if necessary
        if not save:
            return

        if self.m is None:
            self.m = igl.massmatrix(self.v, self.f, igl.MASSMATRIX_TYPE_VORONOI)
        self.cvals, self.ceig = sp.sparse.linalg.eigsh(self.cl, k, self.m, sigma=0, which="LM")

        # save if needed
        if save:
            print('saving complex spectral')
            if load and np.isin('eig', F.files):
                np.savez(save_spectral, eig=F['eig'], val=F['val'], ceig=self.ceig, cval=self.cvals)
                return
            np.savez(save_spectral, ceig=self.ceig, cval=self.cvals)

    def fps_3d(self, n, seed=42):
        """
        farther point sampling (with fixed seed for reproducibility)
        here seed needs to be less than nv
        """
        if n > self.v.shape[0]: n = self.v.shape[0]

        S = np.zeros(n, dtype=int)
        S[0] = seed
        d = np.linalg.norm(self.v - self.v[S[0]], axis=1)
        # avoiding loop would be nice ... but this is already quite fast
        for i in range(1, n):
            m = np.argmax(d)
            S[i] = m
            new_d = np.linalg.norm(self.v - self.v[S[i]], axis=1)
            d = np.min(np.stack([new_d, d], axis=-1), axis=-1)
        self.samples = S
        return S

    def local_basis(self):
        """
        we create an array of beginning halfedges for each vertex
        if (k^i)_j=0..d_i are the neighbors of vertex i, we choose the index h_i of halfedge i->(k^i)_0
        it will be the local basis for the tangent plane at point i
        """
        self.halfedge()
        _, self.he_start = np.unique(self.e[:, 0], return_index=True)  #starting he for each vertex
        return
    
    def embed_he(self, he):
        """
        embed the halfedge in 3d by fetching 3d points coordinate
        used for embedding a vf on the shape
        also useful to visualize local bases
        """
        he_emb = self.v[self.e[he][:, 1]] - self.v[self.e[he][:, 0]]
        return he_emb
    
    def embed_vector_field(self, VF):
        """
        VF is a complex per-vertex, we have to translate this into 3d vectors
        we find the right neighboring face for VF_i (to embed it in 3d)
        by finding the two halfedges concerned (in the right order)
        also keep track of the angle within

        Note: here we focus on making the vector field live on the surface, but we could
        consider the tangent plane as a 2d plane in 3d orthogonal to the vertex normal
        and compute the VF directly in it. This would in fact be faster.
        """
        # define structure if not there
        if self.e is None:
            self.local_basis()
        if self.cl is None:
            self.complex_laplacian()
        ##
        angles = np.angle(VF)
        angles[angles < 0] += 2 * np.pi  # reposition in [0, 2pi]
        ##
        rotate = np.zeros(self.v.shape[0], dtype=int)  # will keep in degree of VF
        he = np.copy(self.he_start)
        last_he = he
        he_VF = np.zeros((self.v.shape[0],2), dtype=int)
        in_ang_VF = np.zeros(self.v.shape[0])
        i = 0
        while np.any(rotate==0):
            # circulate (here CW)
            he = self.op[he]
            he = self.nex[he]
            i += 1

            # compare angles
            he_cur_angle = self.he_angles_norm[he]
            ang_mask = (he_cur_angle > angles)
            rot_mask = (rotate == 0) * ang_mask

            # check if we reached end
            rot_mask = rot_mask | ((rotate==0) * (self.he_start == he))
            #print('size', np.sum(rot_mask))
            rotate[rot_mask] = i

            # get left and right halfedge for embedding face
            he_VF[rot_mask, 0] = last_he[rot_mask]
            he_VF[rot_mask, 1] = he[rot_mask]
            in_ang_VF[rot_mask] = angles[rot_mask] - self.he_angles_norm[last_he[rot_mask]]

            last_he = he

        # renormalize angle
        in_ang_VF *= (1 - self.K/(2*np.pi))
        #print('rot', np.mean(rotate))
        b1 = self.embed_he(he_VF[:, 0])
        b2 = self.embed_he(he_VF[:, 1])

        # Gram-Schmidt the dim2 basis
        b1 /= np.linalg.norm(b1, axis=1)[:, None]
        b2 -= np.sum(b1*b2, axis=1)[:, None] * b1
        b2 /= np.linalg.norm(b2, axis=1)[:, None]

        # embed VF
        VF_embed = np.cos(in_ang_VF)[:, None] * b1 + np.sin(in_ang_VF)[:, None] * b2
        VF_embed *= np.abs(VF)[:, None]
        return VF_embed
    
    def complex_laplacian(self):
        """
        complex Laplacian - also called connection Laplacian. The discretization is introduced
        in Sharp et al., 2019 : The Vector Heat Method
        Here we use igl to define the halfedge structure and broadcast operations to the whole mesh
        so that we only need to loop over the 1-ring (max. ~10 iterations)
        """
        angles = igl.internal_angles(self.v, self.f)
        # here we have an angle deficit on vertices
        # we need to flatten it for our vert Laplacian
        he_angles = np.concatenate([angles[:, 1], angles[:, 2], angles[:, 0]])
        self.local_basis()
        he_angles[self.he_start] = 0  # angle on basis = 0
        self.K = igl.gaussian_curvature(self.v, self.f)  # angle deficit

        vert_angle_sum = 2 * np.pi - self.K
        self.he_angles_norm = 2 * np.pi * he_angles/vert_angle_sum[self.e[:, 0]]
        # print(self.he_angles_norm[self.e[:,0]==4] *180/np.pi) ## precision 1e-12
        
        # now we have to circle around vertices just to get cumulative angle sums
        rotate = np.zeros(self.v.shape[0], dtype=int)  # will keep in degree of vertex
        he = np.copy(self.he_start)
        last_he = he
        i = 0
        while np.any(rotate == 0):
            # circulate CW
            he = self.op[he]
            he = self.nex[he]
            i += 1

            rot_mask = (self.he_start == he) * (rotate == 0)
            rotate[rot_mask] = i
            self.he_angles_norm[he[rotate == 0]] += self.he_angles_norm[last_he[rotate == 0]]
            last_he = he
        # print(self.he_angles_norm[self.e[:,0]==4] *180/np.pi) ## precision 1e-12
        
        # then simply  get the rho and its corresponding complex rotation
        self.rho = (self.he_angles_norm[self.op] + np.pi) - self.he_angles_norm
        r = np.cos(self.rho) + np.sin(self.rho) * 1j
        r_op = r[self.op]

        # now reshape
        r = r.reshape(3, self.f.shape[0]).T
        r_op = r_op.reshape(3, self.f.shape[0]).T
        cot_ = 0.5 / np.tan(angles)
        cot = cot_ * r
        cot_op = cot_ * r_op
        # get values
        S_ = np.concatenate([cot_[:, 2], cot_[:, 0], cot_[:, 1]])
        S = np.concatenate([cot[:, 2], cot[:, 0], cot[:, 1]])
        S_op = np.concatenate([cot_op[:, 2], cot_op[:, 0], cot_op[:, 1]])
        
        # build sparse matrix
        I = np.concatenate([self.f[:, 0], self.f[:, 1], self.f[:, 2]])
        J = np.concatenate([self.f[:, 1], self.f[:, 2], self.f[:, 0]])
        In = np.concatenate([I, J, I, J])
        Jn = np.concatenate([J, I, I, J])
        Sn = np.concatenate([-S_op, -S, S_, S_])
        ##
        A = sp.sparse.csr_matrix((Sn, (In, Jn)), shape=(self.v.shape[0],self.v.shape[0]))
        return A

    #################################################
    # D_fi operators (transfer from TVF to Fun space)
    #################################################

    def grad_vert_op(self):
        """
        gradient on vertices operator. We use all the 1-ring to find a linear & local approximation
        for an input function which yields the formula below
        """
        if self.e is None:  # to get the halfedge structure
            self.local_basis()
        if self.cl is None:  # to get the he_angles normalized (flat tangent plane structure)
            self.complex_laplacian()


        I = []
        J = []
        V = []

        ###
        Vjs = []
        rotate = np.zeros(self.v.shape[0], dtype=int)  # will keep in degree of vertex
        he = np.copy(self.he_start)
        i = 0
        while np.any(rotate == 0):
            # get [vj] and store it in X; same for fj - fi in f
            lij = np.linalg.norm(self.v[self.e[he][:, 1]] - self.v[self.e[he][:, 0]], axis=1)
            aij = self.he_angles_norm[he]
            vj = lij[:, None] * np.cos(np.stack([aij, np.pi/2 - aij], axis=-1))
            vj[rotate>0]=0 # do not add values if cycle done
            Vjs += [vj]

            # circulate CW
            he = self.op[he]
            he = self.nex[he]
            i += 1

            # update rot mask once cycle is done
            rot_mask = (self.he_start == he) * (rotate == 0)
            rotate[rot_mask] = i

        # build and invert local systems
        Vjs = np.stack(Vjs, axis=1)
        Vjs_inv = np.linalg.pinv(Vjs)

        # new rotation around the vertex to add in the coefficients to the sparse matrix
        rotate = np.zeros(self.v.shape[0], dtype=int)  # will keep in degree of vertex
        he = np.copy(self.he_start)
        i = 0
        while np.any(rotate==0):
            # fill in the values
            jdv = self.e[he][:,1]; idv = self.e[he][:,0];
            I += [2*idv, 2*idv, 2*idv+1, 2*idv+1]
            J += [idv, jdv, idv, jdv]
            V += [-Vjs_inv[:,0,i], Vjs_inv[:,0,i], -Vjs_inv[:,1,i], Vjs_inv[:,1,i]]

            # circulate CW
            he = self.op[he]
            he = self.nex[he]
            i += 1
            # update rot mask
            rot_mask = (self.he_start == he) * (rotate == 0)
            rotate[rot_mask] = i

        # build sparse matrix
        I = np.concatenate(I)
        J = np.concatenate(J)
        V = np.concatenate(V)
        #
        A = sp.sparse.csr_matrix((V, (I, J)), shape=(2 * self.v.shape[0], self.v.shape[0]))
        self.degv = rotate
        self.gradv = A
        return A
    
    def grad_vert(self, f):
        if not hasattr(self, 'gradv') or self.gradv is None:
            self.grad_vert_op()
        ##
        idv = np.arange(self.v.shape[0])
        gf = self.gradv @ f

        # print(gf)
        gf1 = gf[2*idv]
        gf2 = gf[2*idv+1]
        return gf1 + 1j * gf2

    def grad_fun_scal(self, f):
        """
        compute operator D_f which takes an tangent vector field X as input and yields the function
        D_f(X) = <grad f , X>
        """
        I = []
        J = []
        V = []
        #
        if not hasattr(self, 'gradv') or self.gradv is None:
            self.grad_vert_op()
        gf = self.gradv @ f
        idv = np.arange(self.v.shape[0])
        gf1 = gf[2*idv]
        gf2 = gf[2*idv+1]

        # build sparse matrix
        I = np.concatenate([idv, idv])
        J = np.concatenate([2*idv, 2*idv+1])
        V = np.concatenate([gf1, gf2])
        #
        A = sp.sparse.csr_matrix((V, (I, J)), shape=(self.v.shape[0], 2 * self.v.shape[0]))
        return A

    def grad_fun_scal_op(self, f, k1, k2):
        """
        previous operator but in spectral basis
        """
        if not hasattr(self, 'eig_trans') or self.eig_trans is None:
            if self.m is None:
                self.m = igl.massmatrix(self.v, self.f, igl.MASSMATRIX_TYPE_VORONOI)
            self.eig_trans = (self.m @ self.eig).T

        # conversion from complex to 2nv * 2k2 matrix
        if not hasattr(self, 'ceig_real') or self.ceig_real is None:
            a = self.ceig.real
            b = self.ceig.imag
            c1 = np.stack([a, b], 1)
            d1 = c1.reshape(2*self.v.shape[0], self.ceig.shape[1])
            c2 = np.stack([-b, a], 1)
            d2 = c2.reshape(2*self.v.shape[0], self.ceig.shape[1])
            d = np.stack([d1, d2], -1).reshape(2*self.v.shape[0], 2*self.ceig.shape[1])
            #
            self.ceig_real = d
        #
        Df_spec = self.eig_trans[:k1] @ self.grad_fun_scal(f) @ self.ceig_real[:, :2*k2]
        return Df_spec
    
    def spec_grad(self, k):
        """
        spectral gradient simply returns directly the spectral complex coefficients of a gradient
        """
        if not hasattr(self, 'ceig_trans_real') or self.ceig_trans_real is None:
            self.ceig_trans = np.conjugate(self.m @ self.ceig).T
            a = self.ceig.real; b = self.ceig.imag
            c1 = np.stack([a,b], 0)
            d1 = c1.reshape(self.ceig.shape[1], 2*self.v.shape[0])
            c2 = np.stack([-b,a], 0)
            d2 = c2.reshape(self.ceig.shape[1], 2*self.v.shape[0])
            d = np.stack([d1,d2], 1).reshape(2*self.ceig.shape[1], 2*self.v.shape[0])
            #
            self.ceig_trans_real = d
        #
        sg = self.ceig_trans_real[:k] @ self.gradv
        self.spec_gradv = sg
        return sg

    ###################################################
    # Now same operators but with the VFs
    # we use this energy to fit the equation
    # <X, grad f> o T = <dT . X, grad (f o T)>
    # here that translates as C D_X = D_QX C, for all X
    ###################################################

    def VF_fun_scal(self, X):
        I = []
        J = []
        V = []
        #
        if not hasattr(self, 'gradv'): self.grad_vert_op()
        #
        idv = np.arange(self.v.shape[0])
        #
        I = np.concatenate([idv, idv])
        J = np.concatenate([2*idv, 2*idv+1])
        V = np.concatenate([X.real, X.imag])
        #
        A = sp.sparse.csr_matrix((V, (I, J)), shape=(self.v.shape[0], 2 * self.v.shape[0]))
        return A @ self.gradv

    def VF_fun_scal_op(self, X, k1):
        #
        if not hasattr(self, 'eig_trans') or self.eig_trans is None:
            self.eig_trans = (self.m @ self.eig).T
        #
        Df_spec = self.eig_trans[:k1] @ self.VF_fun_scal(X) @ self.eig[:, :k1]
        return Df_spec

    ##################################################
    # here we regroup grad fun scal ops in a tensor Df
    ##################################################

    def fun_scal_op_basis(self, k1=10, k2=10):
        Df = []
        for i in range(k1):
            Df += [self.grad_fun_scal_op(self.eig[:, i], k1, k2)]
        Df = np.stack(Df, axis=0)
        self.Df = Df
        return Df

    #####################################################################
    # divergence operators for Q->p2p. Although one can use dual gradient
    #####################################################################

    def div_c_vert_op(self):
        I = []
        J = []
        V = []

        #
        rotate = np.zeros(self.v.shape[0], dtype=int)  # will keep in degree of vertex
        he = np.copy(self.he_start)
        i = 0
        while np.any(rotate == 0):
            # get [vj] and store it in X; same for fj - fi in f
            lij = np.linalg.norm(self.v[self.e[he][:, 1]] - self.v[self.e[he][:, 0]], axis=1)
            aij = self.he_angles_norm[he]
            d = self.degv
            alpha = 1/lij * 1/d
            vj = alpha * (np.cos(aij) - 1j * np.sin(aij))  # conjugate vj ?
            vj[rotate > 0] = 0  # do not add values if cycle done

            # fill in the values
            jdv = self.e[he][:, 1]
            idv = self.e[he][:, 0]
            #
            # cc = np.cos(self.rho[he]); ss = np.sin(self.rho[he]);
            rr = (np.cos(self.rho) - np.sin(self.rho) * 1j)[he]

            J += [jdv, idv]
            I += [idv, idv]
            V += [rr * vj, -vj]

            # circulate CW
            he = self.op[he]
            he = self.nex[he]
            i += 1

            # update rot mask once cycle is done
            rot_mask = (self.he_start == he) * (rotate == 0)
            rotate[rot_mask] = i

        # build sparse matrix
        I = np.concatenate(I)
        J = np.concatenate(J)
        V = np.concatenate(V)
        #
        A = sp.sparse.csr_matrix((V, (I, J)), shape=(self.v.shape[0], self.v.shape[0]))
        return A
    
    # slightly different divergence operator (using Stokes' formula)
    def div_c_vert_op2(self):
        I = []
        J = []
        V = []

        #
        rotate = np.zeros(self.v.shape[0], dtype=int)  #will keep in degree of vertex
        he = np.copy(self.he_start)
        i = 0
        while np.any(rotate==0):
            # get [vj] and store it in X; same for fj - fi in f
            # he2 = self.nex[he]
            lij = np.linalg.norm(self.v[self.e[he][:, 1]] - self.v[self.e[he][:, 0]], axis=1)
            aij = self.he_angles_norm[he]
            he2 = self.nex[self.op[he]]
            lij2 = np.linalg.norm(self.v[self.e[he2][:, 1]] - self.v[self.e[he2][:, 0]], axis=1)
            aij2 = self.he_angles_norm[he2]

            vj = lij * (np.cos(aij) - 1j * np.sin(aij))
            vj2 = lij2 * (np.cos(aij2) - 1j * np.sin(aij2))
            vj = -1j * (vj2 - vj)
            vj[rotate > 0] = 0  # do not add values if cycle done

            # fill in the values
            jdv = self.e[he][:, 1]
            idv = self.e[he][:, 0]
            #
            # cc = np.cos(self.rho[he]); ss = np.sin(self.rho[he]);
            rr = (np.cos(self.rho) - np.sin(self.rho) * 1j)[he]

            J += [jdv]
            I += [idv]
            V += [rr * vj]

            # circulate CW
            he = self.op[he]
            he = self.nex[he]
            i += 1

            # update rot mask once cycle is done
            rot_mask = (self.he_start == he) * (rotate == 0)
            rotate[rot_mask] = i

        # build sparse matrix
        I = np.concatenate(I)
        J = np.concatenate(J)
        V = np.concatenate(V)
        #
        A = sp.sparse.csr_matrix((V, (I, J)), shape=(self.v.shape[0], self.v.shape[0]))
        inv_m = sp.sparse.diags(1/self.m.diagonal())
        return inv_m @ A

######################################################
######################################################
######################################################
