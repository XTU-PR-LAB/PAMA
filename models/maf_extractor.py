# This script is borrowed and extended from https://github.com/shunsukesaito/PIFu/blob/master/lib/model/SurfaceClassifier.py

from packaging import version
import torch
import scipy
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from core.cfgs import cfg
from utils.geometry import projection

import logging
logger = logging.getLogger(__name__)


class MAF_Extractor(nn.Module):
    ''' Mesh-aligned Feature Extrator

    As discussed in the paper, we extract mesh-aligned features based on 2D projection of the mesh vertices.
    The features extrated from spatial feature maps will go through a MLP for dimension reduction.
    '''

    def __init__(self, device=torch.device('cuda')):
        super().__init__()

        self.device = device
        self.filters = []
        self.num_views = 1
        filter_channels = cfg.MODEL.MLP_DIM  #[256,128,64,5]
        self.last_op = nn.ReLU(True) 

        for l in range(0, len(filter_channels) - 1):
            if 0 != l:
                self.filters.append(
                    nn.Conv1d(
                        filter_channels[l] + filter_channels[0],
                        filter_channels[l + 1],
                        1))
            else:
                self.filters.append(nn.Conv1d(
                    filter_channels[l],
                    filter_channels[l + 1],
                    1))

            self.add_module("conv%d" % l, self.filters[l])
        
        self.im_feat = None
        self.cam = None

        # downsample SMPL mesh and assign part labels
        # from https://github.com/nkolot/GraphCMR/blob/master/data/mesh_downsampling.npz
        smpl_mesh_graph = np.load('data/mesh_downsampling.npz', allow_pickle=True, encoding='latin1')

        A = smpl_mesh_graph['A']
        U = smpl_mesh_graph['U']
        D = smpl_mesh_graph['D'] # shape: (2,)

        # downsampling
        ptD = []
        for i in range(len(D)):
            d = scipy.sparse.coo_matrix(D[i])  #(1723,6890)
            i = torch.LongTensor(np.array([d.row, d.col]))  #tensor,(2,1723)
            v = torch.FloatTensor(d.data)  #shape:(1723,),data:array([1.,1.,1.,...])
            ptD.append(torch.sparse.FloatTensor(i, v, d.shape))
        
        # downsampling mapping from 6890 points to 431 points
        # ptD[0].to_dense() - Size: [1723, 6890]
        # ptD[1].to_dense() - Size: [431. 1723]
        Dmap = torch.matmul(ptD[1].to_dense(), ptD[0].to_dense()) # 6890 -> 431
        self.register_buffer('Dmap', Dmap)

    def reduce_dim(self, feature):
        '''
        Dimension reduction by multi-layer perceptrons
        :param feature: list of [B, C_s, N] point-wise features before dimension reduction
        :return: [B, C_p x N] concatantion of point-wise features after dimension reduction
        '''
        y = feature
        tmpy = feature
        for i, f in enumerate(self.filters):
            y = self._modules['conv' + str(i)](
                y if i == 0
                else torch.cat([y, tmpy], 1)
            )
            if i != len(self.filters) - 1:
                y = F.leaky_relu(y)
            if self.num_views > 1 and i == len(self.filters) // 2:
                y = y.view(
                    -1, self.num_views, y.shape[1], y.shape[2]
                ).mean(dim=1)
                tmpy = feature.view(
                    -1, self.num_views, feature.shape[1], feature.shape[2]
                ).mean(dim=1)

        y = self.last_op(y)  #(B,5,N)

        y = y.view(y.shape[0], -1)  #(B,5*N)
        return y

    def sampling(self, points, im_feat=None, z_feat=None):
        '''
        Given 2D points, sample the point-wise features for each point, 
        the dimension of point-wise features will be reduced from C_s to C_p by MLP.
        Image features should be pre-computed before this call.
        :param points: [B, N, 2] image coordinates of points
        :im_feat: [B, C_s, H_s, W_s] spatial feature maps 
        :return: [B, C_p x N] concatantion of point-wise features after dimension reduction
        '''
        if im_feat is None:  
            im_feat = self.im_feat  

        batch_size = im_feat.shape[0]

        if version.parse(torch.__version__) >= version.parse('1.3.0'):
            # Default grid_sample behavior has changed to align_corners=False since 1.3.0.
            point_feat = torch.nn.functional.grid_sample(im_feat, points.unsqueeze(2), align_corners=True)[..., 0]  #(B,C_s,21*21,1)->(B,C_s,21*21),i.e.,(B,256,21*21)或(B,256,431)
        else:
            point_feat = torch.nn.functional.grid_sample(im_feat, points.unsqueeze(2))[..., 0]

        mesh_align_feat = self.reduce_dim(point_feat)  #(B,5*21*21)或(B,5*431)
        return mesh_align_feat
    
    def limb_Dmap(self, limb):
        Dmap_limb = self.Dmap  #(431,6890)
        data = self.d.data[0]
        ds_limb = Dmap_limb[:, limb]  #(431,254)
        idx = torch.nonzero(ds_limb == data)
        idx_x = idx[:, 0]  #(15,)
        Dmap_limb = ds_limb[idx_x, :]  #(15,254)
        return Dmap_limb, len(idx_x)

    def forward(self, p, s_feat=None, cam=None, **kwargs):
        ''' Returns mesh-aligned features for the 3D mesh points.

        Args:
            p (tensor): [B, N_m, 3] mesh vertices
            s_feat (tensor): [B, C_s, H_s, W_s] spatial feature maps
            cam (tensor): [B, 3] camera
        Return:
            mesh_align_feat (tensor): [B, C_p x N_m] mesh-aligned features
        '''
        if cam is None:
            cam = self.cam 
        p_proj_2d = projection(p, cam, retain_z=False)  #(B,N_m,2),(B,431,2)
        mesh_align_feat = self.sampling(p_proj_2d, s_feat)
        return mesh_align_feat
