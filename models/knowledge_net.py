import torch
import torch.nn as nn
import numpy as np
from .pose_resnet import get_resnet_encoder
from core.cfgs import cfg
from utils.geometry import rot6d_to_rotmat, projection, rotation_matrix_to_angle_axis
from .maf_extractor import MAF_Extractor
from .smpl import SMPL, SMPL_MODEL_DIR, SMPL_MEAN_PARAMS, H36M_TO_J14
from .hmr import ResNet_Backbone
from .iuv_predictor import IUV_predict_layer
from .attention import Encoder
from .utils import judge_vert_vis

import logging
logger = logging.getLogger(__name__)

import json
with open('./data/smpl_vert_segmentation.json', encoding='utf-8') as file:
    data = json.load(file)
    limb = []
    limb.append((data['rightArm'], data['leftArm']))
    limb.append((data['rightForeArm'], data['leftForeArm']))
    limb.append((data['rightUpLeg'], data['leftUpLeg']))
    limb.append((data['rightLeg'], data['leftLeg']))


BN_MOMENTUM = 0.1

class Regressor(nn.Module):
    def __init__(self, feat_dim, smpl_mean_params):
        super().__init__()

        npose = 24 * 6
        nbbox = 3

        # self.fc1 = nn.Linear(feat_dim + npose + 13, 1024)
        self.fc1 = nn.Linear(feat_dim + nbbox + npose + 13, 1024)
        self.drop1 = nn.Dropout()
        self.fc2 = nn.Linear(1024, 1024)
        self.drop2 = nn.Dropout()
        self.decpose = nn.Linear(1024, npose)
        self.decshape = nn.Linear(1024, 10)
        self.deccam = nn.Linear(1024, 3)
        nn.init.xavier_uniform_(self.decpose.weight, gain=0.01)
        nn.init.xavier_uniform_(self.decshape.weight, gain=0.01)
        nn.init.xavier_uniform_(self.deccam.weight, gain=0.01)

        self.smpl = SMPL(
            SMPL_MODEL_DIR,
            batch_size=64,
            create_transl=False
        )  #加载SMPL_NEUTRAL.pkl中的参数

        mean_params = np.load(smpl_mean_params)
        init_pose = torch.from_numpy(mean_params['pose'][:]).unsqueeze(0)
        init_shape = torch.from_numpy(mean_params['shape'][:].astype('float32')).unsqueeze(0)
        init_cam = torch.from_numpy(mean_params['cam']).unsqueeze(0)
        self.register_buffer('init_pose', init_pose)
        self.register_buffer('init_shape', init_shape)
        self.register_buffer('init_cam', init_cam)

    def forward(self, x, bbox=None, init_pose=None, init_shape=None, init_cam=None, n_iter=1, J_regressor=None):
        batch_size = x.shape[0]

        if init_pose is None:
            init_pose = self.init_pose.expand(batch_size, -1)
        if init_shape is None:
            init_shape = self.init_shape.expand(batch_size, -1)
        if init_cam is None:
            init_cam = self.init_cam.expand(batch_size, -1)

        pred_pose = init_pose  #(B,24*6),用一个6D三维旋转表示的
        pred_shape = init_shape
        pred_cam = init_cam
        for i in range(n_iter):
            # xc = torch.cat([x, pred_pose, pred_shape, pred_cam], 1)
            xc = torch.cat([x, bbox, pred_pose, pred_shape, pred_cam], 1)  #x:(B,5*21*21),xc:(B,5*21*21+24*6+13),i.e.(B,2362)
            xc = self.fc1(xc)  #(B,C)->(B,1024)
            xc = self.drop1(xc)
            xc = self.fc2(xc)
            xc = self.drop2(xc)
            pred_pose = self.decpose(xc) + pred_pose
            pred_shape = self.decshape(xc) + pred_shape
            pred_cam = self.deccam(xc) + pred_cam

        pred_rotmat = rot6d_to_rotmat(pred_pose).view(batch_size, 24, 3, 3)  #(B,24,3,3)

        pred_output = self.smpl(
            betas=pred_shape,
            body_pose=pred_rotmat[:, 1:],
            global_orient=pred_rotmat[:, 0].unsqueeze(1),
            pose2rot=False
        )  

        pred_vertices = pred_output.vertices
        pred_joints = pred_output.joints
        pred_smpl_joints = pred_output.smpl_joints
        pred_keypoints_2d = projection(pred_joints, pred_cam)
        pose = rotation_matrix_to_angle_axis(pred_rotmat.reshape(-1, 3, 3)).reshape(-1, 72)

        if J_regressor is not None:
            pred_joints = torch.matmul(J_regressor, pred_vertices)
            pred_pelvis = pred_joints[:, [0], :].clone()
            pred_joints = pred_joints[:, H36M_TO_J14, :]
            pred_joints = pred_joints - pred_pelvis

        output = {
            'theta'  : torch.cat([pred_cam, pred_shape, pose], dim=1),
            'verts'  : pred_vertices,
            'kp_2d'  : pred_keypoints_2d,
            'kp_3d'  : pred_joints,
            'smpl_kp_3d' : pred_smpl_joints,
            'rotmat' : pred_rotmat,
            'pred_cam': pred_cam,
            'pred_shape': pred_shape,
            'pred_pose': pred_pose,
        }
        return output

    def forward_init(self, x, init_pose=None, init_shape=None, init_cam=None, n_iter=1, J_regressor=None):
        batch_size = x.shape[0]

        if init_pose is None:
            init_pose = self.init_pose.expand(batch_size, -1)  #(B,24*6),i.e.(B,144)
        if init_shape is None:
            init_shape = self.init_shape.expand(batch_size, -1)  #(B,10)
        if init_cam is None:
            init_cam = self.init_cam.expand(batch_size, -1)  #(B,3)

        pred_pose = init_pose
        pred_shape = init_shape
        pred_cam = init_cam

        pred_rotmat = rot6d_to_rotmat(pred_pose.contiguous()).view(batch_size, 24, 3, 3)

        pred_output = self.smpl(
            betas=pred_shape,
            body_pose=pred_rotmat[:, 1:],
            global_orient=pred_rotmat[:, 0].unsqueeze(1),
            pose2rot=False
        )

        pred_vertices = pred_output.vertices
        pred_joints = pred_output.joints
        pred_smpl_joints = pred_output.smpl_joints
        pred_keypoints_2d = projection(pred_joints, pred_cam)
        pose = rotation_matrix_to_angle_axis(pred_rotmat.reshape(-1, 3, 3)).reshape(-1, 72)

        if J_regressor is not None:
            pred_joints = torch.matmul(J_regressor, pred_vertices)
            pred_pelvis = pred_joints[:, [0], :].clone()
            pred_joints = pred_joints[:, H36M_TO_J14, :]
            pred_joints = pred_joints - pred_pelvis

        output = {
            'theta'  : torch.cat([pred_cam, pred_shape, pose], dim=1),
            'verts'  : pred_vertices,  #(B,V,3)
            'kp_2d'  : pred_keypoints_2d,
            'kp_3d'  : pred_joints,  #(B,49,3)
            'smpl_kp_3d' : pred_smpl_joints,  #(B,24,3)
            'rotmat' : pred_rotmat,
            'pred_cam': pred_cam,   #(B,3)
            'pred_shape': pred_shape,
            'pred_pose': pred_pose,
        }
        return output


class PAMA(nn.Module):
    def __init__(self, smpl_mean_params=SMPL_MEAN_PARAMS, pretrained=True):
        super().__init__()
        self.global_mode = not cfg.MODEL.MAF_ON  #True
        self.feature_extractor = get_resnet_encoder(cfg, global_mode=self.global_mode)

        # deconv layers
        self.inplanes = self.feature_extractor.inplanes  #64
        self.deconv_with_bias = cfg.RES_MODEL.DECONV_WITH_BIAS
        self.deconv_layers = self._make_deconv_layer(
            cfg.RES_MODEL.NUM_DECONV_LAYERS,
            cfg.RES_MODEL.NUM_DECONV_FILTERS,
            cfg.RES_MODEL.NUM_DECONV_KERNELS,
        )

        self.maf_extractor = nn.ModuleList()
        for _ in range(cfg.MODEL.N_ITER):  #3
            self.maf_extractor.append(MAF_Extractor())
        ma_feat_len = self.maf_extractor[-1].Dmap.shape[0] * cfg.MODEL.MLP_DIM[-1]  #431*5
        
        grid_size = 21
        xv, yv = torch.meshgrid([torch.linspace(-1, 1, grid_size), torch.linspace(-1, 1, grid_size)])  #(21,21)
        points_grid = torch.stack([xv.reshape(-1), yv.reshape(-1)]).unsqueeze(0)  #(1,2,21*21)
        self.register_buffer('points_grid', points_grid)
        grid_feat_len = grid_size * grid_size * cfg.MODEL.MLP_DIM[-1]

        self.regressor = nn.ModuleList()
        self.right_vert_ds = []
        self.left_vert_ds = []
        for i in range(cfg.MODEL.N_ITER):  #3
            if i == 0:
                ref_infeat_dim = grid_feat_len  #21*21*5
            else:
                ref_infeat_dim = ma_feat_len
                body_len = 0
                limb_dimesion = 0
                limb_vert = 0
                for (limb0, limb1) in limb:  
                    body_len += (len(limb0) + len(limb1)) * cfg.MODEL.MLP_DIM[-1]
                    limb_dimesion += len(limb0) * cfg.MODEL.MLP_DIM[-1]
                    limb_vert += len(limb0)

                    
                # ref_infeat_dim = ma_feat_len + body_len   #(B,2,X'*5)
                ref_infeat_dim = ma_feat_len + limb_dimesion  #(B,1,X'*5)
            self.regressor.append(Regressor(feat_dim=ref_infeat_dim, smpl_mean_params=smpl_mean_params))

        self.feature_fusion = Encoder(n_layers=1, n_head=8, d_model=limb_dimesion, d_v=32, d_k=32, d_inner=256)
        

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def _make_deconv_layer(self, num_layers, num_filters, num_kernels):
        """
        Deconv_layer used in Simple Baselines:
        Xiao et al. Simple Baselines for Human Pose Estimation and Tracking
        https://github.com/microsoft/human-pose-estimation.pytorch
        """
        assert num_layers == len(num_filters), \
            'ERROR: num_deconv_layers is different len(num_deconv_filters)'
        assert num_layers == len(num_kernels), \
            'ERROR: num_deconv_layers is different len(num_deconv_filters)'
        
        def _get_deconv_cfg(deconv_kernel, index):
            if deconv_kernel == 4: #
                padding = 1
                output_padding = 0
            elif deconv_kernel == 3:
                padding = 1
                output_padding = 1
            elif deconv_kernel == 2:
                padding = 0
                output_padding = 0

            return deconv_kernel, padding, output_padding

        layers = []
        for i in range(num_layers):
            kernel, padding, output_padding = _get_deconv_cfg(num_kernels[i], i)

            planes = num_filters[i]
            layers.append(
                nn.ConvTranspose2d(
                    in_channels=self.inplanes,
                    out_channels=planes,
                    kernel_size=kernel,
                    stride=2,
                    padding=padding,
                    output_padding=output_padding,
                    bias=self.deconv_with_bias))
            layers.append(nn.BatchNorm2d(planes, momentum=BN_MOMENTUM))
            layers.append(nn.ReLU(inplace=True))
            self.inplanes = planes

        return nn.Sequential(*layers)

    def forward(self, x, bbox=None, J_regressor=None):

        batch_size = x.shape[0]  #x:(B,C,H,W)

        # spatial features and global features
        s_feat, g_feat = self.feature_extractor(x)
        # print(s_feat.shape) #(B,2048,7,7)

        assert cfg.MODEL.N_ITER >= 0 and cfg.MODEL.N_ITER <= 3
        if cfg.MODEL.N_ITER == 1:
            deconv_blocks = [self.deconv_layers]
        elif cfg.MODEL.N_ITER == 2:
            deconv_blocks = [self.deconv_layers[0:6], self.deconv_layers[6:9]]
        elif cfg.MODEL.N_ITER == 3:
            deconv_blocks = [self.deconv_layers[0:3], self.deconv_layers[3:6], self.deconv_layers[6:9]]

        out_list = {}

        # initial parameters
        # TODO: remove the initial mesh generation during forward to reduce runtime
        # by generating initial mesh the beforehand: smpl_output = self.init_smpl
        smpl_output = self.regressor[0].forward_init(g_feat, J_regressor=J_regressor)

        out_list['smpl_out'] = [smpl_output]
        out_list['dp_out'] = []

        # for visulization
        vis_feat_list = [s_feat.detach()]

        # parameter predictions
        for rf_i in range(cfg.MODEL.N_ITER):
            pred_cam = smpl_output['pred_cam']
            pred_shape = smpl_output['pred_shape']
            pred_pose = smpl_output['pred_pose']

            pred_cam = pred_cam.detach()
            pred_shape = pred_shape.detach()
            pred_pose = pred_pose.detach()

            s_feat_i = deconv_blocks[rf_i](s_feat) #(B,256,14,14),(B,256,28,28),(B,256,56,56)
            s_feat = s_feat_i
            vis_feat_list.append(s_feat_i.detach())

            self.maf_extractor[rf_i].im_feat = s_feat_i  
            self.maf_extractor[rf_i].cam = pred_cam      
            if rf_i == 0:
                sample_points = torch.transpose(self.points_grid.expand(batch_size, -1, -1), 1, 2)  #(B,21*21,2)
                ref_feature = self.maf_extractor[rf_i].sampling(sample_points)  #(B,5*N),即(B,5*21*21)
            else:
                pred_smpl_verts = smpl_output['verts'].detach()  #(B,6890,3)

                # TODO: use a more sparse SMPL implementation (with 431 vertices) for acceleration
                pred_smpl_verts_ds = torch.matmul(self.maf_extractor[rf_i].Dmap.unsqueeze(0), pred_smpl_verts) # [B, 431, 3]
                ref_feature = self.maf_extractor[rf_i](pred_smpl_verts_ds) # [B, 431 * n_feat],即(B,431*5)
                
                #true
                full_limb_feature = []
                # visibility, arg_min_r = judge_vert_vis(cfg, pred_smpl_verts, pred_cam, self.regressor[rf_i].smpl.faces)  #(B,6890)
                for i,(limb0, limb1) in enumerate(limb):
                    pred_right_vert = pred_smpl_verts[:, limb0, :]  #(4,284,3)
                    pred_left_vert = pred_smpl_verts[:, limb1, :]
                    
                    right_feature = self.maf_extractor[rf_i](pred_right_vert).unsqueeze(1)  #(B,1,284*5)
                    left_feature = self.maf_extractor[rf_i](pred_left_vert).unsqueeze(1)

                    limb_feature = torch.cat([right_feature, left_feature], dim=1)  #(B,2,X*5)，即(B,2,284*5)
                    full_limb_feature.append(limb_feature)
                limb_feat = torch.cat(full_limb_feature,dim=2) #(B,2,X'*5)
                att_feature, = self.feature_fusion(limb_feat) 
                # print('att_feature', att_feature.shape)
                att_feature = att_feature.view(batch_size, -1) #(B,1*X'*5)
                ref_feature = torch.cat([ref_feature, att_feature], dim=1)


            smpl_output = self.regressor[rf_i](ref_feature, bbox, pred_pose, pred_shape, pred_cam, n_iter=1, J_regressor=J_regressor)
            out_list['smpl_out'].append(smpl_output)


        return out_list, vis_feat_list

def knowledge_net(smpl_mean_params, pretrained=True):
    """ Constructs an model with ResNet50 backbone.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = PAMA(smpl_mean_params, pretrained)
    return model