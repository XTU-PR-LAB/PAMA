# This script is borrowed and extended from https://github.com/nkolot/SPIN/blob/master/datasets/base_dataset.py

from __future__ import division

import cv2
import torch
import random
import numpy as np
from os.path import join
from torch.utils.data import Dataset
from torchvision.transforms import Normalize

from core import path_config, constants
from core.cfgs import cfg
from utils.imutils import crop, flip_img, flip_pose, flip_kp, transform, transform_pts, rot_aa, kp_transform, center_transform
from models.smpl import SMPL

from PIL import Image

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import pickle

class BaseDataset(Dataset):
    """
    Base Dataset Class - Handles data loading and augmentation.
    Able to handle heterogeneous datasets (different annotations available for different datasets).
    You need to update the path to each dataset in utils/path_config.py.
    """

    def __init__(self, options, dataset, ignore_3d=False, use_augmentation=True, is_train=True):
        super().__init__()
        self.dataset = dataset
        self.is_train = is_train
        self.options = options
        self.img_dir = path_config.DATASET_FOLDERS[dataset]  #/opt/data/private/datasets/Datasets/coco
        self.normalize_img = Normalize(mean=constants.IMG_NORM_MEAN, std=constants.IMG_NORM_STD)

        if not is_train and dataset == 'h36m-p2' and options.eval_pve:
            self.data = np.load(path_config.DATASET_FILES[is_train]['h36m-p2-mosh'], allow_pickle=True)
        else:
            self.data = np.load(path_config.DATASET_FILES[is_train][dataset], allow_pickle=True)

        self.imgname = self.data['imgname']  #(N,)
        self.dataset_dict = {dataset: 0}

        logger.info('len of {}: {}'.format(self.dataset, len(self.imgname)))

        # Get paths to gt masks, if available
        try:
            self.maskname = self.data['maskname']
        except KeyError:
            pass
        try:
            self.partname = self.data['partname']
        except KeyError:
            pass

        # Bounding boxes are assumed to be in the center and scale format
        self.scale = self.data['scale']  #(N,), corresponding to the height of bounding box
        self.center = self.data['center']  #(N,2)
        
        # If False, do not do augmentation
        self.use_augmentation = use_augmentation
        
        # Get gt SMPL parameters, if available
        try:
            self.pose = self.data['pose'].astype(np.float) # (N, 72)
            self.betas = self.data['shape'].astype(np.float) # (N, 10)

            ################# generate final_fits file in case of it is missing #################
            # import os
            # params_ = np.concatenate((self.pose, self.betas), axis=-1)
            # out_file_fit = os.path.join('data/final_fits', self.dataset)
            # if not os.path.exists('data/final_fits'):
            #     os.makedirs('data/final_fits')
            # np.save(out_file_fit, params_)
            # raise ValueError('Please copy {}.npy file to data/final_fits, and delete this code block.'.format(self.dataset))
            ########################################################################

            if 'has_smpl' in self.data:
                self.has_smpl = self.data['has_smpl']
            else:
                self.has_smpl = np.ones(len(self.imgname), dtype=np.float32)  #(N,)
        except KeyError:
            self.has_smpl = np.zeros(len(self.imgname), dtype=np.float32)
        if ignore_3d:
            self.has_smpl = np.zeros(len(self.imgname), dtype=np.float32)

        # Get SMPL 2D keypoints
        try:
            self.smpl_2dkps = self.data['smpl_2dkps']
            self.has_smpl_2dkps = 1
        except KeyError:
            self.has_smpl_2dkps = 0  #here

        try:
            self.focal_length = self.data['focal_l']
            self.has_focal_l = 1
        except KeyError:
            self.has_focal_l = 0

        # Get gt 3D pose, if available
        try:
            self.pose_3d = self.data['S']
            self.has_pose_3d = 1
        except KeyError:
            self.has_pose_3d = 0  #here
        if ignore_3d:
            self.has_pose_3d = 0
        
        # Get 2D keypoints
        try:
            keypoints_gt = self.data['part']  #(N,24,3)
        except KeyError:
            keypoints_gt = np.zeros((len(self.imgname), 24, 3))
        try:
            keypoints_openpose = self.data['openpose']
        except KeyError:
            keypoints_openpose = np.zeros((len(self.imgname), 25, 3))  #(N,25,3)
        self.keypoints = np.concatenate([keypoints_openpose, keypoints_gt], axis=1)  #(N,49,3)---49 is related to CONSTANT.JOINTMAP?

        # Get gender data, if available
        try:
            gender = self.data['gender']
            self.gender = np.array([0 if str(g) == 'm' else 1 for g in gender]).astype(np.int32)
        except KeyError:
            self.gender = -1*np.ones(len(self.imgname)).astype(np.int32)  #(N,)
        
        self.length = self.scale.shape[0]  #N

        self.smpl = SMPL(path_config.SMPL_MODEL_DIR,
                    batch_size=cfg.TRAIN.BATCH_SIZE,
                    create_transl=False)  #get data from /data/smpl/SMPL_NEUTRAL.pkl
        
        self.faces = self.smpl.faces  #(13776,3)


    def augm_params(self):
        """Get augmentation parameters."""
        flip = 0            # flipping
        pn = np.ones(3)  # per channel pixel-noise
        rot = 0            # rotation
        sc = 1            # scaling

        if self.is_train:
            # We flip with probability 1/2
            if np.random.uniform() <= 0.5:
                flip = 1
            
            # Each channel is multiplied with a number 
            # in the area [1-opt.noiseFactor,1+opt.noiseFactor]
            pn = np.random.uniform(1-self.options.noise_factor, 1+self.options.noise_factor, 3)
            
            # The rotation is a number in the area [-2*rotFactor, 2*rotFactor]
            rot = min(2*self.options.rot_factor,
                    max(-2*self.options.rot_factor, np.random.randn()*self.options.rot_factor))
            
            # The scale is multiplied with a number
            # in the area [1-scaleFactor,1+scaleFactor]
            sc = min(1+self.options.scale_factor,
                    max(1-self.options.scale_factor, np.random.randn()*self.options.scale_factor+1))
            # but it is zero with probability 3/5
            if np.random.uniform() <= 0.6:
                rot = 0
        
        return flip, pn, rot, sc
    def crop_cliff(self, img, center, scale, res):
        """
        Crop image according to the supplied bounding box.
        res: [rows, cols]
        """
        # Upper left point
        ul = np.array(transform([1, 1], center, scale, res, invert=1)) - 1
        # Bottom right point
        br = np.array(transform([res[1] + 1, res[0] + 1], center, scale, res, invert=1)) - 1

        # Padding so that when rotated proper amount of context is included
        pad = int(np.linalg.norm(br - ul) / 2 - float(br[1] - ul[1]) / 2)

        new_shape = [br[1] - ul[1], br[0] - ul[0]]
        if len(img.shape) > 2:
            new_shape += [img.shape[2]]
        new_img = np.zeros(new_shape, dtype=np.float32)

        # Range to fill new array
        new_x = max(0, -ul[0]), min(br[0], len(img[0])) - ul[0]
        new_y = max(0, -ul[1]), min(br[1], len(img)) - ul[1]
        # Range to sample from original image
        old_x = max(0, ul[0]), min(len(img[0]), br[0])
        old_y = max(0, ul[1]), min(len(img), br[1])
        try:
            new_img[new_y[0]:new_y[1], new_x[0]:new_x[1]] = img[old_y[0]:old_y[1], old_x[0]:old_x[1]]
        except Exception as e:
            print(e)

        new_img = cv2.resize(new_img, (res[1], res[0]))  # (cols, rows)

        return new_img, ul, br
    
    def rgb_processing(self, rgb_img, center, scale, rot, flip, pn):
        """Process rgb image and do augmentation."""
        rgb_img = crop(rgb_img, center, scale, 
                      [constants.IMG_RES, constants.IMG_RES], rot=rot)  #(224,224,3)
        # flip the image 
        if flip:
            rgb_img = flip_img(rgb_img)
        # in the rgb image we add pixel noise in a channel-wise manner
        rgb_img[:,:,0] = np.minimum(255.0, np.maximum(0.0, rgb_img[:,:,0]*pn[0]))
        rgb_img[:,:,1] = np.minimum(255.0, np.maximum(0.0, rgb_img[:,:,1]*pn[1]))
        rgb_img[:,:,2] = np.minimum(255.0, np.maximum(0.0, rgb_img[:,:,2]*pn[2]))
        # (3,224,224),float,[0,1]
        rgb_img = np.transpose(rgb_img.astype('float32'), (2,0,1)) / 255.0
        return rgb_img
    
    def process_image(self, orig_img_rgb, center, scale, rot, flip, pn):
        """
        Read image, do preprocessing and possibly crop it according to the bounding box.
        If there are bounding box annotations, use them to crop the image.
        If no bounding box is specified but openpose detections are available, use them to get the bounding box.
        """

        # img, ul, br = self.crop_cliff(orig_img_rgb, center, scale, [constants.IMG_H, constants.IMG_W])  #(256,192,3)
        img, ul, br = self.crop_cliff(orig_img_rgb, center, scale, [constants.IMG_RES, constants.IMG_RES])  #(224,224,3)
        # crop_img = img.copy()
        # # flip the image 
        # if flip:
        #     img = flip_img(img)

        img = img / 255.
        mean = np.array(constants.IMG_NORM_MEAN, dtype=np.float32)
        std = np.array(constants.IMG_NORM_STD, dtype=np.float32)
        norm_img = (img - mean) / std
        norm_img = np.transpose(norm_img, (2, 0, 1))
        return norm_img


    def j2d_processing(self, kp, center, scale, r, f, is_smpl=False):
        """Process gt 2D keypoints and apply all augmentation transforms."""
        nparts = kp.shape[0]
        for i in range(nparts):
            kp[i,0:2] = transform(kp[i,0:2]+1, center, scale, 
                                  [constants.IMG_RES, constants.IMG_RES], rot=r)
        # convert to normalized coordinates
        kp[:,:-1] = 2.*kp[:,:-1] / constants.IMG_RES - 1.
        # flip the x coordinates
        if f:
             kp = flip_kp(kp, is_smpl)
        kp = kp.astype('float32')
        return kp

    def orig_2d_processing(self, kp, scale, orig_shape, r, f, is_smpl=False):
        """Process gt 2D keypoints and apply all augmentation transforms."""
        nparts = kp.shape[0]
        h, w = orig_shape[:2]
        center = (w / 2, h / 2)
        for i in range(nparts):
            kp[i,0:2] = kp_transform(kp[i,0:2]+1, center, scale, orig_shape, rot=r)

        # convert to normalized coordinates
        kp[:, 0] = 2. * kp[:, 0] / orig_shape[1] - 1.
        kp[:, 1] = 2. * kp[:, 1] / orig_shape[0] - 1.
        # flip the x coordinates
        if f:
             kp = flip_kp(kp, is_smpl)
        kp = kp.astype('float32')
        return kp
    
    def j3d_processing(self, S, r, f, is_smpl=False):
        """Process gt 3D keypoints and apply all augmentation transforms."""
        # in-plane rotation
        rot_mat = np.eye(3)
        if not r == 0:
            rot_rad = -r * np.pi / 180
            sn,cs = np.sin(rot_rad), np.cos(rot_rad)
            rot_mat[0,:2] = [cs, -sn]
            rot_mat[1,:2] = [sn, cs]
        S[:, :-1] = np.einsum('ij,kj->ki', rot_mat, S[:, :-1]) 
        # flip the x coordinates
        if f:
            S = flip_kp(S, is_smpl)
        S = S.astype('float32')
        return S

    def pose_processing(self, pose, r, f):
        """Process SMPL theta parameters  and apply all augmentation transforms."""
        # rotation or the pose parameters
        pose[:3] = rot_aa(pose[:3], r)
        # flip the pose parameters
        if f:
            pose = flip_pose(pose)
        # (72),float
        pose = pose.astype('float32')
        return pose

    def estimate_focal_length(self, img_h, img_w):
        return (img_w * img_w + img_h * img_h) ** 0.5  # fov: 55 degree

    def __getitem__(self, index):
        item = {}
        scale = self.scale[index].copy()
        center = self.center[index].copy()

        # Get augmentation parameters
        flip, pn, rot, sc = self.augm_params()

        # Load image
        imgname = join(self.img_dir, self.imgname[index])
        try:
            img = cv2.imread(imgname)[:,:,::-1].copy().astype(np.float32)  #BGR->RGB
            orig_shape = np.array(img.shape)[:2]
        except:
            logger.error('fail while loading {}'.format(imgname))

        kp_is_smpl = True if self.dataset == 'surreal' else False  #False

        # Get SMPL parameters, if available
        if self.has_smpl[index]:
            pose = self.pose[index].copy()  #(72,)
            betas = self.betas[index].copy()  #(10,)
            pose = self.pose_processing(pose, rot, flip)
        else:
            pose = np.zeros(72)
            betas = np.zeros(10)

        
        # Process image
        img = self.rgb_processing(img, center, sc*scale, rot, flip, pn) 
        img = torch.from_numpy(img).float()
        # Store image before normalization to use it in visualization
        item['img'] = self.normalize_img(img)
        # item['img'] = img
        item['pose'] = torch.from_numpy(pose).float()
        item['betas'] = torch.from_numpy(betas).float()
        item['imgname'] = imgname

        # if self.has_smpl[index]:
        #     betas_th = item['betas'].unsqueeze(0)
        #     pose_th = item['pose'].unsqueeze(0)
        #     smpl_out = self.smpl(betas=betas_th, body_pose=pose_th[:, 3:], global_orient=pose_th[:, :3],
        #                             pose2rot=True)
        #     verts = smpl_out.vertices[0]
        #     item['verts'] = verts
        # else:
        #     item['verts'] = torch.zeros(6890, 3, dtype=torch.float32)
        
        # Get 2D SMPL joints
        if self.has_smpl_2dkps:
            smpl_2dkps = self.smpl_2dkps[index].copy()
            smpl_2dkps = self.j2d_processing(smpl_2dkps, center, sc * scale, rot, f=0)
            smpl_2dkps[smpl_2dkps[:, 2] == 0] = 0
            if flip:
                smpl_2dkps = smpl_2dkps[constants.SMPL_JOINTS_FLIP_PERM]
                smpl_2dkps[:, 0] = - smpl_2dkps[:, 0]
            item['smpl_2dkps'] = torch.from_numpy(smpl_2dkps).float()
        else:
            item['smpl_2dkps'] = torch.zeros(24, 3, dtype=torch.float32)  #here

        # Get 3D pose, if available
        if self.has_pose_3d:
            S = self.pose_3d[index].copy()
            item['pose_3d'] = torch.from_numpy(self.j3d_processing(S, rot, flip, kp_is_smpl)).float()
        else:
            item['pose_3d'] = torch.zeros(24,4, dtype=torch.float32)  #here,(N,24,4)

        # Get 2D keypoints and apply augmentation transforms
        keypoints = self.keypoints[index].copy()
        item['keypoints'] = torch.from_numpy(self.j2d_processing(keypoints, center, sc*scale, rot, flip, kp_is_smpl)).float()  #(N,49,3)

        item['has_smpl'] = self.has_smpl[index]  #shape:(N,),data:[1.,1.,1.,...]
        item['has_pose_3d'] = self.has_pose_3d  #0
        # item['scale'] = float(sc * scale)  #zheli
        item['center'] = center.astype(np.float32)
        item['orig_shape'] = orig_shape
        item['is_flipped'] = flip
        item['rot_angle'] = np.float32(rot)
        item['gender'] = self.gender[index]
        item['sample_index'] = index
        item['dataset_name'] = self.dataset
        
        img_h, img_w = orig_shape
        if self.has_focal_l:
            focal_length = np.float32(self.focal_length[index].copy())           #numpy.float32
        else:
            focal_length = np.float32(self.estimate_focal_length(img_h, img_w))  #numpy.float64
        item['img_h'] = img_h
        item['img_w'] = img_w
        cx, cy = center.astype(np.float32)
        # item['scale'] = float(sc * scale)
        s = np.float32(sc * scale)

        bbox_info = np.stack([cx - img_w / 2., cy - img_h / 2., s*200])
        bbox_info[:2] = bbox_info[:2] / focal_length * 2.8  # [-1, 1]
        bbox_info[2] = (bbox_info[2] - 0.24 * focal_length) / (0.06 * focal_length)  # [-1, 1]

        item['bbox_info'] = np.float32(bbox_info)
        item['focal_length'] = focal_length

        item['scale'] = s
        keypoints_orig = self.keypoints[index].copy()
        item['keypoints_orig'] = torch.from_numpy(self.orig_2d_processing(keypoints_orig, sc*scale, orig_shape, rot, flip, kp_is_smpl)).float()
        
        

        try:
            item['maskname'] = self.maskname[index]
        except AttributeError:
            item['maskname'] = ''
        try:
            item['partname'] = self.partname[index]
        except AttributeError:
            item['partname'] = ''

        return item

    def __len__(self):
        return len(self.imgname)