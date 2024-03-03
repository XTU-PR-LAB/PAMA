"""
This file contains functions that are used to perform data augmentation.
"""
import cv2
import torch
import numpy as np
import skimage.transform
from PIL import Image

from core import constants

import math

def get_transform(center, scale, res, rot=0):
    """Generate transformation matrix."""
    h = 200 * scale
    t = np.zeros((3, 3))
    t[0, 0] = float(res[1]) / h
    t[1, 1] = float(res[0]) / h
    t[0, 2] = res[1] * (-float(center[0]) / h + .5)
    t[1, 2] = res[0] * (-float(center[1]) / h + .5)
    t[2, 2] = 1
    if not rot == 0:
        rot = -rot # To match direction of rotation from cropping
        rot_mat = np.zeros((3,3))
        rot_rad = rot * np.pi / 180
        sn,cs = np.sin(rot_rad), np.cos(rot_rad)
        rot_mat[0,:2] = [cs, -sn]
        rot_mat[1,:2] = [sn, cs]
        rot_mat[2,2] = 1
        # Need to rotate around center
        t_mat = np.eye(3)
        t_mat[0,2] = -res[1]/2
        t_mat[1,2] = -res[0]/2
        t_inv = t_mat.copy()
        t_inv[:2,2] *= -1
        t = np.dot(t_inv,np.dot(rot_mat,np.dot(t_mat,t)))
    return t

def transform(pt, center, scale, res, invert=0, rot=0):
    """Transform pixel location to different reference."""
    t = get_transform(center, scale, res, rot=rot)
    if invert:
        t = np.linalg.inv(t)
    new_pt = np.array([pt[0] - 1, pt[1] - 1, 1.]).T
    new_pt = np.dot(t, new_pt)
    return new_pt[:2].astype(int) + 1


def kp_transform(pt, center, scale, res, invert=0, rot=0):
    """Transform pixel location to different reference."""
    t = kp_get_transform(center, scale, res, rot=rot)
    if invert:
        t = np.linalg.inv(t)
    new_pt = np.array([pt[0] - 1, pt[1] - 1, 1.]).T
    new_pt = np.dot(t, new_pt)
    return new_pt[:2].astype(int) + 1

def kp_get_transform(center, scale, res, rot=0):
    """Generate transformation matrix."""
    # Compute the transformation matrix for scaling and rotation
    t = np.zeros((3, 3))
    t[0, 0] = t[1, 1] = scale
    t[0, 2] = center[0] * (1 - scale)
    t[1, 2] = center[1] * (1 - scale)
    t[2, 2] = 1
    
    if rot != 0:
        # Compute the rotation matrix
        rot_rad = np.deg2rad(rot)
        sn, cs = np.sin(rot_rad), np.cos(rot_rad)
        rot_mat = np.array([[cs, -sn, 0], [sn, cs, 0], [0, 0, 1]])
        
        # Combine scaling and rotation matrices
        t = np.dot(rot_mat, t)
    
    return t

"""def full_transform(pt, res, invert=0, rot=0):
    t = full_get_transform(res, rot=rot)
    if invert:
        t = np.linalg.inv(t)
    new_pt = np.array([pt[0] - 1, pt[1] - 1, 1.]).T
    new_pt = np.dot(t, new_pt)
    return new_pt[:2].astype(int) + 1
def full_get_transform(res, rot=0):
    t = np.eye(3)
    if not rot == 0:
        rot = -rot # To match direction of rotation from cropping
        rot_mat = np.zeros((3,3))
        rot_rad = rot * np.pi / 180
        sn,cs = np.sin(rot_rad), np.cos(rot_rad)
        rot_mat[0,:2] = [cs, -sn]
        rot_mat[1,:2] = [sn, cs]
        rot_mat[2,2] = 1
        # Need to rotate around center
        t_mat = np.eye(3)
        t_mat[0,2] = -res[1]/2
        t_mat[1,2] = -res[0]/2
        t_inv = t_mat.copy()
        t_inv[:2,2] *= -1
        t = np.dot(t_inv,np.dot(rot_mat,np.dot(t_mat,t)))
    return t"""

'''def center_get_transform(scale, res, rot=0):
    """Generate transformation matrix."""
    h = 200 * scale
    t = np.eye(3)
    t[0, 0] = float(res[1]) / h
    t[1, 1] = float(res[0]) / h
    if not rot == 0:
        rot = -rot # To match direction of rotation from cropping
        rot_mat = np.zeros((3,3))
        rot_rad = rot * np.pi / 180
        sn,cs = np.sin(rot_rad), np.cos(rot_rad)
        rot_mat[0,:2] = [cs, -sn]
        rot_mat[1,:2] = [sn, cs]
        rot_mat[2,2] = 1
        # Need to rotate around center
        t_mat = np.eye(3)
        t_mat[0,2] = -res[1]/2
        t_mat[1,2] = -res[0]/2
        t_inv = t_mat.copy()
        t_inv[:2,2] *= -1
        t = np.dot(t_inv,np.dot(rot_mat,np.dot(t_mat,t)))
    return t'''
def center_get_transform(res, rot=0):
    """Generate transformation matrix."""
    t = np.eye(3)
    if not rot == 0:
        rot = -rot # To match direction of rotation from cropping
        rot_mat = np.zeros((3,3))
        rot_rad = rot * np.pi / 180
        sn,cs = np.sin(rot_rad), np.cos(rot_rad)
        rot_mat[0,:2] = [cs, -sn]
        rot_mat[1,:2] = [sn, cs]
        rot_mat[2,2] = 1
        # Need to rotate around center
        t_mat = np.eye(3)
        t_mat[0,2] = -res[1]/2
        t_mat[1,2] = -res[0]/2
        t_inv = t_mat.copy()
        t_inv[:2,2] *= -1
        t = np.dot(t_inv,np.dot(rot_mat,np.dot(t_mat,t)))
    return t

'''def center_transform(pt, scale, res, invert=0, rot=0):
    """Transform pixel location to different reference."""
    t = center_get_transform(scale, res, rot=rot)
    if invert:
        t = np.linalg.inv(t)
    new_pt = np.array([pt[0] - 1, pt[1] - 1, 1.]).T
    new_pt = np.dot(t, new_pt)
    return new_pt[:2].astype(int) + 1'''
def center_transform(pt, res, invert=0, rot=0):
    """Transform pixel location to different reference."""
    t = center_get_transform(res, rot=rot)
    if invert:
        t = np.linalg.inv(t)
    new_pt = np.array([pt[0] - 1, pt[1] - 1, 1.]).T
    new_pt = np.dot(t, new_pt)
    return new_pt[:2].astype(int) + 1

def transform_pts(coords, center, scale, res, invert=0, rot=0):
    """Transform coordinates (N x 2) to different reference."""
    new_coords = coords.copy()
    for p in range(coords.shape[0]):
        new_coords[p, 0:2] = transform(coords[p, 0:2], center, scale, res, invert, rot)
    return new_coords

def crop(img, center, scale, res, rot=0):
    """Crop image according to the supplied bounding box."""
    # Upper left point
    ul = np.array(transform([1, 1], center, scale, res, invert=1))-1
    # Bottom right point
    br = np.array(transform([res[0]+1, 
                             res[1]+1], center, scale, res, invert=1))-1
    
    # Padding so that when rotated proper amount of context is included
    pad = int(np.linalg.norm(br - ul) / 2 - float(br[1] - ul[1]) / 2)
    if not rot == 0:
        ul -= pad
        br += pad

    new_shape = [br[1] - ul[1], br[0] - ul[0]]
    if len(img.shape) > 2:
        new_shape += [img.shape[2]]
    new_img = np.zeros(new_shape)

    # Range to fill new array
    new_x = max(0, -ul[0]), min(br[0], len(img[0])) - ul[0]
    new_y = max(0, -ul[1]), min(br[1], len(img)) - ul[1]
    # Range to sample from original image
    old_x = max(0, ul[0]), min(len(img[0]), br[0])
    old_y = max(0, ul[1]), min(len(img), br[1])
    new_img[new_y[0]:new_y[1], new_x[0]:new_x[1]] = img[old_y[0]:old_y[1], old_x[0]:old_x[1]]

    if not rot == 0:
        # Remove padding
        new_img = skimage.transform.rotate(new_img, rot).astype(np.uint8)
        new_img = new_img[pad:-pad, pad:-pad]

    new_img = np.array(Image.fromarray(new_img.astype(np.uint8)).resize(res))
    
    return new_img



def uncrop(img, center, scale, orig_shape, rot=0, is_rgb=True):
    """'Undo' the image cropping/resizing.
    This function is used when evaluating mask/part segmentation.
    """
    res = img.shape[:2]
    # Upper left point
    ul = np.array(transform([1, 1], center, scale, res, invert=1))-1
    # Bottom right point
    br = np.array(transform([res[0]+1,res[1]+1], center, scale, res, invert=1))-1
    # size of cropped image
    # crop_shape = [br[1] - ul[1], br[0] - ul[0]] 
    crop_shape = [br[0] - ul[0], br[1] - ul[1]]

    new_shape = [br[1] - ul[1], br[0] - ul[0]]
    if len(img.shape) > 2:
        new_shape += [img.shape[2]]
    new_img = np.zeros(orig_shape, dtype=np.uint8) #(187.184)
    
    # Range to fill new array
    new_x = max(0, -ul[0]), min(br[0], orig_shape[1]) - ul[0]
    new_y = max(0, -ul[1]), min(br[1], orig_shape[0]) - ul[1]
    
    # Range to sample from original image
    old_x = max(0, ul[0]), min(orig_shape[1], br[0])
    old_y = max(0, ul[1]), min(orig_shape[0], br[1])

    # print('crop_shape',crop_shape) #(125,124)
    img = np.array(Image.fromarray(img.astype(np.uint8)).resize(crop_shape)) #(124,125)

    new_img[old_y[0]:old_y[1], old_x[0]:old_x[1]] = img[new_y[0]:new_y[1], new_x[0]:new_x[1]]
    
    return new_img

def rot_aa(aa, rot):
    """Rotate axis angle parameters."""
    # pose parameters
    R = np.array([[np.cos(np.deg2rad(-rot)), -np.sin(np.deg2rad(-rot)), 0],
                  [np.sin(np.deg2rad(-rot)), np.cos(np.deg2rad(-rot)), 0],
                  [0, 0, 1]])
    # find the rotation of the body in camera frame
    per_rdg, _ = cv2.Rodrigues(aa)
    # apply the global rotation to the global orientation
    resrot, _ = cv2.Rodrigues(np.dot(R,per_rdg))
    aa = (resrot.T)[0]
    return aa

def flip_img(img):
    """Flip rgb images or masks.
    channels come last, e.g. (256,256,3).
    """
    img = np.fliplr(img)
    return img

def flip_kp(kp, is_smpl=False):
    """Flip keypoints."""
    flipped_parts = []
    if len(kp) == 24:
        if is_smpl:
            flipped_parts = constants.SMPL_JOINTS_FLIP_PERM
        else:
            flipped_parts = constants.J24_FLIP_PERM
    elif len(kp) == 49:
        if is_smpl:
            flipped_parts = constants.SMPL_J49_FLIP_PERM
        else:
            flipped_parts = constants.J49_FLIP_PERM
    kp = kp[flipped_parts]
    kp[:,0] = - kp[:,0]
    return kp

def flip_kp_mat(kp, is_smpl=False):
    """Flip keypoints."""
    flipped_parts = []
    if kp.shape[1] == 24:
        if is_smpl:
            flipped_parts = constants.SMPL_JOINTS_FLIP_PERM
        else:
            flipped_parts = constants.J24_FLIP_PERM
    elif kp.shape[1] == 49:
        if is_smpl:
            flipped_parts = constants.SMPL_J49_FLIP_PERM
        else:
            flipped_parts = constants.J49_FLIP_PERM
    kp = kp[:,flipped_parts,:]
    kp[:,:,0] = - kp[:,:,0]
    return kp

def flip_pose(pose):
    """Flip pose.
    The flipping is based on SMPL parameters.
    """
    flipped_parts = constants.SMPL_POSE_FLIP_PERM
    pose = pose[flipped_parts]
    # we also negate the second and the third dimension of the axis-angle
    pose[1::3] = -pose[1::3]
    pose[2::3] = -pose[2::3]
    return pose

def normalize_2d_kp(kp_2d, crop_size=224, inv=False):
    # Normalize keypoints between -1, 1
    if not inv:
        ratio = 1.0 / crop_size
        kp_2d = 2.0 * kp_2d * ratio - 1.0
    else:
        ratio = 1.0 / crop_size
        kp_2d = (kp_2d + 1.0)/(2*ratio)

    return kp_2d

def generate_heatmap(joints, heatmap_size, sigma=1, joints_vis=None):
    '''
    param joints:  [num_joints, 3]
    param joints_vis: [num_joints, 3]
    return: target, target_weight(1: visible, 0: invisible)
    '''
    num_joints = joints.shape[0]
    device = joints.device
    cur_device = torch.device(device.type, device.index)
    if not hasattr(heatmap_size, '__len__'):
        # width  height
        heatmap_size = [heatmap_size, heatmap_size]
    assert len(heatmap_size) == 2
    target_weight = np.ones((num_joints, 1), dtype=np.float32)
    if joints_vis is not None:
        target_weight[:, 0] = joints_vis[:, 0]
    target = torch.zeros((num_joints,
                          heatmap_size[1],
                          heatmap_size[0]),
                         dtype=torch.float32,
                         device=cur_device)

    tmp_size = sigma * 3

    for joint_id in range(num_joints):
        mu_x = int(joints[joint_id][0] * heatmap_size[0] + 0.5)
        mu_y = int(joints[joint_id][1] * heatmap_size[1] + 0.5)
        # Check that any part of the gaussian is in-bounds
        ul = [int(mu_x - tmp_size), int(mu_y - tmp_size)]
        br = [int(mu_x + tmp_size + 1), int(mu_y + tmp_size + 1)]
        if ul[0] >= heatmap_size[0] or ul[1] >= heatmap_size[1] \
                or br[0] < 0 or br[1] < 0:
            # If not, just return the image as is
            target_weight[joint_id] = 0
            continue

        # # Generate gaussian
        size = 2 * tmp_size + 1
        # x = np.arange(0, size, 1, np.float32)
        # y = x[:, np.newaxis]
        # x0 = y0 = size // 2
        # # The gaussian is not normalized, we want the center value to equal 1
        # g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))
        # g = torch.from_numpy(g.astype(np.float32))

        x = torch.arange(0, size, dtype=torch.float32, device=cur_device)
        y = x.unsqueeze(-1)
        x0 = y0 = size // 2
        # The gaussian is not normalized, we want the center value to equal 1
        g = torch.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))

        # Usable gaussian range
        g_x = max(0, -ul[0]), min(br[0], heatmap_size[0]) - ul[0]
        g_y = max(0, -ul[1]), min(br[1], heatmap_size[1]) - ul[1]
        # Image range
        img_x = max(0, ul[0]), min(br[0], heatmap_size[0])
        img_y = max(0, ul[1]), min(br[1], heatmap_size[1])

        v = target_weight[joint_id]
        if v > 0.5:
            target[joint_id][img_y[0]:img_y[1], img_x[0]:img_x[1]] = \
                g[g_y[0]:g_y[1], g_x[0]:g_x[1]]

    return target, target_weight

def cam_crop2full(crop_cam, center, scale, full_img_shape, focal_length):
    """
    convert the camera parameters from the crop camera to the full camera
    :param crop_cam: shape=(N, 3) weak perspective camera in cropped img coordinates (s, tx, ty)
    :param center: shape=(N, 2) bbox coordinates (c_x, c_y)
    :param scale: shape=(N, 1) square bbox resolution  (b / 200)
    :param full_img_shape: shape=(N, 2) original image height and width
    :param focal_length: shape=(N,)
    :return:
    """
    img_h, img_w = full_img_shape[:, 0], full_img_shape[:, 1]
    cx, cy, b = center[:, 0], center[:, 1], scale * 200
    w_2, h_2 = img_w / 2., img_h / 2.
    bs = b * crop_cam[:, 0] + 1e-9
    tz = 2 * focal_length / bs
    tx = (2 * (cx - w_2) / bs) + crop_cam[:, 1]
    ty = (2 * (cy - h_2) / bs) + crop_cam[:, 2]
    full_cam = torch.stack([tx, ty, tz], dim=-1)
    return full_cam

def convert_crop_to_full_img_cam(crop_cam, center, scale, full_img_shape, focal_length):
    """
    convert the camera parameters from the crop camera to the full camera
    :param crop_cam: shape=(N, 3) weak perspective camera in cropped img coordinates (s, tx, ty)
    :param center: shape=(N, 2) bbox coordinates (c_x, c_y)
    :param scale: shape=(N, 1) square bbox resolution  (b / 200)
    :param full_img_shape: shape=(N, 2) original image height and width
    :param focal_length: shape=(N,)
    :return:
    """
    s, tx, ty = crop_cam[:,0], crop_cam[:,1], crop_cam[:,2]
    res = 224
    img_h, img_w = full_img_shape[:, 0], full_img_shape[:, 1]
    b = scale * 200
    r = b / res
    tz = 2 * focal_length / (r * res * s)
    
    cx = 2 * (center[:, 0] - img_w / 2.) / (b * s)
    cy = 2 * (center[:, 1] - img_h / 2.) / (b * s)
    full_cam = torch.stack([tx + cx, ty + cy, tz], dim=-1)
    return full_cam

def get_transform_t(center, scale, res, rot=0):
    """Generate transformation matrix."""
    h = 200 * scale
    t = torch.zeros((3, 3))
    t[0, 0] = float(res[1]) / h
    t[1, 1] = float(res[0]) / h
    t[0, 2] = res[1] * (-float(center[0]) / h + .5)
    t[1, 2] = res[0] * (-float(center[1]) / h + .5)
    t[2, 2] = 1
    if not rot == 0:
        rot = -rot # To match direction of rotation from cropping
        rot_mat = torch.zeros((3,3))
        rot_rad = rot * math.pi / 180
        # rot_rad = torch.tensor(rot_rad)
        sn,cs = torch.sin(rot_rad), torch.cos(rot_rad)
        rot_mat[0,:2] = torch.tensor([cs, -sn])
        rot_mat[1,:2] = torch.tensor([sn, cs])
        rot_mat[2,2] = 1
        # Need to rotate around center
        t_mat = torch.eye(3)
        t_mat[0,2] = -res[1]/2
        t_mat[1,2] = -res[0]/2
        t_inv = t_mat.clone()
        t_inv[:2,2] *= -1
        t = torch.matmul(t_inv,torch.matmul(rot_mat,torch.matmul(t_mat,t)))
    return t

def transform_t(pt, center, scale, res, invert=0, rot=0):
    """Transform pixel location to different reference."""
    t = get_transform_t(center, scale, res, rot=rot)
    if invert:
        t = torch.linalg.inv(t)
    new_pt = torch.tensor([pt[0] - 1, pt[1] - 1, 1.]).t()
    new_pt = torch.matmul(t, new_pt)
    return new_pt[:2].type(torch.int) + 1

def get_transform_mat(center, scale, res, rot=0):
    """Generate transformation matrix."""
    h = 200 * scale
    bs = center.shape[0]
    t = torch.zeros((bs, 3, 3))
    t[:, 0, 0] = float(res[1]) / h
    t[:, 1, 1] = float(res[0]) / h
    t[:, 0, 2] = res[1] * (-center[:,0] / h[:] + .5)
    t[:, 1, 2] = res[0] * (-center[:,1] / h[:] + .5)
    t[:, 2, 2] = 1
    new_t = t.clone()
    rot_mat = torch.zeros((bs,3,3))
    for i in range(bs):
        if not rot[i] == 0:
            rot[i] = -rot[i] # To match direction of rotation from cropping
            
            rot_rad = rot[i] * math.pi / 180
            # rot_rad = torch.tensor(rot_rad)
            sn,cs = torch.sin(rot_rad), torch.cos(rot_rad)
            rot_mat[i,0,:2] = torch.tensor([cs, -sn])
            rot_mat[i,1,:2] = torch.tensor([sn, cs])
            rot_mat[i,2,2] = 1
            # Need to rotate around center
            t_mat = torch.eye(3)
            t_mat[0,2] = -res[1]/2
            t_mat[1,2] = -res[0]/2
            t_inv = t_mat.clone()
            t_inv[:2,2] *= -1
            new_t[i,:] = torch.matmul(t_inv,torch.matmul(rot_mat[i,:],torch.matmul(t_mat,t[i,:])))
    
    return new_t

def transform_mat(pt, center, scale, res, invert=0, rot=0):
    """Transform pixel location to different reference."""
    t = get_transform_mat(center, scale, res, rot=rot)
    if invert:
        t = torch.linalg.inv(t)
    ph = torch.ones(pt.shape[0],pt.shape[1],1).cuda()
    new_pt = torch.transpose(torch.cat([(pt[:,:,0]-1).unsqueeze(2),(pt[:,:,1]-1).unsqueeze(2),ph],dim=2),1,2)
    new_pt = torch.transpose(torch.matmul(t.cuda(), new_pt),1,2)
    return new_pt[:,:,:2].type(torch.int) + 1

def j2d_processing(kp, center, scale, r, f, is_smpl=False):
    """Process gt 2D keypoints and apply all augmentation transforms."""
    nparts = kp.shape[1]
    '''for batch_size in range(kp.shape[0]):
        for i in range(nparts):
            kp[batch_size,i,0:2] = transform_t(kp[batch_size,i,0:2]+1, center[batch_size,:], scale[batch_size], 
                                    [constants.IMG_RES, constants.IMG_RES], rot=r[batch_size])'''
    
    kp[:,:,0:2] = transform_mat(kp[:,:,0:2]+1, center, scale, 
                            [constants.IMG_RES, constants.IMG_RES], rot=r)
    # convert to normalized coordinates
    kp[:,:,:-1] = 2.*kp[:,:,:-1] / constants.IMG_RES - 1.
    # flip the x coordinates
    for batch_size in range(kp.shape[0]):
        if f[batch_size]:
            kp[batch_size,:] = flip_kp(kp[batch_size,:], is_smpl)
    kp = kp.type(torch.float32)
    return kp

