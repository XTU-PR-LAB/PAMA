# -*- coding: utf-8 -*-
import numpy as np
import torch
import cv2


def norm(n):
    dim = len(n.shape) # len(n.get_shape().as_list())
    return torch.sqrt(torch.sum(torch.square(n), dim-1)) 

def normalize(n):
    return n/norm(n).unsqueeze(-1) # n/tf.expand_dims(norm(n), [-1])

kEpsilon = 1e-8
def get_visibility_raycast(tf_v, f, reduce_step=4):
    # tf_v: batch_size x mesh_num x 3
    # f: num_faces x 3
    batch_size, mesh_num, _  = tf_v.shape 
    num_faces = f.shape[0] 

    #f = torch.from_numpy(f).int() # f = tf.constant(f, tf.int32)
    f = f / 1.0
    f = torch.LongTensor(f)
    #f = torch.cat(f)  #TypeError: expected Tensor as element 0 in argument 0, but got numpy.ndarray
    #f = torch.stack(f)  #TypeError: expected Tensor as element 0 in argument 0, but got numpy.ndarray
    idx0 = f[:, 0]
    idx1 = f[:, 1]
    idx2 = f[:, 2]
    idx0 = torch.reshape(torch.arange(0, batch_size), [-1, 1]).repeat(1, num_faces) * mesh_num + torch.reshape(idx0, [1, -1])  #(b_s,num_faces)*mesh_num+(1,13776)->(b_s,num_faces)
    idx1 = torch.reshape(torch.arange(0, batch_size), [-1, 1]).repeat(1, num_faces) * mesh_num + torch.reshape(idx1, [1, -1])
    idx2 = torch.reshape(torch.arange(0, batch_size), [-1, 1]).repeat(1, num_faces) * mesh_num + torch.reshape(idx2, [1, -1])
    
    # batch x #face(13776) x 3 
    temp_tf_v = torch.reshape(tf_v, [-1, 3])  #(b_s*mesh_num,3),即(b_s*6890,3)
    v0 = temp_tf_v[idx0]  #(b_s,num_faces,3),v0表示所有batch下mesh面片中第一个顶点的x,y,z值
    v1 = temp_tf_v[idx1]  #(b_s,num_faces,3),v0表示所有batch下mesh面片中第二个顶点的x,y,z值
    v2 = temp_tf_v[idx2]  #(b_s,num_faces,3),v0表示所有batch下mesh面片中第三个顶点的x,y,z值
    v0v1 = v1 - v0
    v0v2 = v2 - v0

    reduce_idx = torch.arange(0, num_faces, reduce_step).cuda() 
    num_faces_r = reduce_idx.shape[0] 
    # 5 x 4592 x 3
    v0_reshape = torch.reshape(v0, [-1, 3])  #(b_s*num_faces,3)
    v_idx = torch.reshape(torch.arange(0, batch_size), [-1 ,1]).repeat(1, num_faces_r).cuda() * mesh_num + torch.reshape(reduce_idx, [1, -1])
    v0_r = v0_reshape[v_idx]  #(b_s,num_faces_r,3)
    v1_reshape = torch.reshape(v1, [-1, 3])
    v1_r = v1_reshape[v_idx]
    v2_reshape = torch.reshape(v2, [-1, 3])
    v2_r = v2_reshape[v_idx]    
   
    face_center = (v0_r + v1_r + v2_r)/3.0
    # tf_project: 5 x 4592 x 2
    tf_project =  torch.div(face_center[:, :, 0:2], face_center[:, :, 2].view(batch_size, -1, 1)) #
    # dir_ 5 x 4592 x 3
    dir_ = normalize(torch.cat([tf_project, torch.ones([batch_size, num_faces_r, 1]).cuda()], 2)) # normalize(tf.concat([tf_project, tf.ones([batch_size, num_faces_r, 1])], 2))

    # N: 5 x 13776 x 3 
    N = torch.cross(v0v1, v0v2) 
    NdotRayDirection = torch.matmul(N, dir_.permute(0, 2, 1).contiguous())  #(5,13776,4592)
    isNotParallel = torch.where(torch.lt(torch.abs(NdotRayDirection), kEpsilon), torch.zeros_like(NdotRayDirection), torch.ones_like(NdotRayDirection))


    # find P
    d = torch.mul(N, v0).sum(dim=2) 
    t = d.unsqueeze(2)/NdotRayDirection 
    isNotBehind = torch.where(torch.lt(t, 0), torch.zeros_like(NdotRayDirection), torch.ones_like(NdotRayDirection))
    # p: batch_size x 13776 x 4592 x 3
    P = t.unsqueeze(3) * dir_.unsqueeze(1) 

    # batch x #face(13776) x 1 x 3 
    edge0 = (v1 - v0).unsqueeze(2).repeat(1,1, num_faces_r, 1)
    # vp0: batch_size x 13776 x 4592 x 3
    vp0 = P - v0.unsqueeze(2) # P - tf.expand_dims(v0, 2)
    C = torch.cross(edge0, vp0) 
    inner = torch.mul(N.unsqueeze(2), C).sum(3)  
    isInTri0= torch.where(torch.lt(inner, 0), torch.zeros_like(NdotRayDirection), torch.ones_like(NdotRayDirection))
    edge1 = (v2 - v1).unsqueeze(2).repeat(1,1, num_faces_r, 1) 
    vp1 = P - v1.unsqueeze(2) # P - tf.expand_dims(v1, 2)
    C = torch.cross(edge1, vp1) 
    inner = torch.mul(N.unsqueeze(2), C).sum(3) 
    isInTri1= torch.where(torch.lt(inner, 0), torch.zeros_like(NdotRayDirection), torch.ones_like(NdotRayDirection))
    edge2 = (v0 - v2).unsqueeze(2).repeat(1,1, num_faces_r, 1) 
    # vp0: batch_size x 13776 x 4592 x 3
    vp2 = P - v2.unsqueeze(2) # P - tf.expand_dims(v2, 2)
    C = torch.cross(edge2, vp2) 
    inner = torch.mul(N.unsqueeze(2), C).sum(3) 
    isInTri2= torch.where(torch.lt(inner, 0), torch.zeros_like(NdotRayDirection), torch.ones_like(NdotRayDirection))
    # vp0: batch_size x 13776 x 4592
    final_decision = isNotParallel * isNotBehind *  isInTri0 * isInTri1 * isInTri2
    dist = torch.where(torch.greater(final_decision, 0.5), t, 1.0e12 * torch.ones_like(final_decision))
    select_faces = torch.argmin(dist, dim=1) 
    out = torch.eq(reduce_idx, select_faces.int()).unsqueeze(2).repeat(1,1,3)
    f_reduce_idx = f[reduce_idx].unsqueeze(0).repeat(batch_size, 1, 1).cuda()
    collected_face_idx = torch.where(out, f_reduce_idx, (mesh_num) * torch.ones_like(f_reduce_idx).cuda())

    onehot = []
    for id in range(batch_size):
        arg_min_list = torch.unique(torch.reshape(collected_face_idx[id], [-1]))  #(X,Y)
        arg_min_list = torch.reshape(arg_min_list, [-1, 1]).long().cuda()         #(X*Y,1)    
        data_zeros = torch.zeros(arg_min_list.shape[0], mesh_num + 1).cuda()      #(X*Y,V+1)
        oneh = data_zeros.scatter_(1, arg_min_list, 1)
        # print(oneh)
        oneh = oneh.sum(dim=0)
        # temp = oneh.sum()
        onehot.append(oneh)

    visibility = torch.stack(onehot)[:, :mesh_num] 

    return visibility, face_center


def judge_vert_vis(cfg, verts, cam, faces):
    transposed_x = cam[:, 0].view(-1,1)
    transposed_y = cam[:, 1].view(-1,1)
    transposed_z = cam[:, 2].view(-1,1)
    transposed_cams = torch.cat([transposed_x, transposed_y, transposed_z], -1)

    verts = verts * 1000

    _, n_vertex, _ = verts.shape
    translation = transposed_cams.unsqueeze(1)
    translation = translation.repeat(1, n_vertex, 1)

    temp_verts = verts + translation

    visibility, arg_min = get_visibility_raycast(temp_verts, faces, cfg.SMPL_MODEL_DOWNSIZE_SCALE)

    return visibility, arg_min