import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from sklearn.cluster import KMeans


def init_map_blocks(model_r, block_num, v_range, r_range):
    masks = cluster_map(model_r, block_num, 0.5)
    masks = torch.from_numpy(masks).float()
    avg_r = torch.zeros(block_num)
    for k in range(block_num):
        avg_r[k] = torch.mean(model_r[masks[k]==1])
    masks = masks.int()
    avg_r = avg_r.float()
    
    # *to velocity domain 
    # norm the avg_depth to [0, 1]
    avg_r_norm = (avg_r - r_range[0]) / (r_range[1] - r_range[0])
    # map to velocity domain 
    init_vel_vec = v_range[0] + (v_range[1]-v_range[0]) * avg_r_norm
    # visual inital map
    map_v_init = torch.zeros_like(model_r)
    for i, init_vel in enumerate(init_vel_vec):
        map_v_init += masks[i] * init_vel
        
    return init_vel_vec, masks, map_v_init


def vec2map(v_vec, masks):
    v_map_split = masks.permute(1, 2, 0) * v_vec
    v_map = torch.sum(v_map_split, axis=-1)
    return v_map

def mapping_func(map, ori_range, new_range):
    map_scale  = (map - ori_range[0]) / (ori_range[1] - ori_range[0])
    map_new = new_range[0] + (new_range[1]-new_range[0]) * map_scale
    return map_new

def cluster_map(r_map, block_num, r_weight=0.6):
    # map to point set
    point_set = []
    for i in range(r_map.shape[0]):
        for j in range(r_map.shape[1]):
            point_set.append([i, j, r_map[i, j]])
    point_set = np.array(point_set)
    
    # normalize the point set
    w_x, w_y, w_r = (1-r_weight)*0.4, (1-r_weight)*0.6, r_weight
    point_set_scale = (point_set - np.min(point_set, axis=0)) / (np.max(point_set, axis=0)-np.min(point_set, axis=0))
    point_set_scale = point_set_scale * np.array([w_x, w_y, w_r])
    
    # k-means clustering
    kmeans = KMeans(n_clusters=block_num)
    kmeans.fit(point_set_scale)
    predicted_labels = kmeans.predict(point_set_scale)
    
    # generate masks
    masks = np.zeros((block_num, r_map.shape[0], r_map.shape[1]))
    for i in range(block_num):
        sel_ind = point_set[predicted_labels == i, :2].astype(np.int32)
        masks[i][sel_ind[:, 0], sel_ind[:, 1]] = 1
    
    return masks