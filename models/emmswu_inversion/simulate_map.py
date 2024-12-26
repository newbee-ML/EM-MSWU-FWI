import numpy as np
import pickle

import sys
sys.path.append('.')
sys.path.append('..')
sys.path.append('../..')
from models.emmswu_inversion.vel2rest import Vel2Rest
from util.data_loader import read_velocity_model
from util.plot_tools import plot_two_model
from config import model_info
import os

"""
Simulate the resistivity model based on velocity model
---
author: WANG hogntao
email:  colin315wht@gmail.com

"""


def simulate_map(model_name:str, map_method:str, map_para_dict:dict, base_name:str):
    #* load the velocity model 
    grid_info = model_info[model_name]['grid_info']
    vel_path = model_info[model_name]['path']
    model_v = read_velocity_model(vel_path, grid_info['n_offset'], grid_info['n_depth']).T.numpy()
    syn_root = model_info[model_name]['synthetic_CSEM'] 
    
    #* map to resistivity model
    if map_method == 'linear':
        model_r, map_info = Vel2Rest.linear(model_v, **map_para_dict)
    elif map_method == 'nonlinear':
        model_r, map_info = Vel2Rest.nonlinear(model_v, **map_para_dict)
    elif map_method == 'random_dict':
        model_r, map_info = Vel2Rest.random_dict(model_v, **map_para_dict)
    elif map_method == 'IZ_fold':
        model_r, map_info = Vel2Rest.IZ_fold(model_v)
    elif map_method == 'OBR_fold':
        model_r, map_info = Vel2Rest.OBR_fold(model_v)
    else:
        raise ValueError('map method not supported: %s' % map_method)
    
    #* save to bin file
    # if root folder not exist, then create it 
    os.makedirs(syn_root, exist_ok=True)
    # save to .npy file
    np.save(os.path.join(syn_root, '%s.npy'%base_name), model_r.T)
    np.save(os.path.join(syn_root, '%s_map_info.npy'%base_name), map_info)
    #* plot two model figure in a figure
    plot_two_model(
        model_v,
        np.log10(model_r),
        {
            'title': 'Velocity',
            'bar_name': r'$V_p$ (m/s)'
        },
        {
            'title': 'Resistivity',
            'bar_name': r'Resistivity (log10 $\Omega\cdot m$)'
        },
        save_path=os.path.join(syn_root, '%s_compare.png'%base_name),
        dpi=300,
        figsize=(10, 12)
    )
    
    
if __name__ == '__main__':
    simulate_map('mar_small', 'IZ_fold', {}, 'IZ_fold')