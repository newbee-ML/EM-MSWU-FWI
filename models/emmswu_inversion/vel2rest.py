"""
Functions for mapping velocity to restistivity

Author: Hongtao Wang (colin315wht@gmail.com)
"""

import numpy as np
import math
class Vel2Rest:
    """
    Class for mapping velocity to restistivity
    """
    def __init__(self):
        pass
    
    @staticmethod
    def linear(model_v, r_min, r_max, v_min=None, v_max=None):
        if v_min is None or v_max is None:
            v_min, v_max = np.min(model_v), np.max(model_v)
        model_r = (model_v - v_min) / (v_max - v_min) * (r_max - r_min) + r_min
        map_info = {'linear': [v_min, v_max, r_min, r_max]}
        return model_r, map_info
    
    @staticmethod
    def IZ_fold(model_v):
        model_r = 10 ** ((model_v - 1000) / 1396.656)
        return model_r, {}

    @staticmethod
    def OBR_fold(model_v):
        model_r = 10 ** ((model_v - 1000) / 1085.029)
        return model_r, {}
    
    @staticmethod
    def nonlinear(model_v, r_min, r_max, p=2.0, v_min=None, v_max=None):
        if v_min is None or v_max is None:
            v_min, v_max = np.min(model_v), np.max(model_v)
        model_r = ((model_v - v_min) / (v_max - v_min)) ** p * (r_max - r_min)  + r_min
        map_info = {'nonlinear': [v_min, v_max, r_min, r_max, p]}
        return model_r, map_info
    
    @staticmethod
    def random_dict(model_v, r_min, r_max, grid_n=10):
        # generate & split the grids
        v_min, v_max = np.min(model_v), np.max(model_v)
        v_vec = np.linspace(v_min, v_max, grid_n+1, endpoint=True)
        r_vec = np.linspace(r_min, r_max, grid_n+1, endpoint=True)
        
        # random rank the index
        random_rank = np.arange(len(r_vec)-1)
        np.random.shuffle(random_rank)
        
        # map each grid
        model_v_1d = np.reshape(model_v, (-1))
        model_r_1d = np.zeros_like(model_v_1d)
        for v_i in range(len(v_vec)-1):
            v_i_min, v_i_max = v_vec[v_i], v_vec[v_i+1]
            sel_ind = np.where((model_v_1d >= v_i_min)&(model_v_1d < v_i_max))[0]
            v_values = model_v_1d[sel_ind]
            model_r_1d[sel_ind] = Vel2Rest.linear(v_values, r_vec[random_rank[v_i]], r_vec[random_rank[v_i]+1], v_min=v_i_min, v_max=v_i_max)[0]
        model_r = np.reshape(model_r_1d, model_v.shape)
        map_info = {'random_dict': [v_vec, r_vec, random_rank]}
        return model_r, map_info
    

if  __name__ == '__main__':
    v_min, v_max = 1000, 7000
    v_value = np.linspace(v_min, v_max, 20, endpoint=True)
    
    save_root = r'models\cascade_inversion\figs'
    import os
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker
    os.makedirs(save_root, exist_ok=True)
    fig, ax = plt.subplots(ncols=2, nrows=1, figsize=(15, 3))
    ax[0].plot(v_value, Vel2Rest.IZ_fold(v_value)[0], 'r-*')
    ax[0].set_title('IZ Fold')
    ax[1].plot(v_value, Vel2Rest.OBR_fold(v_value)[0], 'b-^')
    ax[1].set_title('OBR Fold')
    for i in range(2):
        ax[i].set_xlabel(r'$V_p$ (m/s)')
        ax[i].set_ylabel(r'$\rho$ (ohmm)')
        ax[i].grid(True)
        ax[i].xaxis.set_major_formatter(ticker.ScalarFormatter())
        ax[i].set_yscale('log')
    plt.savefig(os.path.join(save_root, 'vel2rest.png'), bbox_inches='tight', pad_inches=0.1, dpi=100)
    plt.close('all')
