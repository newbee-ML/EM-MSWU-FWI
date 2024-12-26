############################################################
# loading velocity models
# ------
# author: Hongtao Wang(汪泓涛)
# email:  colin_wht@stu.xjtu.edu.cn
############################################################
import torch
import numpy as np
import scipy
import sys
sys.path.append('../..')
from config import SA_init_model

# load velocity model from file (.bin file)
def read_velocity_model(model_path, n_offset, n_depth):
    # (x, depth)
    v_model = torch.from_file(model_path, size=n_offset*n_depth).reshape(n_offset, n_depth)
    return v_model


def create_initial_model(model_true_gpu, init_method, fix_value_depth=0, device='cpu'):
    """
        Create 2D initial guess model for inversion ('line','lineminmax','const','GS')
        reference: https://doi.org/10.5281/zenodo.7028832
        
    """
    assert init_method in ['line', 'lmm', 'const', 'gs', 'SA']
    model_true = model_true_gpu.T.cpu().detach().numpy()   
    shape = model_true.shape
    if fix_value_depth > 0:
        const_value = model_true[:fix_value_depth,:]
    
    if init_method == 'line':
    # generate the line increased initial model
        lipar = 1.0
        value = np.linspace(model_true[fix_value_depth,np.int32(shape[1]/2)],
                            model_true[-1,np.int32(shape[1]/2)]*lipar,num=shape[0]-fix_value_depth,
                            endpoint=True,dtype=float).reshape(-1,1)
        value = value.repeat(shape[1],axis=1)        
    elif init_method == 'lmm':
    # generate the line increased initial model (different min/max value)
        lipar = 1.1
        value = np.linspace(model_true.min()*lipar,
                            model_true.max(),
                            num=shape[0]-fix_value_depth,
                            endpoint=True,dtype=float).reshape(-1,1)
        
        value = value.repeat(shape[1],axis=1)      
    elif init_method == 'const': # generate the constant initial model
        value = model_true[fix_value_depth, int(np.floor(shape[1] / 2))] * np.ones((shape[0]-fix_value_depth, shape[1]))
    elif init_method == 'gs':  # generate the initial model by using Gaussian smoothed function
        value = scipy.ndimage.gaussian_filter(model_true[fix_value_depth:,:], sigma=5)
    elif init_method == 'SA':  # generate the initial model by using Gaussian smoothed function
        value = np.load(SA_init_model).T    
    if fix_value_depth > 0:
        model_init = np.concatenate([const_value,value],axis=0)
    else:
        model_init = value
        
    model_init = torch.tensor(model_init.T, dtype=torch.float32)
    # Make a copy so at the end we can see how far we came from the initial model
    model = model_init.clone()
   
    model = model.to(device)
    # set the requires_grad to True to update the model
    model.requires_grad = True
    
    return model, model_init


def init_forward_process():
    
    return 


def add_noise():
    return