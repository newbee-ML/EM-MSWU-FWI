import torch
import torch.nn as nn

import deepwave
from deepwave import scalar

import sys
sys.path.append('.')
sys.path.append('..')
sys.path.append('../..')
from util.plot_tools import plot_model
from util.data_loader import read_velocity_model
"""
2D forward model
---
Using Deepwave
"""

"""
# grid defination
n_offset = 2301  # offset axis
n_depth = 751   # depth axis
dx = 4.0

# source defination
n_shots = 115
n_sources_per_shot = 1
d_source = 20              # 20 * 4m = 80m
first_source = 10          # 10 * 4m = 40m
source_depth = 2           # 2  * 4m = 8m

# receiver defination
n_receivers_per_shot = 384
d_receiver = 6             # 6 * 4m = 24m
first_receiver = 0         # 0 * 4m = 0m
receiver_depth = 2         # 2 * 4m = 8m

# source amplitude 
freq = 25
nt = 750
dt = 0.004
peak_time = 1.5 / freq

"""


class seismic_farword(torch.nn.Module):
    def __init__(self, info_dict, forward_order=8, device='cpu'):
        super(seismic_farword, self).__init__()
        self.farword_order = forward_order
        self.device = device
        self.grid_info = info_dict['grid_info']
        self.s_r_loc = {**info_dict['src_info'], **info_dict['rec_info']}
        self.s_amp_info = info_dict['wave_info']
        self._define_s_r_loc()
        self._define_s_amp()
    
    def _define_s_r_loc(self):
        self.s_loc, self.r_loc = define_s_r_loc(**self.s_r_loc)
    
    def _define_s_amp(self):
        self.s_amp = define_s_amp(**self.s_amp_info, **self.s_r_loc)
        
    def forward(self, v_map, s_amp_bs=None, s_loc_bs=None, r_loc_bs=None):
        s_amp_use = self.s_amp if s_amp_bs is None else s_amp_bs
        s_loc_use = self.s_loc if s_loc_bs is None else s_loc_bs
        r_loc_use = self.r_loc if r_loc_bs is None else r_loc_bs 
        
        # forward modelling
        rec_gth = forward_model(
            v_map, 
            self.grid_info['dx'], 
            self.s_amp_info['dt'], 
            s_amp_use.to(self.device), 
            s_loc_use.to(self.device), 
            r_loc_use.to(self.device), 
            self.s_amp_info['freq'], 
            order=self.farword_order
        )
        return rec_gth
    
    
# define the source and receivers
def define_s_r_loc(**loc):
    # source_locations
    source_locations = torch.zeros(loc['n_shots'], loc['n_sources_per_shot'], 2,
                               dtype=torch.long)
    source_locations[..., 1] = loc['source_depth']
    source_locations[:, 0, 0] = (torch.arange(loc['n_shots']) * loc['d_source'] +
                                loc['first_source'])
    # receiver_locations
    receiver_locations = torch.zeros(loc['n_shots'], loc['n_receivers_per_shot'], 2,
                                    dtype=torch.long)
    receiver_locations[..., 1] = loc['receiver_depth']
    receiver_locations[:, :, 0] = (
        (torch.arange(loc['n_receivers_per_shot']) * loc['d_receiver'] +
        loc['first_receiver'])
        .repeat(loc['n_shots'], 1)
    )
    return source_locations, receiver_locations


# define the amplitudes of sources
def define_s_amp(**s_amp_info):
    source_amplitudes = (
        deepwave.wavelets.ricker(s_amp_info['freq'], s_amp_info['nt'], s_amp_info['dt'], s_amp_info['peak_time'])
        .repeat(s_amp_info['n_shots'], s_amp_info['n_sources_per_shot'], 1)
    )
    return source_amplitudes.float()


# main function of the forward modeling
def forward_model(v, dx, dt, source_amplitudes, source_locations, receiver_locations, pml_freq, order=8):
    out = scalar(
        v, dx, dt, source_amplitudes=source_amplitudes,
        source_locations=source_locations,
        receiver_locations=receiver_locations,
        accuracy=order,
        pml_freq=pml_freq,
        pml_width=[20, 20, 20, 20]
        )
    return out[-1]

if __name__ == "__main__":
    # define parameters
    para_dict = {
        # grid defination
        'n_offset': 2301, 
        'n_depth': 751, 
        'dx': 4.0, 
        # source defination
        'n_shots': 10, 
        'n_sources_per_shot': 1, 
        'd_source': 20, 
        'first_source': 10, 
        'source_depth': 2, 
        # receiver defination
        'n_receivers_per_shot': 384, 
        'd_receiver': 6, 
        'first_receiver': 0, 
        'receiver_depth': 2, 
        # source amplitude 
        'freq': 25, 
        'nt': 750, 
        'dt': 0.004, 
        'peak_time': 1.5/25,
        'device': 0
    }
    
    # get velocity model
    
    v_model = read_velocity_model(r'data/velocity_models/marmousi2_vp.bin', para_dict['n_offset'], para_dict['n_depth']).to(0)
    # get locations of source and receivers
    s_loc, r_loc = define_s_r_loc(**para_dict)
    
    # get the amplitudes of sources
    s_amp = define_s_amp(**para_dict)
    
    # forward modelling
    rec_amp = forward_model(
        v_model, para_dict['dx'], para_dict['dt'],
        s_amp.to(0), s_loc.to(0), r_loc.to(0), para_dict['freq'], 8
        )
    # plot results
    receiver_amplitudes = rec_amp
    receiver_amplitudes.cpu().numpy().tofile(r'data/velocity_models/marmousi_data.bin')