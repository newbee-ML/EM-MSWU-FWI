import torch
import sys
sys.path.append('.')
sys.path.append('..')
sys.path.append('../..')
from models.forward_models.forward_model import define_s_r_loc, define_s_amp, forward_model


"""
2D forward model
---
Using Deepwave
"""


class seismic_cas_farword(torch.nn.Module):
    def __init__(self, info_dict, mapping_class, forward_order=8, device='cpu'):
        super(seismic_cas_farword, self).__init__()
        self.map_opt = mapping_class
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
        
    def forward(self, s_amp_bs=None, s_loc_bs=None, r_loc_bs=None):
        v_map = self.map_opt.update()
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
        return rec_gth, v_map
    
