############################################################
# generate the shot gathers for the inversion models
# ------
# author: Hongtao Wang(汪泓涛)
# email:  colin_wht@stu.xjtu.edu.cn
# reference: https://github.com/guoketing/deepwave-order.git
############################################################


import numpy as np
import sys
sys.path.append('.')
sys.path.append('..')
sys.path.append('../..')

from config import save_root, model_info
from models.forward_models.forward_model import seismic_farword


def fw_shot_gth(model_name, inv_model, shot_id, device=-1):
    model_info_dict = model_info[model_name]
    FM_opt = seismic_farword(model_info_dict, 8, device)
    batch_src_amps = FM_opt.s_amp[shot_id].unsqueeze(0)
    batch_x_s = FM_opt.s_loc[shot_id].unsqueeze(0)
    batch_x_r = FM_opt.r_loc[shot_id].unsqueeze(0)
    batch_rcv_amps_pred = FM_opt(inv_model, batch_src_amps, batch_x_s, batch_x_r)
    return batch_rcv_amps_pred