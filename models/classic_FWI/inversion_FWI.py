############################################################
# classic inversion method 
# ------
# author: Hongtao Wang(汪泓涛)
# email:  colin315wht@gmail.com
# reference: https://github.com/guoketing/deepwave-order.git
############################################################

#######################  import part   #####################
# packages
import os
import torch
import shutil
import argparse
import pandas as pd
import numpy as np
from time import time
import sys
sys.path.append('.')
sys.path.append('..')
sys.path.append('../..')

# my functions
from config import save_root, model_info
from util.misfit import Wasserstein1, shot_max_normalize
from util.data_loader import read_velocity_model, create_initial_model
from models.forward_models.forward_model import seismic_farword
from util.metrics_tools import compute_metrics
from util.plot_tools import plot_model, plot_comp_figs, plot_log
from util.training_tools import set_seed
###########################################################

def inversion_main():
    global save_root, model_info
    # *====== hyper-parameters defination ======*
    parser = argparse.ArgumentParser()
    # path setting
    parser.add_argument('--save_root', type=str, default=save_root)
    parser.add_argument('--save_group', type=str, default='FWI-try')
    parser.add_argument('--save_name', type=str)
    parser.add_argument('--device', type=int, default=-1)
    parser.add_argument('--rerun_forward', type=int, default=0)
    
    # model setting
    parser.add_argument('--vel_model_name', type=str, default='mar_small')
    parser.add_argument('--forward_order', type=int, default=8, help='An int specifying the finite difference order of accuracy. (2, 4, 6, and 8)')
    parser.add_argument('--init_method', type=str, default='SA')
    parser.add_argument('--init_model_path', type=str, default=None)
    parser.add_argument('--data_norm', type=int, default=0, help='whether to normalize the data in the loss function')
    
    # training setting
    parser.add_argument('--train_lr', type=float, default=30)
    parser.add_argument('--loss', type=str, default='L1', help='L1, L2, or W1')
    parser.add_argument('--max_epoch', type=int, default=300)
    parser.add_argument('--train_bs', type=int, default=8)
    parser.add_argument('--log_interval', type=int, default=10)
    
    # random setting
    parser.add_argument('--seed', type=int, default=1)
    args = parser.parse_args() 
    
    # *====== init wandb ======*
    args.save_name = 'FWI-%s-%s-%s-lr%.1e-ep%d-bs%d-facc=%d-dn%d-s%d' % (args.vel_model_name, args.init_method, args.loss, args.train_lr, args.max_epoch, args.train_bs, args.forward_order, args.data_norm, args.seed) if args.save_name is None else args.save_name
    if args.device == -1:
        device = 'cpu'
    else:
        device = args.device
        
    # *====== set random seed ======*
    set_seed(args.seed)
    
    # *====== init save folder ======*
    save_root = os.path.join(args.save_root, args.save_group, args.save_name)
    if args.rerun_forward and os.path.exists(save_root):
        shutil.rmtree(save_root)
    for folder in ['figs', 'log', 'models']:
        os.makedirs(os.path.join(save_root, folder), exist_ok=True)
    
    # *====== load true velocity model ======*
    # laod basic information of the velocity model
    model_info_dict = model_info[args.vel_model_name]
    n_offset, n_depth, dx = model_info_dict['grid_info'].values()
    # true velocity model
    model_true = read_velocity_model(model_info_dict['path'], n_offset, n_depth)
    model_true = model_true.to(device=device)
    
    # *====== initial velocity model ======*
    if args.init_model_path is None:
        if args.init_method in ['line', 'lmm', 'const', 'gs', 'SA']:
            vp_model, model_init_map = create_initial_model(
                model_true, 
                args.init_method,
                device=device
                )
        else:
            raise ValueError('initial method not supported, %s' % args.init_method)
    else:
        vp_model = torch.tensor(np.load(args.init_model_path), device=device)
        vp_model.requires_grad = True
        model_init_map = vp_model.detach().cpu().numpy()
        
    # plot initial map 
    plot_model(model_init_map.T, os.path.join(save_root, 'figs', 'init_model.png'), title='Initial Velocity Model', dpi=500, figsize=(8, 3), dx=dx)
    
    # *====== forward method ======*
    FM_opt = seismic_farword(model_info_dict, args.forward_order, device)
    
    # *====== load receiver amplitudes of true velocity model ======*
    if os.path.exists(model_info_dict['rec_path']) and args.rerun_forward == 0:
        rcv_amp_gth = np.load(model_info_dict['rec_path'])
        rcv_amp_gth = torch.tensor(rcv_amp_gth, device=device)
        print('# Receiver amplitudes of true velocity model are loaded.')
    else:
        print('# Generating Receiver amplitudes of true velocity model')
        with torch.no_grad():
            rcv_amp_gth = FM_opt(model_true)
        os.makedirs(os.path.split(model_info_dict['rec_path'])[0], exist_ok=True)
        np.save(model_info_dict['rec_path'], rcv_amp_gth.detach().cpu().numpy())
        print('# Receiver amplitudes of true velocity model are loaded.')
        
    # *====== inverison method ======*
    # no model needed
    
    # *====== define training tools ======*
    # optimizer
    optimizer = torch.optim.Adam([{'params': vp_model, 'lr':args.train_lr, 'betas':(0.5, 0.9), 'eps':1e-8, 'weight_decay':0}])
    
    # loss function
    if args.loss == 'L1':
        criterion = torch.nn.L1Loss()
    elif args.loss == 'L2':
        criterion = torch.nn.MSELoss()
    elif args.loss == 'W1':  # transform type for 1-D W1
        trans_type = 'linear' # linear, square, exp, softplus, abs
    else:
        raise NotImplementedError
    
    # *====== define metrics tools & log ======*
    met_opt = compute_metrics(device=device)
    log_head = ['epoch', 'iter', 'SSIM', 'SNR', 'RSE', 'loss_k', 'loss_mean']
    log_list = []
    
    # *====== inversion processing ======*
    shot_num = model_info_dict['src_info']['n_shots']
    bs = shot_num if args.train_bs < 0 else args.train_bs
    iter_num = int(shot_num/bs)
    start = time()
    for epoch in range(args.max_epoch):
        loss_epoch = 0.0
        # optimization 
        for iter_k in range(iter_num):
            itx = epoch * iter_num + iter_k
            optimizer.zero_grad()
            # split the batch
            batch_src_amps = FM_opt.s_amp[iter_k::iter_num]
            batch_rcv_amps_true = rcv_amp_gth[iter_k::iter_num]
            batch_x_s = FM_opt.s_loc[iter_k::iter_num]
            batch_x_r = FM_opt.r_loc[iter_k::iter_num]
            
            # forward modeling
            batch_rcv_amps_pred = FM_opt(vp_model, batch_src_amps, batch_x_s, batch_x_r)
            
            # compute the loss
            if args.loss == 'L1' or args.loss == 'L2':             
                if args.data_norm:
                    # normalize the amplitude of each shot 
                    batch_rcv_amps_true = shot_max_normalize(batch_rcv_amps_true.permute(1,0,2).unsqueeze(1)).squeeze(1).permute(1,0,2)*100
                    batch_rcv_amps_pred = shot_max_normalize(batch_rcv_amps_pred.permute(1,0,2).unsqueeze(1)).squeeze(1).permute(1,0,2)*100
                    loss = criterion(batch_rcv_amps_pred, batch_rcv_amps_true)            
                else:
                    loss = criterion(batch_rcv_amps_pred, batch_rcv_amps_true)
            elif args.loss == 'W1':
                loss = Wasserstein1(batch_rcv_amps_pred, batch_rcv_amps_true,trans_type, theta=1.1)            
            else:
                raise NotImplementedError
            loss_epoch += loss.item()
            loss.backward()
            # Clips gradient value of model
            torch.nn.utils.clip_grad_value_(vp_model, torch.quantile(vp_model.grad.detach().abs(), 0.98)) 
            optimizer.step()
            # clip the model value that keep the minimum value is larger than 0
            vp_model.data=torch.clamp(vp_model.data, min=1e-12)
            # evaluate the metrics
            if itx % args.log_interval == 0:
                # compute the metrics
                met_dict = met_opt(vp_model.detach(), model_true)
                # log
                print('# epoch %-3d idx %-5d SSIM %-10.6f SNR %-10.6f RSE %-10.6f loss_it %-12.8f loss_mean %-12.8f' % (epoch, itx, met_dict['SSIM'], met_dict['SNR'], met_dict['RSE'], loss_epoch/(iter_k+1), loss.item()))
                log_list.append([epoch, itx, met_dict['SSIM'], met_dict['SNR'], met_dict['RSE'], loss_epoch/(iter_k+1), loss.item()])
                
        # save results
        np.save(os.path.join(save_root, 'models', 'epoch_%d_model.npy' % epoch), vp_model.detach().cpu().numpy())
        # plot resutls
        plot_model(vp_model.T.detach().cpu().numpy(), os.path.join(save_root, 'figs', 'epoch_%d_plot.png' % epoch), title='FWI Epoch=%d' % epoch, dpi=500, figsize=(8, 3), dx=dx)
        # plot log 
        plot_log(log_list, ['SSIM', 'SNR', 'RSE', 'Misfit'], os.path.join(save_root, 'log'))
    time_cost = time() - start   
    # *====== evaluate final results & log the results ======*
    # save FWI results
    np.save(os.path.join(save_root, 'models', 'Final_%d_model.npy' % epoch), vp_model.detach().cpu().numpy())
    
    # metrics
    met_dict = met_opt(vp_model.detach(), model_true)
    final_met = [[epoch, itx, met_dict['SSIM'], met_dict['SNR'], met_dict['RSE'], time_cost]]
    FWI_df = pd.DataFrame(final_met, columns=log_head[:-2]+['cost'])
    FWI_df.to_excel(os.path.join(save_root, 'log', 'Final_metrics.xlsx'), index=False)
    
    # comparative maps
    plot_comp_figs(vp_model.T.detach().cpu().numpy(), model_true.T.cpu().numpy(), os.path.join(save_root, 'figs', 'Comp_results.png'), title='Classic FWI', dpi=500, figsize=(8, 12), dx=dx)
    plot_model(vp_model.T.detach().cpu().numpy(), os.path.join(save_root, 'figs', 'Final_FWI_Vp.png'), title='Final FWI $V_p$ Model', dpi=500, figsize=(8, 3), dx=dx)
    plot_model(model_true.T.cpu().numpy(), os.path.join(save_root, 'figs', 'True_Vp.png'), title='True $V_p$ Model', dpi=500, figsize=(8, 3), dx=dx)
    
    # training details: loss functions
    training_detail_df = pd.DataFrame(log_list, columns=log_head)
    training_detail_df.to_excel(os.path.join(save_root, 'log', 'training_details.xlsx'), index=False)
    

if __name__ == "__main__":
    inversion_main()