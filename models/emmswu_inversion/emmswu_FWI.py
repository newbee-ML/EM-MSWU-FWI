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
from models.emmswu_inversion.mapping_method import init_map_blocks, vec2map, mapping_func
from util.metrics_tools import compute_metrics
from util.plot_tools import plot_model, plot_comp_figs, plot_log
from util.training_tools import set_seed
###########################################################

def gauss_noise(gth, noise_snr):
    # 0 reshape to (h, bs*c*w)
    bs, h, w = gth.shape
    gth = gth.permute(1, 0, 2)  # (h, bs, c, w)
    gth = gth.reshape(h, -1)  # (h, bs*c*w)
    # add noise 
    # 1 measure power of signal
    singal_power = torch.var(gth, dim=0)  # len = bs*c*w
    noise_std = torch.sqrt(singal_power/(10**(noise_snr/10)))
    # 2 generate Gaussian noise with specific SNR
    noise = torch.randn(gth.shape).to(gth.device) * noise_std  # (h, bs*c*w)
    # 3 add to gth
    gth_noise = gth + noise
    # 4 reshape original shape
    gth_noise = gth_noise.reshape((h, bs, w))
    gth_noise = gth_noise.permute(1, 0, 2)
    return gth_noise

def inversion_main():
    global save_root, model_info
    # *====== hyper-parameters defination ======*
    parser = argparse.ArgumentParser()
    # path setting
    parser.add_argument('--save_root', type=str, default=save_root)
    parser.add_argument('--save_group', type=str, default='CFWI-try')
    parser.add_argument('--save_name', type=str)
    parser.add_argument('--device', type=int, default=-1)
    parser.add_argument('--rerun_forward', type=int, default=0)
    parser.add_argument('--rerun_optim', type=int, default=1)
    parser.add_argument('--vel_model_name', type=str, default='mar_small')
    parser.add_argument('--forward_order', type=int, default=8, help='An int specifying the finite difference order of accuracy. (2, 4, 6, and 8)')
    parser.add_argument('--mapping_method', type=str, default='line')
    parser.add_argument('--fw_freq', type=float, default=5)
    parser.add_argument('--noise_SNR', type=float, default=10000)
    
    # ############### Warm Up FWI ######################
    parser.add_argument('--wu_cluster_k', type=int, default=100)
    parser.add_argument('--wu_data_norm', type=int, default=0, help='whether to normalize the data in the loss function')
    parser.add_argument('--wu_train_lr', type=float, default=30)
    parser.add_argument('--wu_loss', type=str, default='L2', help='L1, L2, or W1')
    parser.add_argument('--wu_max_epoch', type=int, default=150)
    parser.add_argument('--wu_train_bs', type=int, default=2)
    parser.add_argument('--wu_vel_range', type=str, default='None')

    # ############## Classic FWI ########################
    parser.add_argument('--c_do', type=int, default=1, help='whether to conduct refined FWI')
    parser.add_argument('--c_data_norm', type=int, default=0, help='whether to normalize the data in the loss function')
    parser.add_argument('--c_train_lr', type=float, default=30)
    parser.add_argument('--c_loss', type=str, default='W1', help='L1, L2, or W1')
    parser.add_argument('--c_max_epoch', type=int, default=300)
    parser.add_argument('--c_train_bs', type=int, default=8)
    parser.add_argument('--c_opt_stg', type=str, default='const', help='const, StepLR')
    
    # random setting
    parser.add_argument('--seed', type=int, default=1)
    
    args = parser.parse_args() 
    
    # model parameter: vel_model_name mapping_method
    # tuning parameter: train_bs train_lr loss data_norm
    # *====== init wandb ======*
    args.save_name = 'CascadeFWI-%s-%s-k=%d-wlr%.1e-wbs%d-wdn%d-wl=%s-wep%d-clr%.1e-cbs%d-cdn%d-cl=%s-cep%d-f=%.1f-cstg=%s-SNR=%d-%s-s%d' % (args.vel_model_name, args.mapping_method, args.wu_cluster_k, args.wu_train_lr, args.wu_train_bs, args.wu_data_norm, args.wu_loss, args.wu_max_epoch, args.c_train_lr, args.c_train_bs, args.c_data_norm, args.c_loss, args.c_max_epoch, args.fw_freq, args.c_opt_stg, args.noise_SNR, args.wu_vel_range, args.seed)
    if args.device == -1:
        device = 'cpu'
    else:
        device = args.device
        
    # *====== set random seed ======*
    set_seed(args.seed)
    
    # *====== init save folder ======*
    save_root = os.path.join(args.save_root, args.save_group, args.save_name)
    if args.rerun_optim and os.path.exists(save_root):
        shutil.rmtree(save_root)
    for subfolder in ['warm_up', 'classic_FWI']:
        for folder in ['figs', 'log', 'models', 'processing']:
            os.makedirs(os.path.join(save_root, subfolder, folder), exist_ok=True)
    
    # *====== load true velocity model ======*
    # laod basic information of the velocity model
    model_info_dict = model_info[args.vel_model_name]
    n_offset, n_depth, dx = model_info_dict['grid_info'].values()
    # true velocity model
    model_true = read_velocity_model(model_info_dict['path'], n_offset, n_depth)
    plot_model(model_true.T, os.path.join(save_root, 'warm_up', 'figs', 'true_model_vp.png'), dx=dx, title='True Velocity Model', dpi=500, figsize=(8, 3))
    model_true = model_true.to(device=device)
    
    # *====== initial velocity model ======*
    # load resistivity model
    r_m_path = os.path.join(model_info_dict['synthetic_CSEM'])
    r_model = torch.tensor(np.log10(np.load(r_m_path)), dtype=torch.float32)
    r_min, r_max = model_info_dict['r_info']['min'], model_info_dict['r_info']['max']
    # plot initial map 
    plot_model(r_model.T.detach().numpy(), os.path.join(save_root, 'warm_up', 'figs', 'init_model_r.png'), dx=dx, bar_name=r'Resistivity (Log10 ohm$\cdot$m)', title='Reference Resistivity Model', dpi=500, figsize=(8, 3))
    
    # *====== forward method ======*
    freq = args.fw_freq
    peak_amp = 1/freq
    model_info_dict['wave_info']['freq'] = freq
    model_info_dict['wave_info']['peak_time'] = peak_amp
    FM_opt = seismic_farword(model_info_dict, args.forward_order, device)
    
    # *====== load receiver amplitudes of true velocity model ======*
    if os.path.exists(model_info_dict['rec_path']) and not args.rerun_forward:
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
    
    # *add Gaussian noise
    if args.noise_SNR < 10000:
        rcv_amp_gth = gauss_noise(rcv_amp_gth, args.noise_SNR)
    
    if args.wu_cluster_k > 0:
        # *====== inverison method ======*
        if args.wu_vel_range == 'None':
            vel_min, vel_max = model_true.min().item(), model_true.max().item()
            print(vel_min, vel_max)
        else:
            vel_min, vel_max = map(float, args.wu_vel_range.split('='))
        v_vec, masks, map_v_init = init_map_blocks(r_model, block_num=args.wu_cluster_k, 
                                                v_range=[vel_min, vel_max],
                                                r_range=[np.log10(r_min), np.log10(r_max)])
        plot_model(map_v_init.T.detach().cpu().numpy(), os.path.join(save_root, 'warm_up', 'figs', 'mask_model_v.png'), dx=dx, bar_name='Velocity (m/s)', title='Divide Initial Velocity Model', dpi=500, figsize=(8, 3))
        v_vec = v_vec.to(device=device)
        masks = masks.to(device=device)
        v_vec.requires_grad = True
            
        # *====== define training tools ======*
        # optimizer
        optimizer = torch.optim.Adam([v_vec], lr=args.wu_train_lr)
        
        # loss function
        if args.wu_loss == 'L1':
            criterion = torch.nn.L1Loss()
        elif args.wu_loss == 'L2':
            criterion = torch.nn.MSELoss()
        elif args.wu_loss == 'W1':  # transform type for 1-D W1
            trans_type = 'linear' # linear, square, exp, softplus, abs
        else:
            raise NotImplementedError
        
        # *====== define metrics tools & log ======*
        met_opt = compute_metrics(device=device)
        log_head = ['epoch', 'iter', 'SSIM', 'SNR', 'loss_k', 'loss_mean']
        log_list = []
        
        # *====== warm up FWI ======*
        shot_num = model_info_dict['src_info']['n_shots']
        bs = shot_num if args.wu_train_bs < 0 else args.wu_train_bs
        iter_num = int(shot_num/bs)
        start = time()
        for epoch in range(args.wu_max_epoch):
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
                vp_model = vec2map(v_vec, masks)
                # clip the model value that keep the minimum value is larger than 0
                vp_model.data=torch.clamp(vp_model.data, min=1)
                batch_rcv_amps_pred = FM_opt(vp_model, batch_src_amps, batch_x_s, batch_x_r)
                
                # compute the loss
                if args.wu_loss == 'L1' or args.wu_loss == 'L2':             
                    if args.wu_data_norm:
                        # normalize the amplitude of each shot 
                        batch_rcv_amps_true = shot_max_normalize(batch_rcv_amps_true.permute(1,0,2).unsqueeze(1)).squeeze(1).permute(1,0,2)*100
                        batch_rcv_amps_pred = shot_max_normalize(batch_rcv_amps_pred.permute(1,0,2).unsqueeze(1)).squeeze(1).permute(1,0,2)*100
                        loss = criterion(batch_rcv_amps_pred, batch_rcv_amps_true)            
                    else:
                        loss = criterion(batch_rcv_amps_pred, batch_rcv_amps_true)
                elif args.wu_loss == 'W1':
                    loss = Wasserstein1(batch_rcv_amps_pred, batch_rcv_amps_true,trans_type, theta=1.1)            
                else:
                    raise NotImplementedError
                loss_epoch += loss.item()
                loss.backward()
                
                # *** check inversion results *** #
                if iter_k==0:
                    # plot gradient image
                    vp_model_grad = vec2map(v_vec.grad.detach(), masks)
                    np.save(os.path.join(save_root, 'warm_up', 'processing', 'epoch_%d_iter_%d_grad.npy' % (epoch, itx)), vp_model_grad.detach().cpu().numpy())
                # Clips gradient value of model
                # torch.nn.utils.clip_grad_value_(v_vec, torch.quantile(v_vec.grad.detach().abs(), 0.98)) 
                optimizer.step()
                
            # compute the metrics
            met_dict = met_opt(vp_model.detach(), model_true)
            # log
            print('# epoch %-3d idx %-5d SSIM %-10.6f SNR %-10.6f RSE %-10.6f loss_it %-12.8f loss_mean %-12.8f' % (epoch, itx, met_dict['SSIM'], met_dict['SNR'], met_dict['RSE'], loss_epoch/(iter_k+1), loss.item()))
            log_list.append([epoch, itx, met_dict['SSIM'], met_dict['SNR'], loss_epoch/(iter_k+1), loss.item()])
            plot_log(log_list, ['SSIM', 'SNR', 'RSE', 'Misfit'], os.path.join(save_root, 'warm_up', 'log'))        
            # plot resutls
            plot_model(vp_model.T.detach().cpu().numpy(), os.path.join(save_root, 'warm_up', 'figs', 'epoch_%d_plot.png' % epoch), dx=dx, title='FWI Epoch=%d' % epoch, dpi=500, figsize=(8, 3))
            np.save(os.path.join(save_root, 'warm_up', 'models', 'epoch_%d_model.npy' % epoch), vp_model.detach().cpu().numpy())
        
        time_fw = time() - start  
        # *====== evaluate final results & log the results ======*
        # save FWI results
        np.save(os.path.join(save_root, 'warm_up', 'models', 'Final_model.npy'), vp_model.detach().cpu().numpy())
        # metrics
        met_dict = met_opt(vp_model.detach(), model_true)
        final_met = [[epoch, itx, met_dict['SSIM'], met_dict['SNR']]]
        FWI_df = pd.DataFrame(final_met, columns=log_head[:-2])
        FWI_df.to_excel(os.path.join(save_root, 'warm_up', 'log', 'Final_metrics.xlsx'), index=False)
        
        # comparative maps
        plot_comp_figs(vp_model.T.detach().cpu().numpy(), model_true.T.cpu().numpy(), os.path.join(save_root, 'warm_up', 'figs', 'Comp_results.png'), dx=dx, title='Warm Up FWI', dpi=500, figsize=(8, 15))
        plot_model(vp_model.T.detach().cpu().numpy(), os.path.join(save_root, 'warm_up', 'figs', 'Final_FWI_Vp.png'), dx=dx, title='Warmed Initial $V_p$ Model', dpi=500, figsize=(8, 3))
        plot_model(model_true.T.cpu().numpy(), os.path.join(save_root, 'warm_up', 'figs', 'True_Vp.png'), dx=dx, title='True $V_p$ Model', dpi=500, figsize=(8, 3))
        # training details: loss functions
        training_detail_df = pd.DataFrame(log_list, columns=log_head)
        training_detail_df.to_excel(os.path.join(save_root, 'warm_up', 'log', 'training_details.xlsx'), index=False)
    
    else:
        vp_model = mapping_func(r_model.detach().numpy(), [np.log10(r_min), np.log10(r_max)], [model_true.min().item(), model_true.max().item()])
        np.save(os.path.join(save_root, 'warm_up', 'models', 'Final_model.npy'), vp_model)
        time_fw = 0
        
        
    # *====== classic FWI ======*
    if args.c_do:
        # initial velocity model
        vp_model = torch.tensor(np.load(os.path.join(save_root, 'warm_up', 'models', 'Final_model.npy')), device=device)
        vp_model.requires_grad = True
        model_init_map = vp_model.detach().cpu().numpy()
        
        # visualize inital velocity model
        plot_model(model_init_map.T, os.path.join(save_root, 'classic_FWI', 'figs', 'init_model_v.png'), dx=dx, bar_name='Vp', title='Warmed Initial Velocity Model', dpi=500, figsize=(8, 3))
        set_seed(args.seed)
        
        # define optimizer and loss function
        # optimizer
        optimizer = torch.optim.Adam([vp_model], lr=args.c_train_lr)
        
        # loss function
        if args.c_loss == 'L1':
            criterion = torch.nn.L1Loss()
        elif args.c_loss == 'L2':
            criterion = torch.nn.MSELoss()
        elif args.c_loss == 'W1':  # transform type for 1-D W1
            trans_type = 'linear' # linear, square, exp, softplus, abs
        else:
            raise NotImplementedError
        
        # *====== define metrics tools & log ======*
        met_opt = compute_metrics(device=device)
        log_head = ['epoch', 'iter', 'SSIM', 'SNR', 'RSE', 'loss_mean', 'lr']
        log_list = []
        
        # *====== classic FWI ======*
        shot_num = model_info_dict['src_info']['n_shots']
        bs = shot_num if args.c_train_bs < 0 else args.c_train_bs
        iter_num = int(shot_num/bs)
        
        if args.c_opt_stg == 'const':
            scheduler = None
        elif args.c_opt_stg == 'StepLR':
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer, 200, gamma=0.8, last_epoch=-1)
        elif args.c_opt_stg == 'MSLR':
            scheduler = torch.optim.lr_scheduler.MultiStepLR(
                optimizer, 
                [int(args.c_max_epoch*0.5), int(args.c_max_epoch*0.75)], 
                gamma=0.5, 
                last_epoch=-1)
        elif args.c_opt_stg == 'cos':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=args.c_max_epoch, eta_min=args.c_train_lr*0.5, last_epoch=-1)
        else:
            raise ValueError('Optimizer scheduler is not defined.')
        
        start = time()
        for epoch in range(args.c_max_epoch):
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
                if args.c_loss == 'L1' or args.c_loss == 'L2':             
                    if args.c_data_norm:
                        # normalize the amplitude of each shot 
                        batch_rcv_amps_true = shot_max_normalize(batch_rcv_amps_true.permute(1,0,2).unsqueeze(1)).squeeze(1).permute(1,0,2)*100
                        batch_rcv_amps_pred = shot_max_normalize(batch_rcv_amps_pred.permute(1,0,2).unsqueeze(1)).squeeze(1).permute(1,0,2)*100
                        loss = criterion(batch_rcv_amps_pred, batch_rcv_amps_true)            
                    else:
                        loss = criterion(batch_rcv_amps_pred, batch_rcv_amps_true)
                elif args.c_loss == 'W1':
                    loss = Wasserstein1(batch_rcv_amps_pred, batch_rcv_amps_true,trans_type, theta=1.1)            
                else:
                    raise NotImplementedError
                loss_epoch += loss.item()
                loss.backward()
                if iter_k==0:
                    # plot gradient image
                    np.save(os.path.join(save_root, 'classic_FWI', 'processing', 'epoch_%d_iter_%d_grad.npy' % (epoch, itx)), vp_model.grad.detach().cpu().numpy())
                # Clips gradient value of model
                torch.nn.utils.clip_grad_value_(vp_model, torch.quantile(vp_model.grad.detach().abs(), 0.98)) 
                optimizer.step()
                # clip the model value that keep the minimum value is larger than 0
                vp_model.data=torch.clamp(vp_model.data, min=100)
                
            # compute the metrics
            met_dict = met_opt(vp_model.detach(), model_true)
            # log
            print('# epoch %-3d idx %-5d SSIM %-10.6f SNR %-10.6f RSE %-10.6f loss_it %-12.8f loss_mean %-12.8f' % (epoch, itx, met_dict['SSIM'], met_dict['SNR'], met_dict['RSE'], loss_epoch/(iter_k+1), loss.item()))
            log_list.append([epoch, itx, met_dict['SSIM'], met_dict['SNR'], met_dict['RSE'], loss_epoch/(iter_k+1), optimizer.param_groups[0]['lr']])
            plot_log(log_list, ['SSIM', 'SNR', 'RSE', 'Misfit', 'Lr'], os.path.join(save_root, 'classic_FWI', 'log'))   
            # plot resutls
            plot_model(vp_model.T.detach().cpu().numpy(), os.path.join(save_root, 'classic_FWI', 'figs', 'epoch_%d_plot.png' % epoch), title='FWI Epoch=%d' % epoch, dpi=500, figsize=(8, 3))        
            np.save(os.path.join(save_root, 'classic_FWI', 'models', 'epoch_%d_model.npy' % epoch), vp_model.detach().cpu().numpy())

            if scheduler is not None:
                scheduler.step()
                    
        time_ri = time() - start
        # *====== evaluate final results & log the results ======*
        # save FWI results
        np.save(os.path.join(save_root, 'classic_FWI', 'models', 'Final_model.npy'), vp_model.detach().cpu().numpy())
        
        # metrics
        met_dict = met_opt(vp_model.detach(), model_true)
        final_met = [[epoch, itx, met_dict['SSIM'], met_dict['SNR'], met_dict['RSE'], time_fw, time_ri]]
        FWI_df = pd.DataFrame(final_met, columns=log_head[:-2]+['cost_fw', 'cost_ri'])
        FWI_df.to_excel(os.path.join(save_root, 'classic_FWI', 'log', 'Final_metrics.xlsx'), index=False)
        
        # comparative maps
        plot_comp_figs(vp_model.T.detach().cpu().numpy(), model_true.T.cpu().numpy(), os.path.join(save_root, 'classic_FWI', 'figs', 'Comp_results.png'), dx=dx, title='Cascade FWI', dpi=500, figsize=(10, 15))
        plot_model(vp_model.T.detach().cpu().numpy(), os.path.join(save_root, 'classic_FWI', 'figs', 'Final_FWI_Vp.png'), dx=dx, title='CascadeFWI $V_p$ Model', dpi=500, figsize=(8, 3))
        plot_model(model_true.T.cpu().numpy(), os.path.join(save_root, 'classic_FWI', 'figs', 'True_Vp.png'), dx=dx, title='True $V_p$ Model', dpi=500, figsize=(8, 3))
        # training details: loss functions
        training_detail_df = pd.DataFrame(log_list, columns=log_head)
        training_detail_df.to_excel(os.path.join(save_root, 'classic_FWI', 'log', 'training_details.xlsx'), index=False)

    
if __name__ == "__main__":
    inversion_main()