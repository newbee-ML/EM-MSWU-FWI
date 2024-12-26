############################################################
# * FWIGAN *
# ------
# author: Hongtao Wang(汪泓涛)
# email:  colin315wht@gmail.com
# reference: @author: fangshuyang (yangfs@hit.edu.cn)
############################################################

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
from models.FWIGAN.Discriminator import weights_init, Discriminator
from util.metrics_tools import compute_metrics
from util.misfit import calc_gradient_penalty
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
    parser.add_argument('--save_group', type=str, default='FWIGAN-try')
    parser.add_argument('--save_name', type=str)
    parser.add_argument('--device', type=int, default=-1)
    parser.add_argument('--rerun_forward', type=int, default=0)
    parser.add_argument('--noise_SNR', type=float, default=10000)
    
    # model setting
    parser.add_argument('--vel_model_name', type=str, default='mar_small')
    parser.add_argument('--forward_order', type=int, default=8, help='An int specifying the finite difference order of accuracy. (2, 4, 6, and 8)')
    parser.add_argument('--init_method', type=str, default='line')
    parser.add_argument('--init_model_path', type=str, default=None)
    parser.add_argument('--data_norm', type=int, default=0, help='whether to normalize the data in the loss function')
    
    # training setting
    parser.add_argument('--train_lr', type=float, default=30)
    parser.add_argument('--loss', type=str, default='W1', help='L1, L2, or W1')
    parser.add_argument('--max_epoch', type=int, default=300)
    parser.add_argument('--train_bs', type=int, default=5)
    parser.add_argument('--log_interval', type=int, default=10)
    
    # random setting
    parser.add_argument('--seed', type=int, default=1)
    args = parser.parse_args() 
    
    # *====== init wandb ======*
    args.save_name = 'FWIGAN-%s-%s-%s-lr%.1e-ep%d-bs%d-facc=%d-dn%d-SNR=%d-s%d' % (args.vel_model_name, args.init_method, args.loss, args.train_lr, args.max_epoch, args.train_bs, args.forward_order, args.data_norm, args.noise_SNR, args.seed) if args.save_name is None else args.save_name
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
    model_true = model_true.to(device)
    
    # *====== initial velocity model ======*
    if args.init_model_path is None:
        if args.init_method in ['line', 'lmm', 'const', 'gs']:
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
    
    # *====== define discriminator network ======*
    shot_num = model_info_dict['src_info']['n_shots']
    num_shots_per_batch = shot_num if args.train_bs < 0 else args.train_bs
    iter_num = int(shot_num/num_shots_per_batch)
    criticIter = 6
    DFilter  = 32 
    Filters = np.array([DFilter,2*DFilter,4*DFilter,8*DFilter,16*DFilter,32*DFilter],dtype=int)
    netD = Discriminator(
        batch_size=num_shots_per_batch,ImagDim=[model_info_dict['wave_info']['nt'],n_offset], LReLuRatio=0.1,filters=Filters, leak_value=0
        )  
    netD.apply(lambda m: weights_init(m, 0))
    netD = netD.to(device)
    
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
    
    # *add Gaussian noise
    if args.noise_SNR < 10000:
        rcv_amp_gth = gauss_noise(rcv_amp_gth, args.noise_SNR)
        
    # *====== define training tools ======*
    # optimizer
    optim_d = torch.optim.Adam(netD.parameters(),lr=1e-3,betas=(0.5, 0.9), \
                    eps=1e-8, weight_decay=0)
    optim_g = torch.optim.Adam([{'params' : vp_model, 'lr':args.train_lr, 'betas':(0.5, 0.9), 'eps':1e-8, 'weight_decay':0}])
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
    start = time()
    for epoch in range(args.max_epoch):
        loss_epoch = 0.0
        # optimization 
        for iter_k in range(iter_num):
            itx = epoch * iter_num + iter_k
            
            # *====== Discriminator training ======*
            for p in netD.parameters():  # reset requires_grad
                p.requires_grad_(True)    # they are set to False below in training G
                
            for j in range(criticIter):
                # set netD in training stage
                netD.train()                
                netD.zero_grad()     
                # for inner loop training of Discirminator 
                if iter_k*criticIter+j < iter_num:
                    start_id = iter_k*criticIter+j 
                else:  # take the shots per batch from starting
                    start_id = (iter_k*criticIter+j) % iter_num
                if start_id < shot_num % iter_num:
                    # drop one sample randomly
                    ori_id = np.arange(num_shots_per_batch+1)
                    drop_id = np.random.choice(ori_id, 1, replace=False)
                    sel_id = ori_id[ori_id!=drop_id]
                else:
                    sel_id = np.arange(num_shots_per_batch)
                # split samples into mini-batch
                batch_rcv_amps_true = rcv_amp_gth[start_id::iter_num][sel_id]
                batch_src_amps = FM_opt.s_amp[start_id::iter_num][sel_id]
                batch_x_s = FM_opt.s_loc[start_id::iter_num][sel_id]
                batch_x_r = FM_opt.r_loc[start_id::iter_num][sel_id]
                d_real = batch_rcv_amps_true.detach()
                with torch.no_grad():                    
                    model_fake = vp_model  # totally freeze G, training D 
                    learn_snr_fake = None
                ### obtain the simulated data from current model
                d_fake = FM_opt(model_fake, batch_src_amps, batch_x_s, batch_x_r)
                
                # train with real data
                # change the dim from [num_shots, num_receiver, nt] to [num_shots,nt,num_receiver]
                d_real = d_real.permute(0, 2, 1)
                # insert one dim at the second place, so that the dim is [num_shots,1,nt,num_receiver]
                d_real = d_real.unsqueeze(1)
                disc_real = netD(d_real)
                disc_real = disc_real.mean()
    
                # train with fake data
                d_fake = d_fake.permute(0, 2, 1)
                d_fake = d_fake.unsqueeze(1)
                disc_fake = netD(d_fake)
                disc_fake = disc_fake.mean()
    
                # train with interpolates data                
                gradient_penalty = calc_gradient_penalty(
                    netD, d_real, d_fake, 
                    batch_size=num_shots_per_batch, 
                    channel=1, lamb=10, 
                    device=device)
                             
                disc_cost = disc_fake - disc_real + gradient_penalty
                print ('Epoch: %03d  Ite: %05d  DLoss: %f' % (epoch+1, itx, disc_cost.item()))
                disc_cost.backward()
                # Clips gradient norm of netD parameters
                torch.nn.utils.clip_grad_norm_(netD.parameters(),1e6) #1e3 for smoothed initial model; 1e6 for linear model
                # optimize
                optim_d.step()                 

            # *====== Inversion Optimization ======*
            for p in netD.parameters():
                p.requires_grad_(False)  # freeze D,to avoid computation
                
            optim_g.zero_grad()
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
                loss = Wasserstein1(batch_rcv_amps_pred, batch_rcv_amps_true, trans_type, theta=1.1)            
            else:
                raise NotImplementedError
            loss_epoch += loss.item()
            loss.backward()
            # Clips gradient value of model
            torch.nn.utils.clip_grad_value_(vp_model, torch.quantile(vp_model.grad.detach().abs(), 0.98)) 
            optim_g.step()
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