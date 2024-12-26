############################################################
# visualization tools 
# ------
# author: Hongtao Wang(汪泓涛)
# email:  colin_wht@stu.xjtu.edu.cn
# packages: matplotlib
############################################################
import os
import numpy as np
import matplotlib.pyplot as plt

def plot_model(v_model, save_path, 
               dx=None, title='2D Velocity Model', bar_name='Velocity (m/s)',
               figsize=(10, 3), dpi=500, pure_plot=False, v_range=None):
    fig = plt.figure(figsize=figsize, dpi=dpi)
    ax = fig.add_subplot(111)
    n_depth, n_offset = v_model.shape
    if pure_plot:
        ax.imshow(v_model, cmap='jet')
        ax.set_axis_off()
    else:
        cax = ax.imshow(v_model, cmap='jet')
        if v_range is not None:
            cax.set_clim(*v_range)
        fig.colorbar(cax, ax=ax, label=bar_name)
        if dx is not None:
            ax.set_xticks(np.arange(0, n_offset, 50), np.arange(0, int(n_offset*dx), int(50*dx)))
            ax.set_yticks(np.arange(0, n_depth, 30), np.arange(0, int(n_depth*dx), int(30*dx)))
        ax.set_title(title)
        ax.set_xlabel('Offset (m)')
        ax.set_ylabel('Depth (m)')
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    plt.close('all')
    
    
def plot_comp_figs(model_pred, model_true, save_path, dx, title='FWI', bar_name='Velocity (m/s)', figsize=(10, 9), dpi=500):
    fig, ax = plt.subplots(nrows=3, ncols=1, figsize=figsize, dpi=dpi)
    n_depth, n_offset = model_pred.shape
    v_min = model_true.min()
    v_max = model_true.max()
    # FWI
    cax_1 = ax[0].imshow(model_pred, cmap='jet')
    cax_1.set_clim(v_min, v_max)
    fig.colorbar(cax_1, ax=ax[0], label=bar_name, pad=0.02, fraction=0.02)
    ax[0].set_xticks(np.arange(0, n_offset, 50), np.arange(0, int(n_offset*dx), int(50*dx)))
    ax[0].set_yticks(np.arange(0, n_depth, 30), np.arange(0, int(n_depth*dx), int(30*dx)))
    ax[0].set_title('%s $V_p$ Model' % title)
    ax[0].set_xlabel('Offset (m)')
    ax[0].set_ylabel('Depth (m)')
    # True
    cax_2 = ax[1].imshow(model_true, cmap='jet')
    cax_2.set_clim(v_min, v_max)
    fig.colorbar(cax_2, ax=ax[1], label=bar_name, pad=0.02, fraction=0.02)
    ax[1].set_xticks(np.arange(0, n_offset, 50), np.arange(0, int(n_offset*dx), int(50*dx)))
    ax[1].set_yticks(np.arange(0, n_depth, 30), np.arange(0, int(n_depth*dx), int(30*dx)))
    ax[1].set_title('True $V_p$ Model')
    ax[1].set_xlabel('Offset (m)')
    ax[1].set_ylabel('Depth (m)')
    # bias 
    cax_3 = ax[2].imshow(model_true-model_pred, cmap='seismic')
    fig.colorbar(cax_3, ax=ax[2], label=bar_name, pad=0.02, fraction=0.02)
    ax[2].set_xticks(np.arange(0, n_offset, 50), np.arange(0, int(n_offset*dx), int(50*dx)))
    ax[2].set_yticks(np.arange(0, n_depth, 30), np.arange(0, int(n_depth*dx), int(30*dx)))
    ax[2].set_title('Bias between %s and True' % title)
    ax[2].set_xlabel('Offset (m)')
    ax[2].set_ylabel('Depth (m)')
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0.1)
    plt.close('all')


def plot_two_model(model_1, model_2, model1_info, model2_info, figsize=(10, 3), save_path=None, dpi=500, cmap='jet'):
    fig, ax = plt.subplots(nrows=2, ncols=1, figsize=figsize, dpi=dpi)
    # model 1
    cax_1 = ax[0].imshow(model_1, cmap=cmap, aspect='auto')
    fig.colorbar(cax_1, ax=ax[0], label=model1_info['bar_name'])
    ax[0].set_title(model1_info['title'])
    ax[0].set_xlabel('Offset')
    ax[0].set_ylabel('Depth')
    # model 2
    cax_2 = ax[1].imshow(model_2, cmap=cmap, aspect='auto')
    fig.colorbar(cax_2, ax=ax[1], label=model2_info['bar_name'])
    ax[1].set_title(model2_info['title'])
    ax[1].set_xlabel('Offset')
    ax[1].set_ylabel('Depth')
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0.1)
    plt.close('all')


def plot_grad(grad_map, save_path, dx=None):
    fig = plt.figure(figsize=(10, 3), dpi=500)
    ax = fig.add_subplot(111)
    n_depth, n_offset = grad_map.shape
    cax = ax.imshow(grad_map, cmap='jet', aspect='auto')
    fig.colorbar(cax, ax=ax, label='gradient')
    if dx is not None:
        ax.set_xticks(np.arange(0, n_offset, 50), np.arange(0, int(n_offset*dx), int(50*dx)))
        ax.set_yticks(np.arange(0, n_depth, 30), np.arange(0, int(n_depth*dx), int(30*dx)))
    ax.set_xlabel('Offset (m)')
    ax.set_ylabel('Depth (m)')
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    plt.close('all')

def plot_gth(gth_fw, gth_true, name, save_path, dx=None, dt=None):
    fig, ax = plt.subplots(3, 1, figsize=(8, 20))
    n_t, n_offset = gth_fw.T.shape
    gth_fw_k = gth_fw.T
    gth_t_k = gth_true.T
    diff = gth_t_k-gth_fw_k
    abs_max = max(np.abs(gth_fw_k).max(), np.abs(gth_t_k).max())
    gth_fw_k = gth_fw_k / abs_max
    gth_t_k = gth_t_k / abs_max
    cax1 = ax[0].imshow(gth_t_k, cmap='seismic', aspect='auto')
    cax1.set_clim(-1, 1)
    fig.colorbar(cax1, ax=ax[0], label='Amplitude')
    ax[0].set_title('True')
    cax2 = ax[1].imshow(gth_fw_k, cmap='seismic', aspect='auto')
    cax2.set_clim(-1, 1)
    fig.colorbar(cax2, ax=ax[1], label='Amplitude')
    ax[1].set_title('Forward of %s' % name)
    cax3 = ax[2].imshow(diff, cmap='seismic', aspect='auto')
    cax3.set_clim(-20, 20)
    ax[2].set_title('Bias between %s and True' % name)
    fig.colorbar(cax3, ax=ax[2], label='Amplitude')
        
    for j in range(3):
        if dx is not None:
            ax[j].set_xticks(np.arange(0, n_offset, 50), np.arange(0, int(n_offset*dx), int(50*dx)))
            ax[j].set_yticks(np.arange(0, n_t, 300), np.arange(0, int(n_t*dt), int(300*dt)))
        ax[j].set_xlabel('Offset (m)')
        ax[j].set_ylabel('Time (ms)')
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0.1)
    plt.close('all')
    

def plot_log(log_list, col_names, save_root):
    log_array = np.array(log_list)
    for k, name in enumerate(col_names):
        fig = plt.figure(figsize=(6, 4), dpi=200)
        ax = fig.add_subplot(111)
        ax.plot(log_array[:, 0], log_array[:, k+2], 'o', ls='--', color='gray',
                linewidth=1, markersize=2, markeredgewidth=0.1, markerfacecolor='tab:blue', markeredgecolor='red')
        ax.set_xlabel('Epoch')
        ax.set_ylabel(name)
        plt.savefig(os.path.join(save_root, 'log_%s.png' % name), bbox_inches='tight', pad_inches=0.1)
        plt.close('all')


# def plot_rec_amp(rec_amp, shot_n, rec_n, save_path):
#     vmin, vmax = torch.quantile(rec_amp[0],
#                                 torch.tensor([0.05, 0.95]).to(para_dict['device']))
#     fig, ax = plt.subplots(1, 2, figsize=(10.5, 7), sharey=True)
#     cax0 = ax[0].imshow(receiver_amplitudes[shot_n].cpu().T, aspect='auto',
#                 cmap='seismic', vmin=vmin, vmax=vmax)
#     cax1 = ax[1].imshow(receiver_amplitudes[:, rec_n].cpu().T, aspect='auto',
#                 cmap='seismic', vmin=vmin, vmax=vmax)
#     ax[0].set_xlabel("Channel")
#     ax[0].set_ylabel("Time Sample")
#     ax[1].set_xlabel("Shot")
#     fig.colorbar(cax0, ax=ax[0])
#     fig.colorbar(cax1, ax=ax[1])
#     plt.savefig(save_path, bbox_inches='tight', pad_inches=0.1, dpi=500)
#     plt.close('all')

