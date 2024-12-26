"""
Metrics tools
---
SSIM:
SNR:
RE
@author: hongtao wang (colin_wht@stu.xjtu.edu.cn)


"""

import torch
import torch.nn.functional as F

from torchmetrics.image import StructuralSimilarityIndexMeasure as SSIM
from torchmetrics.image import PeakSignalNoiseRatio as PSNR


###############################################################################
# metrics used in validation
###############################################################################

class compute_metrics(torch.nn.Module):
    def __init__(self, device=0):
        super(compute_metrics, self).__init__()
        self.my_SSIM = SSIM(k1=1e-4, k2=9*1e-4).to(device)
    
    def forward(self, model_pred, model_true):
        h, w = model_pred.shape
        # SSIM 
        ssim_n = self.my_SSIM(model_pred.reshape(1, 1, h, w), model_true.reshape(1, 1, h, w))
        # SNR
        snr_n = self.my_SNR(model_pred, model_true)
        # error 
        error_n = self.my_error(model_pred, model_true)
        # to dict
        met_dict = {
            'SSIM': ssim_n.item(),
            'SNR': snr_n.item(),
            'RSE': error_n.item()
        }
        return met_dict

    @staticmethod
    def my_SNR(model_pred, model_true):
        SNR_n = 10*torch.log10(torch.sum(torch.square(model_true))/torch.sum(torch.square(model_true-model_pred)))
        return SNR_n

    @staticmethod
    def my_error(model_pred, model_true):
        RSE_n = (torch.sum(torch.square(model_true-model_pred))/torch.sum(torch.square(model_true))) ** 0.5
        return RSE_n
    

if __name__ == "__main__":
    # * do a test
    device = 'cpu'
    preds = torch.rand([256, 256]).to(device)
    target = preds * 0.75
    met_opt = compute_metrics('cpu')
    print(met_opt(preds, target))