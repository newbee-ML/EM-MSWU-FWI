import numpy as np
import torch
import random


# setup seed 
def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)  # cpu 
    torch.cuda.manual_seed(seed)  # gpu 
    torch.backends.cudnn.deterministic = True  # cudnn
    np.random.seed(seed)  # numpy
    
