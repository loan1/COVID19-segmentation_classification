import os
import numpy as np
import torch
import random

def seed_everything(seed):        

    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed) # set python seed

    np.random.seed(seed) # seed the global NumPy RNG

    torch.manual_seed(seed) # seed the RNG for all devices (both CPU and CUDA):
    torch.cuda.manual_seed_all(seed)