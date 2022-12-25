import os.path

import numpy as np
import matplotlib.pyplot as plt
import torch
from models.unet import UNet
import torch.optim as optim
import torch.fft as fft
import torch.nn as nn
from tqdm.notebook import tqdm
from skimage.metrics import peak_signal_noise_ratio as compute_psnr
import glob
from torch.utils.tensorboard import SummaryWriter
import os
import argparse
from Parser import parse_args

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark =True
dtype = torch.cuda.FloatTensor

args = argparse.ArgumentParser(description='Training taxonomy expansion model')
args.add_argument('--config', default=None, type=str, help='config file path (default: None)')
args = args.parse_args()
config = parse_args(args.config)

log_dir = "../log/super_resolution"
if not os.path.exists(os.path.join(log_dir, config.name)):
    os.mkdir(os.path.join(log_dir, config.name))
writer = SummaryWriter(os.path.join(log_dir, config.name))

over_parameterization = config.over_parameterization
raw_data = np.load(config.data_path)