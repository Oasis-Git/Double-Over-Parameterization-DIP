import os.path
import sys
sys.path.append("..")
sys.path.extend([os.path.join(root, name) for root, dirs, _ in os.walk("../") for name in dirs])

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.optim as optim
import torch.nn as nn
from tqdm.notebook import tqdm
from skimage.metrics import peak_signal_noise_ratio as compute_psnr
import glob
from torch.utils.tensorboard import SummaryWriter
import os
import argparse
from Parser import parse_args
from skimage.metrics import peak_signal_noise_ratio
from skimage.util import random_noise

from models.unet import UNet
from models import *
from utils_dip.common_utils import *
from utils_dip.mri_utils import *


torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark =True
dtype = torch.cuda.FloatTensor

args = argparse.ArgumentParser(description='Training taxonomy expansion model')
args.add_argument('--config', default=None, type=str, help='config file path (default: None)')
args = args.parse_args()
config = parse_args(args.config)

log_dir = "../log/mri_reconstruction"
if not os.path.exists(os.path.join(log_dir, config.name)):
    os.mkdir(os.path.join(log_dir, config.name))
writer = SummaryWriter(os.path.join(log_dir, config.name))

over_parameterization = config.over_parameterization
ksp, mps, gt = read_data(config.data_path, select=config.select)

noise = random_noise(gt, mode=config.noise_mode, amount=config.noise_amount)
ksp = mps_and_gt_to_ksp(mps, noise)

mask = torch.tensor(make_vdrs_mask(640, 372, 372//4, 20), dtype=torch.complex64)

input_depth = 32
INPUT =     'noise'
pad   =     'reflection'
OPT_OVER =  'net'
KERNEL_TYPE='lanczos2'
net_input = get_noise(input_depth, INPUT, (gt.shape[0], gt.shape[1])).type(dtype).detach()
net = get_net(input_depth, 'skip', pad,
              n_channels=2,
              skip_n33d=128,
              skip_n33u=128,
              skip_n11=4,
              num_scales=5,
              upsample_mode='bilinear').type(dtype)

# Loss
if config.loss == "L2":
    loss = torch.nn.MSELoss().type(dtype)
elif config.loss == "L1":
    loss = torch.nn.L1Loss().type(dtype)
else:
    assert -1

r_img_cor_p_torch = torch.zeros_like(torch.tensor(gt)).type(dtype).normal_()*1e-5
r_img_cor_n_torch = torch.zeros_like(torch.tensor(gt)).type(dtype).normal_()*1e-5
r_img_cor_p_torch.requires_grad = True
r_img_cor_n_torch.requires_grad = True

# Optimize
optimizer1 = get_optimizer(config.optimizer_1, net.parameters(), config.l1)
if config.optimizer_2 is not None:
    optimizer2 = get_optimizer(config.optimizer_2, [r_img_cor_p_torch, r_img_cor_n_torch], config.l2)
else:
    optimizer2 = None
tv = config.tv

net = net.cuda()
criterion = loss.cuda()
mps = mps.cuda()
ksp = ksp.cuda()
mask = mask.cuda()
r_img_cor_p_torch = r_img_cor_p_torch.cuda()
r_img_cor_n_torch = r_img_cor_n_torch.cuda()

for i in range(0, config.num_iter):
    optimizer1.zero_grad()
    if over_parameterization:
        optimizer2.zero_grad()
    out = net(net_input).squeeze()
    out_complex = torch.view_as_complex(out.permute(1,2,0).contiguous())
    r_img_cor_torch = r_img_cor_p_torch ** 2 - r_img_cor_n_torch ** 2

    # loss = criterion(mask * (18000 * ksp1 + r_img_cor_p_torch **2 - r_img_cor_n_torch **2) , mask * pred_ksp.squeeze())
    if over_parameterization:
        total_loss = loss(mask * 14431031 * ksp,
                          (mask * mps_and_gt_to_ksp(mps, out_complex) + r_img_cor_torch).squeeze())
    else:
        total_loss = loss(mask * 14431031 * ksp,
                          mask * mps_and_gt_to_ksp(mps, out_complex).squeeze())

    if tv:
        total_loss += 2e-6 * tv_loss(out, beta=0.6)

    total_loss.backward()
    optimizer1.step()
    if over_parameterization:
        optimizer2.step()

    writer.add_scalar('total loss', total_loss.item(), i)
    psnr = peak_signal_noise_ratio(np.array(gt, dtype=np.float), np.array(out_complex.detach().cpu(), dtype=np.float))
    writer.add_scalar('PSNR', psnr, i)

    if i % config.snapshot == 0:
        writer.add_image(tag='output of epoch' + str(i), img_tensor=np.array(out_complex.unsqueeze(dim=0).detach().cpu()), global_step=i)

writer.add_image(tag='gt', img_tensor=gt.unsqueeze(dim=0).detach().cpu())
writer.add_image(tag='noise', img_tensor=noise.unsqueeze(dim=0).detach().cpu())
