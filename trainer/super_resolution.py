import matplotlib.pyplot as plt
import argparse
import os
import sys
sys.path.append("..")
sys.path.extend([os.path.join(root, name) for root, dirs, _ in os.walk("../") for name in dirs])

import numpy as np
import scipy.sparse
import scipy
import torch
import torch.optim
from torch.utils.tensorboard import SummaryWriter
from skimage.metrics import peak_signal_noise_ratio
from skimage.util import random_noise

from models import *
from models.downsampler import Downsampler
from utils_dip.sr_utils import *
from utils_dip.utils import *
from utils_dip.common_utils import *
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

# data preparation
factor = config.factor
imgs = load_LR_HR_imgs_sr(config.data_path, -1, factor, None)


# net preparation
input_depth = 32
INPUT =     'noise'
pad   =     'reflection'
OPT_OVER =  'net'
KERNEL_TYPE='lanczos2'
net_input = get_noise(input_depth, INPUT, (imgs['HR_pil'].size[1], imgs['HR_pil'].size[0])).type(dtype).detach()

NET_TYPE = 'skip' # UNet, ResNet
net = get_net(input_depth, 'skip', pad,
              skip_n33d=128,
              skip_n33u=128,
              skip_n11=4,
              num_scales=5,
              upsample_mode='bilinear').type(dtype)

downsampler = Downsampler(n_planes=3, factor=factor, kernel_type=KERNEL_TYPE, phase=0.5, preserve_size=True).type(dtype)

# Loss
if config.loss == "L2":
    loss = torch.nn.MSELoss().type(dtype)
elif config.loss == "L1":
    loss = torch.nn.L1Loss().type(dtype)

c, h, w = imgs['LR_np'].shape
if config.noise is True:
    img_LR_var = np_to_torch(random_noise(imgs['LR_np'], mode=config.noise_mode, amount=config.noise_amount)).type(dtype)
else:
    img_LR_var = np_to_torch(imgs['LR_np']).type(dtype)

r_img_cor_p_torch = torch.zeros_like(torch.tensor(img_LR_var)).type(dtype).normal_()*1e-5
r_img_cor_n_torch = torch.zeros_like(torch.tensor(img_LR_var)).type(dtype).normal_()*1e-5
r_img_cor_p_torch.requires_grad = True
r_img_cor_n_torch.requires_grad = True

# Optimize
optimizer1 = get_optimizer(config.optimizer_1, net.parameters(), config.l1)
if config.optimizer_2 is not None:
    optimizer2 = get_optimizer(config.optimizer_2, [r_img_cor_p_torch, r_img_cor_n_torch], config.l2)
else:
    optimizer2 = None
tv = config.tv

for i in range(0, config.num_iter):
    optimizer1.zero_grad()
    if over_parameterization:
        optimizer2.zero_grad()
    out_HR = net(net_input)
    out_LR = downsampler(out_HR)
    r_img_cor_torch = r_img_cor_p_torch ** 2 - r_img_cor_n_torch ** 2
    recon = torch_to_np(out_HR)
    noise_recon = r_img_cor_torch.data.squeeze(0).cpu().numpy()
    data_loss = np.linalg.norm(recon - imgs['HR_np']) / np.linalg.norm(imgs['HR_np'])
    if over_parameterization:
        total_loss = loss(out_LR + r_img_cor_torch, img_LR_var)
    else:
        total_loss = loss(out_LR, img_LR_var)

    if tv:
        total_loss += 2e-6 * tv_loss(out_LR, beta=0.6)

    total_loss.backward()
    optimizer1.step()
    if over_parameterization:
        optimizer2.step()

    writer.add_scalar('total loss', total_loss.item(), i)
    psnr = peak_signal_noise_ratio(imgs['HR_np'], torch_to_np(out_HR))
    writer.add_scalar('PSNR', psnr, i)

    if i % config.snapshot == 0:
        writer.add_image(tag='output of epoch' + str(i), img_tensor=torch_to_np(out_HR), global_step=i)
