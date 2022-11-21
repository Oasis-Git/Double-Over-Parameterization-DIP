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

from models import *
from models.downsampler import Downsampler
from utils_dip.inpainting_utils import *
from utils_dip.utils import *
from Parser import parse_args
from torch.utils.tensorboard import SummaryWriter
from skimage.metrics import peak_signal_noise_ratio


torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark =True
dtype = torch.cuda.FloatTensor

args = argparse.ArgumentParser(description='Training taxonomy expansion model')
args.add_argument('--config', default=None, type=str, help='config file path (default: None)')
args = args.parse_args()
config = parse_args(args['config'])

log_dir = "../log/inpainting"
if not os.path.exists(os.path.join(log_dir, config['name'])):
    os.mkdir(os.path.join(log_dir, config['name']))
writer = SummaryWriter(os.path.join(log_dir, config['name']))

over_parameterization = config["over parameterization"]

# data preparation
imgs = load_image_and_mask(config['data_path'], config['mask_path'])

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

# Losses
mse = torch.nn.MSELoss().type(dtype)

c, h, w = imgs.shape
eta = sparse_noise_3d(h, w)
if config["noise"] is True:
    img_var = np_to_torch(eta+imgs).type(dtype)
else:
    img_var = np_to_torch(imgs).type(dtype)

mask_var = np_to_torch(imgs['mask']).type(dtype)

r_img_cor_p_torch = torch.zeros_like(torch.tensor(img_var)).type(dtype).normal_()*1e-5
r_img_cor_n_torch = torch.zeros_like(torch.tensor(img_var)).type(dtype).normal_()*1e-5
r_img_cor_p_torch.requires_grad = True
r_img_cor_n_torch.requires_grad = True

# Optimize
optimizer1 = get_optimizer(config['optimizer 1'], net.parameters(), config['l1'])
if config['optimizer 2'] is not None:
    optimizer2 = get_optimizer(config['optimizer 2'], [r_img_cor_p_torch, r_img_cor_n_torch], config['l2'])
else:
    optimizer2 = None

for i in range(0, config['num iter']):
    optimizer1.zero_grad()
    if over_parameterization:
        optimizer2.zero_grad()
    output = net(net_input)
    r_img_cor_torch = r_img_cor_p_torch ** 2 - r_img_cor_n_torch ** 2
    recon = torch_to_np(output)
    noise_recon = r_img_cor_torch.data.squeeze(0).cpu().numpy()
    data_loss = np.linalg.norm(recon - imgs['HR_np']) / np.linalg.norm(imgs['HR_np'])
    if over_parameterization:
        total_loss = mse((output + r_img_cor_torch) * mask_var, img_var * mask_var)
    else:
        total_loss = mse(output * mask_var, img_var * mask_var)

    total_loss.backward()
    optimizer1.step()
    if over_parameterization:
        optimizer2.step()

    writer.add_scalar('total loss', total_loss.item(), i)
    psnr = peak_signal_noise_ratio(imgs['HR_np'], torch_to_np(imgs))
    writer.add_scalar('PSNR', psnr)

    if i % config['snapshot']:
        writer.add_figure(tag='output of epoch' + str(i), figure=torch_to_np(recon), global_step=i)
