# -*- codeing = utf-8 -*-
# @time:2022/9/11 上午11:57
# Author:Xuewen Shen
# @File:fig_analyse.py
# @Software:PyCharm

# This file is for experimental results(figures) loading, andlyzing and reconstruction


import matplotlib.pyplot as plt
import torch
import cv2
import math

filepath = 'Analysis_fig_YP//'
figpath = filepath + 'fig_YP//'
savepath = filepath + 'results//'


def tt(x, device='cpu'):
    return torch.from_numpy(x).to(device)

def tn(x):
    return x.detach().cpu().numpy()

def read_fig(figname: str, device='cpu'):
    '''
    :param figname: name of figures in dictionary figpath
    :param device: project figures to device default 'cpu', else use 'cuda:0' or 'cuda:1'
    :return:
    '''
    img_bgrm = cv2.imread(figpath + figname, cv2.IMREAD_GRAYSCALE)
    return tt(img_bgrm, device=device)


def conv2d(data: torch.Tensor, kernel: torch.Tensor, step=1,device='cpu'):
    '''
    :param data: [M,N] shaped torch.Tensor of data matrix
    :param kernel:[m,n] shaped torch.Tensor of kernel used for convolution
    :param step: int, steps of interval for convolution
    :return:
    '''
    data=data.to(device)
    kernel=kernel.to(device)
    M, N = data.shape
    m, n = kernel.shape
    N_m = math.floor((M - m) / step + 1)
    N_n = math.floor((N - n) / step + 1)
    data_sorted = torch.Tensor(
        [[data[k_m * step:k_m * step + m, k_n * step:k_n * step + n] for k_n in range(N_n)] for k_m in range(N_m)])
    data_conv=torch.einsum('ijmn,mn->ij',data_sorted,kernel)
    return data_conv
