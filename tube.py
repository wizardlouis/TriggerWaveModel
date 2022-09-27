# -*- codeing = utf-8 -*-
# @time:2022/5/13 下午7:52
# Author:Xuewen Shen
# @File:tube.py
# @Software:PyCharm

import torch
import torch.nn as nn
import numpy as np
import os
import matplotlib.pyplot as plt
from RD_param import *


def tn(x):
    return x.detach().clone().cpu().numpy()


class Tube:
    # generate finite length tube
    default_param = dict(
        length=200,  # um
    )

    def __init__(self, length=None):
        if length == None:
            self.length = Tube.default_param['length']
        else:
            self.length = length

    def gen_cell(self, n_components, dlength, device='cpu'):
        self.cell = torch.zeros(n_components, int(self.length / dlength)).to(device)
        return self.cell


class LabeledTensor:
    offset = 0.

    # using labeled tensor to mark the first dimension of concentration matrix data, thus simplify the calculation
    def __init__(self, name: list, data: torch.Tensor, device='cpu'):
        assert len(name) == data.shape[0]
        self.name = name
        self.data = data.to(device)
        self.device = device
        for attr in name:
            self.__setattr__(attr, self.data[self.name.index(attr)])

    def __getitem__(self, item):
        return self.data[self.name.index(item)]

    def get_data(self):
        return self.data

    def get_name(self):
        return self.name

    def set_data(self, data):
        self.data = data.to(self.device)

    def __add__(self, other):
        assert self.name == other.name
        return LabeledTensor(self.name, self.data + other.data, device=self.device)

    def __sub__(self, other):
        assert self.name == other.name
        return LabeledTensor(self.name, self.data - other.data, device=self.device)

    def copy(self, device='cpu'):
        return LabeledTensor(self.name, self.data, device=device)

    def non_neg(self):
        self.data = torch.where(self.data > 0, self.data,
                                LabeledTensor.offset * torch.ones_like(self.data, device=self.data.device))
        return LabeledTensor(self.name, self.data, device=self.device)


class RD_Simulator:
    # Simulator of reaction-diffusion system,the formulation of reaction term must come from extra definition
    default_param = dict(
        dlength=1.,
        dt=0.1,
        D=10.,
        name=['Cb', 'M', 'Mp', 'Mb', 'Mbp', 'AP0', 'AP1', 'I0', 'I1', 'E0', 'E1'],
    )

    def __init__(self, P=None):
        if P == None:
            self.P = RD_Simulator.default_param
        else:
            self.P = P
        self.alpha = self.P['D'] * self.P['dt'] / self.P['dlength'] ** 2

    def gen_cell(self, length=None, device='cpu'):
        tube = Tube(length=length)
        cell = tube.gen_cell(self.P['n_components'], self.P['dlength'], device=device)
        labeledcell = LabeledTensor(self.P['name'], cell, device=device)
        return labeledcell

    def __getattr__(self, item):
        return None

    def V_Diffusion(self, labeledcell):
        data = labeledcell.get_data()
        Vp = torch.cat([data[:, -1:], data[:, :-1]], dim=1)
        Vm = torch.cat([data[:, 1:], data[:, :1]], dim=1)
        return Vp + Vm - 2 * data

    def set_V_Reaction(self, func):
        self.V_Reaction = func

    def set_Reaction_adjacent(self, func):
        self.Reaction_adjacent = func

    def update(self, labeledcell, device='cpu'):
        if self.Reaction_adjacent is None:
            DReaction = self.P['dt'] * self.V_Reaction(labeledcell, self.P)
            DDiffusion = self.alpha * self.V_Diffusion(labeledcell)
            return (labeledcell + LabeledTensor(self.P['name'], DReaction, device=device) + LabeledTensor(
                self.P['name'], DDiffusion, device=device)).non_neg()
        else:
            adjacent = self.Reaction_adjacent(labeledcell, self.P, device=device)
            DReaction = self.P['dt'] * self.V_Reaction(labeledcell, self.P, adjacent)
            DDiffusion = self.alpha * self.V_Diffusion(labeledcell)
            return (labeledcell + LabeledTensor(self.P['name'], DReaction, device=device) + LabeledTensor(
                self.P['name'],
                DDiffusion, device=device)).non_neg(), (adjacent).non_neg()

    def update_bound_fixed(self, labeledcell, device='cpu'):
        if self.Reaction_adjacent is None:
            DReaction = self.P['dt'] * self.V_Reaction(labeledcell, self.P)
            DDiffusion = self.alpha * self.V_Diffusion(labeledcell)
            L = labeledcell + LabeledTensor(self.P['name'], DReaction, device=device) + LabeledTensor(
                self.P['name'], DDiffusion, device=device).non_neg()
            L.data[:, 0] = labeledcell.data[:, 0]
            L.data[:,-1]=labeledcell.data[:,-1]
            return L
        else:
            adjacent = self.Reaction_adjacent(labeledcell, self.P, device=device)
            DReaction = self.P['dt'] * self.V_Reaction(labeledcell, self.P, adjacent)
            DDiffusion = self.alpha * self.V_Diffusion(labeledcell)
            L = (labeledcell + LabeledTensor(self.P['name'], DReaction, device=device) + LabeledTensor(
                self.P['name'],
                DDiffusion, device=device)).non_neg()
            L.data[:, 0] = labeledcell.data[:, 0]
            L.data[:, -1] = labeledcell.data[:, -1]
            return L, (adjacent).non_neg()



    def get_labeledcell(self):
        return self.labeledcell

    def run(self, labeledcell: LabeledTensor, T: int, device='cpu', bound_fixed=False):
        assert labeledcell is not None
        assert self.V_Reaction is not None
        n_steps = int(T / self.P['dt'])
        LC = labeledcell.copy(device=device)
        if self.Reaction_adjacent is None:
            result = LC.copy(device=device).data.unsqueeze(dim=0)
            for s in range(n_steps):
                if not bound_fixed:
                    LC = self.update(LC, device=device)
                else:
                    LC = self.update_bound_fixed(LC, device=device)
                result = torch.cat((result, LC.copy().data.unsqueeze(dim=0)), dim=0)
            return result
        else:
            result = LC.copy(device=device).data.unsqueeze(dim=0)
            result_adj = self.Reaction_adjacent(LC, self.P, device=device).copy(device=device).data.unsqueeze(dim=0)
            for s in range(n_steps):
                if not bound_fixed:
                    LC, adjacent = self.update(LC, device=device)
                else:
                    LC, adjacent = self.update_bound_fixed(LC, device=device)
                result = torch.cat((result, LC.copy(device=device).data.unsqueeze(dim=0)), dim=0)
                result_adj = torch.cat([result_adj, adjacent.copy(device=device).data.unsqueeze(dim=0)], dim=0)
            return result, result_adj


def Reaction_simp_adjacent(LC, P, device='cpu'):
    s_AP = P['Ek_AP'] * P['ATP'] * LC.Mb
    s_Wee1 = P['Ek_Wee1'] * P['ATP'] * LC.Mb
    s_Cdc25 = P['Ek_Cdc25'] * P['ATP'] * (LC.Mb + P['kinase'])
    AP_A = P['AP_T'] * s_AP ** (P['L'] - 1) / sum([s_AP ** l for l in range(P['L'])])
    Wee1_A = P['Wee1_T'] * sum([s_Wee1 ** m for m in range(P['M'] - 1)]) / sum([s_Wee1 ** m for m in range(P['M'])])
    Cdc25_A = P['Cdc25_T'] * s_Cdc25 ** (P['N'] - 1) / sum([s_Cdc25 ** n for n in range(P['N'])])
    return LabeledTensor(['AP_A', 'Wee1_A', 'Cdc25_A'],
                         torch.cat([V.reshape((1,) + V.shape) for V in [AP_A, Wee1_A, Cdc25_A]], dim=0), device=device)


AP_offset = 1.


def Reaction_simp_adjacent_Mb(Mb, P, device='cpu'):
    s_AP = P['Ek_AP'] * P['ATP'] * Mb
    s_Wee1 = P['Ek_Wee1'] * P['ATP'] * Mb
    s_Cdc25 = P['Ek_Cdc25'] * P['ATP'] * (Mb + P['kinase'])
    AP_A = AP_offset + P['AP_T'] * (s_AP ** (P['L'] - 1)) / sum([s_AP ** l for l in range(P['L'])])
    Wee1_A = P['Wee1_T'] * sum([s_Wee1 ** m for m in range(P['M'] - 1)]) / sum([s_Wee1 ** m for m in range(P['M'])])
    Cdc25_A = P['Cdc25_T'] * (s_Cdc25 ** (P['N'] - 1)) / sum([s_Cdc25 ** n for n in range(P['N'])])
    return LabeledTensor(['AP_A', 'Wee1_A', 'Cdc25_A'],
                         torch.cat([V.reshape((1,) + V.shape) for V in [AP_A, Wee1_A, Cdc25_A]], dim=0), device=device)


def plot_simp_nullcline(LC, P, Cdk_range, Traj=None):
    Cdk_list = torch.linspace(*Cdk_range, 100)
    adj = Reaction_simp_adjacent_Mb(Cdk_list, P)
    Cb1 = P['v_B1'] / (P['k_degB'] * adj.AP_A)
    Cb2 = P['k_B-'] / (P['k_B+'] * (
            P['M_T'] / (Cdk_list * (1 + P['k_+1'] * P['ATP'] * adj.Wee1_A / P['k_+2'] / adj.Cdc25_A)) - 1))
    plt.plot(tn(Cb1), tn(Cdk_list), color='red')
    plt.plot(tn(Cb2), tn(Cdk_list), color='blue')
    plt.xlim(0, 100)
    if Traj is not None:
        Traj = tn(Traj)
        plt.plot(Traj[0], Traj[1], color='purple')
    plt.show()


simp_name_list = [
    'Cb', 'Mb', 'Mbp'
]

simp_name_list_low_pass = [
    'Cb', 'Mb', 'Mbp', 'AP_A', 'Wee1_A', 'Cdc25_A'
]

timescale = 2.
timescale2 = 1e2
simp_param_0524_for_oscillator = {
    'n_components': 3,
    'ATP': 1e6, 'ADP': 1e4, 'Pi': 1e6,
    'v_B1': 3e-1 / timescale, 'k_degB': 2e-4 / timescale, 'k_B+': 3e-4 / timescale, 'k_B-': 5e-2 / timescale,
    'k_+1': 1e-7 / timescale, 'k_+2': 1.05 / timescale,
    'Ek_AP': 5.4e-8, 'Ek_Wee1': 5.7e-9, 'Ek_Cdc25': 6.6e-9, 'L': 17, 'M': 5, 'N': 5,
    'M_T': 500, 'AP_T': 100, 'Wee1_T': 60, 'Cdc25_T': 150, 'kinase': 30
}

simp_param = {
    'n_components': 3,
    'ATP': 1e6, 'ADP': 1e4, 'Pi': 1e6,
    'v_B1': 2e-1 / timescale, 'k_degB': 2e-3 / timescale, 'k_B+': 3e-4 / timescale, 'k_B-': 5e-2 / timescale,
    'k_+1': 1e-7 / timescale2, 'k_+2': 1.05 / timescale2,
    'Ek_AP': 9.e-8, 'Ek_Wee1': 5.7e-8, 'Ek_Cdc25': 1.67e-8, 'L': 17, 'M': 5, 'N': 11,
    'M_T': 500, 'AP_T': 100, 'Wee1_T': 60, 'Cdc25_T': 150, 'kinase': 30
}

simp_param_low_pass = {
    'n_components': 6,
    'ATP': 1e6, 'ADP': 1e4, 'Pi': 1e6,
    'v_B1': 2e-1 / timescale, 'k_degB': 2e-3 / timescale, 'k_B+': 3e-4 / timescale, 'k_B-': 5e-2 / timescale,
    'k_+1': 1e-7 / timescale, 'k_+2': 1.05 / timescale,
    'Ek_AP': 9.e-8, 'Ek_Wee1': 5.7e-8, 'Ek_Cdc25': 1.67e-8, 'L': 15, 'M': 5, 'N': 11,
    'M_T': 500, 'AP_T': 100, 'Wee1_T': 60, 'Cdc25_T': 150, 'kinase': 30
}

if __name__ == "__main__":
    # Cb = 20.
    # Mt = 240.
    # APt = 200.,
    # It = 60.
    # Et = 150.
    # # init_Var = torch.Tensor([20., 100., 100., 20., 20., 40., 10., 10., 50., 20., 130.]).unsqueeze(dim=1)
    # init_Var = torch.Tensor([20., 100., 100., 20., 20., 190., 10., 50., 10., 130., 20.]).unsqueeze(dim=1)

    init_Var = torch.Tensor([30., 3., 50.]).unsqueeze(dim=1)
    adj_name_list = ['AP_A', 'Wee1_A', 'Cdc25']

    device = 'cpu'
    param_j = {**RD_Simulator.default_param, **simp_param}
    param_j.update(dict(name=simp_name_list))
    RDS = RD_Simulator(param_j)
    RDS.set_V_Reaction(V_Reaction_simp)
    RDS.set_Reaction_adjacent(Reaction_simp_adjacent)
    labeledcell = RDS.gen_cell(length=1., device=device)

    # plot_simp_nullcline(labeledcell,param_j,[1e-2,100.])

    # initialization of cells:

    labeledcell.set_data(init_Var)
    T = 3000
    result, result_adj = RDS.run(labeledcell, T, device=device)

    for i, name in enumerate(simp_name_list):
        plt.plot(result[:, i, 0], label=name, linewidth=2)

    for i, name in enumerate(adj_name_list):
        plt.plot(result_adj[:, i, 0], label=name, linewidth=2)
    plt.legend()
    plt.show()

    plot_simp_nullcline(labeledcell, param_j, [.1, 140.], result[:, :2, 0].T)
