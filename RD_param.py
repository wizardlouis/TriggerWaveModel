# -*- codeing = utf-8 -*-
# @time:2022/5/23 下午11:39
# Author:Xuewen Shen
# @File:RD_param.py
# @Software:PyCharm

import torch

# Full model paramters
default_namelist = [
    'Cb', 'M', 'Mp', 'Mb', 'Mbp', 'AP0', 'AP1', 'I0', 'I1', 'E0', 'E1'
]

param = {'n_components': 11,
         'ATP': 4e5, 'ADP': 1e4, 'Pi': 1e6, 'v_B1': 3., 'k_degB': 0.002, 'k_B+': 1e-4, 'k_B-': 1e-2,
         'k_+1': 1e-6, 'k_-1': 1e-9, 'k_+2': 1, 'k_-2': 1e-11,
         'k_AP+1': 1e-9, 'k_AP-1': 1e-12, 'k_AP+2': 1e-3, 'k_AP-2': 1e-11,
         'k_I+1': 1e-6, 'k_I-1': 1e-9, 'k_I+2': 1., 'k_I-2': 1e-11,
         'k_E+1': 1e-6, 'k_E-1': 1e-9, 'k_E+2': 1., 'k_E-2': 1e-11,
         'P_AP': 30., 'P_I': 30, 'P_E': 30
         }


# definition of reaction terms
def V_Reaction0(LC, P):
    V_Cb = P['v_B1'] - P['k_degB'] * LC.AP1 * LC.Cb - P['k_B+'] * (LC.M + LC.Mp) * LC.Cb + P['k_B-'] * (LC.Mb + LC.Mbp)
    V_M = P['k_B-'] * LC.Mb - P['k_B+'] * LC.M * LC.Cb
    V_Mp = P['k_B-'] * LC.Mbp - P['k_B+'] * LC.Mp * LC.Cb
    V_Mb = P['k_B+'] * LC.M * LC.Cb - P['k_B-'] * LC.Mb - (
            P['k_+1'] * LC.Mb * P['ATP'] - P['k_-1'] * LC.Mbp * P['ADP']) * LC.I0 + (
                   P['k_+2'] * LC.Mbp - P['k_-2'] * LC.Mb * P['Pi']) * LC.E1
    V_Mbp = P['k_B+'] * LC.Mp * LC.Cb - P['k_B-'] * LC.Mbp + (
            P['k_+1'] * LC.Mb * P['ATP'] - P['k_-1'] * LC.Mbp * P['ADP']) * LC.I0 - (
                    P['k_+2'] * LC.Mbp - P['k_-2'] * LC.Mb * P['Pi']) * LC.E1
    V_AP0 = -(P['k_AP+1'] * P['ATP'] * LC.AP0 - P['k_AP-1'] * P['ADP'] * LC.AP1) * LC.Mb + (
            P['k_AP+2'] * LC.AP1 - P['k_AP-2'] * P['Pi'] * LC.AP0) * P['P_AP']
    V_AP1 = (P['k_AP+1'] * P['ATP'] * LC.AP0 - P['k_AP-1'] * P['ADP'] * LC.AP1) * LC.Mb - (
            P['k_AP+2'] * LC.AP1 - P['k_AP-2'] * P['Pi'] * LC.AP0) * P['P_AP']
    V_I0 = -(P['k_I+1'] * P['ATP'] * LC.I0 - P['k_I-1'] * P['ADP'] * LC.I1) * LC.Mb + (
            P['k_I+2'] * LC.I1 - P['k_I-2'] * P['Pi'] * LC.I0) * P['P_I']
    V_I1 = (P['k_I+1'] * P['ATP'] * LC.I0 - P['k_I-1'] * P['ADP'] * LC.I1) * LC.Mb - (
            P['k_I+2'] * LC.I1 - P['k_I-2'] * P['Pi'] * LC.I0) * P['P_I']
    V_E0 = -(P['k_E+1'] * P['ATP'] * LC.E0 - P['k_E-1'] * P['ADP'] * LC.E1) * LC.Mb + (
            P['k_E+2'] * LC.E1 - P['k_E-2'] * P['Pi'] * LC.E0) * P['P_E']
    V_E1 = (P['k_E+1'] * P['ATP'] * LC.E0 - P['k_E-1'] * P['ADP'] * LC.E1) * LC.Mb - (
            P['k_E+2'] * LC.E1 - P['k_E-2'] * P['Pi'] * LC.E0) * P['P_E']
    return torch.cat(
        [V.reshape((1,) + V.shape) for V in [V_Cb, V_M, V_Mp, V_Mb, V_Mbp, V_AP0, V_AP1, V_I0, V_I1, V_E0, V_E1]],
        dim=0)


# simplified model of 3 components

timescale = 2.
timescale2=5.
simp_param = {
    'n_components': 3,
    'ATP':1e6, 'ADP': 1e4, 'Pi': 1e6,
    'v_B1': 2e-1 / timescale, 'k_degB': 2e-3 / timescale, 'k_B+': 3e-4 / timescale, 'k_B-': 8e-2 / timescale,
    'k_+1': 1e-7 / timescale2, 'k_+2': 1.05 / timescale2,
    'Ek_AP': 9.e-8, 'Ek_Wee1': 1e-8, 'Ek_Cdc25': 1.67e-8, 'L': 17, 'M': 3, 'N': 11,
    'M_T': 500, 'AP_T': 100, 'Wee1_T': 60, 'Cdc25_T': 150, 'kinase': 30
}

def V_Reaction_simp(LC, P, adj):
    # Here is differentials
    V_Cb = P['v_B1'] - P['k_degB'] * adj.AP_A * LC.Cb - P['k_B+'] * (P['M_T'] - LC.Mb - LC.Mbp) * LC.Cb+P['k_B-']*(LC.Mb+LC.Mbp)
    V_Mb = -P['k_B-'] * LC.Mb - P['k_+1'] * LC.Mb * P['ATP'] * adj.Wee1_A + P['k_+2'] * LC.Mbp * adj.Cdc25_A
    V_Mbp = P['k_B+'] * (P['M_T'] - LC.Mb - LC.Mbp) * LC.Cb - P['k_B-'] * LC.Mbp + P['k_+1'] * LC.Mb * P[
        'ATP'] * adj.Wee1_A - P['k_+2'] * LC.Mbp * adj.Cdc25_A
    return torch.cat([V.reshape((1,) + V.shape) for V in [V_Cb, V_Mb, V_Mbp]], dim=0)

g=1e-1
def V_Reaction_simp_low_pass(LC,P,adj):
    V_Cb = P['v_B1'] - P['k_degB'] * LC.AP_A * LC.Cb - P['k_B+'] * (P['M_T'] - LC.Mb - LC.Mbp) * LC.Cb+P['k_B-']*(LC.Mb+LC.Mbp)
    V_Mb = -P['k_B-'] * LC.Mb - P['k_+1'] * LC.Mb * P['ATP'] * LC.Wee1_A + P['k_+2'] * LC.Mbp * LC.Cdc25_A
    V_Mbp = P['k_B+'] * (P['M_T'] - LC.Mb - LC.Mbp) * LC.Cb - P['k_B-'] * LC.Mbp + P['k_+1'] * LC.Mb * P[
        'ATP'] * LC.Wee1_A - P['k_+2'] * LC.Mbp * LC.Cdc25_A
    V_AP=-g*(LC.AP_A-adj.AP_A)
    V_Wee1=-g*(LC.Wee1_A-adj.Wee1_A)
    V_Cdc25=-g*(LC.Cdc25_A-adj.Cdc25_A)
    return torch.cat([V.reshape((1,) + V.shape) for V in [V_Cb, V_Mb, V_Mbp,V_AP,V_Wee1,V_Cdc25]], dim=0)


def V_Reaction_simp_A(LC, P, adj):
    # Here is differentials
    V_Cb = P['v_B1'] - P['k_degB'] * adj.AP_A * LC.Cb
    V_Mb = P['k_B+'] * (P['M_T'] - LC.Mb - LC.Mbp) * LC.Cb - P['k_B-'] * LC.Mb - P['k_+1'] * LC.Mb * P[
        'ATP'] * adj.Wee1_A + P['k_+2'] * LC.Mbp * adj.Cdc25_A
    V_Mbp = - P['k_B-'] * LC.Mbp + P['k_+1'] * LC.Mb * P[
        'ATP'] * adj.Wee1_A - P['k_+2'] * LC.Mbp * adj.Cdc25_A
    return torch.cat([V.reshape((1,) + V.shape) for V in [V_Cb, V_Mb, V_Mbp]], dim=0)
# Non-degradable cyclinB:
