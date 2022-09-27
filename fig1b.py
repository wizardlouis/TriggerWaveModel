# -*- codeing = utf-8 -*-
# @time:2022/3/25 下午10:49
# Author:Xuewen Shen
# @File:fig1b.py
# @Software:PyCharm

### fig1b:CdK_A--CdK_T plotting under certain hyperparameters

import numpy as np
import scipy
from sympy import *
from sympy.solvers.solveset import solveset_real
from scipy import integrate
from multiprocessing import *
from utils import *

import os
import time
import math
from multiprocessing import Pool, TimeoutError

from model import *

ATP=0.8e6
ADP=0.23e6
## saving formula:[CdK,bistable(True/False),CdK_act]
def search_bistable_thru_CdK(CdK_list,P):
    '''
    :param CdK_list: list of CdK_total concentration
    :param P: hyperparameters
    :return: a list of [CdK_total,[bistable points of CdK_act]]
    '''
    N=len(CdK_list)
    bistable=np.zeros(N)
    CdKa_list=np.zeros((N,3))
    p = P.copy()
    for i in range(N):
        p.update({'Xt':CdK_list[i]})
        model=Static_model(p)
        model.solve_fixed_point_detail()
        if model.mode=='bistable':
            bistable[i]=True
            CdKa_list[i]=np.array(model.fixed_point)
        elif model.mode=='attractor':
            bistable[i]=False
            CdKa_list[i][0]=np.array(model.fixed_point)
        else:
            print('Error occurred in bistable checking procedure!!!')
    return bistable,CdKa_list


if __name__=="__main__":
    ATP=[0.8e6,0.9e6,1.e6]
    ADP=[0.23e6,0.13e6,0.03e6]
    def main_func1(ATP,ADP):
        savepath='20220325//'
        param = default_param.copy()
        param.update(dict(ATP=ATP, ADP=ADP))
        CdK_list = np.arange(40,201,1)
        bistable, CdKa_list = search_bistable_thru_CdK(CdK_list, param)
        np.savez(savepath+'ATP={}_ADP={}'.format(str(ATP),str(ADP)),CdK=CdK_list,bistable=bistable,CdKa=CdKa_list)
    with Pool(processes=3) as pool:
        for i in range(3):
            pool.apply_async(main_func1, (ATP[i],ADP[i],))
        pool.close()
        pool.join()


