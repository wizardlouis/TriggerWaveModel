# -*- codeing = utf-8 -*-
# @time:2022/3/26 上午12:10
# Author:Xuewen Shen
# @File:fig1d.py
# @Software:PyCharm

### fig1d:Bistable region on delta G--ln CdK_T plotting under certain hyperparameters

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

from model import Static_model,default_param

ATP_interval=1e3
N_ATP_prior=100
ATP_tol=1.
#search lower and upper boundary of ATP for bistable region
def search_bound_prior(P,ATP_bound):
    '''
    :param P:
    :param ATP_bound:
    :param ADP:
    :return: [lower_bound,upper_bound], and [],[] if not bistable
    '''
    N_ATP = math.ceil((ATP_bound[1] - ATP_bound[0]) / ATP_interval)
    if N_ATP>N_ATP_prior:
        N=N_ATP+1
    else:
        N=N_ATP_prior+1
    ATP_list=np.array([ATP_bound[0]+(ATP_bound[1]-ATP_bound[0])*i/(N-1) for i in range(N)])

    type = []
    for i in range(N):
        par = P.copy()
        par.update({'ATP': ATP_list[i]})
        M = Static_model(par)
        M.solve_fixed_point()
        type.append(M.mode)
        if i > 0 and (type[-2] == 'attractor' and type[-1] == 'bistable'):
            lower_bound = [ATP_list[i - 1], ATP_list[i]]
        if i > 0 and (type[-2] == 'bistable' and type[-1] == 'attractor'):
            upper_bound = [ATP_list[i - 1], ATP_list[i]]
    if all([tp != 'bistable' for tp in type]):
        # print('bistable region not found!!!')
        return [], []
    else:
        return lower_bound, upper_bound


# get wave front direction, 0 for not bistable, 1 for positive, -1 for negative
def pos_wave(P, ATP):
    par = P.copy()
    par.update({'ATP': ATP})
    M = Static_model(par)
    M.solve_fixed_point_detail()
    if M.mode != 'bistable':
        return 0
    else:
        M.get_potential()
        if M.potential[0] > 0:
            return 1
        else:
            return -1


# searching lower and upper bound within the ATP_tol level error:
def search_lower_bound(P, lower_bound):
    '''
    :param P: model parameters
    :param lower_bound:
    :return:
    '''
    state = True
    rpfile = P['filepath'] + '_rp.txt'
    bound = lower_bound

    par_l = P.copy()
    par_l.update({'ATP': bound[0]})
    M_l = Static_model(par_l)
    M_l.solve_fixed_point()
    par_u = P.copy()
    par_u.update({'ATP': bound[1]})
    M_u = Static_model(par_u)
    M_u.solve_fixed_point()
    if M_l.mode != 'attractor' or M_u.mode != 'bistable':
        f = open(rpfile, 'a')
        f.write('Lower bound prior is not correct!!!\n')
        f.close()
        state = False
        return bound, state
    else:
        while bound[1] - bound[0] > ATP_tol:
            new_bound = np.mean(bound)
            # f = open(rpfile, 'a')
            # f.write('lower bound: Try ATP{} using {} s!:\n'.format(str(new_bound),str(time.time()-start)))
            # f.close()

            par = P.copy()
            par.update({'ATP': new_bound})
            M = Static_model(par)
            M.solve_fixed_point()
            if M.mode == 'attractor':
                bound = [new_bound, bound[1]]
            elif M.mode == 'bistable':
                bound = [bound[0], new_bound]
            else:
                f = open(rpfile, 'a')
                f.write('Error occurred in searching lower bound!!!\n')
                f.close()
                state = False

                f = open(rpfile, 'a')
                f.write('Finished lower bound: using {} s!:\n'.format(str(time.time() - start)))
                f.close()
                return bound, state
        f = open(rpfile, 'a')
        f.write('Finished lower bound: using {} s!:\n'.format(str(time.time() - start)))
        f.close()
        return bound, state


def search_upper_bound(P, upper_bound):
    '''
    :param P:
    :param upper_bound:
    :return:
    '''
    state = True
    rpfile = P['filepath'] + '_rp.txt'
    bound = upper_bound

    par_l = P.copy()
    par_l.update({'ATP': bound[0]})
    M_l = Static_model(par_l)
    M_l.solve_fixed_point()
    par_u = P.copy()
    par_u.update({'ATP': bound[1]})
    M_u = Static_model(par_u)
    M_u.solve_fixed_point()
    if M_l.mode != 'bistable' or M_u.mode != 'attractor':
        f = open(rpfile, 'a')
        f.write('Upper bound prior is not correct!!!\n')
        f.close()
        state = False
        return bound, state
    else:
        while bound[1] - bound[0] > ATP_tol:
            new_bound = np.mean(bound)
            # f = open(rpfile, 'a')
            # f.write('upper bound: Try ATP{} using {} s!:\n'.format(str(new_bound),str(time.time()-start)))
            # f.close()

            par = P.copy()
            par.update({'ATP': new_bound})
            M = Static_model(par)
            M.solve_fixed_point()
            if M.mode == 'attractor':
                bound = [bound[0], new_bound]
            elif M.mode == 'bistable':
                bound = [new_bound, bound[1]]
            else:
                f = open(rpfile, 'a')
                f.write('Error occurred in searching upper bound!!!\n')
                f.close()
                state = False

                f = open(rpfile, 'a')
                f.write('Finished upper bound: using {} s!:\n'.format(str(time.time() - start)))
                f.close()
                return bound, state
        f = open(rpfile, 'a')
        f.write('Finished upper bound: using {} s!:\n'.format(str(time.time() - start)))
        f.close()
        return bound, state


# searching phase change line within ATP_tol level error in ATP_bound region
def search_phase_change_line(P, ATP_bound):
    '''
    :param P:
    :param ATP_bound: [a,b];a:right of lower bound;b:left of upper bound;
    :param ADP:
    :return:
    '''
    state = True
    rpfile = P['filepath'] + '_rp.txt'
    bound = ATP_bound
    wl = pos_wave(P, bound[0])
    wu = pos_wave(P, bound[1])
    if wl == -1 and wu == -1:
        f = open(rpfile, 'a')
        f.write('Type 1 Finished phase change line: using {} s!:\n'.format(str(time.time() - start)))
        f.close()
        return [bound[1], bound[1]], state
    elif wl == 1 and wu == 1:
        f = open(rpfile, 'a')
        f.write('Type 1 Finished phase change line: using {} s!:\n'.format(str(time.time() - start)))
        f.close()
        return [bound[0], bound[0]], state
    elif wl == -1 and wu == 1:
        while bound[1] - bound[0] > ATP_tol:
            new_bound = np.mean(bound)
            par = P.copy()
            par.update({'ATP': new_bound})
            M = Static_model(par)
            M.solve_fixed_point_detail()
            if M.mode != 'bistable':
                f = open(rpfile, 'a')
                f.write('Type 0 Finished phase change line: using {} s!:\n'.format(str(time.time() - start)))
                f.close()
                state = False
                return bound, state
            else:
                M.get_potential()
                if M.potential[0] > 0:
                    bound = [bound[0], new_bound]
                else:
                    bound = [new_bound, bound[1]]
        f = open(rpfile, 'a')
        f.write('Type 1 Finished phase change line: using {} s!:\n'.format(str(time.time() - start)))
        f.close()
        return bound, state
    else:
        state = False
        f = open(rpfile, 'a')
        f.write('Type 2 Finished phase change line: using {} s!:\n'.format(str(time.time() - start)))
        f.close()
        return bound, state

#searching for lower_bound,upper_bound under certain CdK
def search_bistable(P, lower_bound_prior, upper_bound_prior):
    if lower_bound_prior == [] or upper_bound_prior == []:
        rpfile = P['filepath'] + '_rp.txt'
        f = open(rpfile, 'a')
        f.write('bound_prior is blank!!!')
        f.close()
        return [[], []],False
    else:
        lower_bound, l_state = search_lower_bound(P, lower_bound_prior)
        upper_bound, u_state = search_upper_bound(P, upper_bound_prior)
        return [lower_bound, upper_bound],True

#searching for lower_bound,upper_bound,phase_change_line and phase_change state under certain CdK
def search_all(P, lower_bound_prior, upper_bound_prior):
    if lower_bound_prior == [] or upper_bound_prior == []:
        rpfile = P['filepath'] + '_rp.txt'
        f = open(rpfile, 'a')
        f.write('bound_prior is blank!!!')
        f.close()
        return [[], [], []], False
    else:
        lower_bound, l_state = search_lower_bound(P, lower_bound_prior)
        upper_bound, u_state = search_upper_bound(P, upper_bound_prior)
        if not l_state or not u_state:
            return [lower_bound, upper_bound, []], False
        else:
            phase_change, p_state = search_phase_change_line(P, (lower_bound[1], upper_bound[0]))
            return [lower_bound, upper_bound, phase_change], p_state

def search_thru_CdK(P,CdK_list,ATP_bound):
    N_CdK=len(CdK_list)
    rpfile = P['filepath'] + '_rp.txt'
    f = open(rpfile, 'a')
    f.write('Report of Simulation:\n')
    f.close()

    np.save(P['filepath'],np.empty((0,5)))
    for i in range(N_CdK):
        f = open(rpfile, 'a')
        f.write('Try CdK_T {}: using {} s!\n'.format(str(CdK_list[i]), str(time.time() - start)))
        f.close()
        par=P.copy()
        par.update(dict(Xt=CdK_list[i]))
        lower_bound,upper_bound=search_bound_prior(par,ATP_bound)
        lbp=lower_bound
        ubp=upper_bound
        res,state=search_bistable(par,lbp,ubp)
        if state:
            matrix_update(P['filepath'],np.array([[CdK_list[i],res[0][0],res[0][1],res[1][0],res[1][1]]]),0)
    f = open(rpfile, 'a')
    f.write('End of Simulation:\n')
    f.close()



if __name__=="__main__":
    start = time.time()
    # filepath = 'CdK_20220113//'
    filepath = '20220325//fig1d//'
    if not os.path.exists(filepath):
        os.makedirs(filepath)
    CdK_list = np.arange(40., 641., 5.)
    np.save(filepath+'params.npy',CdK_list)
    ATP_bound=[1.,2e6]
    ADP_list=np.arange(0.1,1.31,0.1)*1e6
    # CdK_list = np.arange(50., 501., 5.)
    param = default_param.copy()
    w_dict(filepath + '//df_param.txt', param)

    def main_func(i):
        par = param.copy()
        par.update({'ADP': ADP_list[i], 'filepath': filepath + str(i)})
        search_thru_CdK(par,CdK_list,ATP_bound)


    # main_func(52)
    with Pool(processes=13) as pool:
        for i in range(13):
            pool.apply_async(main_func, (i,))
        pool.close()
        pool.join()
