#!/usr/bin/python
# -*- codeing = utf-8 -*-
# @time:2021/5/29 afternoon3:17
# Author:Xuewen Shen
# @File:model.py
# @Software:PyCharm

import numpy as np
import scipy
from sympy import *
from sympy.solvers.solveset import solveset_real
from scipy import integrate
from multiprocessing import *
from utils import *

# from sympy.abc import x, y, z
import os
import time
import math
from multiprocessing import Pool, TimeoutError

# Set default param before simulation
# other variable reference values are:
# ATP=5e5,ADP=1e4,Xt=120
default_param = dict(
    k_p1=1e-6, k_p2=1, k_m1=1e-9, k_m2=1e-11,
    P1=30., P2=30., kin=12., Xt=70., It=60., Et=150.,
    L=300., Q=300., Pi=1e6, N=5,
)

# default hyperparameters
X_acc = 0.1
X_tol = 1e-5

N_ATP_prior = 100
ATP_interval = 1e4
ATP_min_bound = 1.
ATP_max_bound = 2e6

ATP_tol = 1.

ADP_initial=1e-4
ADP_interval = 1e6
ADP_min_bound = 1.
ADP_max_bound = 1e8

ADP_prior = [ADP_initial*(2. ** i) for i in range(int(np.log2(ADP_interval/ADP_initial)) + 1)]

Slowing_bound = [3., 9.]


# get phase change line from specific parameters


class Static_model(object):
    def __init__(self, param):
        self.P = param.copy()
        self.P['K_p1'] = self.P['k_p1'] * self.P['ATP']
        self.P['K_m1'] = self.P['k_m1'] * self.P['ADP']
        self.P['K_p2'] = self.P['k_p2']
        self.P['K_m2'] = self.P['k_m2'] * self.P['Pi']
        self.mode = 'underdetermined'

    def sigmaI(self, X):
        return (self.P['K_p1'] * X + self.P['K_m2'] * self.P['P1']) / (
                self.P['K_m1'] * X + self.P['K_p2'] * self.P['P1'])

    def sigmaE(self, X):
        return (self.P['K_p1'] * (X + self.P['kin']) + self.P['K_m2'] * self.P['P2']) / (
                self.P['K_m1'] * (X + self.P['kin']) + self.P['K_p2'] * self.P['P2'])

    # get deviation of CdK
    def dX(self, X, I, E):
        return self.P['Xt'] * (self.P['K_m1'] * I + self.P['K_p2'] * E) - X * (
                (self.P['K_p1'] + self.P['K_m1']) * I + (self.P['K_p2'] + self.P['K_m2']) * E)

    # get deviation of Wee1
    def dI(self, X, I):
        return I - self.P['It'] * sum(self.sigmaI(X) ** i for i in range(0, self.P['N'])) / sum(
            self.sigmaI(X) ** i for i in range(0, self.P['N'] + 1))

    # get deviation of Cdc25
    def dE(self, X, E):
        return E - self.P['Et'] * (self.sigmaE(X) ** self.P['N']) / sum(
            self.sigmaE(X) ** i for i in range(0, self.P['N'] + 1))

    def I(self, X):
        return self.P['It'] * sum(self.sigmaI(X) ** i for i in range(0, self.P['N'])) / sum(
            self.sigmaI(X) ** i for i in range(0, self.P['N'] + 1))

    def E(self, X):
        return self.P['Et'] * (self.sigmaE(X) ** self.P['N']) / sum(
            self.sigmaE(X) ** i for i in range(0, self.P['N'] + 1))

    # manually searching for solutions
    def solve_fixed_point(self):
        N_iter = int(self.P['Xt'] / X_acc) + 1
        X_list_prior = np.array([X_acc * i for i in range(N_iter)])
        dX_list_prior = np.array([self.dX(X, self.I(X), self.E(X)) for X in X_list_prior])
        X_label = dX_list_prior > 0
        X_label_multiply = np.array([X_label[i] ^ X_label[i + 1] for i in range(len(X_label) - 1)])
        X_label_idx = np.argwhere(X_label_multiply == True)
        N_fixed = X_label_idx.shape[0]
        self.N_fixed = N_fixed
        if N_fixed == 3:
            self.mode = 'bistable'
        elif N_fixed == 1:
            self.mode = 'attractor'
        elif N_fixed == 0:
            self.mode = 'unstable'
        else:
            self.mode = 'undefined'
        # self.fixed_point_tol = []
        # self.fixed_point = []
        # for i in range(N_fixed):
        #     Xrange = X_list_prior[X_label_idx[i, 0]:X_label_idx[i, 0] + 2]
        #     while Xrange[1] - Xrange[0] > X_tol:
        #         new_X = Xrange.mean()
        #         X_list = np.array([Xrange[0], new_X, Xrange[1]])
        #         dX_list = [self.dX(X, self.I(X), self.E(X)) for X in X_list]
        #         label2d = dX_list > 0
        #         devide = [label2d[0] ^ label2d[1], label2d[1] ^ label2d[2]]
        #         if devide[0]:
        #             Xrange = np.array([Xrange[0], new_X])
        #         else:
        #             Xrange = np.array([new_X, Xrange[1]])
        #     self.fixed_point_tol.append(list(Xrange))
        #     self.fixed_point.append(Xrange.mean())

    def solve_fixed_point_detail(self):
        N_iter = int(self.P['Xt'] / X_acc) + 1
        X_list_prior = np.array([X_acc * i for i in range(N_iter)])
        dX_list_prior = np.array([self.dX(X, self.I(X), self.E(X)) for X in X_list_prior])
        X_label = dX_list_prior > 0
        X_label_multiply = np.array([X_label[i] ^ X_label[i + 1] for i in range(len(X_label) - 1)])
        X_label_idx = np.argwhere(X_label_multiply == True)
        N_fixed = X_label_idx.shape[0]
        self.N_fixed = N_fixed
        if N_fixed == 3:
            self.mode = 'bistable'
        elif N_fixed == 1:
            self.mode = 'attractor'
        elif N_fixed == 0:
            self.mode = 'unstable'
        else:
            self.mode = 'undefined'
        self.fixed_point_tol = []
        self.fixed_point = []
        for i in range(N_fixed):
            Xrange = [X_list_prior[X_label_idx[i, 0]], X_list_prior[X_label_idx[i, 0] + 1]]
            while Xrange[1] - Xrange[0] > X_tol:
                new_X = np.mean(Xrange)
                X_list = np.array([Xrange[0], new_X, Xrange[1]])
                dX_list = np.array([self.dX(X, self.I(X), self.E(X)) for X in X_list])
                label2d = dX_list > 0
                devide = [label2d[0] ^ label2d[1], label2d[1] ^ label2d[2]]
                if devide[0]:
                    Xrange = [Xrange[0], new_X]
                else:
                    Xrange = [new_X, Xrange[1]]
            self.fixed_point_tol.append(Xrange)
            self.fixed_point.append(np.mean(Xrange))

    # sovle fixed point equations which depend on parameters given
    # def solve_fixed_point(self):
    #     x = symbols('x', real=True)
    #     result = list(solveset(self.dX(x, self.I(x), self.E(x)), x, domain=Interval(0, self.P['Xt'])))
    #     self.fixed_point = result
    #     if len(result) == 3:
    #         self.mode = 'bistable'
    #     elif len(result) == 1:
    #         self.mode = 'attractor'
    #     elif len(result) == 0:
    #         self.mode = 'unstable'
    #     else:
    #         self.mode = 'undefined'

    # solve fake potential dependent on
    def get_potential(self):
        if self.mode == 'bistable':
            xlist = [self.fixed_point[i] for i in range(3)]
            xmin = min(xlist)
            xmax = max(xlist)
            self.potential = integrate.quad(lambda x: self.dX(x, self.I(x), self.E(x)), xmin, xmax)
        else:
            self.potential = None

    # output: model result
    def _out(self):
        return [[self.P['Xt'], self.P['ATP'], self.P['ADP']], self.fixed_point, self.potential]


# to search the lower and upper boundary of ATP for bistable region
def search_bound_prior(P, ATP_bound, ADP):
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
        par.update({'ATP': ATP_list[i], 'ADP': ADP})
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
def pos_wave(P, ATP, ADP):
    par = P.copy()
    par.update({'ATP': ATP, 'ADP': ADP})
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
def search_lower_bound(P, lower_bound, ADP):
    '''
    :param P: model parameters
    :param lower_bound:
    :return:
    '''
    state = True
    rpfile = P['filepath'] + '_rp.txt'
    bound = lower_bound

    par_l = P.copy()
    par_l.update({'ATP': bound[0], 'ADP': ADP})
    M_l = Static_model(par_l)
    M_l.solve_fixed_point()
    par_u = P.copy()
    par_u.update({'ATP': bound[1], 'ADP': ADP})
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
            par.update({'ATP': new_bound, 'ADP': ADP})
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


def search_upper_bound(P, upper_bound, ADP):
    '''
    :param P:
    :param upper_bound:
    :return:
    '''
    state = True
    rpfile = P['filepath'] + '_rp.txt'
    bound = upper_bound

    par_l = P.copy()
    par_l.update({'ATP': bound[0], 'ADP': ADP})
    M_l = Static_model(par_l)
    M_l.solve_fixed_point()
    par_u = P.copy()
    par_u.update({'ATP': bound[1], 'ADP': ADP})
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
            par.update({'ATP': new_bound, 'ADP': ADP})
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
def search_phase_change_line(P, ATP_bound, ADP):
    '''
    :param P:
    :param ATP_bound: [a,b];a:right of lower bound;b:left of upper bound;
    :param ADP:
    :return:
    '''
    state = True
    rpfile = P['filepath'] + '_rp.txt'
    bound = ATP_bound
    wl = pos_wave(P, bound[0], ADP)
    wu = pos_wave(P, bound[1], ADP)
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
            par.update({'ATP': new_bound, 'ADP': ADP})
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


#searching for lower_bound,upper_bound,phase_change_line and phase_shange state under certain ADP
def search_all(P, ADP, lower_bound_prior, upper_bound_prior):
    if lower_bound_prior == [] or upper_bound_prior == []:
        rpfile = P['filepath'] + '_rp.txt'
        f = open(rpfile, 'a')
        f.write('bound_prior is blank!!!')
        f.close()
        return [ADP, [], [], []], False
    else:
        lower_bound, l_state = search_lower_bound(P, lower_bound_prior, ADP)
        upper_bound, u_state = search_upper_bound(P, upper_bound_prior, ADP)
        if not l_state or not u_state:
            return [ADP, lower_bound, upper_bound, []], False
        else:
            phase_change, p_state = search_phase_change_line(P, (lower_bound[1], upper_bound[0]), ADP)
            return [ADP, lower_bound, upper_bound, phase_change], p_state

#Searching for phase_change graph uner ATP/ADP range
def search_through_ADP(P, slow=tuple(Slowing_bound), ADP_prior=tuple(ADP_prior), ADP_max_bound=ADP_max_bound,
                       ADP_interval=ADP_interval, ATP_min_bound=ATP_min_bound, ATP_max_bound=ATP_max_bound):
    rpfile = P['filepath'] + '_rp.txt'
    f = open(rpfile, 'a')
    f.write('Report of Simulation:\n')
    f.close()

    # initialization of bounds
    res = np.empty((0,7))
    np.save(P['filepath'],res)

    # ger ADP prior list answer
    n_prior=len(ADP_prior)
    prior_count=0
    while prior_count<n_prior:
        ADP=ADP_prior[prior_count]
        f = open(rpfile, 'a')
        f.write('Try ADP {}: using {} s!\n'.format(str(ADP), str(time.time() - start)))
        f.close()
        if prior_count==0:
            lower_bound, upper_bound = search_bound_prior(P, (ATP_min_bound, ATP_max_bound), ADP)
            lower_bound_prior=lower_bound
            upper_bound_prior=upper_bound
        else:
            lower_bound, upper_bound = search_bound_prior(P, (res[1][0], res[2][1]), ADP)
        res, state = search_all(P, ADP, lower_bound, upper_bound)

        prior_count+=1
        if state:
            matrix_update(P['filepath'],np.array([[res[0], res[1][0], res[1][1], res[2][0], res[2][1], res[3][0], res[3][1]]]),0)
        else:
            f = open(rpfile, 'a')
            f.write('End of Simulation:\n')
            f.close()
            return 0

    # After ADP_prior, automatically detect ADP spaces!!!
    bound_length = upper_bound_prior[1] - lower_bound_prior[0]
    temporal_ADP_interval = ADP_interval
    count = 0
    max_count = len(slow)
    ADP = ADP_prior[-1]
    while ADP < ADP_max_bound and (res[2][1] - res[1][0] > 3 * ATP_tol):
        if count < max_count:
            if res[2][1] - res[1][0] < bound_length / slow[count]:
                f = open(rpfile, 'a')
                f.write('Slowing ADP rate for times={}!!!\n'.format(str(count)))
                f.close()
                temporal_ADP_interval /= 10
                count += 1
        ADP += temporal_ADP_interval
        f = open(rpfile, 'a')
        f.write('Try ADP {}:\n'.format(str(ADP)))
        f.close()
        lower_bound, upper_bound = search_bound_prior(P, (res[1][0], res[2][1]), ADP)
        res, state = search_all(P, ADP, lower_bound, upper_bound)
        if state:
            matrix_update(P['filepath'],
                          np.array([[res[0], res[1][0], res[1][1], res[2][0], res[2][1], res[3][0], res[3][1]]]), 0)
        else:
            f = open(rpfile, 'a')
            f.write('End of Simulation:\n')
            f.close()
            return 0

    f = open(rpfile, 'a')
    f.write('End of Simulation:\n')
    f.close()


if __name__ == '__main__':
    # Xtlist = np.arange(50, 250, 10)
    # ATPlim = [1, 1e7]
    # ADPlim = [1, 1e8]
    # start = time.time()
    # result = []
    # for i in range(len(Xtlist)):
    #     p = default_param.copy()
    #     p.update({'Xt': Xtlist[i], 'ATP': 5e5, 'ADP': 0})
    #     m = Static_model(p)
    #     m.solve_fixed_point()
    #     m.get_potential()
    #     result.append(m._out())
    #     print(str(i) + 'th result complete!!!')
    # end = time.time()
    # print(end - start)
    # save_obj(result, 'result')
    start = time.time()
    # filepath = 'CdK_20220113//'
    filepath='Wee1_20220228_Xt=40//'
    if not os.path.exists(filepath):
        os.makedirs(filepath)
    Wee1_list = np.arange(0., 121., 5.)

    # CdK_list = np.arange(50., 501., 5.)
    param=default_param.copy()
    param['Xt']=40.
    w_dict(filepath + '//df_param.txt', param)


    def main_func(i):
        par = param.copy()
        par.update({'It': Wee1_list[i], 'filepath': filepath + str(i)})
        search_through_ADP(par)


    # main_func(52)
    with Pool(processes=25) as pool:
        for i in range(len(Wee1_list)):
            pool.apply_async(main_func, (i,))
        pool.close()
        pool.join()

    # # Param = []
    # for i in range(Wee1_list.shape[0]):
    #     par = default_param.copy()
    #     par.update({'It': Wee1_list[i], 'filepath': filepath + str(i)})
    #     # Param.append(par)
    #     search_through_ADP(par)

    # for i in range(Wee1_list.shape[0]):
    #     search_through_ADP(Param[i])

    # with Pool(processes=60) as pool:
    #     # pool.map(search_through_ADP, tuple(Param))
    #     # for i in range(Wee1_list.shape[0]):
    #     res = pool.apply_async(search_through_ADP, tuple(Param))

    # filepath = 'Wee1_20220112//'
    # par = default_param.copy()
    # par.update({'It': 80., 'filepath': filepath + '0'})
    # search_through_ADP(par)
