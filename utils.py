#!/usr/bin/python
# -*- codeing = utf-8 -*-
# @time:2022/1/11 下午7:16
# Author:Xuewen Shen
# @File:utils.py
# @Software:PyCharm

import pickle
import json
import numpy as np
import os
import math

# save and load param dictionaries(if not needed in fast reading)
def save_obj(obj, path):
    with open(path + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(path):
    with open(path + '.pkl', 'rb') as f:
        return pickle.load(f)

def w_dict(filepath, dict):
    f = open(filepath, 'w')
    f.write(str(dict))
    f.close()

def create(filepath):
    if filepath is not None and not os.path.exists(filepath):
        os.makedirs(filepath)

#update vector to savepath's matrix
def matrix_update(savepath,vector,axis):
    data=np.load(savepath+'.npy',allow_pickle=True)
    data_=np.concatenate((data,vector),axis=axis)
    np.save(savepath,data_)