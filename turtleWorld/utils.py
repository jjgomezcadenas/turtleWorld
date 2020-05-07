import numpy as np
import pandas as pd
from enum import Enum
import glob
import os
import sys


class PrtLvl(Enum):
    Mute     = 1
    Concise  = 2
    Detailed = 3
    Verbose  = 4


def in_range(x, xmin, xmax):
    if x >= xmin and x < xmax:
        return True
    else:
        return False


def print_level(prtl, PRT):
    if prtl.value >= PRT.value:
        return True
    else:
        return False


def throw_dice(dice):
    atry =  np.random.random_sample()
    if atry < dice:
        return True
    else:
        return False

def get_files(path, mdir, sep=' '):
    mpath = os.path.join(path, mdir)
    FLS = glob.glob(mpath+"/*.csv", recursive=False)
    FND = {}   # file name dict
    DFD = {}
    for f in FLS:
        fl = f.split('/')
        fk = fl[-1]
        ff = fk.split('.')
        ffk = ff[0]
        FND[ffk] = f
        DFD[ffk] = pd.read_csv(f, sep=sep)
    return DFD, FND


# def get_files(path, F=True, S=False):
#     def get_file_type(f):
#         fl = f.split('_')
#         fl1 = fl[1].split('/')
#         fk = fl1[-1]           # Gives S or F
#         fll = fl[-1].split('.')
#         fkl = fll[0]           # Give "average" or number
#         return fk, fkl
#
#     FND = {}   # file name dict
#     DFD = {}   # df dict
#
#     FLS = glob.glob(path+"/*.csv", recursive=False)
#
#     for f in FLS:
#         fk, fkl = get_file_type(f)
#         if fk=='PZ':
#             FND['PZ'] = f
#         else:
#             FND[f'{fk}:{fkl}'] = f
#
#     for key, value in FND.items():
#         if key == 'PZ':
#             ds = pd.read_csv(value, header=None, squeeze=True, index_col=0)
#             DFD[key] = ds
#         else:
#             DFD[key] = pd.read_csv(value, sep=' ')
#
#     return FND, DFD
