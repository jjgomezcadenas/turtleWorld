import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from . utils import get_files

def peak_position(dft, ticks_per_day=1):
    return dft.NumberOfInfected.idxmax()/ticks_per_day, dft.NumberOfInfected.max()


def r0(dft, tmax=25, tr=5, ticks_per_day=5):
    T = dft.index.values
    E = dft.NumberOfExposed.values
    I = dft.NumberOfInfected.values
    S = dft.NumberOfSusceptible.values
    R = dft.NumberOfRecovered.values
    N = S[0] + I[0]
    r0 = np.array([(E[t] - E[t-1]) / (S[t-1]/N) / I[t-1] for t in T[1:tmax]])
    return T[1:tmax], r0 * tr * ticks_per_day


def r0_series(DFD, tmax=25, tr=5, ticks_per_day=5):
    R0D = {}
    AR0D = {}
    TD  = {}

    for key, value in DFD.items():
        if key == 'STA':
            continue
        TD[key], R0D[key] = r0(value, tmax, tr, ticks_per_day)
        AR0D[key] = R0D[key].mean()
    return TD, R0D, AR0D


def plot_average_I(DFD, F=True, S=True, P=True,
                   T=' Infected: R0 = 3.5, ti = 5.5, tr = 5', figsize=(8,8)):
    fig = plt.figure(figsize=figsize)
    ax=plt.subplot(111)

    ftot = DFD['F:average'].NumberOfInfected[0] + DFD['F:average'].NumberOfSusceptible[0]
    stot = DFD['S:average'].NumberOfInfected[0] + DFD['S:average'].NumberOfSusceptible[0]
    if F:
        plt.plot(DFD['F:average'].index, DFD['F:average'].NumberOfInfected/ftot, 'r',
        lw=2, label='Fixed' )
    if S:
        plt.plot(DFD['S:average'].index, DFD['S:average'].NumberOfInfected/stot, 'b',
        lw=2, label='NB'  )
    if P:
        plt.plot(DFD['P:average'].index, DFD['P:average'].NumberOfInfected/stot, 'g',
        lw=2, label='PO'  )

    plt.xlabel('time (days)')
    plt.ylabel('Fraction of Infected')
    plt.legend()
    plt.title(T)
    plt.show()

def plot_runs_I(DFD, T='Infected: R0 = 3.5, ti = 5.5, tr = 5',
                maxlbls = 10, figsize=(12,12)):
    fig = plt.figure(figsize=figsize)

    dft = DFD['DFT_run_average']
    ftot = dft.NumberOfInfected[0] + dft.NumberOfSusceptible[0]

    ax=plt.subplot(2,1,1)
    plt.plot(dft.index, dft.NumberOfInfected/ftot, 'r', lw=2, label='average' )
    plt.xlabel('time (days)')
    plt.ylabel('Fraction of Infected')
    plt.title(T)

    ax=plt.subplot(2,1,2)

    i = 0
    for key, value in DFD.items():
        if key == 'STA':
            continue
        name = key.split("_")
        if name[2] != 'average':
            dft = DFD[key]
            i+=1
            if i < maxlbls:
                plt.plot(dft.index, dft.NumberOfInfected/ftot, lw=2, label=key )
            else:
                plt.plot(dft.index, dft.NumberOfInfected/ftot, lw=2 )

    plt.xlabel('time (days)')
    plt.ylabel('Fraction of Infected')
    plt.legend()

    plt.tight_layout()
    plt.show()


def plot_compar_I(path, DIRS, LBLS, T='Comparison', figsize=(12,12)):
    fig = plt.figure(figsize=figsize)
    ax=plt.subplot(1,1,1)
    for i, d in enumerate(DIRS):
        #print(i)
        DFD, _ = get_files(path=path, mdir = d)
        #print(DFD.keys())
        dft = DFD['DFT_run_average']
        ftot = dft.NumberOfInfected[0] + dft.NumberOfSusceptible[0]

        plt.plot(dft.index, dft.NumberOfInfected/ftot, lw=2, label=LBLS[i] )
    plt.xlabel('time (days)')
    plt.ylabel('Fraction of Infected')
    plt.title(T)
    plt.legend()

# def plot_runs_I(DFD, F=True, S=True, P=True,
#                 T='Infected: R0 = 3.5, ti = 5.5, tr = 5', figsize=(8,8)):
#     fig = plt.figure(figsize=figsize)
#     ax=plt.subplot(111)
#     ftot =1
#     stot =1
#     ptot =1
#     if F:
#         ftot = DFD['F:average'].NumberOfInfected[0] + DFD['F:average'].NumberOfSusceptible[0]
#     if S:
#         stot = DFD['S:average'].NumberOfInfected[0] + DFD['S:average'].NumberOfSusceptible[0]
#     if P:
#         ptot = DFD['P:average'].NumberOfInfected[0] + DFD['P:average'].NumberOfSusceptible[0]
#
#     for key, value in DFD.items():
#         if key == 'PZ':
#             ds = value
#
#     i = 0
#     for key, value in DFD.items():
#         if key == 'PZ':
#             continue
#         name = key.split(":")
#         if name[1] != 'average':
#
#             lbl = f'{key}-PZ:{ds[i]*10000:.1f}'
#             if name[0] == 'F' and F:
#                 plt.plot(value.index, value.NumberOfInfected/ftot, lw=2, label=lbl  )
#             if name[0] == 'S' and S:
#                 plt.plot(value.index, value.NumberOfInfected/stot, lw=2, label=lbl  )
#             if name[0] == 'P' and P:
#                 plt.plot(value.index, value.NumberOfInfected/ptot, lw=2, label=lbl  )
#             i+=1
#     plt.xlabel('time (days)')
#     plt.ylabel('Fraction of Infected')
#     plt.legend()
#     plt.title(T)
#     plt.show()


def plot_I_E(DFD, F=True, S=True, T=' Infected/Exposed: R0 = 3.5, ti = 5.5, tr = 5',
             figsize=(8,8)):
    fig = plt.figure(figsize=figsize)
    ax=plt.subplot(111)

    ftot = DFD['F:average'].NumberOfInfected[0] + DFD['F:average'].NumberOfSusceptible[0]
    stot = DFD['S:average'].NumberOfInfected[0] + DFD['S:average'].NumberOfSusceptible[0]
    if F:
        plt.plot(DFD['F:average'].index, DFD['F:average'].NumberOfInfected/ftot, 'r', lw=2,
        label='I : Fixed' )
        plt.plot(DFD['F:average'].index, DFD['F:average'].NumberOfExposed/ftot, 'b', lw=2,
        label='E : Fixed' )
    if S:
        plt.plot(DFD['S:average'].index, DFD['S:average'].NumberOfInfected/stot, 'g', lw=2,
        label='I : Stochastic'  )
        plt.plot(DFD['S:average'].index, DFD['S:average'].NumberOfExposed/stot, 'y',  lw=2,
        label='E : Stochastic'  )
    plt.xlabel('time (days)')
    plt.ylabel('Fraction of Infected')
    plt.legend()
    plt.title(T)
    plt.show()


def plot_S_R(DFD, F=True, S=True, T=' R0 = 3.5, ti = 5.5, tr = 5', figsize=(8,8)):
    fig = plt.figure(figsize=figsize)
    ax=plt.subplot(111)

    ftot = DFD['F:average'].NumberOfInfected[0] + DFD['F:average'].NumberOfSusceptible[0]
    stot = DFD['S:average'].NumberOfInfected[0] + DFD['S:average'].NumberOfSusceptible[0]
    if F:
        plt.plot(DFD['F:average'].index, DFD['F:average'].NumberOfSusceptible/ftot, 'r', lw=2, label='S : Fixed' )
        plt.plot(DFD['F:average'].index, DFD['F:average'].NumberOfRecovered/ftot, 'b', lw=2, label  ='R : Fixed' )
    if S:
        plt.plot(DFD['S:average'].index, DFD['S:average'].NumberOfSusceptible/stot, 'g', lw=2, label='S : Stochastic'  )
        plt.plot(DFD['S:average'].index, DFD['S:average'].NumberOfRecovered/stot, 'y',  lw=2, label ='R : Stochastic'  )
    plt.xlabel('time (days)')
    plt.ylabel('Fraction of Infected')
    plt.legend()
    plt.title(T)
    plt.show()
