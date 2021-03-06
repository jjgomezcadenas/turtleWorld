from scipy.stats import gamma
from turtleWorld.BarrioTortugaSEIR import BarrioTortugaNX
from turtleWorld.networks import build_ed_network, build_ba_network
import pandas as pd
import networkx as nx
import os
import sys
import shutil

#turtles        = 20000
#k              = 0.002

#turtles        = 10000
#k              = 0.05

#turtles        = 20000
#k              = 20

# print(f'Defining ED network for {turtles} turtles, k = {k}')
# G, n           = build_ed_network(turtles, k)

def run_turtles(turtles        = 20000,
                k              = 0.002,
                steps          = 500,
                fprint         = 25,
                ticks_per_day  = 5,
                i0             = 20,
                r0             = 3.5,
                ti             = 5.5,
                tr             = 6.5,
                ti_dist        = 'F',    # F for fixed, E for exp G for Gamma
                tr_dist        = 'F',
                p_dist         = 'F',    # F for fixed, S for Binomial, P for Poissoin
                network        = 'ER'   # ER = random netwok BA: preferential attachment
                ):


    print(f'Defining network for {turtles} turtles, k = {k}')

    if network == 'ER':
        G, n           = build_ed_network(turtles, k)
    else:
        G, n           = build_ba_network(turtles, k)

    print(f" Running Simulation with netwok {network}   for {steps} steps.")
    bt = BarrioTortugaNX(G, n, ticks_per_day, i0, r0, ti, tr,
                         ti_dist, tr_dist, p_dist)

    for i in range(steps):
        if i%fprint == 0:
            print(f' step {i}')
        bt.step()
    print('Done!')

    STATS = {}
    STATS['Ti'] = bt.Ti
    STATS['Tr'] = bt.Tr
    STATS['P']  = bt.P
    return bt.datacollector.get_model_vars_dataframe(), pd.DataFrame.from_dict(STATS)


def run_series(ns=100,
               csv            = False,
               path           ="/Users/jjgomezcadenas/Projects/Development/turtleWorld/data",
               turtles        = 20000,
               k              = 0.002,
               steps          = 500,
               fprint         = 25,
               ticks_per_day  = 5,
               i0             = 20,
               r0             = 3.5,
               ti             = 5.5,
               tr             = 6.5,
               ti_dist        = 'F',    # F for fixed, E for exp G for Gamma
               tr_dist        = 'F',
               p_dist         = 'F',    # F for fixed, S for Binomial, P for Poissoin
               network        = 'ER'   # F for fixed, S for Binomial, P for Poissoin
               ):

    if csv:
        fn1 = f'Nx_{network}_Turtles_{turtles}_steps_{steps}_i0_{i0}_r0_{r0}_ti_{ti}_tr_{tr}'
        fn2 = f'Tid_{ti_dist}_Tir_{tr_dist}_Pdist_{p_dist}'
        dirname =f'{fn1}_{fn2}'
        mdir = os.path.join(path, dirname)

        try:
            shutil.rmtree(mdir, ignore_errors=False, onerror=None)
            print(f"Directory {mdir} has been removed" )
        except OSError as error:
            print(error)
            print("Directory {mdir} not removed")

        print(f" Creating Directory {mdir} created")
        try:
            os.mkdir(mdir)
            print(f"Directory {mdir} has been created" )
        except OSError as error:
            print(error)
            print("Directory {mdir} not created")
            sys.exit()


    STATS = []
    DFT   = []
    for i in range(ns):
        print(f' series number {i}')
        dft, stats = run_turtles(turtles, k, steps, fprint, ticks_per_day, i0, r0, ti, tr,
                                 ti_dist, tr_dist, p_dist, network)

        STATS.append(stats)
        DFT.append(dft)
        if csv:
            file =f'DFT_run_{i}.csv'
            mfile = os.path.join(mdir, file)
            dft.to_csv(mfile, sep=" ")
            if i == 0:
                file=f'STA.csv'
                mfile = os.path.join(mdir, file)
                stats.to_csv(mfile, sep=" ")

    df  = pd.concat(DFT)
    dfs = pd.concat(STATS)

    if csv:

        file=f'DFT_run_average.csv'
        mfile = os.path.join(mdir, file)
        df.groupby(df.index).mean().to_csv(mfile, sep=" ")


run_series(ns             = 1,
           csv            = True,
           path           ="/Users/jjgomezcadenas/Projects/Development/turtleWorld/data",
           turtles        = 40000,
           k              = 20,
           steps          = 500,
           fprint         = 25,
           ticks_per_day  = 5,
           i0             = 10,
           r0             = 3.5,
           ti             = 5.5,
           tr             = 6.5,
           ti_dist        = 'F',    # F for fixed, E for exp G for Gamma
           tr_dist        = 'F',
           p_dist         = 'F',    # F for fixed, S for Binomial, P for Poissoin
           network        = 'BA'   # F for fixed, S for Binomial, P for Poissoin
           )
