
import numpy as np
import networkx as nx
from networkx import *

def build_ed_network(turtles=20000, k=0.002):
    G = nx.erdos_renyi_graph(turtles, k)
    D = np.array([degree(G,n) for n in nodes(G)])
    n = np.mean(D)
    print(f' mean number of neighbors ={n}')
    return G, n
