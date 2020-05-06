import numpy as np
import pandas as pd
from networkx import *
import glob
import os
import sys


"""
Create an G{n,m} random graph with n nodes and m edges
and report some properties.

This graph is sometimes called the Erdős-Rényi graph
but is different from G{n,p} or binomial_graph which is also
sometimes called the Erdős-Rényi graph.
"""


n=10 # 10 nodes
m=20 # 20 edges

G=gnm_random_graph(n,m)

# some properties
print("node degree clustering")
for v in nodes(G):
    print('%s %d %f' % (v,degree(G,v),clustering(G,v)))

# print the adjacency list to terminal
try:
    write_adjlist(G,sys.stdout)
except TypeError: # Python 3.x
    write_adjlist(G,sys.stdout.buffer)
