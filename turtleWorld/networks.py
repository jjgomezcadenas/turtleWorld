
import numpy as np
import networkx as nx
from networkx import *
from . utils import PrtLvl, print_level, throw_dice

prtl=PrtLvl.Concise

def build_ed_network(turtles=20000, k=0.002):
    G = nx.erdos_renyi_graph(turtles, k)
    D = np.array([degree(G,n) for n in nodes(G)])
    n = np.mean(D)
    print(f' mean number of neighbors ={n}')
    return G, n


def build_ba_network(turtles=20000, k=20):
    G = nx.barabasi_albert_graph(turtles, k)
    D = np.array([degree(G,n) for n in nodes(G)])
    n = np.mean(D)
    print(f' mean number of neighbors ={n}')
    return G, n


def degree_list(g):
    D = np.array([degree(g,n) for n in nodes(g)])
    return D


def max_k(g):
    return np.max(degree_list(g))


def mean_k(g):
    return np.mean(degree_list(g))


def sum_k(g):
    return np.sum(degree_list(g))


def node_is_active(G, n):
    if G.nodes[n]['state'] == 1:
        return True
    else:
        return False

def node_list(G):
    return np.array([n for n in nodes(G)])


def node_list_ki_kj(G, ki, kj):
    nl = node_list(G)

    return nl[ki:kj]


def node_list_rnd(G):
    return np.random.shuffle(node_list(G))


def node_list_rnd_ki_kj(G, ki, kj):
    nl = node_list_ki_kj(G, ki, kj)
    np.random.shuffle(nl)
    return nl


def active_nodes(G):
    return [n for n in nodes(G) if node_is_active(G, n) ]


def node_color(G):
    C =[]
    for node in G:
        if node_is_active(G, node):
            C.append('r')
        else:
            C.append('b')
    return C


def prob_k(G, n):
    """
    The deactivation probability of a node n in the
    ER algorithm is 1 /k where k is the degree (number of links)
    of the node

    """
    return  1. / degree(G,n)


def ke_deactivation_prob_norm(G):
    """
    The normalisation of the deactivation probability
    is sum_j (1/k_j)

    """
    return np.sum([prob_k(G, n) for n in active_nodes(G)])


def ke_deactivation_prob(G, a):
    """
    Deactivation probability for each node.
    This is a dictionary ordered by node with normalised
    deactivation probabilities as value

    """
    return {n:a * prob_k(G, n) for n in active_nodes(G)}


def ke_pa_prob(G, n):
    """
    The probability of linking to a node in the KE algorithm
    uses preferential attachment, e.g, is proportional to the degree of the node.
    """
    #print(f'node = {n}, degree ={degree(G, n)}')
    return degree(G, n) / sum_k(G)


def ke_pa_activation_prob(G, WG):
    """
    The probability of linking to a node n is an ordered dictionary per node.
    """
    return {n:ke_pa_prob(G, n) for n in nodes(G) if n not in WG}


def KE_network_init(m=10):
    """
    Inits the network with m fully connected nodes
    The nodes are initially in the active state

    """
    G = nx.complete_graph(m)

    for n in range(m):
        G.nodes[n]['state'] = 1
    return G


def select_random_node(G, WG, norm0):

    def pa_norm(g, WG):
        norm = 0
        for n in WG:
            norm += degree(G, n)
        return norm

    rnodes = node_list_rnd_ki_kj(G, 0, -1)
    #print(rnodes)
    for n in rnodes:
        if n in WG:
            if print_level(prtl, PrtLvl.Verbose):
                print(f' node  = {n} already selected')
            continue
        else:

            pa = degree(G, n) / norm0
            if print_level(prtl, PrtLvl.Verbose):
                print(f' node  = {n} pa = {pa}, degree = {degree(G, n)}, norm = {norm0}')

            if throw_dice(pa):
                if print_level(prtl, PrtLvl.Verbose):
                    print(f' selecting random node = {n} with prob = {pa}')
                break
    return n

def KE_newtwork_step(G, mu):
    """
    Steps the KE network

    - 1. A new node joins the network in the following way:
        - i.   For each of the m links of the new node it is decided randomly if the link
               connects to the active node or to a random node
        - ii.  The probability to attach to a random node is specified by the parameter $\mu$.
        - iii. If a random node is chosen, the criterium is linear preference attachment,
               that is the probability that node $j$ grabs a link is proportional to
               the node's degree $k_j$.
    - 2. The new node becomes active.
    - 3. One of the actives node is deactivated according to the following criterium:
    - i. The probability that node $i$ is chosen for deactivation
         is $p_i = \frac{a}{k_i}$ with normalisation $ a = \sum_j \frac{1}{k_j}$

    """

    def deactivate_node(Pd):
        for node, pd in Pd.items():
            if throw_dice(pd):      # random node
                if print_level(prtl, PrtLvl.Verbose):
                    print(f' deactivating node = {node} with prob = {pd}')
                break
        Pd.pop(node)
        return node

    m = len(G)
    norm = sum_k(G)
    # adding new node
    new_node = m
    G.add_node(new_node)
    G.nodes[new_node]['state'] = 1
    WG = []
    KG = [0.]

    for node in np.arange(m):
        if print_level(prtl, PrtLvl.Verbose):
            print(f' Now considering link from node = {new_node} to node = {node}')

        if node in WG:
            if print_level(prtl, PrtLvl.Verbose):
                print(f' node already selected ')
            continue

        else:
            if throw_dice(mu): # attach to a random node
                rnode = select_random_node(G, WG, norm)
                #print(f' selected node = {rnode}, WG = {WG}')
                if rnode not in WG:
                    WG.append(rnode)

                    edge = (new_node, rnode)
                    G.add_edge(*edge)
                    if print_level(prtl, PrtLvl.Verbose):
                         print(f' Attach random node = {rnode}')
                else:
                    if print_level(prtl, PrtLvl.Verbose):
                         print(f' node = {rnode} already selected')

            else : # attach to the active node

                edge = (new_node, node)
                G.add_edge(*edge)
                WG.append(node)
                if print_level(prtl, PrtLvl.Verbose):
                    print(f' Now choosing active node = {node}')

    if print_level(prtl, PrtLvl.Verbose):
        print(f' List of selected nodes = {WG}')

    a = ke_deactivation_prob_norm(G)
    Pd = ke_deactivation_prob(G, a)

    if print_level(prtl, PrtLvl.Detailed):
        print(f' deactivation prob normalisation = {a}')
        print(f' deactivation prob for active nodes  = {Pd}')

    node = deactivate_node(Pd)
    if print_level(prtl, PrtLvl.Detailed):
        print(f'now deactivating node = {node}')

    G.nodes[node]['state'] = 0


def KE_network(N, m=10, mu=0.5):
    """
    The KE network is build up to N nodes starting from
    m fully connected nodes.

    """

    G = KE_network_init(m)

    for step in np.arange(N-m):
        KE_newtwork_step(G, mu)

    return G
