from stella.sasca.Node import *

import networkx as nx

def initialize_graph(distri=None,Nk=None):
    """
        initialize the complete factor graph
        distri: (#ofVnode,Nk) or None. The row of distri are assigned to 
            the VNode with the row index
        Nk: the number of possible values for the variable nodes
    """
    for p,node in enumerate(VNode.buff):
        if distri is not None:
            d = distri[p,:]
            Nk = len(d)
        else:
            d = None
        node.initialize(distri=d,Nk=Nk)
    for node in FNode.buff:
        node.initialize(Nk=Nk)
def build_nx_graph(fnodes):
    G = nx.Graph()
    off = 0
    for F in fnodes:
        for vnode in F._inputs:
            G.add_edges_from([(vnode,F)])
        G.add_edges_from([(F,F._output)])
    return G

def plot_graph(fnodes=None,G=None,cycle=None,pos=None):
    if fnodes is None:
        fnodes = FNode.buff
    if G is None:
        G = build_nx_graph(fnodes)
    color_map=[]
    for node in G.nodes:
        if isinstance(node,VNode):
            color_map.append('r')
        else:
            color_map.append('g')
    edge_map = []
    for ed in G.edges:
        if (cycle is not None) and (ed[0] in cycle) and (ed[1] in cycle):
            edge_map.append('r')
        else:
            edge_map.append('k')

    if pos is None:
        pos = nx.spring_layout(G)
    nx.draw(G,with_labels=True,pos=pos,node_color=color_map,edge_color=edge_map)

def longest_path(fnodes):
    G = build_nx_graph(fnodes)
    return nx.algorithms.dag.dag_longest_path(G)
