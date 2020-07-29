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
