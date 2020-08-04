from stella.sasca.Node import *
import os
import numpy as np
import networkx as nx
import ctypes

class Graph():
    """
        Graph allows to interface with the C Belief Progation Library.
    """
    @staticmethod
    def wrap_function(lib, funcname, restype, argtypes):
        """Simplify wrapping ctypes functions"""
        func = lib.__getattr__(funcname)
        func.restype = restype
        func.argtypes = argtypes
        return func

    def __init__(self,Nk,nthread=16,vnodes=None,fnodes=None,DIR=None):
        """
            Nk: number of possible value. i.e. if running on 8-bit value, Nk=256
            nthread: number of CPU used for BP
            vnodes: a list of vnodes. If None, VNode.buff is taken
            fnodes: a list of fnodes. If None, FNode.buff is taken
            DIR: directory of shared lib, is None, get the one from stella
        """
        if vnodes is None:
            vnodes = VNode.buff
        self._vnodes = vnodes

        if fnodes is None:
            fnodes = FNode.buff
        self._fnodes = fnodes

        self._nthread = nthread
        self._Nk = Nk

        self._fnodes_array = (FNode*len(fnodes))()
        self._vnodes_array = (VNode*len(vnodes))()

        if DIR == None:
            DIR = os.path.dirname(__file__)+"/../lib/"

        self._lib = ctypes.CDLL(DIR+"./libbp.so")
        self._run_bp = self.wrap_function(self._lib,"run_bp",None,[ctypes.POINTER(VNode),
                ctypes.POINTER(FNode),
                ctypes.c_uint32,
                ctypes.c_uint32,
                ctypes.c_uint32,
                ctypes.c_uint32,
                ctypes.c_uint32,
                ctypes.c_uint32,
                ctypes.POINTER(ctypes.c_uint32),
                ctypes.c_double])

    def initialize_nodes(self,data_in,
                        data_out):
        """
            initialize the fnodes and the vnodes

            distri_in: zipped list of tags and distributions. 
                Will set the input distribution of nodes with the same tag to distri
            distri_out: zipped list of tags and distributions.
                same as distri_in but for output distribution
        """
        Nk = self._Nk

        self._vnodes = list(sorted(self._vnodes,key=lambda node: node._id))
        data_in = list(sorted(data_in,key=lambda f: f[0]["id"]))
        data_out = list(sorted(data_out,key=lambda f: f[0]["id"]))
        from tqdm import tqdm
        for node in tqdm(self._vnodes):
            if len(data_in) > 0 and node._flag == data_in[0][0]:
                distri_i = data_in[0][1]
                data_in.pop(0)
            else:
                distri_i = None

            if len(data_out) > 0 and node._flag == data_out[0][0]:
                distri_o = data_out[0][1]
                data_out.pop(0)
            else:
                distri_o = None

            node.initialize(Nk=Nk,distri_orig=distri_i,
                    distri=distri_o)

        assert len(data_out) == 0
        assert len(data_in) == 0

        for node in self._fnodes:
            node.initialize(Nk=Nk)


        for i,node in enumerate(self._fnodes):
            self._fnodes_array[i] = node
        for i,node in enumerate(self._vnodes):
            self._vnodes_array[i] = node

    def run_bp(self,it=1,mode=0,alpha=0.0):
        """
            run belief propagation algorithm on the fed graph
            it: number of iterations
            mode:   0 -> on distribution,for attack
                    1 -> on information metrics for LRPM
        """
        if mode == 1 and self._Nk != 1:
            raise Exception("For LRPM, Nk should be equal to 1")

        if FNode.tab is None:
            FNode.tab = np.zeros((2,self._Nk),dtype=np.uint32)

        self._run_bp(self._vnodes_array,
            self._fnodes_array,
            ctypes.c_uint32(self._Nk),
            ctypes.c_uint32(len(self._vnodes)),
            ctypes.c_uint32(len(self._fnodes)),
            ctypes.c_uint32(it),
            ctypes.c_uint32(self._nthread),
            ctypes.c_uint32(mode),
            FNode.tab.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
            ctypes.c_double(alpha))

    def eval(self,nodes):
        """
            this function returns an array with the same length as nodes.
            This array contains the value of nodes
        """
        for node in self._vnodes: node._evaluated = False

        ret = [None for _ in nodes]
        for node in self._vnodes:
            if node in nodes:
                ret[nodes.index(node)] = node.eval()

        return ret

    def get_nodes(self,func):
        """
            return the nodes of all the nodes maching with func

            func should return a boolean function
        """
        return list(filter(func,self._vnodes))

    ###############
    # utils methods
    ###############
    def build_nx_graph(self):
        fnodes = self._fnodes
        G = nx.Graph()
        off = 0
        for F in fnodes:
            for vnode in F._inputs:
                G.add_edges_from([(vnode,F)])
            G.add_edges_from([(F,F._output)])
        return G

    def plot(self):
        fnodes = self._fnodes
        G = self.build_nx_graph()

        color_map=[]
        for node in G.nodes:
            if isinstance(node,VNode):
                color_map.append('r')
            else:
                color_map.append('g')
        nx.draw(G,with_labels=True,node_color=color_map)

    def get_diameter(self):
        G = self.build_nx_graph()
        return nx.algorithms.distance_measures.diameter(G)

    def get_shortest_path(self):
        G = self.build_nx_graph()
        return nx.minimum_cycle_basis(G)

