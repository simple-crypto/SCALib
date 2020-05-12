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
            DIR: directory of shared lib.
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

        for i,node in enumerate(fnodes):
            self._fnodes_array[i] = node

        for i,node in enumerate(vnodes):
            self._vnodes_array[i] = node

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
                ctypes.POINTER(ctypes.c_uint32)])
    def run_bp(self,it=1,mode=0):
        """
            run belief propagation algorithm on the fed graph
            it: number of iterations
            mode:   0 -> on distribution,for attack
                    1 -> on information metrics for LRPM
        """
        if mode == 1 and Nk != 1:
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
            FNode.tab.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)))
