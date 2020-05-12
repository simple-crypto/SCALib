import numpy as np
import networkx as nx
import ctypes

############################
#     Function for nodes
############################ 
bxor = np.bitwise_xor   #ID 2
band = np.bitwise_and   #ID 0
binv = np.invert        #ID 1
def ROL16(a, offset):   #ID 3
    Nk = 2**16
    a = a
    if offset == 0:
        return a
    rs = int(np.log2(Nk) - offset)
    return  (((a) << offset) ^ (a >> (rs)))%Nk
def tab_call(a,offset): #ID4
    return FNode.tab[offset,a];
##############################
distribution_dtype = np.double
all_functions = [band,binv,bxor,ROL16,tab_call]

class Graph():
    @staticmethod
    def wrap_function(lib, funcname, restype, argtypes):
        """Simplify wrapping ctypes functions"""
        func = lib.__getattr__(funcname)
        func.restype = restype
        func.argtypes = argtypes
        return func

    def __init__(self,Nk,nthread=16,vnodes=None,fnodes=None,DIR=""):
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
            mode:   0 -> on distributions; 
                    1 -> on information metrics for LRPM
        """
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
