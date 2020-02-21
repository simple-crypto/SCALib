import numpy as np
from tqdm import tqdm
import networkx as nx

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
##############################
distribution_dtype = np.float64
all_functions = [band,binv,bxor,ROL16]
class VNode:
    """
        This object contains the variable nodes of the factor graph

        It contains multiple variables:
            - value: is the value of the actual node. This can be a scallar of a
            numpy array (do to // computationon the factor graph)
            - id: is the id of then node. The ids are distributed in ordre
            - result_of: is the function node that outputs this variable node
            - used_by: is the function node that use this variable node
    """
    N = 0
    buff = []

    @classmethod
    def reset_all():
        for b in buff:
            del b
        buff = []
        N = 0
    def __init__(self,value=None,result_of=None):
        """
            value: is the value of the node
            result_of: is the function node that output this variable
                (None if no function is involved)
        """
        self._value = value
        self._result_of = result_of
        self._id = VNode.N
        VNode.N += 1
        VNode.buff.append(self)

        # say to the funciton node that this is its output. 
        if result_of is not None: 
            result_of.add_output(self)

        # all the function nodes taking self as input
        self._used_by = []

    def eval(self):
        """
            returns the value of this variable node. To do so, 
            search of the output of the parent node
        """
        if self._value is None:
            self._value = self._result_of.eval()
        return self._value

    def used_by(self,fnode):
        """
            add the fnode to the list of fnodes using this variable
        """
        self._used_by.append(fnode)
    def __str__(self):
        return str(self._id)

    def initialize(self,Nk=None,distri=None):
        """ Initialize the variable node. It goes in all its neighboors and
            searchs for its relative position with their lists

            args:
                - distri: the initial distribution of the node
                - Nk: the number of possible values that this node can take
            the two cannot be None at the same time ... 
        """
        if Nk is None and distri is None:
            raise Exception("Nk and distri cannot be None at the same time")

        if distri is None:
            distri = np.ones(Nk,dtype=distribution_dtype)/Nk
        else:
            distri = distri.astype(distribution_dtype)

        # relative contains the position of this variable node
        # at in input of each of the functions that use it
        self._relative = [fnode._inputs.index(self) for fnode in self._used_by]
        self._distri = distri
        self._distri_orig = distri.copy()

    def dump(self):
        tobytes = np.ndarray.tobytes
        if self._i is None:
            Ni = 0
        else:
            Ni = 1
        if self._fnodes is None:
            Nf = 0
        else:
            Nf = len(self._fnodes)

        header = tobytes(np.array([self._id,Ni,Nf,self._Nk,self._update]).astype(np.uint32))
        ids = tobytes(np.concatenate(([i for i in self._relative],)).astype(np.uint32))
        self._all_related = np.concatenate(([self._i._id for i in range(Ni)],[self._fnodes[i]._id for i in range(Nf)])).astype(np.uint32)
        all_related = tobytes(self._all_related)
        distri = tobytes(np.concatenate((self._distri_orig,)).astype(doublet))

        header = np.frombuffer(header,dtype=np.uint8)
        distri = np.frombuffer(distri,dtype=np.uint8)
        ids = np.frombuffer(ids,dtype=np.uint8)
        r = np.frombuffer(all_related,dtype=np.uint8)
        buff = np.concatenate((header,ids,r,distri)).astype(np.uint8)
        self._buff = buff

        return buff


class FNode:
    """
        This object contains the function nodes of the factor graph

        It contains multiple variables:
            - func: is the function that is applyed to the the node. All the
              available function are above in this file
            - id: is the id of then node. The ids are distributed in ordre
            - inputs: are the variable nodes at the input of this function
            - output: is the ouput of this function node.
    """

    N = 0
    buff = []
    @classmethod
    def reset_all():
        for b in buff:
            del b
        buff = []
        N = 0

    def __init__(self,func,inputs=None,offset=None):
        """
            func: the function implemented by the nodes
            input: a list with the input variable nodes that are the 
            inputs of this node
            offset: is the constant second argument of func
        """

        #add this node the the list
        self._id = FNode.N
        FNode.N +=1
        FNode.buff.append(self)

        self._func = func
        self._func_id = all_functions.index(func)
        self._inputs = inputs
        self._offset = offset

        # notify that all the inputs that they are used here
        if inputs is not None:
            for n in inputs:
                n.used_by(self)

    def __str__(self):
        return str(self._id)

    def eval(self):
        I = []
        for v in self._i:
            I.append(v.eval())

        if len(I) == 1:
            if self._offset is not None:
                return self._func(I[0],self._offset)
            else:
                return self._func(I[0])
        else:
            return self._func(I[0],I[1])

    def add_output(self,vnode):
        self._output = vnode

    def dump(self):
        tobytes = np.ndarray.tobytes
        FILE = "fnode_%d.dat"%(self._id)
        has_offset = self._offset is not None
        if has_offset == True:
            offset = self._offset
        else:
            offset = 0

        if self._func == band:
            func_id = 0
        elif self._func == binv:
            func_id = 1
        elif self._func == bxor:
            func_id = 2
        elif self._func == ROL16:
            func_id = 3
        elif self._func == SBOX:
            func_id = 4
        elif self._func == SBOX_inv:
            func_id = 5
        elif self._func == xtime:
            func_id = 6
        elif self._func == delta:
            func_id = 7

        self._relatives = np.array([np.where(vnode._all_related==self._id)[0] for vnode in self._i]).astype(np.uint32)
        relatives = tobytes(self._relatives) # for the inputs, outputed value is always at index 0 in the variable node
        header = tobytes(np.array([self._id,len(self._i),has_offset,offset,func_id]).astype(np.uint32))
        ids = tobytes(np.concatenate(([i._id for i in self._i],[self._outputvnode._id])).astype(np.uint32))

        header = np.frombuffer(header,dtype=np.uint8)
        ids = np.frombuffer(ids,dtype=np.uint8)
        relatives = np.frombuffer(relatives,dtype=np.uint8)
        buff = np.concatenate((header,ids,relatives)).astype(np.uint8)
        self._buff = buff
        return buff

def apply_func(func=bxor,inputs=[None],offset=None):
    FTMP = FNode(func=func,inputs=inputs,offset=offset)
    return VNode(result_of=FTMP)

def build_nx_grah(fnodes):
    G = nx.DiGraph()
    off = 0
    for F in fnodes:
        for vnode in F._inputs:
            G.add_edges_from([(vnode,F)])
        G.add_edges_from([(F,F._output)])
    return G
def plot_graph(fnodes=None):
    if fnodes is None:
        fnodes = FNode.buff
    G = build_nx_grah(fnodes)
    color_map=[]
    for node in G.nodes:
        if isinstance(node,VNode):
            color_map.append('r')
        else:
            color_map.append('g')
    nx.draw(G,with_labels=True,node_color=color_map)

def longest_path(fnodes):
    G = build_nx_graph(fnodes)
    return nx.algorithms.dag.dag_longest_path(G)
