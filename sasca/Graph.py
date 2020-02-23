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

    @staticmethod
    def reset_all():
        for b in VNode.buff:
            del b
        VNode.buff = []
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

            created state:
                - relative contains the position of this variable in its functions nodes
                - distri extrinsic distribution of the node
                - distri_orig intrinsic distriution of the node
                - header: information about this node:
                    | ID | is a result | len(used_by)|
                - id_neighboor: if of the neighboors, starting with the result_of
        """
        if Nk is None and distri is None:
            raise Exception("Nk and distri cannot be None at the same time")

        if distri is None:
            distri = np.ones(Nk,dtype=distribution_dtype)/Nk
        else:
            distri = distri.astype(distribution_dtype)
            Nk = len(distri)

        # header
        Ni = self._result_of is not None
        Nf = len(self._used_by)
        self._header = np.array([self._id,
                            Ni,
                            Nf,
                            Nk,
                            1]).astype(np.uint32)
        # relative contains the position of this variable node
        # at in input of each of the functions that use it. In fnodes, 
        # the msg with index 0 is always the output. There comes the 1+. 
        self._relative = np.array([1 + fnode._inputs.index(self) for fnode in self._used_by]).astype(np.uint32)
        self._distri = distri
        self._distri_orig = distri.copy()

        # one message to result_of and on to each function using this node
        nmsg = Ni + Nf
        self._msg = np.zeros((nmsg,Nk),dtype=distribution_dtype)
        
        # function node that outputs this node
        if Ni > 0:
            self._id_input = np.uint32(self._result_of._id)
        else:
            self._id_input = np.uint32(0)

        # function node that uses this node
        if Nf > 0:
            self._id_output = np.array([node._id for node in self._used_by],dtype=np.uint32)
        else:
            self._id_output = np.array([],dtype=np.uint32)
        
        tmp = []
        if self._result_of is not None:
            tmp.append(self._result_of._id)
        for node in self._used_by:
            tmp.append(node._id)
        self._id_neighboor = np.array(tmp,dtype=np.uint32)

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
    @staticmethod
    def reset_all():
        for b in FNode.buff:
            del b
        FNode.buff = []
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
        if offset is None:
            self._has_offset = False
            self._offset = np.uint32(0)
        else:
            self._has_offset = True
            self._offset = np.uint32(offset)

        # notify that all the inputs that they are used here
        if inputs is not None:
            for n in inputs:
                n.used_by(self)

    def __str__(self):
        return str(self._id)

    def eval(self):
        """
            apply the function to its inputs and return 
            the output
        """
        I = []
        for v in self._inputs:
            I.append(v.eval())

        if len(I) == 1:
            if self._has_offset:
                return self._func(I[0],self._offset)
            else:
                return self._func(I[0])
        else:
            return self._func(I[0],I[1])

    def add_output(self,vnode):
        self._output = vnode
    
    def initialize(self,Nk):
        """ initialize the message memory for this function node"""
        nmsg = len(self._inputs) + 1
        self._header = np.array([self._id,
                    len(self._inputs),
                    self._has_offset,
                    self._offset,
                    self._func_id]).astype(np.uint32)

        ## Position of the inputs in the variable nodes. 
        # The output node is always first in the variable node
        self._i = np.array([node._id for node in self._inputs]).astype(np.uint32)
        self._o = np.uint32(self._output._id)
        self._relatives = np.array([np.where(vnode._id_neighboor==self._id)[0] for vnode in self._inputs]).astype(np.uint32)
        self._msg = np.zeros((nmsg,Nk),dtype=distribution_dtype)
        self._indexes = np.zeros((3,Nk),dtype=np.uint32)
        for i in range(3):
            self._indexes[i,:] = np.arange(Nk)

def apply_func(func=bxor,inputs=[None],offset=None):
    """ apply the functionc func to the inputs and 
        returns the output node 
    """
    FTMP = FNode(func=func,inputs=inputs,offset=offset)
    return VNode(result_of=FTMP)

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
