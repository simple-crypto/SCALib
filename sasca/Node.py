import ctypes
import numpy as np
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

def apply_func(func=bxor,inputs=[None],offset=None):
    """ apply the functionc func to the inputs and 
        returns the output node 
    """
    FTMP = FNode(func=func,inputs=inputs,offset=offset)
    return VNode(result_of=FTMP)


class FNode(ctypes.Structure):
    """
        This object contains the function nodes of the factor graph

        It contains multiple variables:
            - func: is the function that is applyed to the the node. All the
              available function are above in this file
            - id: is the id of then node. The ids are distributed in ordre
            - inputs: are the variable nodes at the input of this function
            - output: is the ouput of this function node.
    """

    # interface with C library
    _fields_ = [('id', ctypes.c_uint32),                # node id
            ('li', ctypes.c_uint32),                    # number of inputs
            ('has_offset', ctypes.c_uint32),            # is there a constant 
            ('offset', ctypes.POINTER(ctypes.c_uint32)),
            ('func_id', ctypes.c_uint32),
            ('i', ctypes.POINTER(ctypes.c_uint32)),
            ('o', ctypes.c_uint32),
            ('relative', ctypes.POINTER(ctypes.c_uint32)),
            ('msg', ctypes.POINTER(ctypes.c_double)),
            ('repeat', ctypes.c_double),
            ('lf',ctypes.c_double)]


    N = 0
    buff = []
    tab = None
    @staticmethod
    def reset_all():
        for b in FNode.buff:
            del b
        FNode.buff = []
        FNode.N = 0

    def __init__(self,func,inputs=None,offset=None,str=None,lf=1,repeat=1):
        """
            func: the function implemented by the nodes
            input: a list with the input variable nodes that are the 
            inputs of this node
            offset: is the constant second argument of func. It can be a numpy array with
                dtype=np.uint32 or an integer.
            str: name of the node

            lf: used in LRPM, loss factor
            repeat: used in LRPM, number of time this operation is performed i.e. key addition. This will 
                will be a multiplicative factor of "repeat" if the node receiving it has flag acc set high.
        """

        #add this node the the list
        self._id = FNode.N
        FNode.N +=1
        FNode.buff.append(self)

        self._func = func
        self._func_id = all_functions.index(func)
        self._inputs = inputs
        self._output = None
        self._lf=lf
        self._repeat = repeat

        # offset setting
        if offset is None:
            self._has_offset = False
            self._offset = np.array([0],dtype=np.uint32)
        else:
            self._has_offset = True
            if isinstance(offset,np.ndarray):
                assert offset.dtype == np.uint32
                self._offset = offset
            else:
                self._offset = np.array([offset],dtype=np.uint32)

        # notify that all the inputs that they are used here
        # this will update their structure to send message to this
        # function node
        for n in inputs:
            n.used_by(self)
        
        if str is None:
            if self._func_id == 0:
                str = "AND"
            elif self._func_id == 2:
                str = "XOR"
            else:
                str = " f %d"%(self._func_id)# + " " + str(self._id) 
        self._str = str
    
    def __str__(self):
        return self._str

    def eval(self):
        """
            apply the function to its inputs and return 
            the output
        """
        if len(self._inputs) == 1:
            if self._has_offset:
                return self._func(self._inputs[0].eval(),self._offset)
            else:
                return self._func(self._inputs[0].eval())
        else:
            return self._func(self._inputs[0].eval(),self._inputs[1].eval())

    def add_output(self,vnode):
        """
            add vnode as the output of this function node.
        """
        if self._output is not None:
            raise Exception("FNode can not have multiple outputs")
        self._output = vnode
    
    def initialize(self,Nk):
        """ 
            initialize the message memory for this function node
            Nk: the lenght of messages passed.i.e. is VNodes are on 8-bit, 
                Nk is set to 256

            the messages to passed are contained in an array of shape (1+li,Nk). 
            The first row is the messaged passed to the output node. 
            The second row is the messaged passer to the first input, etc ... 
        """
        if self._output is None:
            raise Exception("Initialize FNode which has no output node")
        

        ## number of messages to passed
        nmsg = len(self._inputs) + 1
        self._msg = np.zeros((nmsg,Nk),dtype=distribution_dtype)

        # Position of the inputs in the variable nodes. 
        # The output node is always first in the variable node
        self._i = np.array([node._id for node in self._inputs]).astype(np.uint32)
        self._o = np.uint32(self._output._id)

        # relative is the position of the function node within the inputs vnodes. 
        # this is used to get the approriate messages from the variable nodes
        self._relative = np.array([np.where(vnode._id_neighboor==self._id)[0] for vnode in self._inputs]).astype(np.uint32)

        # index is depreciated
        self._indexes = np.zeros((3,Nk),dtype=np.uint32)
        for i in range(3):
            self._indexes[i,:] = np.arange(Nk)

        ## conversion to ctype pointer
        self.id = np.uint32(self._id)
        self.li = np.uint32(len(self._i))
        self.has_offset = np.uint32(self._has_offset)
        self.offset = self._offset.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32))
        self.func_id = np.uint32(self._func_id)

        self.i = self._i.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32))
        self.o = np.uint32(self._o)
        self.relative = self._relative.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32))
        self.msg = self._msg.ctypes.data_as(ctypes.POINTER(ctypes.c_double))

        # LRPM info
        self.lf = np.double(self._lf)
        self.repeat = np.double(self._repeat)

    def __hash__(self):
        return self._id  | 0xf00000


class VNode(ctypes.Structure):
    """
        This object contains the variable nodes of the factor graph

        It contains multiple variables:
            - value: is the value of the actual node. This can be a scallar of a
            numpy array (do to // computationon the factor graph)
            - id: is the id of then node. The ids are distributed in ordre
            - result_of: is the function node that outputs this variable node
            - used_by: is the function node that use this variable node
            - relative: position of this vnode wihtin the connected functions

            - msg: the messages passed to the neighboor function nodes. 
            - distri_orig: prior distribution of the node
            - distri: posterior distribution of the variable (after BP)

    """
    N = 0
    buff = []
    default_flag = {"profile":True,"method":"LDA","and_output":False}

    _fields_ = [('id', ctypes.c_uint32),
            ('Ni', ctypes.c_uint32),
            ('Nf', ctypes.c_uint32),
            ('Ns', ctypes.c_uint32),
            ('use_log', ctypes.c_uint32),
            ('acc', ctypes.c_uint32),
            ('relative', ctypes.POINTER(ctypes.c_uint32)),
            ('id_input', ctypes.c_uint32),
            ('id_output', ctypes.POINTER(ctypes.c_uint32)),
            ('msg', ctypes.POINTER(ctypes.c_double)),
            ('distri_orig', ctypes.POINTER(ctypes.c_double)),
            ('distri', ctypes.POINTER(ctypes.c_double))]
    @staticmethod
    def reset_all():
        for b in VNode.buff:
            del b
        VNode.buff = []
        VNode.N = 0
    def __hash__(self):
        return self._id

    def __init__(self,value=None,result_of=None,str=None,use_log=0,flag=None,acc=0):
        """
            value: is the value of the node, can be a int of a numpy array
            result_of: is the function node that output this variable
                (None if no function is involved)
                eiter value or result must not be None
            use_log: if set, the VNode BP is run on log distribitions. This is needed
                when many there it alot of neighboors since it avoids computational errors
            flag: the flag of the node. It is a tuple with the first element being True if the node
                must be profiled. Other elements are free. Default flag=(True,id,..)
            acc: used only in LRPM. acc if is used in several encryptions (i.e. key schedule node)
        """
        if value is None and result_of is None:
            raise Exception("value and result_of are None")

        if value is not None and result_of is not None:
            raise Exception("value and result_of are not None together")

        self._value = value
        self._result_of = result_of
        self._acc = acc

        self._id = VNode.N
        VNode.N += 1
        VNode.buff.append(self)

        if flag is None:
            flag = VNode.default_flag.copy()
        flag["id"] = self._id
        self._flag = flag

        self._use_log = use_log

        # say to the funciton node that this is its output.
        if result_of is not None: 
            result_of.add_output(self)

        # all the function nodes taking self as input
        self._used_by = []

        if str is None:
            str =  "v %d"%(self._id)
        self._str = str
        self._is_initialized = False
        self._evaluated = False

    def eval(self):
        """
            returns the value of this variable node. To do so, 
            search of the output of the parent node
        """

        # if being node not evaluated yet, update 
        if self._evaluated == False and self._result_of is not None:
            if self._value is None: # value has not alread been declared
                self._value = self._result_of.eval()
            else: #update the value array
                self._value[:] = self._result_of.eval()

        self._evaluated = True
        return self._value

    def used_by(self,fnode):
        """
            add the fnode to the list of fnodes using this variable
        """
        self._used_by.append(fnode)

    def __str__(self):
        if "variable" in self._flag:
            self._str = self._flag["variable"]
        return self._str

    def initialize(self,Nk=None,distri_orig=None,distri=None,copy=False):
        """ Initialize the variable node. It goes in all its neighboors and
            searchs for its relative position with their lists

            args:
                - distri: prosterior node distribution
                - distri_orig: prior node distribution
                - Nk: the number of possible values that this node can take
            the two cannot be None at the same time ...

            before starting BP, distri_orig is copied in msg. Each run of bp is so independent.

            created state:
                - relative contains the position of this variable in its functions nodes
                - distri extrinsic distribution of the node
                - distri_orig intrinsic distriution of the node
                - id_neighboor: if of the neighboors, starting with the result_of

        """
        if Nk is None and distri_orig is None:
            raise Exception("Nk and distri_orig cannot be None at the same time")

        if self._is_initialized == True:
            raise Exception("Node cannot be initialized twice")
        # setting up distri and Nk
        if distri_orig is None:
            distri_orig = np.ones(Nk,dtype=distribution_dtype)/Nk

        if distri_orig.dtype != distribution_dtype:
            raise Exception("distri_orig has not the correct type")
        self._distri_orig = distri_orig

        # setting up distri and Nk
        if distri is None:
            distri = np.ones(Nk,dtype=distribution_dtype)/Nk
        if distri.dtype != distribution_dtype:
            raise Exception("distri_orig has not the correct type")
        self._distri = distri

        # header
        self.id = np.uint32(self._id)
        self.Ni = np.uint32(self._result_of is not None)
        self.Nf = np.uint32(len(self._used_by))
        self.Ns = np.uint32(Nk)
        self.use_log = np.uint32(self._use_log)
        
        # relative contains the position of this variable node
        # at in input of each of the functions that use it. In fnodes, 
        # the msg with index 0 is always the output. There comes the 1+. 
        self._relative = np.array([1+fnode._inputs.index(self) for fnode in self._used_by]).astype(np.uint32)
        self.acc = np.uint32(self._acc)


        nmsg = self.Ni + self.Nf
        # one message to result_of and on to each function using this node
        # will be initialized to distri_orig before running BP by the C lib.
        self._msg = np.zeros((nmsg,Nk),dtype=distribution_dtype)
        
        # function node that outputs this node
        if self.Ni > 0:
            self._id_input = np.uint32(self._result_of._id)
        else:
            self._id_input = np.uint32(0)

        # function node that uses this node
        if self.Nf > 0:
            self._id_output = np.array([node._id for node in self._used_by],dtype=np.uint32)
        else:
            self._id_output = np.array([],dtype=np.uint32)

        tmp = []
        if self._result_of is not None:
            tmp.append(self._result_of._id)
        for node in self._used_by:
            tmp.append(node._id)
        self._id_neighboor = np.array(tmp,dtype=np.uint32)

        self.relative = self._relative.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32))
        self.id_output = self._id_output.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32))
        self.id_input = self._id_input
        self.msg = self._msg.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        self.distri_orig = self._distri_orig.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        self.distri = self._distri.ctypes.data_as(ctypes.POINTER(ctypes.c_double))

        self._is_initialized = True
    def __and__(self,other):
        if isinstance(other,VNode):
            ret = apply_func(band,inputs=[self,other])
            ret._flag["method"]="LR"
            ret._flag["and_output"]=True
            return ret
        else:
            return apply_func(band,inputs=[self],offset=other)

    def __xor__(self,other):
        if isinstance(other,VNode):
            return apply_func(bxor,inputs=[self,other])
        else:
            return apply_func(bxor,inputs=[self],offset=other)

    def __invert__(self):
        ret = apply_func(binv,inputs=[self])
        # No need to profile this node since bijectively related to 
        # the input
        self._flag["profile"] = False
        return ret
