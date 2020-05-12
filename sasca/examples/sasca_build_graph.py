from stella.sasca.Graph import *
from stella.sasca.utils import *
from stella.sasca.Node import *
import matplotlib.pyplot as plt

def wrap_function(lib, funcname, restype, argtypes):
    """Simplify wrapping ctypes functions"""
    func = lib.__getattr__(funcname)
    func.restype = restype
    func.argtypes = argtypes
    return func


if __name__ == "__main__":
    #remove run from previous execution
    print(VNode)
    VNode.reset_all();
    FNode.reset_all();
    Nk = 16
    constant = np.array([0x03],dtype=np.uint32)

    print("Example to build a Factor Graph")
    A = VNode(value=0x3)
    #A = A & constant
    B = VNode(value=0x8)
    D = A & B
    C = A ^ B
    print("C has a value %d \n"%(C.eval()))

    # initialize the graph
    graph = Graph(Nk=Nk,nthread=1,vnodes=VNode.buff,fnodes=FNode.buff)
    plt.figure()
    graph.plot()
    plt.show(block=False)

    #########################################
    # init the input and output distributions
    distri_a = np.zeros(Nk)
    distri_a[A.eval()] = 1

    distri_b = np.zeros(Nk)
    distri_b[B.eval()] = 1

    flag_in = [A._flag,B._flag]
    distri_in = [distri_a,distri_b]

    flag_out = [C._flag]
    distri_out = [np.zeros(Nk)]
    
    graph.initialize_nodes(distri_in,flag_in,distri_out,flag_out)
    ###########################################

    ###########################################
    # run iterations of BP
    IT = graph.get_diameter()*2
    graph.run_bp(it=IT)
    ###########################################
    print('The guessed C is %d and expected %d'%(np.argmax(distri_out[0]),C.eval()))
