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
    print("Example to build a Factor Graph")
    A = VNode(value=0x3)
    A = A & (0x03<<2)
    B = VNode(value=0x8)
    D = A & B
    C = A ^ B
    print("C has a value %d \n"%(C.eval()))

    print("plot the factor graph")
    plt.figure()
    plot_graph()
    plt.show(block=False)

    #initialize prior distribution
    N = VNode.N
    distri_in = np.ones((N,Nk),dtype=np.float64)/Nk
    distri_in[A._id,:] = 0
    distri_in[A._id,A.eval()] = 1
    distri_in[B._id,:] = 0
    distri_in[B._id,B.eval()] = 1

    #initialize posterior distribution
    distri_out = np.ones((N,Nk),dtype=np.float64)/Nk

    #init all variable nodes with prior distri and output one
    for i,node in enumerate(VNode.buff):
        node.initialize(Nk=Nk,distri_orig=distri_in[i],
            distri=distri_out[i])

    # same on function nodes
    for node in FNode.buff:
        node.initialize(Nk=Nk)
    print("Initialize the complete Graph")

    # init the C lib
    graph = Graph(Nk=Nk,nthread=1)
    # run 2 iterations of BP
    graph.run_bp(it=4)

    print('The guessed C is %d and expected %d'%(np.argmax(C._distri),C.eval()))
