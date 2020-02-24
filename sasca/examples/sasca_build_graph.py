from stella.sasca.Graph import *
import matplotlib.pyplot as plt

def wrap_function(lib, funcname, restype, argtypes):
    """Simplify wrapping ctypes functions"""
    func = lib.__getattr__(funcname)
    func.restype = restype
    func.argtypes = argtypes
    return func


if __name__ == "__main__":
    #remove run from previous execution
    VNode.reset_all();
    FNode.reset_all();
    Nk = 16
    print("Example to build a Factor Graph")
    A = VNode(value=0x3)
    B = VNode(value=0x8)
    C = apply_func(band,inputs=[A,B])
    print("C has a value %02x \n"%(C.eval()))

    print("plot the factor graph")
    #plot_graph()
    #plt.show()

    print("Initialize the complete Graph")
    distri = np.ones((3,Nk))/Nk
    distri[0,:] = 0
    distri[0,A.eval()] = 1
    distri[1,:] = 0
    distri[1,B.eval()] = 1

    initialize_graph(distri=distri,Nk=Nk)

    graph = Graph(Nk=Nk,DIR="../",nthread=1)
    graph.run_bp(it=1)
    print('The guessed C is %d and expected %d'%(np.argmax(C._distri_all),C.eval()))
