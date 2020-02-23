from stella.sasca.Graph import *
import matplotlib.pyplot as plt
if __name__ == "__main__":
    #remove run from previous execution
    VNode.reset_all();
    FNode.reset_all();

    print("Example to build a Factor Graph")
    A = VNode(value=0xff)
    B = VNode(value=0x0f)
    C = apply_func(band,inputs=[A,B])
    print("C has a value %02x \n"%(C.eval()))

    print("plot the factor graph")
    plot_graph()
    plt.show(block=False)

    print("Initialize the complete Graph")
    initialize_graph(Nk=256)
