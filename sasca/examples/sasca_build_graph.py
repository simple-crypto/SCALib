from stella.sasca.Graph import *

if __name__ == "__main__":
    print("Example to build a Factor Graph")
    A = VNode(value=0xff)
    B = VNode(value=0x01)
    C = apply_func(band,inputs=[A,B])
    plot_graph()
    plt.show()
