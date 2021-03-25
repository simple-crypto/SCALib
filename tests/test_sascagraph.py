import pytest
from scale.attacks import SASCAGraph
import numpy as np

def test_table():
    """
    Test Table lookup
    """
    fgraph = "graph.txt"
    nc = 16
    n = 100
    table = np.random.permutation(nc)

    with open(fgraph, "w") as fp:
        fp.write("""%
table #table
#indeploop
x #profile
y = x -> table
#endindeploop""")
    graph = SASCAGraph(fgraph,nc)
    graph.init_graph_memory(n)
    graph.get_table("table")[:] = table

    distri_x = np.random.randint(1,2048,(n,nc))
    distri_x = (distri_x.T / np.sum(distri_x,axis=1)).T
    graph.get_distribution("x","initial")[:,:] = distri_x
    graph.run_bp(1)
    distri_y = graph.get_distribution("y","current")
    
    distri_y_ref = np.zeros(distri_x.shape)
    for x in range(nc):
        y = table[x]
        distri_y_ref[:,y] = distri_x[:,x]

    assert np.allclose(distri_y_ref,distri_y)

def test_xor_public():
    """
    Test XOR with public data
    """
    fgraph = "graph.txt"
    nc = 16
    n = 100
    public = np.random.randint(0,nc,n)

    with open(fgraph, "w") as fp:
        fp.write("""
public #public
#indeploop
x #profile
y = x + public
#endindeploop""")
    graph = SASCAGraph(fgraph,nc)
    graph.init_graph_memory(n)
    graph.get_public("public")[:] = public

    distri_x = np.random.randint(1,100,(n,nc))
    distri_x = (distri_x.T / np.sum(distri_x,axis=1)).T
    graph.get_distribution("x","initial")[:,:] = distri_x
    graph.run_bp(1)
    distri_y = graph.get_distribution("y","current")
    
    distri_y_ref = np.zeros(distri_x.shape)
    for x in range(nc):
        y = x ^ public
        distri_y_ref[np.arange(n),y] = distri_x[np.arange(n),x]

    assert np.allclose(distri_y_ref,distri_y)

def test_xor():
    """
    Test XOR between distributions
    """
    fgraph = "graph.txt"
    nc = 16
    n = 100

    with open(fgraph, "w") as fp:
        fp.write("""
#indeploop
x #profile
y #profile
z = x ^ y 
#endindeploop""")
    graph = SASCAGraph(fgraph,nc)
    graph.init_graph_memory(n)

    distri_x = np.random.randint(1,100,(n,nc))
    distri_x = (distri_x.T / np.sum(distri_x,axis=1)).T
    graph.get_distribution("x","initial")[:,:] = distri_x
    distri_y = np.random.randint(1,100,(n,nc))
    distri_y = (distri_y.T / np.sum(distri_y,axis=1)).T
    graph.get_distribution("y","initial")[:,:] = distri_y

    graph.run_bp(1)
    distri_z = graph.get_distribution("z","current")
    
    distri_z_ref = np.zeros(distri_z.shape)
    for x in range(nc):
        for y in range(nc):
            distri_z_ref[:,x^y] += distri_x[:,x] * distri_y[:,y]

    assert np.allclose(distri_z_ref,distri_z)

def test_and():
    """
    Test AND between distributions
    """
    fgraph = "graph.txt"
    nc = 16
    n = 100

    with open(fgraph, "w") as fp:
        fp.write("""
#indeploop
x #profile
y #profile
z = x & y 
#endindeploop""")
    graph = SASCAGraph(fgraph,nc)
    graph.init_graph_memory(n)

    distri_x = np.random.randint(1,100,(n,nc))
    distri_x = (distri_x.T / np.sum(distri_x,axis=1)).T
    graph.get_distribution("x","initial")[:,:] = distri_x
    distri_y = np.random.randint(1,100,(n,nc))
    distri_y = (distri_y.T / np.sum(distri_y,axis=1)).T
    graph.get_distribution("y","initial")[:,:] = distri_y

    graph.run_bp(1)
    distri_z = graph.get_distribution("z","current")
    
    distri_z_ref = np.zeros(distri_z.shape)
    for x in range(nc):
        for y in range(nc):
            distri_z_ref[:,x&y] += distri_x[:,x] * distri_y[:,y]

    assert np.allclose(distri_z_ref,distri_z)
