# Setup the simulation settings
D=2
tag = "%dshares_sim"%(D)
DIR_PROFILE = "./traces/profile/"
DIR_ATTACK = "./traces/attack/"
fgraph = "./graph.txt"
fmodels = "./models.pkl"

nfile_profile = 10
nfile_attack = 1
ntraces = 1000

std = .1
ndim = 3

ntraces_profile = nfile_profile * ntraces
ntraces_attack = nfile_attack * ntraces

# File for profiling
files_traces = [DIR_PROFILE+"/traces/"+tag+"_traces_%d.npy"%(i) for i in range(nfile_profile)]
files_labels = [DIR_PROFILE+"/labels/"+tag+"_labels_%d.npz"%(i) for i in range(nfile_profile)]

# Attack files
files_traces_a = [DIR_ATTACK+"/traces/"+tag+"_traces_%d.npy"%(i) for i in range(nfile_attack)]
files_labels_a = [DIR_ATTACK+"/labels/"+tag+"_labels_%d.npz"%(i) for i in range(nfile_attack)]


def print_settings():
    print("=========")
    print("Settings:")
    print("=========")
    print("Number of shares:",D)
    print("Noise variance  :",std)
    print("Traces location :")
    print("   profile:",DIR_PROFILE)
    print("   attack :", DIR_ATTACK)
    print("Number of traces:")
    print("   profile:",ntraces_profile)
    print("   attack :", ntraces_attack)
    print("Graph location  :",fgraph)
    print("Models location :",fmodels)
    print("=========\n")
