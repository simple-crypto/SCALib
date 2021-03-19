from common import gen_traces_serial,sbox
import numpy as np
from stella.preprocessing import SNR
from stella.estimator import LDAClassifier
from stella.utils import DataReader
from stella.attacks.sasca import SASCAGraph
from stella.post_processing import rank_accuracy
from tqdm import tqdm
import pickle

# Setup the simulation settings
D=2
tag = "sim"
DIR_PROFILE = "./traces/profile/"
DIR_ATTACK = "./traces/attack/"
nfile_profile = 10
nfile_attack = 1
ntraces = 1000
std = 2
ndim = 3
fgraph = "./graph.txt"

## Generate the profiling and attack sets
gen_traces_serial(nfile_profile,
        ntraces,std,tag,
        DIR_PROFILE,random_key=True,D=D)
secret_key = gen_traces_serial(nfile_attack,ntraces,
                std,tag,DIR_ATTACK,random_key=False,D=D)[0]

## generate the graph
print("####################")
print("# Generate the graph")
print("####################")
print("-> Write ",fgraph)
with open(fgraph, 'w') as fp:
    fp.write("sbox #table \n")
    for i in range(16): fp.write("k%d #secret \n"%(i))
    fp.write("\n\n#indeploop\n\n")
    for i in range(16): fp.write("p%d #public\n"%(i))
    for i in range(16): fp.write("y%d = k%d + p%d\n"%(i,i,i))
    for i in range(16): fp.write("x%d = y%d -> sbox \n"%(i,i))
    for i in range(16):
        for d in range(D): fp.write("x%d_%d #profile\n"%(i,d))
        for d in range(D): fp.write("y%d_%d #profile\n"%(i,d))
        add = ' ^ '.join(["x%d_%d"%(i,d) for d in range(D)])
        fp.write("x%d = "%(i)+add+"\n")
        add = ' ^ '.join(["y%d_%d"%(i,d) for d in range(D)])
        fp.write("y%d = "%(i)+add+"\n")
    fp.write("\n\n#endindeploop\n\n")

print("-> parsing",fgraph,"to generate graph structur")
graph = SASCAGraph(fgraph,256)
pickle.dump(graph,open("graph.pkl",'wb'))

print("####################")
print("# Start Profiling")
print("####################")
# File for profiling
files_traces = [DIR_PROFILE+"/traces/"+tag+"_traces_%d.npy"%(i) for i in range(nfile_profile)]
files_labels = [DIR_PROFILE+"/labels/"+tag+"_labels_%d.npz"%(i) for i in range(nfile_profile)]

graph = pickle.load(open("graph.pkl","rb"))
profile_var  = {}
for v in graph.get_profile_labels():
    profile_var[v] = {}


print("-> Computing SNR")
#Go over all the profiling traces and update the SNR
for it,(traces,labels) in tqdm(enumerate(zip(DataReader(files_traces,None),
                                DataReader(files_labels,["labels"]))),
                                total=nfile_profile,desc="Files"):
    labels = labels[0][0]
    if it == 0:
        ntraces,ns = traces.shape
        snr = SNR(np=len(labels),nc=256,ns=ns)
        data = np.zeros((len(labels),ntraces),dtype=np.uint16)

    for i,v in enumerate(profile_var):
        data[i,:] = labels[v]
    snr.fit_u(traces,data)
snr_val = snr.get_snr()
for i,v in enumerate(profile_var):
    profile_var[v]["SNR"] = snr_val[i,:]

print("-> Computing POI with %d dims"%(ndim))
for v in profile_var:
    snr = profile_var[v]["SNR"].copy()
    snr[np.where(np.isfinite(snr)==0)] = -1
    profile_var[v]["POI"] = np.argsort(profile_var[v]["SNR"])[-ndim:];

print("-> Computing the models")
ntraces_profile = ntraces * nfile_profile
# set data to store all the POIs and labels for profiling
for v in profile_var:
    profile_var[v]["samples"] = np.zeros((ntraces_profile,len(profile_var[v]["POI"])),dtype=np.int16)
    profile_var[v]["data"] = np.zeros(ntraces_profile,dtype=np.uint16)

# fill the traces and the labels
for (traces,labels,index) in tqdm(zip(DataReader(files_traces,None),
                                DataReader(files_labels,["labels"]),
                                range(0,ntraces_profile,ntraces)),
                                total=nfile_profile,desc="Loading"):
    labels = labels[0][0]
    for i,v in enumerate(profile_var):
        var = profile_var[v]
        var["data"][index:index+ntraces] = labels[v]
        var["samples"][index:index+ntraces] = traces[:,var["POI"]]

for v in tqdm(profile_var,desc="Fit LDA"):
    var = profile_var[v]
    var["model"] = LDAClassifier(var["samples"],var["data"],n_components=1,nc=256)
    var.pop("samples")
    var.pop("data")

pickle.dump(profile_var,open("profile_var.pkl",'wb'))

print("####################")
print("Start Attack")
print("####################")
graph = pickle.load(open("graph.pkl","rb"))
profile_var = pickle.load(open("profile_var.pkl","rb"))
ntraces_attack = nfile_attack * ntraces

print("-> Init graph memory")
graph.init_graph_memory(ntraces_attack)

# Attack files
files_traces = [DIR_ATTACK+"/traces/"+tag+"_traces_%d.npy"%(i) for i in range(nfile_attack)]
files_labels = [DIR_ATTACK+"/labels/"+tag+"_labels_%d.npz"%(i) for i in range(nfile_attack)]

# For each attack file
print("-> Load information in graph")
graph.get_tables()["sbox"][:]=sbox
for (traces,labels,index) in tqdm(zip(DataReader(files_traces,None),
                                DataReader(files_labels,["labels"]),
                                range(0,ntraces*nfile_attack,ntraces)),
                                total=nfile_attack,desc="Files"):
    labels = labels[0][0]
    for v in profile_var:
        var = profile_var[v]
        graph.get_var()[v]["distri_orig"][index:index+ntraces,:] = var["model"].predict_proba(traces[:,var["POI"]])
    
    for v in graph.get_publics():
        graph.get_publics()[v][index:index+ntraces] = labels[v] 

print("-> Running BP")
graph.run_bp(5)

# Display the obtained key
guess = []
rank = []
nlpr = np.zeros((16,256))
for i,k in enumerate(secret_key):
    label = "k%d"%(i)
    key_distri = graph.get_var()[label]["distri"];
    key_distri[np.where(key_distri<1E-100)] = 1E-100
    nlpr[i,:] = -np.log10(key_distri)
    guess.append(np.argmax(key_distri,axis=1)[0])
    rank.append(256 - np.where(np.argsort(key_distri,axis=1)[0] == k)[0])
rmin,r,rmax = rank_accuracy(nlpr,secret_key,2**1.0)
print("\nguess    :", " ".join(["%3x"%(x)for x in guess]))
print("key      :", " ".join(["%3x"%(x)for x in secret_key]))
print("byterank :", " ".join(["%3d"%(x)for x in rank]))
print("key rank : %f < %f < %f"%(np.log2(rmin),np.log2(r),np.log2(rmax)))
