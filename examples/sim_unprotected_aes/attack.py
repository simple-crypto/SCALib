import numpy as np
from common import sbox
from stella.postprocessing import rank_accuracy
from stella.ioutils import DataReader
from stella.attacks import SASCAGraph
from tqdm import tqdm
import pickle
from settings import * 

print("Load models")
profile_var = pickle.load(open(fmodels,"rb"))
print("Generate SASCAGraph from",fgraph)
graph = SASCAGraph(fgraph,256)
print("Initialize SASCAGraph distributions memory")
graph.init_graph_memory(ntraces_attack)
print("Load Sbox into SASCAGraph parameters")
graph.get_table("sbox")[:]=sbox

# For each attack file
print("Accross files to derive profiled var. distributions")
for (traces,labels,index) in tqdm(zip(DataReader(files_traces_a,None),
                                DataReader(files_labels_a,["labels"]),
                                range(0,ntraces_attack,ntraces)),
                                total=nfile_attack,desc="Files"):
    labels = labels[0][0]
    for v in profile_var:
        var = profile_var[v]
        graph.get_distribution(v,"initial")[index:index+ntraces,:] = var["model"].predict_proba(traces[:,var["POI"]])
    
    for v in graph.get_public_labels():
        graph.get_public(v)[index:index+ntraces] = labels[v] 

print("Running iterations of blief propagation")
graph.run_bp(5)

print("Evaluate the attack result, bytewise")
secret_key = pickle.load(open("secret_key.pkl","rb"))

guess = []
rank = []
nlpr = np.zeros((16,256))
for i,k in enumerate(secret_key):
    key_distri = np.clip(graph.get_distribution("k%d"%(i),"current"),1E-50,1)
    nlpr[i,:] = -np.log10(key_distri)
    guess.append(np.argmax(key_distri,axis=1)[0])
    rank.append(256 - np.where(np.argsort(key_distri,axis=1)[0] == k)[0])

print("\nguess    :", " ".join(["%3x"%(x)for x in guess]))
print("key      :", " ".join(["%3x"%(x)for x in secret_key]))
print("byterank :", " ".join(["%3d"%(x)for x in rank]))

print("Compute full key rank")
rmin,r,rmax = rank_accuracy(nlpr,secret_key,2**1.0)
print("log2 key rank : %f < %f < %f"%(np.log2(rmin),np.log2(r),np.log2(rmax)))
