from common import gen_traces_serial,sbox
import numpy as np
from stella.metrics import SNR
from stella.modeling import LDAClassifier
from stella.ioutils import DataReader
from stella.attacks import SASCAGraph
from stella.postprocessing import rank_accuracy
from tqdm import tqdm
import pickle
from settings import * 

print("Generate SASCAGraph from",fgraph)
graph = SASCAGraph(fgraph,256)
print("List variables to model")
profile_var  = {}
for v in graph.get_profile_labels():
    profile_var[v] = {}


print("Passing accross profiling traces to estimate SNR")
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

print("Generate %d Point of Interest (POI) from SNR"%(ndim))
for v in profile_var:
    snr = profile_var[v]["SNR"].copy()
    profile_var[v]["POI"] = np.argsort(profile_var[v]["SNR"])[-ndim:];

print("Save results")
pickle.dump(profile_var,open(fmodels,'wb'))
exit()
