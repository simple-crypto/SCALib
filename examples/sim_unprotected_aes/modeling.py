import numpy as np
from stella.modeling import LDAClassifier
from stella.ioutils import DataReader
from stella.attacks import SASCAGraph
from tqdm import tqdm
import pickle
from settings import * 

print("Load POIs")
profile_var = pickle.load(open(fmodels,"rb"))

print("Generate arrays to store traces at given POIs")
for v in profile_var:
    profile_var[v]["samples"] = np.zeros((ntraces_profile,len(profile_var[v]["POI"])),dtype=np.int16)
    profile_var[v]["data"] = np.zeros(ntraces_profile,dtype=np.uint16)

print("Load traces at POI from profiling files")
for (traces,labels,index) in tqdm(zip(DataReader(files_traces,None),
                                DataReader(files_labels,["labels"]),
                                range(0,ntraces_profile,ntraces)),
                                total=nfile_profile,desc="Loading"):
    labels = labels[0][0]
    for i,v in enumerate(profile_var):
        var = profile_var[v]
        var["data"][index:index+ntraces] = labels[v]
        var["samples"][index:index+ntraces] = traces[:,var["POI"]]

print("Build LDAClassifier from traces")
for v in tqdm(profile_var,desc="Fit LDAClassifiers"):
    var = profile_var[v]
    var["model"] = LDAClassifier(p=1,nc=256)
    var["model"].fit(var["samples"],var["data"])
    var.pop("samples")
    var.pop("data")

pickle.dump(profile_var,open(fmodels,'wb'))

exit()
