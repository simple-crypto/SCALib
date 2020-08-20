import numpy as np
import random
import matplotlib.pyplot as plt
from sasca_flow_settings import *
from gen_traces import *
from gen_file import * 
from stella.attacks.sasca.scripts.graph_parsing import * 
from stella.attacks.sasca.scripts.profiling_flags import *
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA
from stella.estimator.classifiers import * 


if __name__ == "__main__":
    file_name = "example_graph.txt"
    write_file(file_name,nbytes=2)

    print("# Generate Traces")
    gen_traces_sim(nfile_profile,ntraces,DIR_PROFILE_TRACES,tag)
    gen_traces_attack_sim(nfile_attack,ntraces_attack,DIR_ATTACK_TRACES,tag)

    print("# Generate Labels")
    gen_labels(nfile_profile,DIR_PROFILE_TRACES,DIR_PROFILE_LABELS,tag)

    _,profile,_ = extract_flags("example_graph.txt")
    profile = list(map(lambda l: l["label"],profile))

    print("# 1. Computing the SNR on the files")
    write_snr(PREFIX_PROFILE_TRACES,
                PREFIX_PROFILE_LABELS,
                FILE_SNR,
                nfile_profile,
                profile,
                batch_size=-1,Nc=Nk)

    print("# 2. Getting POIs")
    ndim = 1
    write_poi(FILE_SNR,FILE_POI,
                profile,
                lambda snr: np.argsort(snr)[-ndim:] if len(np.where(snr>0.001)[0]) > ndim else np.where(snr>0.001)[0])

    print("# 3. Estimate pis")
    def _tmp (t,l,args): m = QDA(); return m.fit(t,l)
    funcs = [_tmp,lambda t,l,args: LDAClassifier(t,l,dim_projection=1)]
    
    args = [None, None]
    labels = [{"label":"y_0",
                    "models":[{"func":funcs[0],"arg":args[0],"method_tag":"QDA"},
                                {"func":funcs[1],"arg":args[1],"method_tag":"LDA"}]
                                },
            {"label":"y_1",
                    "models":[
                                {"func":funcs[1],"arg":args[1],"method_tag":"LDA"}]
                                }]
    estimate_pi(PREFIX_PROFILE_TRACES,
                    PREFIX_PROFILE_LABELS,
                    FILE_POI,
                    "pi",
                    nfile_profile,
                    labels,
                    kfold = 10,verbose=True,
                    batch_size=-1)

    for l in labels:
        m = np.load("pi_"+l["label"]+".npz",allow_pickle=True)
        plt.figure()
        for single_model in m["models"]:
            plt.loglog(single_model["ntrain"][0,:],
                np.mean(single_model["pi"],axis=0),
                label=single_model["method_tag"])
        plt.legend()
        plt.grid(True,which="both",ls="--")
    plt.show()
