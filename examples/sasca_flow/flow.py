import numpy as np
import os
from stella.sasca.scripts.graph_parsing import * 
from stella.sasca.scripts.profiling_flags import *
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA
DIR_PROFILE_TRACES = "/tmp/stella_ex/profile/traces/"
DIR_ATTACK_TRACES = "/tmp/stella_ex/attack/traces/"
DIR_PROFILE_LABELS = "/tmp/stella_ex/profile/labels/"
DIR_SNR = "/tmp/stella_ex/profile/snr/"
DIR_POI = "/tmp/stella_ex/profile/poi/"
DIR_MODEL = "/tmp/stella_ex/profile/model/"
tag = "example"

PREFIX_PROFILE_TRACES = DIR_PROFILE_TRACES+tag+"_traces"
PREFIX_ATTACK_TRACES = DIR_ATTACK_TRACES+tag+"_traces"
PREFIX_PROFILE_LABELS = DIR_PROFILE_LABELS+tag+"_labels"
FILE_SNR = DIR_SNR+tag+"_snr.npz"
FILE_POI = DIR_POI+tag+"_poi.npz"
FILE_MODEL = DIR_MODEL+tag+"_model.npz"

nfile_profile = 10
nfile_attack = 10
ntraces = 10000
ntraces_attack = 10000

os.system("mkdir -p "+DIR_PROFILE_TRACES)
os.system("mkdir -p "+DIR_PROFILE_LABELS)
os.system("mkdir -p "+DIR_SNR)
os.system("mkdir -p "+DIR_POI)
os.system("mkdir -p "+DIR_MODEL)

os.system("mkdir -p "+DIR_ATTACK_TRACES)

Nk = 256
sbox = np.random.permutation(range(Nk)).astype(np.uint8)
#sbox = np.arange(Nk).astype(np.uint8)
def gen_traces_attack_sim(nfile_attack,ntraces,DIR_TRACES,tag):
    """
        
    """
    for i in range(nfile_attack):
        p = np.random.randint(0,Nk,(ntraces,16),dtype=np.uint8)
        k = np.random.randint(0,Nk,16,dtype=np.uint8)
        x = sbox[p ^ k]
        hw = np.sum(np.unpackbits(np.expand_dims(x,2),axis=2),axis=2).astype(np.uint8)
        traces = hw + np.random.normal(0,10,hw.shape)
        traces -= 4
        traces *= 128
        traces = traces.astype(np.int16)

        np.save(DIR_TRACES+tag+"_traces_%d.npy"%(i),
                        traces)
        np.savez(DIR_TRACES+tag+"_meta_%d.npz"%(i),
                        p=p,
                        k=k,
                        sbox=sbox,
                        allow_pickle=True)
def gen_traces_sim(nfile_profile,ntraces,DIR_TRACES,tag):
    """
        
    """
    for i in range(nfile_profile):
        p = np.random.randint(0,Nk,(ntraces,16),dtype=np.uint8)
        k = np.random.randint(0,Nk,(ntraces,16),dtype=np.uint8)
        x = sbox[p ^ k]
        hw = np.sum(np.unpackbits(np.expand_dims(x,2),axis=2),axis=2).astype(np.uint8)
        traces = hw + np.random.normal(0,10,hw.shape)
        traces -= 4
        traces *= 128
        traces = traces.astype(np.int16)

        np.save(DIR_TRACES+tag+"_traces_%d.npy"%(i),
                        traces)
        np.savez(DIR_TRACES+tag+"_meta_%d.npz"%(i),
                        p=p,
                        k=k,
                        sbox=sbox,
                        allow_pickle=True)


def gen_labels(nfile_profile,DIR_TRACES,DIR_LABELS,tag):
    for i in range(nfile_profile):
        dic = np.load(DIR_TRACES+tag+"_meta_%d.npz"%(i))
        x = dic["p"] ^ dic["k"]
        labels = []
        for j in range(16): labels.append({"label":"x%d"%(j),"val":x[:,j]})
        np.savez(DIR_LABELS+tag+"_labels_%d.npz"%(i),
                        labels=labels,
                        allow_pickle=True)

if __name__ == "__main__":

    print("# Generate Traces")
    gen_traces_sim(nfile_profile,ntraces,DIR_PROFILE_TRACES,tag)
    gen_traces_attack_sim(nfile_attack,ntraces_attack,DIR_ATTACK_TRACES,tag)

    print("# Generate Labels")
    gen_labels(nfile_profile,DIR_PROFILE_TRACES,DIR_PROFILE_LABELS,tag)

    _,profile,_ = extract_flags("example_graph.txt")
    profile = list(map(lambda l: l["label"],profile))

    print("\n# Profiling labels :",profile)
   
    print("# 1. Computing the SNR on the files")
    write_snr(PREFIX_PROFILE_TRACES,
                PREFIX_PROFILE_LABELS,
                FILE_SNR,
                nfile_profile,
                profile,
                batch_size=-1,Nc=Nk)

    print("# 2. Getting POIs")
    ndim = 2
    write_poi(FILE_SNR,FILE_POI,
                profile,
                lambda snr: np.argsort(snr)[-ndim:] if len(np.where(snr>0.001)[0]) > ndim else np.where(snr>0.001)[0])

    print("# 3. Building templates")

    build_model(PREFIX_PROFILE_TRACES,
                    PREFIX_PROFILE_LABELS,
                    FILE_POI,
                    FILE_MODEL,
                    nfile_profile,
                    profile,
                    func=lambda x: QDA(),batch_size=-1)

    print("\n# Attack Part")
    LOOP_IT = ntraces_attack; repeat = 1;
    public,profile,secret = extract_flags("example_graph.txt")
    
    for v in public: v["input"] = np.zeros((LOOP_IT,repeat),dtype=np.uint32) if v["loop"] else np.zeros((1,repeat),dtype=np.uint32)
    print("# 1. Build Graph")
    graph = build_graph_from_file("example_graph.txt",Nk=Nk,it=LOOP_IT,public=public)


    print("# 2. Init Attack Graph")
    secret,profile = initialize_graph_from_file(graph,"example_graph.txt",verbose=False,Nk=Nk,LOOP_IT=LOOP_IT)

    print("# 3. Load models")
    model = np.load(FILE_MODEL,allow_pickle=True)["model"]


    print("# 4. Performing Attacks")
    public_l = np.array([x["label"] for x in public])
    model_l = np.array([x["label"] for x in model])

    for f in range(nfile_attack):
        t = np.load(DIR_ATTACK_TRACES+tag+"_traces_%d.npy"%(f))
        dic = np.load(DIR_ATTACK_TRACES+tag+"_meta_%d.npz"%(f),allow_pickle=True)
        pt = dic["p"]
        k = dic["k"]

        # set the public inputs
        for i in range(16):
            index = np.where(public_l == "p%d"%(i))[0][0]
            public[index]["input"][:,0] = pt[:,i]
      
        # compute proba on all the profiled variables
        for p in profile:
            m = model[np.where(model_l == p["label"])[0][0]]
            p["distri"][:,:] = m["model"].predict_proba(t[:,m["poi"]])
            p["distri"][:,:] = m["model"].predict_proba(t[:,m["poi"]])

        graph.run_bp(1)
        print("\n# Attack #%d"%(f))
        print("# correct key  :",' '.join(["%02x"%(x) for x in k]))
        print("# key found    :",' '.join(["%02x"%(np.argmax(s["distri"])) for s in secret]))
