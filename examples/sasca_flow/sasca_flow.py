import numpy as np
import random
from sasca_flow_settings import * 
from stella.attacks.sasca.scripts.graph_parsing import * 
from stella.attacks.sasca.scripts.profiling_flags import *
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA


def gen_traces_attack_sim(nfile_attack,ntraces,DIR_TRACES,tag):
    """
        
    """
    for i in range(nfile_attack):
        p = np.random.randint(0,Nk,(ntraces,16),dtype=np.uint8)
        k = np.random.randint(0,Nk,16,dtype=np.uint8)

        # leakage is sbox output + plaintext
        x = sbox[p ^ k]
        x = np.concatenate((x,p),axis=1)
        hw = np.sum(np.unpackbits(np.expand_dims(x,2),axis=2),axis=2).astype(np.uint8)
        traces = hw + np.random.normal(0,noise,hw.shape)
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
        x = np.concatenate((x,p),axis=1)
        hw = np.sum(np.unpackbits(np.expand_dims(x,2),axis=2),axis=2).astype(np.uint8)
        traces = hw + np.random.normal(0,noise,hw.shape)
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
        for j in range(16): labels.append({"label":"p%d"%(j),"val":dic["p"][:,j]})
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
    ndim = 1
    write_poi(FILE_SNR,FILE_POI,
                profile,
                lambda snr: np.argsort(snr)[-ndim:] if len(np.where(snr>0.001)[0]) > ndim else np.where(snr>0.001)[0])

    print("# 3. Building templates")
    def func_train_pdf(t,l,label):
        m = QDA()
        m.fit(t,l)
        return m
    build_model(PREFIX_PROFILE_TRACES,
                    PREFIX_PROFILE_LABELS,
                    FILE_POI,
                    FILE_MODEL,
                    nfile_profile,
                    profile,
                    func=func_train_pdf,batch_size=10)

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
        for i,_ in enumerate(public):
            index = np.where(public_l == "p%d"%(i))[0][0]
            public[index]["input"][:,0] = pt[:,i]
      
        # compute proba on all the profiled variables
        for p in profile:
            m = model[np.where(model_l == p["label"])[0][0]]
            p["distri"][:,:] = m["model"].predict_proba(t[:,m["poi"]])
        
        graph.run_bp(5)
        
        print("\n# Attack #%d"%(f))
        print("# correct key  :",' '.join([" %02x"%(x) for x in k]))
        print("# key found    :",' '.join([" %02x"%(np.argmax(np.sum(np.log10(s["distri"]),axis=0))) for s in secret]))
        print("# key rank     :",' '.join(["%03d"%(256 - np.where(np.argsort(np.cumsum(np.log10(s["distri"]),axis=0)[-1,:])==x)[0][0]) for s,x in zip(secret,k)]))
