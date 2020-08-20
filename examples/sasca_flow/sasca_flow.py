import numpy as np
import random
from sasca_flow_settings import *
from gen_traces import * 
from stella.attacks.sasca.scripts.graph_parsing import * 
from stella.attacks.sasca.scripts.profiling_flags import *
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA
from stella.estimator.classifiers import * 


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
        #m = QDA()
        #m.fit(t,l)
        m = LDAClassifier(t,l,dim_projection=1)
        return m
    build_model(PREFIX_PROFILE_TRACES,
                    PREFIX_PROFILE_LABELS,
                    FILE_POI,
                    FILE_MODEL,
                    nfile_profile,
                    profile,
                    func=func_train_pdf,batch_size=-1)

    print("\n# Attack Part")
    LOOP_IT = ntraces_attack; repeat = 1;
    public,profile,secret = extract_flags("example_graph.txt")

    for v in public: v["input"] = np.zeros((LOOP_IT,repeat),dtype=np.uint32) if v["loop"] else np.zeros((1,repeat),dtype=np.uint32)
    print("# 1. Build Graph")
    lookup = sbox.astype(np.uint32).reshape((1,Nk))
    graph = build_graph_from_file("example_graph.txt",Nk=Nk,it=LOOP_IT,public=public,lookup=lookup)

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
        
        # compute proba on all the profiled variables
        for p in profile:
            m = model[np.where(model_l == p["label"])[0][0]]
            p["distri"][:,:] = m["model"].predict_proba(t[:ntraces_attack,m["poi"]])


        # set the public inputs
        for p in public:
            code = p["label"].split('_')[0]
            i = int(p["label"].split('_')[1])
            if code == "p":
                p["input"][:,0] = pt[:ntraces_attack,i]
            else:
                raise Exception("code not found")
            
        graph.run_bp(it=5)

        print("\n# Attack #%d"%(f))
        print("# correct key   :",' '.join([" %02x"%(x) for x in k[:len(secret)]]))
        print("# key found     :",' '.join([" %02x"%(np.argmax(np.sum(np.log10(s["distri"]),axis=0))) for s in secret]))
        print("# key byte rank :",' '.join(["%03d"%(256 - np.where(np.argsort(np.cumsum(np.log10(s["distri"]),axis=0)[-1,:])==x)[0][0]) for s,x in zip(secret,k)]))
