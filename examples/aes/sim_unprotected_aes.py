from utils import sbox,gen_traces
import numpy as np

if __name__ == "__main__":
    nc = 256
    std = 1
    ntraces_a = 10000
    ntraces_p = 1000
    npoi = 5
    
    print("1. Generate simulated traces")
    print(f"     ntraces for profiling: {ntraces_p}")
    print(f"     ntraces for attack:    {ntraces_a}")
    print(f"     noise std:             {std}")
    traces_p,labels_p = gen_traces(ntraces_p,std,random_key=1)
    traces_a,labels_a = gen_traces(ntraces_a,std,random_key=0)

    print("2. Profiling")
    models = {}
    print("    2.1 Compute snr for 16 Sbox outputs xi")
    y = np.zeros((ntraces,16),dtype=np.uint16)
    for i in range(16): y[:,i] = labels_p[f"x{i}"]

    snr = SNR(nc=256,ns=len(traces[0,:]),p=16)
    snr.fit_u(traces_a,y)
    snr_val = snr.get_snr()

    print("    2.2 Keep {npoi} POIs for each xi")
    for i in range(16): 
        models[f"x{i}"]["poi"] = np.argsort(snr_var[i])[:-npoi]
    print("    2.3 Build LDACLassifier for each xi")
    for i in range(16):
        model = models[f"x{i}"]
        lda = LDACLassifier(nc=256,ns=npoi,p=1)
        lda.fit(x=traces_p[:,model["poi"]],
                y=labels_p[f"x{i}"])
        model["lda"] = lda

    print("3. Attack")
    print("    3.1 Create the SASCA Graph")
    graph_desc =f"""
                Nc {nc}
                TABLE sbox   # The Sbox
                """

    for i in range(16):
        graph_desc +=f"""
                VAR SINGLE k{i} # The key
                VAR MULTI p{i}  # The plaintext
                VAR MULTI x{i}
                VAR MULTI y{i}
                PROPERTY x{i} = sbox[y{i}] # Sbox lookup
                PROPERTY x{i} = k{i} ^ p{i}   # Key addition
                """
    sasca = SASCAGraph(graph_desc,n=ntraces_a)
    
    print("    3.2 Add public information in the SASCAGraph")
    sasca.set_table("sbox",sbox)
    for i in range(16):
        sasca.set_public(f"p{i}",labels_a[f"p{i}"])

    print("    3.3 Add xi distributions in the graph")
    for i in range(16):
        model = models[f"x{i}"]
        sasca.set_init_distribution(f"x{i}",
                model["lda"].predict_proba(traces_a[:,model["poi"]]))

    print("    3.4 Run belief propagation")
    sasca.run_bp(it=3)
    
    print("4. Attack evaluation")
    secret_key = []
    key_distribution = []
    for i in range(16):
        secret_key.append(labels_a[f"k{i}"])
        key_distribution = sasca.get_distribution(f"k{i}")
    rmin,r,rmax = rank_accuracy(key_distribution,secret_key,1)
    lrmin,lr,lrmax = (np.log2(rmin),np.log2(r),np.log2(rmax))
    print("    4.1 Estimate full log2 key rank:")
    print("         {lrmin} < {lr} < {lrmax}")
