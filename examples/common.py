import os 
import numpy as np
sbox = np.random.permutation(256)
def gen_traces(nfile,ntraces,std,tag,
               DIR_TRACES,random_key,D=1):
    
    os.system("mkdir -p "+DIR_TRACES+"/traces/")
    os.system("mkdir -p "+DIR_TRACES+"/labels/")
    
    if not random_key:
        k = np.random.randint(0,256,(1,16),dtype=np.uint8)

    for i in range(nfile):
        p_s = np.random.randint(0,256,(ntraces,D,16),dtype=np.uint8)
        x_s = np.random.randint(0,256,(ntraces,D,16),dtype=np.uint8)

        p = np.bitwise_xor.reduce(p_s,axis=1)
        if random_key:
            k = np.random.randint(0,256,(ntraces,16),dtype=np.uint8)

        # leakage is sbox output
        x = sbox[p ^ k]
        x_s[:,-1,:] = x ^ np.bitwise_xor.reduce(x_s[:,:-1,:],axis=1)

        # HW + noise
        hw = np.sum(np.unpackbits(np.expand_dims(x_s,3),axis=3),axis=3).astype(np.uint8)
        hw = np.sum(hw,axis=1)
        traces = hw + np.random.normal(0,std,hw.shape)
        traces *= 256
        traces = traces.astype(np.int16)

        # generate all the labels
        labels = [{"label":"k%d"%(i),"value":k[:,i]} for i in range(16)]
        labels += [{"label":"x%d"%(i),"value":x[:,i]} for i in range(16)]
        labels += [{"label":"p%d"%(i),"value":p[:,i]} for i in range(16)]

        # save the data
        np.save(DIR_TRACES+"/traces/"+tag+"_traces_%d.npy"%(i),traces)
        np.savez(DIR_TRACES+"/traces/"+tag+"_meta_%d.npz"%(i),
                        p=p,
                        k=k,
                        sbox=sbox,
                        allow_pickle=True)
        
        np.savez(DIR_TRACES+"/labels/"+tag+"_labels_%d.npz"%(i),
                        labels=labels,
                        allow_pickle=True)

    return k
