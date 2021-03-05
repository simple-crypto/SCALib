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
        p = np.random.randint(0,256,(ntraces,16),dtype=np.uint8)

        if random_key:
            k = np.random.randint(0,256,(ntraces,16),dtype=np.uint8)

        # leakage is sbox output
        y = p ^ k
        x = sbox[y]
        
        y_s = np.random.randint(0,256,(ntraces,D,16),dtype=np.uint8)
        x_s = np.random.randint(0,256,(ntraces,D,16),dtype=np.uint8)
       
        x_s[:,-1,:] = x ^ np.bitwise_xor.reduce(x_s[:,:-1,:],axis=1)
        y_s[:,-1,:] = y ^ np.bitwise_xor.reduce(x_s[:,:-1,:],axis=1)
    
        data = np.concatenate((x_s,y_s),axis=2)
        # HW + noise
        hw = np.sum(np.unpackbits(np.expand_dims(data,3),axis=3),axis=3).astype(np.uint8)
        hw = np.sum(hw,axis=1)
        traces = hw + np.random.normal(0,std,hw.shape)
        traces *= 256
        traces = traces.astype(np.int16)

        # generate all the labels
        labels = {}
        for j in range(16): labels["k%d"%(j)] = k[:,j]
        for j in range(16): labels["x%d"%(j)] = x[:,j]
        for j in range(16): labels["p%d"%(j)] = p[:,j]

        # save the data
        np.save(DIR_TRACES+"/traces/"+tag+"_traces_%d.npy"%(i),traces)
        np.savez(DIR_TRACES+"/traces/"+tag+"_meta_%d.npz"%(i),
                        p=p,
                        k=k,
                        sbox=sbox,
                        allow_pickle=True)
        
        np.savez(DIR_TRACES+"/labels/"+tag+"_labels_%d.npz"%(i),
                        labels=[labels],
                        allow_pickle=True)

    return k

def gen_traces_serial(nfile,ntraces,std,tag,
               DIR_TRACES,random_key,D=1):
    
    os.system("mkdir -p "+DIR_TRACES+"/traces/")
    os.system("mkdir -p "+DIR_TRACES+"/labels/")
    
    if not random_key:
        k = np.random.randint(0,256,(1,16),dtype=np.uint8)

    for i in range(nfile):
        p = np.random.randint(0,256,(ntraces,16),dtype=np.uint8)

        if random_key:
            k = np.random.randint(0,256,(ntraces,16),dtype=np.uint8)

        # leakage is sbox output
        y = p ^ k
        x = sbox[y]
        
        y_s = np.random.randint(0,256,(ntraces,D,16),dtype=np.uint8)
        x_s = np.random.randint(0,256,(ntraces,D,16),dtype=np.uint8)
        x_s[:,-1,:] = x ^ np.bitwise_xor.reduce(x_s[:,:-1,:],axis=1)
        y_s[:,-1,:] = y ^ np.bitwise_xor.reduce(x_s[:,:-1,:],axis=1)

        
        data = np.concatenate((x_s,y_s),axis=2)
        # HW + noise
        hw = np.sum(np.unpackbits(np.expand_dims(data,3),axis=3),axis=3).astype(np.uint8)
        hw = hw.reshape((ntraces,2*D*16))
        traces = hw + np.random.normal(0,std,hw.shape)
        traces *= 256
        traces = traces.astype(np.int16)

        # generate all the labels
        labels = {}
        for j in range(16): labels["k%d"%(j)] = k[:,j]
        for j in range(16): labels["p%d"%(j)] = p[:,j]
        for j in range(16):
            for d in range(D):
                labels["x%d_%d"%(j,d)] = x_s[:,d,j]
                labels["y%d_%d"%(j,d)] = y_s[:,d,j]

        # save the data
        np.save(DIR_TRACES+"/traces/"+tag+"_traces_%d.npy"%(i),traces)
        np.savez(DIR_TRACES+"/traces/"+tag+"_meta_%d.npz"%(i),
                        p=p,
                        k=k,
                        y_s=y_s,
                        x_s=x_s,
                        sbox=sbox,
                        allow_pickle=True)
        
        np.savez(DIR_TRACES+"/labels/"+tag+"_labels_%d.npz"%(i),
                        labels=[labels],
                        allow_pickle=True)

    return k
