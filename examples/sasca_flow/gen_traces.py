import numpy as np
import random
from sasca_flow_settings import *

def gen_traces_attack_sim(nfile_attack,ntraces,DIR_TRACES,tag):
    """
    """
    for i in range(nfile_attack):
        p = np.random.randint(0,Nk,(ntraces,16),dtype=np.uint8)
        k = np.random.randint(0,Nk,16,dtype=np.uint8)
        transi = np.random.randint(0,Nk,(ntraces,15),dtype=np.uint8)

        # leakage is sbox output + plaintext
        x = sbox[p ^ k]
        for j in range(15): transi[:,j] = x[:,j] ^ x[:,j+1]

        x = np.concatenate((x,p,transi),axis=1)
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
        transi = np.random.randint(0,Nk,(ntraces,15),dtype=np.uint8)
        
        x = sbox[p ^ k]
        for j in range(15): transi[:,j] = x[:,j] ^ x[:,j+1]
        
        x = np.concatenate((x,p,transi),axis=1)
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
        y = sbox[x]
        labels = []
        for j in range(16): labels.append({"label":"x_%d"%(j),"val":x[:,j]})
        for j in range(16): labels.append({"label":"p_%d"%(j),"val":dic["p"][:,j]})
        for j in range(16): labels.append({"label":"y_%d"%(j),"val":y[:,j]})
        for j in range(15): labels.append({"label":"t_%d_%d"%(j,j+1),"val":y[:,j]^y[:,j+1]})
        
        np.savez(DIR_LABELS+tag+"_labels_%d.npz"%(i),
                        labels=labels,
                        allow_pickle=True)

