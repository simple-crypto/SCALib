from common import gen_traces,sbox
import numpy as np
from stella.preprocessing import SNR,SNROrder
from stella.utils import DataReader
from stella.attacks import MCP_DPA

# Setup the simulation settings
D=2
tag = "sim"
DIR_PROFILE = "./traces/profile/"
DIR_ATTACK = "./traces/attack/"
nfile_profile = 10
nfile_attack = 1
ntraces = 10000
std = 2

## Generate the profiling and attack sets
gen_traces(nfile_profile,
        ntraces,std,tag,
        DIR_PROFILE,random_key=True,D=D)
secret_key = gen_traces(nfile_attack,ntraces,
                    std,tag,DIR_ATTACK,random_key=False,D=D)[0]

print("Start Profiling")
# File for profiling
files_traces = [DIR_PROFILE+"/traces/"+tag+"_traces_%d.npy"%(i) for i in range(nfile_profile)]
files_labels = [DIR_PROFILE+"/labels/"+tag+"_labels_%d.npz"%(i) for i in range(nfile_profile)]

profile = ["x%d"%(i) for i in range(16)]

#Go over all the profiling traces and update the SNROrder 
for it,(traces,labels) in enumerate(zip(DataReader(files_traces,None),DataReader(files_labels,["labels"]))):
    labels = labels[0][0]
    if it == 0:
        ntraces,Ns = traces.shape
        snr = SNROrder(Np=16,Nc=256,D=D,Ns=Ns)
        data = np.zeros((16,ntraces))
    for i,p in enumerate(profile):
        data[i,:] = labels[p] 
    snr_val = snr.fit_u(traces,data)

# Get standardized moments at D-th order
SM,u,s = snr.get_sm(D)

print("Start Attack")
# Attack files
files_traces = [DIR_ATTACK+"/traces/"+tag+"_traces_%d.npy"%(i) for i in range(nfile_attack)]
files_labels = [DIR_ATTACK+"/labels/"+tag+"_labels_%d.npz"%(i) for i in range(nfile_attack)]

# For each attack file
for it,(traces,labels) in enumerate(zip(DataReader(files_traces,None),DataReader(files_labels,["labels"]))):
    labels = labels[0][0]
    if it == 0:
        ntraces,Ns = traces.shape
        mcp_dpas = [MCP_DPA(SM[i,:,:],
                            u[i,:,:],
                            s[i,:,:],256,D) for i in range(16)]
        
        # Memory for every guess at the output of the Sbox
        guesses = np.zeros((256,ntraces)).astype(np.uint16)
        # generate all the key guess
        kg = np.repeat(np.arange(256),ntraces).reshape(256,ntraces).astype(np.uint16)

    for i,mcp_dpa in enumerate(mcp_dpas):
        # get a plaintext byte value
        data = labels["p%d"%(i)]
        # derive the Sbox output
        guesses[:,:] = sbox[data^kg]
        # update correlation
        mcp_dpa.fit_u(traces,guesses)

# Display the obtained key
for i,k in enumerate(secret_key):
    guess = np.argmax(np.max(np.abs(mcp_dpas[i]._corr),axis=1))
    print("Key :",k,"guess: ",guess)


