import numpy as np
import stella.lib.rust_stella as rust
from tqdm import tqdm
class SNR:
    def __init__(self,Nc,Ns,Np=1):
        """
            This function computes the Signal-to-Noise ratio between the traces
            and the intermediate values. It is ment to work on traces being 
            int16.

            Nc: Possible values for the intermediate values X
            Ns: Number of samples in a single traces
            Np: Number of intermediates variable to comptue the SNR on. Default
            to 1
        """
        if Nc >= (2**16):
            raise Exception("SNR can be computed on max 16 bit, {} given".format(Nc))
        self._Nc = Nc
        self._Ns = Ns
        self._Np = Np
        # Number of observed traces with given intermediate variable
        self._ns = np.zeros((Np,Nc),dtype=np.uint32)
        # Sum for each class
        self._sum = np.zeros((Np,Nc,Ns),dtype=np.int64)
        # Sum of squared traces for each class
        self._sum2 = np.zeros((Np,Nc,Ns),dtype=np.int64)
        # Mean of each class
        self._means = np.zeros((Np,Nc,Ns),dtype=np.float32)
        # Variance in each class
        self._vars= np.zeros((Np,Nc,Ns),dtype=np.float32)
        # SNR on each class
        self._SNR = np.zeros((Np,Ns),dtype=np.float32)

        self._means[:,:] = np.nan

        self._i = 0
    def __del__(self):
        """
            Delet all the numpy array in the object
        """
        del self._ns,self._sum,self._sum2,self._means
        del self._SNR,self._vars

    def fit_u(self,traces,X,use_rust=True,nchunks=1):
        """
            Updates the SNR status to take the fresh samples into account

            traces: (?,Ns) int16 or int8 array containing the array.
            X: (Np,?) uint16 array coutaining
            use_rust: use low level rust
            nchunks: in how many chunks to // the snr
        """
        if not (traces.dtype == np.int16):
            raise Exception("Trace type not supported {}".format(Trace.dtype))
        if self._Np == 1 and X.ndim == 1:
            X = X.reshape((1,len(X)))
        elif len(X) != self._Np:
            raise Exception("Input X array does not match: Expected {} given {}".format((self._Np,len(traces)),X.shape))
        X = (X%self._Nc).astype(np.uint16)

        if use_rust:
            rust.update_snr(traces,X,self._sum,self._sum2,self._ns,self._means,self._vars,nchunks)
        else:
            n = len(traces[:,0])
            for v in range(self._Np):
                for i in tqdm(range(n),desc="SNR"):
                    self._ns[v,X[v,i]] += 1
                    t = traces[i,:]
                    self._sum[v,X[v,i],:] += t
                    self._sum2[v,X[v,i],:] += t*t.astype(np.int32)

                for c in range(self._Nc):
                    self._means[v,c,:] = (self._sum[v,c,:].T / self._ns[v,c]).T
                    self._vars[v,c,:] = (self._sum2[v,c,:].T/self._ns[v,c]).T - (self._means[v,c,:]**2)

                self._SNR[v,:] = np.var(self._means[v,:],axis=0)/np.mean(self._vars[v,:],axis=0)
        return self._SNR
