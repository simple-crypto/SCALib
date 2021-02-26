import numpy as np
import stella.lib.rust_stella as rust
from scipy.special import comb

class Ttest:
    """
        This Class computes a t-test on vectors (traces). 
        It operates on signals in int16 and updates its
        state at every call to fit_u

        see "Leakage assessment methodology"
        https://link.springer.com/content/pdf/10.1007/s13389-016-0120-y.pdf
    """
    def __init__(self,Ns,D=1):
        """
            Ns: number of samples in a single traces
            D: is the order to the t-test
        """
        self._Ns = Ns
        self._D  = D

        #number of samples in each class
        self._n = np.zeros(2,dtype=np.float64)
        # raw moment 1.
        self._M = np.zeros((2,Ns),dtype=np.float64)
        # central moment up to D*2
        self._CS = np.zeros((2,2*D,Ns),dtype=np.float64)

    def fit_u(self,traces,C,use_rust=True,nchunks=8):
        """
            Updates the Ttest status to take the fresh samples into account

            traces: (?,Ns) int16 or int8 array containing the array.
            C: (?)  uint8 array coutaining the traces id (set 0 or ones)

            use_rust: use low level rust function
            nchunks: number of threads used to compute the ttest. only works
            for use_rust mode.

            return:
            t: (D,Ns) then contains all the ttest up to the order D
        """
        if not (traces.dtype == np.int16):
            raise Exception("Trace type not supported {}".format(Trace.dtype))
        if not (C.dtype == np.uint8):
            raise Exception("C type not supported {}".format(Trace.dtype))
        if C.ndim != 1:
            raise Exception("Input C array does not match: waiting  ndim=1 vector")
        l,x = traces.shape()
        lc = len(C)
        if x != self._Ns:
            raise Exception("Trace len does not match: Expected {} given {}".format(self._Ns,x))
        if lc != l:
            raise Exception("Trace and C dim does not match")

        M = self._M
        n = self._n
        CS = self._CS
        D = self._D

        if use_rust:
            rust.update_ttest(traces,
                    C,
                    n,
                    CS,
                    M,
                    D,nchunks);
        else:
            for i in range(len(traces)):
                y = traces[i,:]
                x = C[i]
                n[x] += 1
                delta = y - M[x,:]
                for d in np.flip(range(2,(D*2)+1)):
                    if n[x]>1:
                        tmp = (((n[x]-1)*delta/n[x])**(d)) * (1 - (-1/(n[x]-1))**(d-1))
                        CS[x,(d-1),:] += tmp
                    for k in range(1,(d-2)+1):
                        cb = comb(d,k);
                        CS[x,(d-1),:] += cb*CS[x,(d-k)-1,:]*(-delta/n[x])**k
                M[x,:] += delta/n[x]
                CS[x,0,:] = M[x,:]

        CM0 = CS[0]/n[0]
        CM1 = CS[1]/n[1]
        t = np.zeros((self._D,len(traces[0,:])))
        for d in range(1,self._D+1):
            if d == 1:
                u0 = M[0,:];u1 = M[1,:]
                v0 = CM0[1,:];v1=CM1[1,:]
            elif d == 2:
                u0 = CM0[1,:];u1=CM1[1,:]
                v0 = CM0[3,:] - CM0[1,:]**2
                v1 = CM1[3,:] - CM1[1,:]**2
            else:
                u0 = CM0[d-1,:]/np.power(CM0[1,:],d/2);
                u1 = CM1[d-1,:]/np.power(CM1[1,:],d/2);
                v0 = (CM0[(d*2)-1,:] - CM0[(d)-1,:]**2)/(CM0[1,:]**d)
                v1 = (CM1[(d*2)-1,:] - CM1[(d)-1,:]**2)/(CM1[1,:]**d)

            t[d-1,:] = (u0-u1)/(np.sqrt((v0/n[0]) + (v1/n[1])))
        self._t = t
        self._CM0 = CM0
        self._CM1 = CM1
        return t

if __name__ == "__main__":
    Nt = 1000
    l = 5000
    D = 2
    np.random.seed(0);
    traces = np.random.normal(0,10,(Nt,l)).astype(np.int16)
    c = np.random.randint(0,2,Nt).astype(np.uint16)
    traces = (traces.T + c).T.astype(np.int16)
    ttest = Ttest(l,D=D)
    ttest.fit_u(traces,c,use_rust=False)

    ttest_r = Ttest(l,D=D)
    ttest_r.fit_u(traces,c,use_rust=True)
