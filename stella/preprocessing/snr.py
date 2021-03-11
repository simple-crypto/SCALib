"""
This module allows to compute SNR for multiple independent variables in
parallel.
"""

import numpy as np
import stella.lib.rust_stella as rust
from tqdm import tqdm
class SNR:
    def __init__(self,Nc,Ns,Np=1):
        r"""Computes the Signal-to-Noise ratio between the traces
        and the intermediate values. It is meant to work on traces being 
        int16.

        Parameters
        ----------
        Nc : int
            Number of possible classes (e.g., 256 for 8-bit target). We force
            that the number of classes is smaller than 2**16.
        Ns : int
            Trace length to process.
        Np : int
            Number of independent variables to process.

        Examples
        --------
        >>> from stella.preprocessing import SNR
        >>> import numpy as np
        >>> traces = np.random.randint(0,1000,(100,200),dtype=np.int16)
        >>> X = np.random.randint(0,256,(10,100),dtype=np.uint8)
        >>> snr = SNR(256,200,10)
        >>> snr.fit_u(traces,X)
        """

        if Nc >= (2**16):
            raise Exception("SNR can be computed on max 16 bit, {} given".format(Nc))

        self._Nc = Nc
        self._Ns = Ns
        self._Np = Np

        # Number of observed traces with given intermediate variable
        self._ns = np.zeros((Np,Nc),dtype=np.uint64)
        # Sum for each class
        self._sum = np.zeros((Np,Nc,Ns),dtype=np.int64)
        # Sum of squared traces for each class
        self._sum2 = np.zeros((Np,Nc,Ns),dtype=np.int64)
        # SNR on each class
        self._SNR = np.zeros((Np,Ns),dtype=np.float64)

    def fit_u(self,traces,X):
        r""" Updates the SNR state to take into account fresh
        traces. This is typically called serially for multiple pairs (traces,X)

        Parameters
        ----------
        traces : array_like
            Array that contains the traces with data type uint16. The array must
            be of dimension (ntraces,Ns). must be int16
        X : array_like
            Labels for each traces. Must be of shape (Np,ntraces). must be uint16
        """
        rust.update_snr_only(traces,X,self._sum,self._sum2,self._ns);

    def get_snr(self):
        rust.finalyze_snr_only(self._sum,self._sum2,self._ns,self._SNR);
        return self._SNR

    def __del__(self):
        del self._ns,self._sum,self._sum2
        del self._SNR


if __name__ == "__main__":
    Nt = int(1E5)
    Ns = 50000
    Nc = 4
    import time
    traces = np.random.randint(0,256,(Nt,Ns)).astype(np.int16)
    labels = np.random.randint(0,256,(Nc,Nt)).astype(np.uint16)

    snr = SNR(256,Ns,Nc)
    start = time.time()
    for i in range(10):
        snr.fit_u(traces,labels)
    snr_val = snr._SNR
    print(start - time.time())
