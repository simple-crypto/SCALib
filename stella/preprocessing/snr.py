"""
This module allows to compute SNR for multiple independent variables in
parallel.
"""

import numpy as np
import stella.lib.rust_stella as rust
from tqdm import tqdm
class SNR:
    def __init__(self,nc,ns,np=1):
        r"""Computes the Signal-to-Noise ratio between the traces
        and the intermediate values. It is meant to work on traces being 
        int16.

        Parameters
        ----------
        nc : int
            Number of possible classes (e.g., 256 for 8-bit target). We force
            that the number of classes is smaller than 2**16.
        ns : int
            Trace length to process.
        np : int
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

        if nc >= (2**16):
            raise Exception("SNR can be computed on max 16 bit, {} given".format(Nc))
      
        self.snr = rust.SNR(nc,ns,np)

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
        self.snr.update(traces,X)

    def get_snr(self):
        return self.snr.get_snr()

    def __del__(self):
        del self.snr


if __name__ == "__main__":
    Nt = int(1E4)
    Ns = 5000
    Np = 4
    import time
    traces = np.random.randint(0,256,(Nt,Ns)).astype(np.int16)
    labels = np.random.randint(0,256,(Np,Nt)).astype(np.uint16)

    snr = SNR(256,Ns,Np)
    start = time.time()
    for i in range(10):
        snr.fit_u(traces,labels)
    snr_val = snr.get_snr()
    print(start - time.time())
