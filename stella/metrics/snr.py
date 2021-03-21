import numpy as np
import stella.lib.rust_stella as rust
from tqdm import tqdm
class SNR:
    r"""Computes the Signal-to-Noise Ratio (SNR) between the traces
    and the intermediate values. The SNR can be updated with its `fit()`
    method. 

    Parameters
    ----------
    nc : int
        Number of possible classes (e.g., 256 for 8-bit target). `nc` must
        be smaller than `65536`.
    ns : int
        Number of samples in a single trace.
    np : int
        Number of independent variables for which SNR must be estimated.

    Examples
    --------
    >>> from stella.preprocessing import SNR
    >>> import numpy as np
    >>> traces = np.random.randint(0,256,(100,200),dtype=np.int16)
    >>> X = np.random.randint(0,256,(10,100),dtype=np.uint16)
    >>> snr = SNR(256,200,10)
    >>> snr.fit_u(traces,X)

    Notes
    -----
    [1] Stefan Mangard, "Hardware Countermeasures against DPA ? A
    Statistical Analysis of Their Effectiveness", CT-RSA 2004: 222-235
    """
    def __init__(self,nc,ns,np=1):

        if nc >= (2**16):
            raise Exception("SNR can be computed on max 16 bit, {} given".format(Nc))
      
        self.nc_ = nc
        self.ns_ = ns
        self.np_ = np
        self.snr = rust.SNR(nc,ns,np)

    def fit_u(self,x,y):
        r""" Updates the SNR estimation with samples of `x` for the classes `y` 
        traces.

        Parameters
        ----------
        x : array_like, int16
            Array that contains the signal. The array must
            be of dimension `(n,ns)` and its type must be `int16`.
        y : array_like, uint16
            Labels for each trace. Must be of shape `(np,n)` and
            must be `uint16`.
        """
        nx,nsx = x.shape
        npy,ny = y.shape
        if not (npy == self.np_ and ny==nx):
            raise Exception("Expected y with shape (%d,%d)"%(self.np_,nx))
        if not (nsx == self.ns_):
            raise Exception("x is too long. Expected second dim of size %d"%(self.ns_))
        self.snr.update(x,y)

    def get_snr(self):
        r"""Return the current SNR estimation with an array of shape `(np,ns)`. 
        """
        return self.snr.get_snr()

    def __del__(self):
        del self.snr


if __name__ == "__main__":
    n = int(1E4)
    ns = 5000
    np = 4
    import time
    traces = np.random.randint(0,256,(n,ns)).astype(np.int16)
    labels = np.random.randint(0,256,(np,n)).astype(np.uint16)

    snr = SNR(256,ns,np)
    start = time.time()
    for i in range(10):
        snr.fit_u(traces,labels)
    snr_val = snr.get_snr()
    print(start - time.time())
