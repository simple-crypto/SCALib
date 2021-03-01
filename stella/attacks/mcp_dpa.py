import numpy as np
import stella.lib.rust_stella as rust
from tqdm import tqdm

class MCP_DPA():
    def __init__(self,SM,mean,std,Nc,D):
        """Performs an MCP-DPA on traces to recover a secret. This is performed
        on standardized moments to avoid computational errors.

        Parameters
        ----------
        SM : array_like
            Standardized moment of the secret. Must be an array of shape (Nc,Ns)
            where Ns is the size of the traces and Nc the size of the secret
            (e.g., 256 for 8-bit secret).
        mean : array_like
            Mean of each class. Must be of shape (Nc,Ns) where each row is the
            mean trace for a single value of the secret. 
        std : array_like 
            Standard deviation of each class. Must be of shape (Nc,Ns) where 
            each row is the standard deviation trace for a single value of the
            secret.           
        Nc : int 
            Number of possible value for the secret to recover. 
        D : int
            Statistical moment to attack.
        """

        l = len(SM[0,:])
        self._SM = SM
        self._mean = mean
        self._std = std
        self._Nc = Nc
        self._D = D

        self._x = np.zeros((Nc,l))
        self._x2 = np.zeros((Nc,l))
        self._xy = np.zeros((Nc,l))
        self._y2 = np.zeros((Nc,l))
        self._y = np.zeros((Nc,l))
        
        self._corr = np.zeros((Nc,l))
        self._n = 0

        if D == 1:
            self._mean[:] = 0
            self._std[:] = 1
        elif D == 2:
            self._std[:] = 1

    def fit_u(self,traces,guesses):
        r"""Updates the correlations estimations with new traces for the
        MCP-DPA.
        
        Parameters
        ----------
        traces : array_like
            New traces to update the correlations with. Must be of type int16
            with shape (ntraces,Ns).
        guesses: array_like
            Guesses for each classes. The array must be of shape (Nc,ntraces). 
            Each row corresponds to the guesses (e.g., Sbox output) for a fixed
            secret (i.e., key byte) value.

        Returns
        -------
        corr : array_like
            Correlation for each secret and is of shape (Nc,Ns) where each row
            corresponds to one secret.

        Examples
        --------
        >>> snr = SNROrder(Np=1,Nc=256,D=D,Ns=Ns)
        # ... update snr ...
        >>> SM,mean,std = snr.get_SM(D)
        >>> mcp_dpa = MCP_DPA(SM[0,:,:],mean[0,:,:],std[0,:,:],Nc=256,D=D)
        >>> corr = mcp_dpa.fit_u(traces,guesses)
        >>> guess = np.argmax(np.max(np.abs(corr),axis=1))
        """

        self._n += len(traces[:,0])
        n = self._n
        
        rust.update_mcp_dpa(traces,guesses,
                self._x,
                self._x2,
                self._xy,
                self._y,
                self._y2,
                
                self._SM,
                self._mean,
                self._std,
                self._D,
                1);

        for g in range(self._Nc):
            num = n * self._xy[g] - (self._x[g] * self._y[g])
            den = 1
            den *= np.sqrt((n * self._x2[g] - (self._x[g]**2))) 
            den *= np.sqrt((n * self._y2[g] - (self._y[g]**2)))
            
            self._corr[g] = num/den
        return self._corr
