import numpy as np
import stella.lib.rust_stella as rust
from tqdm import tqdm
class MCP_DPA():
    def __init__(self,SM,u,s,Ng,D):
        """
        Performs an MCP DPA on traces: https://eprint.iacr.org/2014/409.pdf
        This is performed on standardized moments to avoid computational errors

        SM: (Nk,len) standardized moment matrix, will be used as the model 
        u: (Nk,len) mean of the traces (to center the traces)
        s: (Nk,len) std of the traces (to center the traces)

        Ng: Number of possible guess
        D: order of the attack
        """

        l = len(SM[0,:])
        self._SM = SM
        self._u = u
        self._s = s
        self._Ng = Ng
        self._D = D

        self._x = np.zeros((Ng,l))
        self._x2 = np.zeros((Ng,l))
        self._xy = np.zeros((Ng,l))
        self._y2 = np.zeros((Ng,l))
        self._y = np.zeros((Ng,l))
        
        self._corr = np.zeros((Ng,l))
        self._n = 0

        if D == 1:
            self._u[:] = 0
            self._s[:] = 1
        elif D == 2:
            self._s[:] = 1
    def fit(self,I,G):
        """
            updates the MCP_DPA with traces I and the guesses G
           
           I: (Nt,len) Nt traces  
           G: (Ng,Nt) the guesses corresponding to each traces
        """

        self._n += len(I[:,0])
        n = self._n
        
        rust.update_mcp_dpa(I,G,
                self._x,
                self._x2,
                self._xy,
                self._y,
                self._y2,
                
                self._SM,
                self._u,
                self._s,
                self._D,
                1);
        for g in range(self._Ng):
            """
            v = G[g]
            
            x = ((I-self._u[v])/self._s[v])**self._D
            y = self._SM[v]

            self._xy[g] += np.sum(x*y,axis=0)
            self._y[g] += np.sum(y,axis=0)
            self._y2[g] += np.sum(y**2,axis=0)
            self._x[g] += np.sum(x,axis=0)
            self._x2[g] += np.sum(x**2,axis=0)
            """
            num = n * self._xy[g] - (self._x[g] * self._y[g])
            den = 1
            den *= np.sqrt((n * self._x2[g] - (self._x[g]**2))) 
            den *= np.sqrt((n * self._y2[g] - (self._y[g]**2)))
            
            self._corr[g] = num/den
