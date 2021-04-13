import numpy as np
from scaffe import _scaffe_ext


class Ttest:
    def __init__(self, ns, d):
        self._ns = ns
        self._d = d

        self._ttest = _scaffe_ext.Ttest(ns, d)
    def fit_u(self, l, x):
        r"""Updates the Ttest estimation with samples of `l` for the classes `x`
        traces. This method may be called multiple times.

        Parameters
        ----------
        l : array_like, np.int16
            Array that contains the signal. The array must
            be of dimension `(n, ns)` and its type must be `np.int16`.
        x : array_like, np.uint16
            Labels for each trace. Must be of shape `(n,)` and must be
            `np.uint16`.
        """
        nl, nsl = l.shape
        nx = x.shape[0]
        print(x.shape)
        if not (nx == nl):
            raise ValueError(f"Expected x with shape ({nl},)")
        if not (nsl == self._ns):
            raise Exception(f"l is too long. Expected second dim of size {self._ns}.")
        
        self._ttest.update(l, x)

    def get_ttest(self):
        return self._ttest.get_ttest()
