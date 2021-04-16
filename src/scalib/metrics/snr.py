import numpy as np
from scalib import _scalib_ext


class SNR:
    r"""Computes the Signal-to-Noise Ratio (SNR) between the traces and the
    intermediate values. Informally, SNR allows to quantify the amount of information about a
    random variable :math:`X` contained in the mean of the leakage :math:`L_X`. High SNR
    means more information contained in the mean. The SNR metric is defined
    with the following equation [1]_:

    .. math::
        \mathrm{SNR} = \frac{\mathrm{Var}_{x\leftarrow X}(\mathrm{E}[L_x])}
                {\mathrm{E}_{x\leftarrow X}(\mathrm{Var}[L_x])}

    The SNR is estimated from leakage samples :math:`L_x` and the value `x` of
    the random variable.

    Parameters
    ----------
    nc : int
        Number of possible values for the random variable :math:`X` (e.g., 256 for 8-bit
        target). `nc` must be between :math:`1` and :math:`2^{16}` (included).
    ns : int
        Number of samples in a single trace.
    np : int
        Number of independent variables `X` for which SNR must be estimated.

    Examples
    --------
    >>> from scalib.metrics import SNR
    >>> import numpy as np
    >>> traces = np.random.randint(0,256,(100,200),dtype=np.int16)
    >>> X = np.random.randint(0,256,(100,10),dtype=np.uint16)
    >>> snr = SNR(256,200,10)
    >>> snr.fit_u(traces,X)
    >>> snr_val = snr.get_snr()

    Notes
    -----
    .. [1] "Hardware Countermeasures against DPA ? A Statistical Analysis of
       Their Effectiveness", Stefan Mangard, CT-RSA 2004: 222-235

    """

    def __init__(self, nc, ns, np=1):
        if nc not in range(1, 2 ** 16 + 1):
            raise ValueError(
                f"SNR can be computed on max 16 bit variable, nc={nc} given"
            )

        self._ns = ns
        self._np = np
        self._snr = _scalib_ext.SNR(nc, ns, np)

    def fit_u(self, l, x):
        r"""Updates the SNR estimation with samples of `l` for the classes `x`.
        This method may be called multiple times.

        Parameters
        ----------
        l : array_like, np.int16
            Array that contains the signal. The array must
            be of dimension `(n, ns)` and its type must be `np.int16`.
        x : array_like, np.uint16
            Labels for each trace. Must be of shape `(n, np)` and must be
            `np.uint16`.
        """
        nl, nsl = l.shape
        nx, npx = x.shape
        if not (npx == self._np and nx == nl):
            raise ValueError(f"Expected x with shape ({nl}, {self._np})")
        if not (nsl == self._ns):
            raise Exception(f"l is too long. Expected second dim of size {self._ns}.")
        # _scalib_ext uses inverted axes for x.
        self._snr.update(l, x.transpose())

    def get_snr(self):
        r"""Return the current SNR estimation with an array of shape `(np,ns)`."""
        return self._snr.get_snr()
