import numpy as np

from scalib import _scalib_ext
from scalib.config import get_config
import scalib.utils


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
        target). `nc` must be between :math:`2` and :math:`2^{16}` (included).
    ns : int
        Number of samples in a single trace.
    np : int
        Number of independent variables `X` for which SNR must be estimated.
    use_64bit : bool (default False)
        Use 64 bits for intermediate sums instead of 32 bits.
        When using 64-bit sums, SNR can accumulate up to :math:`2^{32}` traces, while when
        32-bit sums are used, the bound is :math:`n_i < 2^{32}/b`, where b is the
        maximum absolute value of a sample rounded to the next power of 2, and
        :math:`n_i` is the maximum number of times a variable can take a given value.
        Concretely, the total number of traces `n` should be at most
        :math:`(nc \cdot 2^{32}/b) - k `, where :math:`k = O(\sqrt{n})`, typ.
        :math:`k>=3*\sqrt{n}`  (see https://mathoverflow.net/a/273060).



    Examples
    --------
    >>> from scalib.metrics import SNR
    >>> import numpy as np
    >>> # 500 traces of 200 points, 8-bit samples
    >>> traces = np.random.randint(0,256,(500,200),dtype=np.int16)
    >>> # 10 variables on 4 bit (16 classes = 2^4)
    >>> X = np.random.randint(0,16,(500,10),dtype=np.uint16)
    >>> snr = SNR(16,200,10)
    >>> snr.fit_u(traces,X)
    >>> snr_val = snr.get_snr()

    Notes
    -----
    .. [1] "Hardware Countermeasures against DPA ? A Statistical Analysis of
       Their Effectiveness", Stefan Mangard, CT-RSA 2004: 222-235

    """

    def __init__(self, nc, ns, np=1, use_64bit=False):
        if nc not in range(2, 2**16 + 1):
            raise ValueError(
                f"SNR can be computed on max 16 bit variable (and at least 2 classes), {nc=} given."
            )

        self._ns = ns
        self._np = np
        self._snr = _scalib_ext.SNR(nc, ns, np, use_64bit)

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
        if not isinstance(l, np.ndarray):
            raise ValueError("l a numpy array")
        if not isinstance(x, np.ndarray):
            raise ValueError("x a numpy array")
        nl, nsl = l.shape
        nx, npx = x.shape
        if l.dtype != np.int16:
            raise ValueError("l must by array of np.int16")
        if not (npx == self._np and nx == nl):
            raise ValueError(f"Expected x with shape ({nl}, {self._np})")
        if not (nsl == self._ns):
            raise Exception(f"l is too long. Expected second dim of size {self._ns}.")
        if not l.flags.c_contiguous:
            raise Exception(f"l not a C-style array.")
        # _scalib_ext uses inverted axes for x.
        # we can copy when needed, as x should be small, so this should be cheap
        x = x.transpose().astype(np.uint16, order="C", casting="equiv", copy=False)
        with scalib.utils.interruptible():
            self._snr.update(l, x, get_config())

    def get_snr(self):
        r"""Return the current SNR estimation with an array of shape `(np,ns)`."""
        with scalib.utils.interruptible():
            return self._snr.get_snr(get_config())
