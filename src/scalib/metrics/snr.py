import numpy as np
import numpy.typing as npt

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
    >>> snr = SNR(nc=16)
    >>> snr.fit_u(traces,X)
    >>> snr_val = snr.get_snr()

    Notes
    -----
    .. [1] "Hardware Countermeasures against DPA ? A Statistical Analysis of
       Their Effectiveness", Stefan Mangard, CT-RSA 2004: 222-235

    """

    def __init__(self, nc: int, _ns=None, _np=None, use_64bit: bool = False):
        if nc not in range(2, 2**16 + 1):
            raise ValueError(
                "SNR can be computed on max 16 bit variable (and at least 2 classes),"
                f" {nc=} given."
            )
        self._nc = nc
        self._use_64bit = use_64bit

    def fit_u(self, l: npt.NDArray[np.int16], x: npt.NDArray[np.uint16]):
        r"""Updates the SNR estimation with samples of `l` for the classes `x`.
        This method may be called multiple times.

        Parameters
        ----------
        l : Array that contains the leakage traces. The array must be of
        dimension `(n, ns)`.
        x : Labels for each trace. Must be of shape `(n, nv)`.
        """
        scalib.utils.assert_traces(l)
        scalib.utils.assert_classes(x)
        if not hasattr(self, "_snr"):
            self._ns = l.shape[1]
            self._nv = x.shape[1]
            self._snr = _scalib_ext.SNR(self._nc, self._ns, self._nv, self._use_64bit)
        if l.shape[1] != self._ns:
            raise ValueError(
                f"Traces length {l.shape[1]} does not match"
                f"previously-fitted traces ({self._ns})."
            )
        elif x.shape[1] != self._nv:
            raise ValueError(
                f"Number of variables {x.shape[1]} does not match"
                f"previously-fitted classes ({self._nv})."
            )
        elif x.shape[0] != l.shape[0]:
            raise ValueError(
                f"Number of traces {l.shape[0]} does not match size of classes array {x.shape[0]}."
            )
        elif not l.flags.c_contiguous:
            raise Exception("l not a C-style array.")
        # _scalib_ext uses inverted axes for x.
        # we can copy when needed, as x should be small, so this should be cheap
        x = x.transpose().astype(np.uint16, order="C", casting="equiv", copy=False)
        with scalib.utils.interruptible():
            self._snr.update(l, x, get_config())

    def get_snr(self):
        r"""Return the current SNR estimation with an array of shape `(np,ns)`."""
        with scalib.utils.interruptible():
            return self._snr.get_snr(get_config())
