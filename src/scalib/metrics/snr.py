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
    >>> x = np.random.randint(0,16,(500,10),dtype=np.uint16)
    >>> snr = SNR(nc=16)
    >>> snr.fit_u(traces,x)
    >>> snr_val = snr.get_snr()

    Notes
    -----
    .. [1] "Hardware Countermeasures against DPA ? A Statistical Analysis of
       Their Effectiveness", Stefan Mangard, CT-RSA 2004: 222-235

    """

    def __init__(self, nc: int, use_64bit: bool = False):
        if nc not in range(2, 2**16 + 1):
            raise ValueError(
                "SNR can be computed on max 16 bit variable (and at least 2 classes),"
                f" {nc=} given."
            )
        self._nc = nc
        self._ns = None
        self._nv = None
        self._use_64bit = use_64bit
        self._init = False

    def fit_u(self, traces: npt.NDArray[np.int16], x: npt.NDArray[np.uint16]):
        r"""Updates the SNR estimation with samples of `traces` for the classes `x`.
        This method may be called multiple times.

        Parameters
        ----------
        traces :
            Array that contains the leakage traces. The array must be of
            dimension `(n, ns)`.
        x :
            Labels for each trace. Must be of shape `(n, nv)`.
        """
        traces = scalib.utils.clean_traces(traces, self._ns)
        x = scalib.utils.clean_labels(x, self._nv)
        if not self._init:
            self._init = True
            self._ns = traces.shape[1]
            self._nv = x.shape[1]
            self._snr = _scalib_ext.SNR(self._nc, self._ns, self._nv, self._use_64bit)
        if x.shape[0] != traces.shape[0]:
            raise ValueError(
                f"Number of traces {traces.shape[0]} does not match size of classes array {x.shape[0]}."
            )
        # _scalib_ext uses inverted axes for x.
        # we can copy when needed, as x should be small, so this should be cheap
        x = np.ascontiguousarray(x.transpose())
        with scalib.utils.interruptible():
            self._snr.update(traces, x, get_config())

    def get_snr(self) -> npt.NDArray[np.float64]:
        r"""Return the current SNR estimation with an array of shape `(np,ns)`."""
        if not self._init:
            raise ValueError("Need to call .fit_u at least once.")
        with scalib.utils.interruptible():
            return self._snr.get_snr(get_config())
