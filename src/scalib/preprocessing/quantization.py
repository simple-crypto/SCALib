import numpy as np


class QUANTIZER:
    r"""Quantize a side channel traces given as as float into int16.
        The quantizer evaluates the minimum and maximum value of the traces at each point to estimate a shift and
        scale that minimize the loss due to rounding.

    .. math::
        \mathrm{Quantize}(X) = (x - \mathrm{Shift}) * \mathrm{Scale}

    Parameters
    ----------
    ns : int
        Number of samples in a single trace.
    shift : np.ndarray[np.float64]
        The value to shift every traces.
    scale : np.ndarray[np.float64]
        The value to scale every traces.

    Examples
    --------
    >>> from scalib.preprocessing import QUANTIZER
    >>> import numpy as np
    >>> # 500 traces of 200 points
    >>> noisy_traces : np.ndarray[np.float64] = np.random.randn(500,200)
    >>> quantizer = QUANTIZER(200, np.zeros(200),np.ones(200))
    >>> snr.fit_shift_scale(noisy_traces)
    >>> quantized_traces : np.ndarray[np.int16] = quantizer.quantize(noisy_traces)
    >>> # Can be reused directly 5000 new traces for instance
    >>> noisy_traces : np.ndarray[np.float64] = np.random.randn(5000,200)
    >>> quantized_traces : np.ndarray[np.int16] = quantizer.quantize(noisy_traces)
    """

    def __init__(
        self, ns: int, shift: np.ndarray[np.float64], scale: np.ndarray[np.float64]
    ):
        self._ns = ns
        self._shift = shift
        self._scale = scale

    def fit_shift_scale(self, noisy_traces: np.ndarray[np.float64]) -> None:
        r"""Updates the shift and scale estimation from sample of `noisy_traces`
        This method may be called multiple times.

        Parameters
        ----------
        noisy_traces : array_like, np.float64
            Array that contains the traces to estimate the shift and scale in the quantization. The array must
            be of dimension `(n, ns)
        """

        # Max/Min Centering and Multiplication by a constant prior to quantization to avoid information loss via rounding error
        max: np.ndarray[np.float64] = np.amax(noisy_traces, axis=0)
        min: np.ndarray[np.float64] = np.amin(noisy_traces, axis=0)

        # Derive shift and scale accordingly to center the traces
        shift: np.ndarray[np.float64] = (max + min) / 2
        width: np.ndarray[np.float64] = (max - min) / 2
        scale: np.ndarray[np.float64] = (
            2**14
        ) / width  # 2**14 instead of 2**15 as a safety margin.

        # Update the parameter
        self._shift = shift
        self._scale = scale

    def quantize(self, noisy_traces: np.ndarray[np.float64]) -> np.ndarray[np.int16]:
        r"""Quantize the noisy traces provide in `noisy_traces`

        Parameters
        ----------
        noisy_traces : array_like, np.float64
            Array that contains the traces to be quantized into int16. The array must
            be of dimension `(n, ns)
        """
        quantized_traces: np.ndarray[np.int16] = (
            (noisy_traces - self._shift) * self._scale
        ).astype(np.int16)
        return quantized_traces
