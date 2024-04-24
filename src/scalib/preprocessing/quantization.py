import numpy as np
from enum import IntEnum


class Quantizer:
    r"""Quantize a side channel traces given as an array of float into an array of int16.
        The quantizer estimates a shift and scale that minimize the loss due to the rounding operation.

    .. math::
        \mathrm{Quantize}( x) = (x - \mathrm{Shift}) \cdot \mathrm{Scale}

    The shift and scale are computed using `n` samples as

    .. math::
        \mathrm{Shift} = \frac{1}{2} (\max_{i=1}^n x_i + \min_{i=1}^n x_i) \qquad and \qquad  \mathrm{Scale} = \frac{2^{14}}{\max_{i=1}^n x_i - \min_{i=1}^n x_i}.


    Parameters
    ----------
    shift : np.ndarray[np.floating]
        The value to shift every traces.
    scale : np.ndarray[np.floating]
        The value to scale every traces.

    Examples
    --------
    >>> from scalib.preprocessing import Quantizer
    >>> import numpy as np
    >>> # 500 traces of 200 points
    >>> traces : np.ndarray[np.floating] = np.random.randn(500,200)
    >>> quantizer = Quantizer.fit(traces)
    >>> quantized_traces : np.ndarray[np.int16] = quantizer.quantize(traces)
    >>> # Can be reused directly on 5000 new traces for instance
    >>> traces : np.ndarray[np.floating] = np.random.randn(5000,200)
    >>> quantized_traces : np.ndarray[np.int16] = quantizer.quantize(traces)
    """

    def __init__(self, shift: np.ndarray[np.floating], scale: np.ndarray[np.floating]):
        self._shift = shift
        self._scale = scale

    @classmethod
    def fit(cls, traces: np.ndarray[np.floating], gaussian: bool = True):
        r"""Compute the shift and scale estimation from sample of `traces`
        This class method returns an instance of Quantizer with the corresponding shift and scale.

        Parameters
        ----------
        traces : array_like, np.floating
            Array that contains the traces to estimate the shift and scale in the quantization. The array must
            be of dimension `(n, ns)`
        gaussian : boolean
            A boolean parameter set to True by default. In this case, the min and max are estimated under the Gaussianity assumption.
        """

        # Max/Min Centering and Multiplication by a constant prior to quantization to avoid information loss via rounding error
        max: np.ndarray[np.floating] = np.amax(traces, axis=0)
        min: np.ndarray[np.floating] = np.amin(traces, axis=0)

        if gaussian:
            # Gaussian Methods
            mean: np.ndarray[np.floating] = np.amax(traces, axis=0)
            std = np.std(traces, axis=0, ddof=1)

            # Conservative confidence interval.
            g_min = mean - 7 * std
            g_max = mean + 7 * std

            max = np.maximum(max, g_max)
            min = np.minimum(min, g_min)

        # Derive shift and scale accordingly to center the traces
        shift: np.ndarray[np.floating] = (max + min) / 2
        width: np.ndarray[np.floating] = (max - min) / 2
        scale: np.ndarray[np.floating] = (
            2**14
        ) / width  # 2**14 instead of 2**15 as a safety margin.

        # Create Quantizer
        quantizer = cls(shift, scale)

        return quantizer

    def quantize(
        self, traces: np.ndarray[np.floating], clip: bool = False
    ) -> np.ndarray[np.int16]:
        r"""Quantize the traces provide in `traces`

        Parameters
        ----------
        traces : array_like, np.floating
            Array that contains the traces to be quantized into int16. The array must
            be of dimension `(n, ns)`
        clip : bool
            Boolean to bypass the overflow check prior to quantization and clip the overflowing values to the boundaries.
            By default it is set to False.
        """
        adjusted_traces: np.ndarray[np.floating] = (traces - self._shift) * self._scale
        if clip:
            adjusted_traces = np.clip(adjusted_traces, -(2**15), 2**15 - 1)
        else:
            overflow: bool = (adjusted_traces > 2**15 - 1).any() or (
                adjusted_traces < -(2**15)
            ).any()
            if overflow:
                raise ValueError(
                    "Overflow detected in the quantization. Update shift and scale more precisely to avoid the error. "
                )

        quantized_traces: np.ndarray[np.int16] = adjusted_traces.astype(np.int16)
        return quantized_traces
