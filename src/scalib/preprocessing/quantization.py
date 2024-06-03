import numpy as np
import numpy.typing as npt
from enum import Enum, auto


class QFitMethod(Enum):
    r"""An enum class used to specify how the maximum and minimum of the traces
    is estimated based on a set of fitting traces.

    With method = QuantFitMethod.BOUNDS they are estimated as the minimum and
    maximum of the fitting trace respectively.
    With method = QuantFitMethod.MOMENT they are estimated as the average of
    the fitting traces minus/plus seven standard deviations
    """

    BOUNDS = auto()
    MOMENT = auto()


class QuantFitMethod:
    """Method for esimating the scale and shift parameters of Quantizer."""

    @classmethod
    def bounds(cls, margin=2.0):
        """Take the min and max of the training traces, fit such that the [min,
        max] range is mapped to a zero-centered interval covering a ``1/margin``
        fraction of the quantized domain: if the quantized domain is
        ``[-Q,Q]``, ``min` is mapped to ``-Q/margin`` and ``max`` is mapped to
        ``Q/margin``.
        """
        return cls(QFitMethod.BOUNDS, margin=margin)

    @classmethod
    def moment(cls, nstd=7.0):
        """Take the mean and standard deviation of the training traces, fit
        such that ``mean-nstd*std`` is mapped to `-Q` and ``mean+nstd*std`` is
        mapped to `Q`, where `[-Q, Q]` is the quantized domain.
        """
        return cls(QFitMethod.MOMENT, nstd=nstd)

    def __init__(self, method: QFitMethod, **kwargs):
        self.method = method
        self.opts = kwargs


class Quantizer:
    r"""Quantize a side channel traces given as an array of float into an array
    of int16.

    The quantizer estimates a shift and scale that minimize the loss due to the
    rounding operation.

    .. math::
        \mathrm{Quantize}( x) = \mathrm{Round}((x - \mathrm{Shift}) \cdot \mathrm{Scale})

    The shift and scale parameter can be provided explicitly, or can be
    estimated based on a few traces.

    Warning
    ^^^^^^^

    The quantization procedure operates pointwise: each point is shifted and
    scaled by a different value.
    As a consequence the quantized version of the trace probably does not look
    like its non quantized version.

    Parameters
    ----------
    shift : npt.NDArray[np.floating]
        The value to shift every traces.
    scale : npt.NDArray[np.floating]
        The value to scale every traces.

    Examples
    --------
    >>> from scalib.preprocessing import Quantizer
    >>> import numpy as np
    >>> # 500 traces of 200 points
    >>> traces : npt.NDArray[np.floating] = np.random.randn(500,200)
    >>> quantizer = Quantizer.fit(traces)
    >>> quantized_traces : npt.NDArray[np.int16] = quantizer.quantize(traces)
    >>> # Can be reused directly on 5000 new traces for instance
    >>> traces : npt.NDArray[np.floating] = np.random.randn(5000,200)
    >>> quantized_traces : npt.NDArray[np.int16] = quantizer.quantize(traces)
    """

    def __init__(
        self, shift: npt.NDArray[np.floating], scale: npt.NDArray[np.floating]
    ):
        self._shift = shift
        self._scale = scale

    @classmethod
    def fit(
        cls,
        traces: npt.NDArray[np.floating],
        method: QuantFitMethod = QuantFitMethod.bounds(),
    ):
        r"""Compute the shift and scale estimation from sample of `traces`

        This class method returns an instance of Quantizer with the
        corresponding shift and scale.

        Parameters
        ----------
        traces : array_like, np.floating
            Array that contains the traces to estimate the shift and scale in
            the quantization. The array must be of dimension `(n, ns)`
        method : QuantFitMethod
            A member of QuantFitMethod enum class that specifies how the
            minimum and maximum value of the trace to be quantized is
            estimated.
        """

        if method.method == QFitMethod.BOUNDS:
            # Max/Min Centering and Multiplication by a constant prior to
            # quantization to avoid information loss via rounding error
            max = np.amax(traces, axis=0)
            min = np.amin(traces, axis=0)
            shift = (max + min) / 2
            scale = 2**15 / ((max - min) / 2) / method.opts["margin"]
        elif method.method == QFitMethod.MOMENT:
            # Gaussian Methods
            mean = np.mean(traces, axis=0)
            std = np.std(traces, axis=0, ddof=1)
            shift = mean
            scale = 2**15 / (method.opts["nstd"] * std)
        else:
            raise ValueError("method.method should be a QFitMethod object")

        return cls(shift, scale)

    def quantize(
        self, traces: npt.NDArray[np.floating], clip: bool = False
    ) -> npt.NDArray[np.int16]:
        r"""Quantize the traces provide in `traces`

        Parameters
        ----------
        traces : array_like, np.floating
            Array that contains the traces to be quantized into int16. The array must
            be of dimension `(n, ns)`
        clip : bool
            Boolean to bypass the overflow check prior to quantization and clip
            the overflowing values to the boundaries.
            By default it is set to False.
        """
        adjusted_traces: npt.NDArray[np.floating] = (traces - self._shift) * self._scale
        if clip:
            adjusted_traces = np.clip(adjusted_traces, -(2**15), 2**15 - 1)
        else:
            overflow = (adjusted_traces > 2**15 - 1).any() or (
                adjusted_traces < -(2**15)
            ).any()
            if overflow:
                raise ValueError(
                    "Overflow detected in the quantization. Update shift and "
                    "scale more precisely to avoid the error."
                )

        quantized_traces: npt.NDArray[np.int16] = adjusted_traces.astype(np.int16)
        return quantized_traces
