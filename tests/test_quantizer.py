import pytest
from scalib.preprocessing import Quantizer, QuantFitMethod
import numpy as np


def test_quantizer():
    fitting_traces: np.ndarray[np.float64] = np.random.randn(500, 200)
    traces: np.ndarray[np.float64] = np.random.randn(5000, 200)

    quantizer = Quantizer.fit(fitting_traces, QuantFitMethod.MOMENT)
    quantized_traces: np.ndarray[np.int16] = quantizer.quantize(traces)

    quantized_traces: np.ndarray[np.int16] = quantizer.quantize(8 * traces, True)
    with pytest.raises(ValueError):
        quantized_traces: np.ndarray[np.int16] = quantizer.quantize(8 * traces)

    quantizer = Quantizer.fit(fitting_traces, QuantFitMethod.BOUNDS)
    quantized_traces: np.ndarray[np.int16] = quantizer.quantize(traces)

    reconstruction: np.ndarray[np.float64] = (
        quantized_traces / quantizer._scale + quantizer._shift
    ).astype(np.float64)
    reconstruction_error: np.float64 = np.linalg.norm(traces - reconstruction, axis=1)
    assert (reconstruction_error <= 10**-2).all()
