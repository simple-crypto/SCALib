import pytest
from scalib.preprocessing import Quantizer
import numpy as np


def test_quantizer():
    fitting_traces: np.ndarray[np.float64] = np.random.randn(500, 200)
    traces: np.ndarray[np.float64] = np.random.randn(5000, 200)

    quantizer = Quantizer.fit(fitting_traces, False)
    quantized_traces: np.ndarray[np.int16] = quantizer.quantize(traces)

    quantized_traces: np.ndarray[np.int16] = quantizer.quantize(8 * traces, True)
    with pytest.raises(ValueError):
        quantized_traces: np.ndarray[np.int16] = quantizer.quantize(8 * traces)

    quantizer = Quantizer.fit(fitting_traces, True)
    quantized_traces: np.ndarray[np.int16] = quantizer.quantize(traces)
