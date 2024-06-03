import pytest
from scalib.preprocessing import Quantizer, QuantFitMethod
import numpy as np


def test_quantizer():
    ns = 200
    fitting_traces = np.random.randn(500, ns)
    traces = np.random.randn(5000, ns)

    quantizer = Quantizer.fit(fitting_traces, QuantFitMethod.moment())
    quantized_traces = quantizer.quantize(traces)

    quantized_traces = quantizer.quantize(8 * traces, True)
    with pytest.raises(ValueError):
        quantized_traces = quantizer.quantize(8 * traces)

    quantizer = Quantizer.fit(fitting_traces, QuantFitMethod.bounds(4.0))
    quantized_traces = quantizer.quantize(traces)

    reconstruction = (quantized_traces / quantizer._scale + quantizer._shift).astype(
        np.float64
    )
    reconstruction_error: np.float64 = np.linalg.norm(traces - reconstruction, axis=1)
    assert (reconstruction_error <= 10**-2).all()
