import pytest
import numpy as np
import queue
from scalib.metrics import SNR
from scalib.tools import ContextExecutor

n_traces = 10
ns = 5


def snr_inputs():
    snr = SNR(2**8, ns, 16, True)
    while (x := np_array_queue.get()) is not None:
        (traces, k) = x
        snr.fit_u(traces, k)


def get_traces():
    # Models a true acquisition process.
    k = np.zeros((n_traces, 16), dtype=np.uint16)
    traces = np.zeros((n_traces, ns), dtype=np.int16)
    return (traces, k)


def test_context_executor():
    snr = SNR(2**8, ns, 16, True)
    snr_workers = 1
    with ContextExecutor(max_workers=snr_workers) as e:
        # Initialize the future queue with dummy futures
        future_queue = [e.submit(lambda: None) for _ in range(snr_workers)]
        for _ in range(3):
            (traces, k) = get_traces()
            future_queue.append(e.submit(snr.fit_u, traces, k))
            # Wait on one future
            future_queue.pop(0).result()
