from scalib.metrics import Ttest

from utils import gen_traces
import numpy as np


def main():
    # Parameters
    std = 2
    ntraces = 1000

    print(f"Fixed-vs-random TVLA on {ntraces} traces.")
    # Fix-vs-random TVLA (fixed-key vs random-key).
    # We generate new datasets, as the previous ones use random plaintext, and we needed fixed plaintext.
    traces_rk, _ = gen_traces(ntraces, std, random_key=True, random_plaintext=False)
    traces_fk, _ = gen_traces(ntraces, std, random_key=False, random_plaintext=False)
    # Ttest needs the lenght of the traces, and the test order (here, two)
    ttest = Ttest(2)
    # Ttest can be updated multiple times, and the order of the updates does not matter.
    ttest.fit_u(traces_rk, np.zeros((traces_rk.shape[0],), dtype=np.uint16))
    ttest.fit_u(traces_fk, np.ones((traces_fk.shape[0],), dtype=np.uint16))
    tt_result = ttest.get_ttest()
    print(
        "First-order t-statistic: ", ", ".join(f"{t: 5.01f}" for t in tt_result[0, :])
    )
    print(
        "Second-order t-statistic:", ", ".join(f"{t: 5.01f}" for t in tt_result[1, :])
    )


if __name__ == "__main__":
    main()
