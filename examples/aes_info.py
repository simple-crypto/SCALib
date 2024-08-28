from scalib.modeling import LDAClassifier

from utils import gen_traces
import numpy as np


def main():
    nc = 256

    # Parameters
    std = 2
    ntraces_p = 10000
    ntraces_v = 1000

    print(
        "1. Generate simulated traces (Hamming weight + Gaussian noise) with parameters:"
    )
    print(f"    ntraces for profiling: {ntraces_p}")
    print(f"    ntraces for validation: {ntraces_v}")
    print(f"    noise std: {std}")
    print("")
    traces_p, labels_p = gen_traces(
        ntraces_p, std, random_key=True, random_plaintext=True
    )
    traces_v, labels_v = gen_traces(
        ntraces_v, std, random_key=True, random_plaintext=True
    )

    print("2. Profiling the value of the first Sbox.")
    print("")

    lda = LDAClassifier(nc=256, p=1)
    lda.fit_u(traces=traces_p, x=labels_p["x0"].astype(np.uint16))
    lda.solve()

    def eval_info(test_traces, test_labels):
        test_probas = lda.predict_proba(test_traces)
        test_probas = test_probas[range(test_probas.shape[0]), test_labels["x0"]]
        info = np.log2(nc) + np.mean(np.log2(test_probas))
        info_std = np.std(np.log2(test_probas)) / np.sqrt(test_traces.shape[0])
        return info, info_std

    print("3. PI: using validation traces")
    pi, pi_std = eval_info(traces_v, labels_v)
    print("PI [bits]", pi)
    print("PI stddev", pi_std)
    print("")

    print("4. TI: using profiling traces")
    ti, ti_std = eval_info(traces_p, labels_p)
    print("TI [bits]", ti)
    print("TI stddev", ti_std)


if __name__ == "__main__":
    main()
