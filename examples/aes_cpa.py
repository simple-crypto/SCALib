from scalib.attacks import Cpa

from utils import sbox, gen_traces
import numpy as np


def generate_hw_models(nv, nc, ns):
    hw_uv = np.zeros([nv, nc, ns])
    for c in range(nc):
        hw_uv[:, c, :] = sbox[c].bit_count()
    return hw_uv.astype(np.float64)


def cpa_HW_example(pts, traces):
    """
    Perform a Correlation Power Attack against the output of the Sbox layer in
    the first round of the AES.

    pts: shape (n, 16),
        The known plaintexts.
    traces: shape (n, ns)
        The traces, of 'ns' samples, correpsonding to the execution of the
        plaintext with the unknown fixed key.

    returns:
        The key candidate maximising the correlation with an Hamming Weight leakage model.
    """
    # First, we create the models. Models map the value of the intermediate
    # variable to the leakage (represented as a lookup table).
    # Note that this step can be done in an offline first step.
    models = generate_hw_models(16, 256, traces.shape[1])
    # Next, we create the CPA object, using the XOR intermediate function: the
    # intermediate variable will be the XOR of the guessed key and the provided
    # class (plaintext).
    cpa = Cpa(256, Cpa.Xor)
    # Fit the correlation computation (this may be called multiple times with
    # different chunks of traces/plaintexts).
    cpa.fit_u(traces, pts)
    # Compute the correlation for each variable, associated to each key guess
    # and time sample.
    corr = np.abs(cpa.get_correlation(models))
    # Return the key guess maximising the correlation for every variable.
    return np.argmax(np.max(corr, axis=2), axis=1)


def main_cpa_example():
    # Parameters
    std = 2  # Noise standard deviation
    n_traces = 1000  # Amount of traces

    # First, we simulate the traces.
    print(
        f"Generate {n_traces} simulated traces (Hamming weight + Gaussian noise)"
        f"with noise standard deviation of {std}."
    )
    traces, labels = gen_traces(n_traces, std, random_key=False, random_plaintext=True)
    # Recover only the plaintext from the labels.
    pts = np.hstack([labels["p{}".format(i)][:, np.newaxis] for i in range(16)]).astype(
        np.uint16
    )

    # Perform the CPA
    key_guessed = cpa_HW_example(pts, traces)
    print(f"Key guessed: {key_guessed}")

    # Correct key comparison
    key_correct = np.array([int(labels["k{}".format(i)][0]) for i in range(16)])
    print("Correct key: {}".format(key_correct))
    if (key_guessed == key_correct).all():
        print("Attack success")
    else:
        print("Attack failure")


if __name__ == "__main__":
    print("Example HW CPA attack")
    main_cpa_example()
