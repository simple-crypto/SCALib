from scalib.metrics import SNR, Ttest
from scalib.modeling import LDAClassifier
from scalib.attacks import FactorGraph, BPState
from scalib.postprocessing import rank_accuracy

from utils import sbox, gen_traces
import numpy as np


def main():
    nc = 256
    npoi = 2

    # Parameters
    std = 2
    ntraces_a = 40
    ntraces_p = 20000

    print(
        "1. Generate simulated traces (Hamming weight + Gaussian noise) with parameters:"
    )
    print(f"    ntraces for profiling: {ntraces_p}")
    print(f"    ntraces for attack:    {ntraces_a}")
    print(f"    noise std:             {std}")
    traces_p, labels_p = gen_traces(
        ntraces_p, std, random_key=True, random_plaintext=True
    )
    traces_a, labels_a = gen_traces(
        ntraces_a, std, random_key=False, random_plaintext=True
    )

    _, ns = traces_p.shape

    print("2. POI selection with SNR")
    print("    2.1 Compute SNR for 16 Sbox outputs xi")
    # y array with xi values
    x = np.zeros((ntraces_p, 16), dtype=np.uint16)
    for i in range(16):
        x[:, i] = labels_p[f"x{i}"]

    # estimate SNR
    snr = SNR(nc=nc, ns=ns, np=16)
    snr.fit_u(traces_p, x)
    snr_val = snr.get_snr()

    print("    2.2 Select POIs with highest SNR.")
    pois = [np.argsort(snr_val[i])[-npoi:] for i in range(16)]

    print("3. Profiling")
    # We build a LDA model (pooled Gaussian templates) for each of the 16
    # Sboxes (xi).

    print("    3.1 Build LDAClassifier for each xi")
    models = []
    for i in range(16):
        lda = LDAClassifier(nc=nc, ns=npoi, p=1)
        lda.fit_u(l=traces_p[:, pois[i]], x=labels_p[f"x{i}"].astype(np.uint16))
        lda.solve()
        models.append(lda)

    print("    3.2 Get xi distributions from attack traces")
    probas = [models[i].predict_proba(traces_a[:, pois[i]]) for i in range(16)]

    print("4. Attack")
    print("    4.1 Create the SASCA Graph")
    graph_desc = f"""
                NC {nc}
                TABLE sbox   # The Sbox
                """

    for i in range(16):
        graph_desc += f"""
                VAR SINGLE k{i} # The key
                PUB MULTI p{i}  # The plaintext
                VAR MULTI x{i}  # Sbox output
                VAR MULTI y{i}  # Sbox input
                PROPERTY y{i} = k{i} ^ p{i} # Key addition
                PROPERTY x{i} = sbox[y{i}]  # Sbox lookup
                """

    # Initialize FactorGraph with the graph description and the required tables
    factor_graph = FactorGraph(graph_desc, {"sbox": sbox})

    print("    4.2 Create belief propagation state.")
    # We have to give the number of attack traces and the values for the public variables.
    bp = BPState(
        factor_graph,
        ntraces_a,
        {f"p{i}": labels_a[f"p{i}"].astype(np.uint32) for i in range(16)},
    )

    for i in range(16):
        bp.set_evidence(f"x{i}", probas[i])

    print("    4.3 Run belief propagation")
    for i in range(16):
        bp.bp_acyclic(f"k{i}")

    print("5. Attack evaluation")
    print("    5.1 Byte-wise attack")

    # correct secret key
    secret_key = []
    # distribution for each of the key bytes
    key_distribution = []
    # the best key guess of the adversary
    guess_key = []
    # rank for all the key bytes
    ranks = []

    for i in range(16):
        sk = labels_a[f"k{i}"][0]  # secret key byte
        distribution = bp.get_distribution(f"k{i}")

        guess_key.append(np.argmax(distribution))
        ranks.append(256 - np.where(np.argsort(distribution) == sk)[0])

        secret_key.append(sk)
        key_distribution.append(distribution)

    print("")
    print("        secret key (hex):", " ".join(["%3x" % (x) for x in secret_key]))
    print("        best key   (hex):", " ".join(["%3x" % (x) for x in guess_key]))
    print("        key byte ranks  :", " ".join(["%3d" % (x) for x in ranks]))
    print("")

    print(f"   5.2 Estimate full log2 key rank:")
    key_distribution = np.array(key_distribution)

    # Scores are negative log-likelihoods.
    # Put a lower-bound on the probabilities, oterwise we might get NaNs.
    scores = -np.log2(np.maximum(key_distribution, 2**-128))
    rmin, r, rmax = rank_accuracy(scores, secret_key, 1)

    lrmin, lr, lrmax = (np.log2(rmin), np.log2(r), np.log2(rmax))
    print("")
    print(f"        {lrmin} < {lr} < {lrmax}")
    print("")


if __name__ == "__main__":
    main()
