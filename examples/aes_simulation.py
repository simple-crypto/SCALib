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

    print("1. Generate simulated traces (Hamming weight + Gaussian noise) with parameters:")
    print(f"    ntraces for profiling: {ntraces_p}")
    print(f"    ntraces for attack:    {ntraces_a}")
    print(f"    noise std:             {std}")
    traces_p, labels_p = gen_traces(ntraces_p, std, random_key=True, random_plaintext=True)
    traces_a, labels_a = gen_traces(ntraces_a, std, random_key=False, random_plaintext=True)

    _, ns = traces_p.shape

    print("2. TVLA (not further used -- first leakage expected, but no second order leakage)")
    # Ttest needs the lenght of the traces, and the test order (here, two)
    ttest = Ttest(traces_p.shape[1], 2)
    # Fix-vs-random TVLA (fixed-key vs random-key).
    # We generate new datasets, as the previous ones use random plaintext, and we needed fixed plaintext.
    traces_rk, _ = gen_traces(ntraces_p, std, random_key=True, random_plaintext=False)
    traces_fk, _ = gen_traces(ntraces_p, std, random_key=False, random_plaintext=False)
    # Ttest can be updated multiple times, and the order of the updates does not matter.
    ttest.fit_u(traces_rk, np.zeros((traces_rk.shape[0],), dtype=np.uint16))
    ttest.fit_u(traces_fk, np.ones((traces_fk.shape[0],), dtype=np.uint16))
    tt_result = ttest.get_ttest()
    print("")
    print("    First-order t-statistic: ", *(f"{t:.02f}" for t in tt_result[0,:]))
    print("    Second-order t-statistic:", *(f"{t:.02f}" for t in tt_result[1,:]))
    print("")

    print("3. Profiling")
    # For each of the 16 variables at the output of the Sbox (xi)
    # we build a model that store in the dictonnary models.
    #
    # The profiling is done in 3 phases
    # 1. Compute SNR for all xi
    # 2. Select the Point-of-Interest based on the SNR
    # 3. Fit an LDA model for each of the xi

    models = {}
    for i in range(16):
        models[f"x{i}"] = {}

    print("    3.1 Compute SNR for 16 Sbox outputs xi")
    # y array with xi values
    x = np.zeros((ntraces_p, 16), dtype=np.uint16)
    for i in range(16):
        x[:, i] = labels_p[f"x{i}"]

    # estimate SNR
    snr = SNR(nc=256, ns=ns, np=16)
    snr.fit_u(traces_p, x)
    snr_val = snr.get_snr()

    print("    3.2 Keep {npoi} POIs for each xi")
    # POI are the npoi indexes with the largest SNR
    for i in range(16):
        models[f"x{i}"]["poi"] = np.argsort(snr_val[i])[-npoi:]

    print("    3.3 Build LDAClassifier for each xi")
    for i in range(16):
        model = models[f"x{i}"]
        lda = LDAClassifier(nc=256, ns=npoi, p=1)
        lda.fit_u(l=traces_p[:, model["poi"]], x=labels_p[f"x{i}"].astype(np.uint16))
        lda.solve()
        model["lda"] = lda

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

    # Init FactorGraph with the graph description and the required tables
    factor_graph = FactorGraph(graph_desc, {'sbox': sbox})

    print("    4.2 Create belief propagation state.")
    # We have to give the number of attack traces and the values for the public variables.
    bp = BPState(factor_graph, ntraces_a, {f"p{i}": labels_a[f"p{i}"].astype(np.uint32) for i in range(16)})

    print("    4.3 Add xi distributions in the graph")
    for i in range(16):
        model = models[f"x{i}"]
        bp.set_evidence(f"x{i}", model["lda"].predict_proba(traces_a[:, model["poi"]]))

    print("    4.4 Run belief propagation")
    bp.bp_loopy(it=3, initialize_states=True)

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
