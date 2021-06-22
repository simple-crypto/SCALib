from scalib.metrics import SNR
from scalib.modeling import LDAClassifier
from scalib.attacks import SASCAGraph
from scalib.postprocessing import rank_accuracy

from utils import sbox, gen_traces
import numpy as np

if __name__ == "__main__":
    nc = 256
    npoi = 2

    # Parameters
    std = 1
    ntraces_a = 1000
    ntraces_p = 20000

    print("1. Generate simulated traces with parameters:")
    print(f"    ntraces for profiling: {ntraces_p}")
    print(f"    ntraces for attack:    {ntraces_a}")
    print(f"    noise std:             {std}")
    traces_p, labels_p = gen_traces(ntraces_p, std, random_key=1)
    traces_a, labels_a = gen_traces(ntraces_a, std, random_key=0)

    _, ns = traces_p.shape
    print("2. Profiling")
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

    print("    2.1 Compute snr for 16 Sbox outputs xi")
    # y array with xi values
    x = np.zeros((ntraces_p, 16), dtype=np.uint16)
    for i in range(16):
        x[:, i] = labels_p[f"x{i}"]

    # estimate SNR
    snr = SNR(nc=256, ns=ns, np=16)
    snr.fit_u(traces_p, x)
    snr_val = snr.get_snr()

    print(f"    2.2 Keep {npoi} POIs for each xi")
    # POI are the npoi indexes with the largest SNR
    for i in range(16):
        models[f"x{i}"]["poi"] = np.argsort(snr_val[i])[-npoi:]

    print("    2.3 Build LDAClassifier for each xi")
    for i in range(16):
        model = models[f"x{i}"]
        lda = LDAClassifier(nc=256, ns=npoi, p=1)
        lda.fit_u(l=traces_p[:, model["poi"]], x=labels_p[f"x{i}"].astype(np.uint16))
        lda.solve()
        model["lda"] = lda

    print("3. Attack")
    print("    3.1 Create the SASCA Graph")
    graph_desc = f"""
                NC {nc}
                TABLE sbox   # The Sbox
                """

    for i in range(16):
        graph_desc += f"""
                VAR SINGLE k{i} # The key
                VAR MULTI p{i}  # The plaintext
                VAR MULTI x{i}  # Sbox output
                VAR MULTI y{i}  # Sbox input
                PROPERTY y{i} = k{i} ^ p{i} # Key addition
                PROPERTY x{i} = sbox[y{i}]  # Sbox lookup
                """

    # Init SASCAGraph with graph_desc and the number of attack traces
    sasca = SASCAGraph(graph_desc, n=ntraces_a)

    print("    3.2 Add public information in the SASCAGraph")
    sasca.set_table("sbox", sbox)
    for i in range(16):
        sasca.set_public(f"p{i}", labels_a[f"p{i}"].astype(np.uint32))

    print("    3.3 Add xi distributions in the graph")
    for i in range(16):
        model = models[f"x{i}"]
        sasca.set_init_distribution(
            f"x{i}", model["lda"].predict_proba(traces_a[:, model["poi"]])
        )

    print("    3.4 Run belief propagation")
    print("")
    sasca.run_bp(it=3)
    print("")

    print("4. Attack evaluation")
    print("    4.1 Byte-wise attack")

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
        distribution = sasca.get_distribution(f"k{i}")[0, :]

        guess_key.append(np.argmax(distribution))
        ranks.append(256 - np.where(np.argsort(distribution) == sk)[0])

        secret_key.append(sk)
        key_distribution.append(distribution)

    print("")
    print("        secret key (hex):", " ".join(["%3x" % (x) for x in secret_key]))
    print("        best key   (hex):", " ".join(["%3x" % (x) for x in guess_key]))
    print("        key byte ranks  :", " ".join(["%3d" % (x) for x in ranks]))
    print("")

    print(f"    4.2 Estimate full log2 key rank:")
    key_distribution = np.array(key_distribution)

    rmin, r, rmax = rank_accuracy(-np.log10(key_distribution), secret_key, 1)

    lrmin, lr, lrmax = (np.log2(rmin), np.log2(r), np.log2(rmax))
    print("")
    print(f"        {lrmin} < {lr} < {lrmax}")
    print("")
