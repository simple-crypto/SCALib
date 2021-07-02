import json
import matplotlib.pyplot as plt
import numpy as np

baseline = "single_thread_native"
nss = 2 ** np.arange(8, 17)
nc = 256
n = 20 * nc
bench = "snr_update"
threads = 1
for bench, baseline,threads in [
    ("snr_update", "single_thread_native",1),
    ("snr_update", "single_thread",1),
    ("snr_update_threads", "12_thread",12),
    ("snr_update_threads", "12_thread_native",12),
]:
    plt.figure()
    for chunks in [12, 13, 14, 25]:
        c = np.random.rand(3)
        for n_p, ls in [(1, "--"), (16, "-")]:
            ms = []
            for ns in nss:
                f = json.load(
                    open(
                        f"target/criterion/{bench}/chunk_{chunks}_{n_p}/{ns}/{baseline}/estimates.json"
                    )
                )
                ms.append(f["mean"]["point_estimate"] * 1e-9)  # time in s
            ms = np.array(ms)
            print(ms)
            cycles_per_sample = (threads * ms * 2.3e9) / (nss * n * n_p)
            plt.semilogx(
                nss,
                cycles_per_sample,
                label=f"1<<{chunks}, var {n_p}",
                basex=2,
                ls=ls,
                color=c,
            )

    c = np.random.rand(3)
    for n_p, ls in [(1, "--"), (16, "-")]:
        ms = []
        for ns in nss:
            f = json.load(
                open(
                    f"target/criterion/{bench}/old_{n_p}/{ns}/{baseline}/estimates.json"
                )
            )
            ms.append(f["mean"]["point_estimate"] * 1e-9)  # time in s
        ms = np.array(ms)
        print(ms)
        cycles_per_sample = (threads * ms * 2.3e9) / (nss * n * n_p)
        plt.semilogx(
            nss, cycles_per_sample, label=f"master, var {n_p}", basex=2, ls=ls, color=c
        )

    plt.legend()
    plt.title(baseline)
    plt.grid(True, which="both", ls="--")
    plt.xlabel("trace length")
    plt.ylabel("cycles * threads / snr_update")
    plt.savefig(f"/tmp/{bench}_{baseline}.pdf")
