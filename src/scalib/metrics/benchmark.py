import numpy as np
from kNNMutualInformation import kNNInformationEstimator
from scipy.special import logsumexp, binom
from scipy.integrate import quad
from scipy.stats import norm
import matplotlib.pyplot as plt


"""
    MI ESTIMATION
    Numerical estimmation of the mutual information between n bit secret and its HW + AWGN(sigma) leakages
"""

# Integration bounds
lb = -np.inf
ub = np.inf
# Absolute Error and Relative Error Tollerated in Numerical Integration
epsabs = 10**-3
epsrel = 10**-3
# Maximal Number of Subinterval in Adapative Integration Method
limit = 100


def integrand(y, sigma, n):
    b = np.array([binom(n, w) for w in range(n + 1)])
    I = 0
    for u in range(n + 1):
        a = np.array([(y - u) ** 2 - (y - w) ** 2 for w in range(n + 1)]) / (
            2 * sigma**2
        )
        I -= norm.pdf(y - u, scale=sigma) * logsumexp(a, b=b) * binom(n, u) * (2**-n)
    return I


def integral(sigma, n):
    QUAD = quad(
        integrand, lb, ub, args=(sigma, n), limit=limit, epsabs=epsabs, epsrel=epsrel
    )
    return QUAD[0]


def mi_hw_awgn(sigma, n):
    return n + integral(sigma, n) / np.log(2)


def hw_vec(arr):
    t = arr.dtype.type
    mask = t(-1)
    s55 = t(0x5555555555555555 & mask)  # Add more digits for 128bit support
    s33 = t(0x3333333333333333 & mask)
    s0F = t(0x0F0F0F0F0F0F0F0F & mask)
    s01 = t(0x0101010101010101 & mask)
    arr = arr - ((arr >> 1) & s55)
    arr = (arr & s33) + ((arr >> 2) & s33)
    arr = (arr + (arr >> 4)) & s0F
    return (arr * s01) >> (8 * (arr.itemsize - 1))


def gen_sample(M, sigma, n_sample):
    X = np.random.randint(low=0, high=M, size=n_sample)
    Y = hw_vec(X) + sigma * np.random.randn(n_sample)
    Y = np.reshape(Y, (n_sample, 1))
    return X, Y


def multiple_estimations(
    n=8, sigma=2, ensemble_k=[3, 5, 10, 15], n_sample=10**4, repeat=100, dummy_dim=0
):
    M = 1 << n
    KSG_pred = np.zeros(repeat)
    KSG_pred_ensemble = np.zeros((repeat, len(ensemble_k)))
    for i in range(repeat):
        X, Y = gen_sample(M, sigma, n_sample)
        Pure_noise = np.random.randn(n_sample, dummy_dim).reshape((n_sample, dummy_dim))
        Y_with_dummy_dim = np.concatenate([Y, Pure_noise], axis=1)
        KSG = kNNInformationEstimator(M, X, Y_with_dummy_dim)
        prediction, ensemble_pred = KSG.ensemble_predict(ensemble_k, p=2)
        KSG_pred[i] = prediction
        KSG_pred_ensemble[i] = ensemble_pred
    return KSG_pred, KSG_pred_ensemble


n = 8
n_sigma = 15
repeat = 50
n_sample = 5*10**4
dummy_dim = 0
Sigma = np.geomspace(5 * 10**-2, 5*10**2, n_sigma)

ensemble_k = [
    3,
    5,
    7,
    10,
]
c = ["blue", "purple", "green", "cyan"]

MI_true = np.array([mi_hw_awgn(sigma, n) for sigma in Sigma])

MI_predictions = np.zeros((n_sigma, repeat))
MI_predictions_ensemble = np.zeros((n_sigma, repeat, len(ensemble_k)))

for i in range(n_sigma):
    MI_predictions[i], MI_predictions_ensemble[i] = multiple_estimations(
        n,
        Sigma[i],
        ensemble_k=ensemble_k,
        n_sample=n_sample,
        repeat=repeat,
        dummy_dim=dummy_dim,
    )


# To make the points with evaluation equal to zero appear on the graph
EPSILON = 10**-15
MI_predictions = np.where(MI_predictions == 0, EPSILON, MI_predictions)
MI_predictions_ensemble = np.where(
    MI_predictions_ensemble == 0, EPSILON, MI_predictions_ensemble
)

for r in range(repeat):
    plt.scatter(Sigma, MI_predictions[:, r], color="red", alpha=0.5)

for j in range(len(ensemble_k)):
    k = ensemble_k[j]
    plt.plot(
        Sigma,
        np.median(MI_predictions_ensemble[:, :, j], axis=1),
        label=f"k = {k}",
        color=c[j],
    )

plt.plot(
    Sigma, np.median(MI_predictions, axis=1), color="red", label=f"KSG k={ensemble_k}"
)


plt.plot(Sigma, MI_true, color="black", label="Ground Truth")

plt.title(f"Benchmark on Hamming Weight Model + AWGN Noise, Nsample= {n_sample}")
plt.xlabel(r"$\sigma$")
plt.ylabel(r"$\hat{I(X;Y)}$")
plt.semilogx(base=10)
plt.semilogy(base=10)
plt.grid()
plt.legend()
plt.show()
