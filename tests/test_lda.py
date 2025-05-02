import pickle

import pytest
import numpy as np
import scipy.linalg
import scipy.stats
import scipy.special

from scalib import ScalibError
from scalib.modeling import LDAClassifier, MultiLDA, Lda, LdaAcc

from utils_test import get_rng

import copy


class RefLda:
    def __init__(self, traces, x, nc, p):
        self.nc = nc
        self.p = p
        n = traces.shape[0]
        self.ns = traces.shape[1]
        tr_x = [traces[x == i, :] for i in range(nc)]
        self.nis = [(x == i).sum() for i in range(nc)]
        self.mus = np.array([tr.mean(axis=0) for tr in tr_x])
        # between-class scatter: s_b = Si(ni*cmui**2)
        self.s_b = n * np.cov(self.mus.T, ddof=0, fweights=self.nis)
        # within-class scatter: s_w = Si(Sxi((xi-mui)**2))
        self.s_w = np.sum(
            [ni * np.cov(tr.T, ddof=0) for ni, tr in zip(self.nis, tr_x)], axis=0
        )
        # Factor n on s_w important to get properly scaled projection for finalize.
        # (such that evec*(s_w/n)*evec^T=1).
        self.evals, self.evecs = scipy.linalg.eigh(self.s_b, self.s_w / (n - nc))
        self.finalize(projection=self.evecs[:, np.argsort(self.evals)[::-1][: self.p]])

    def finalize(self, projection):
        self.projection = projection
        self.coef = np.dot(self.mus, self.projection).dot(self.projection.T)
        self.intercept = -0.5 * np.diag(np.dot(self.mus, self.coef.T))

    def project(self, traces):
        return np.dot(traces, self.projection)

    def predict_proba(self, traces):
        scores = traces @ self.coef.T + self.intercept[np.newaxis, :]
        return scipy.special.softmax(scores, axis=1)


# To test
# 1. pickle: can pickle objects, loading pickled data results in **strictly** identical behavior.
# 2. test projection, means, scatter against SKLearn (test_lda) - in single var case.
# 3. check errors in invalid cases
# 4. single var vs multi-var comparison
#
# functions to test: fit, predict_proba, predict_log2_proba_class, select
#
# Test cases with non auto-generated data.
#
# LDAClassifier, MultiLDA


def pois_uniform_indep(rng, ns, nv, npois, **params):
    assert ns <= params["maxl"]
    pois = np.tile(np.arange(ns), (nv, 1))
    return rng.permuted(pois, axis=1)[:, :npois].tolist()


def data_gaussian_default(
    rng, ns, nv, nc, n, n_batches, signal_std=50.0, noise_std=20.0, **_
):
    all_x = []
    all_traces = []
    class_means = rng.normal(0.0, signal_std, (nc, ns))
    for _ in range(n_batches):
        x = rng.integers(0, nc, (n, nv), dtype=np.uint16)
        noise = rng.normal(0.0, noise_std, (n, ns))
        traces = class_means[x, :].sum(axis=1) + noise
        rtraces = np.round(traces).astype(np.int16)
        all_x.append(x)
        all_traces.append(rtraces)
    return all_traces, all_x


class LdaTestCase:
    def __init__(
        self,
        *,
        ns,
        nc,
        nv,
        npois,
        n,
        n_batches,
        p,
        test_lp,
        maxl,
        n_sel=None,
        nv_sel=None,
    ):
        self.pois = None
        self.traces = None
        self.x = None
        self.params = dict(
            ns=ns,
            nc=nc,
            nv=nv,
            npois=npois,
            n=n,
            n_batches=n_batches,
            p=p,
            test_lp=test_lp,
            maxl=maxl,
            n_sel=n_sel,
            nv_sel=nv_sel,
        )

    def __str__(self):
        return str(self.params)

    def with_params(self, **kwargs):
        cls = type(self)
        return cls(**(self.params | kwargs))

    def with_data(self, x=None, traces=None):
        res = copy.deepcopy(self)
        if x is not None:
            res.x = x
        if traces is not None:
            res.traces = traces
        return res

    def with_pois(self, pois=None):
        res = copy.deepcopy(self)
        if pois is not None:
            res.pois = pois
        return res

    def get_data(self, fpois=pois_uniform_indep, fdata=data_gaussian_default):
        rng = get_rng(**self.params)
        pois = fpois(rng, **self.params) if self.pois is None else self.pois
        if self.traces is None or self.x is None:
            traces, x = fdata(rng, **self.params)
            x = x if self.x is None else self.x
            traces = traces if self.traces is None else self.traces
        else:
            traces, x = self.traces, self.x
        return pois, traces, x

    def __getitem__(self, key):
        return self.params[key]


def normalize(x):
    return x / np.linalg.norm(x, axis=0, keepdims=True)


def is_parallel(x, y):
    z = np.abs(x.dot(y))
    z2 = np.linalg.norm(x) * np.linalg.norm(y)
    return np.allclose(z, z2)


# 1. Test that univariate LDA results in similar results that the one from ref.
lda_ref_uni_bc = LdaTestCase(
    ns=2,
    nc=2,
    nv=1,
    npois=2,
    n=20,
    n_batches=1,
    p=1,
    test_lp=True,
    maxl=2**15,
)
lda_ref_uni_cases = [
    lda_ref_uni_bc,
    lda_ref_uni_bc.with_params(n=200),
    lda_ref_uni_bc.with_params(ns=3, npois=3).with_pois(pois=[[0, 1, 2]]),
    lda_ref_uni_bc.with_params(ns=3, npois=3),
    lda_ref_uni_bc.with_params(ns=10, npois=3),
    lda_ref_uni_bc.with_params(ns=10, npois=3, p=2, nc=4, n=100),
    lda_ref_uni_bc.with_params(ns=3, npois=3, p=1, nc=2, n=5, n_batches=3),
    lda_ref_uni_bc.with_params(ns=10, npois=3, p=2, nc=4, n=100, n_batches=5),
]


@pytest.mark.parametrize("case", lda_ref_uni_cases)
def test_univariate_lda_ref(case):
    pois, btraces, bx = case.get_data()
    # LdaAc
    lda_acc = LdaAcc(pois=pois, nc=case["nc"])
    for bi, (traces, x) in enumerate(zip(btraces, bx)):
        lda_acc.fit_u(traces, x)
    # LDARef
    traces = np.vstack(btraces)
    x = np.vstack(bx)
    lda_ref = RefLda(traces[:, pois[0]], x[:, 0], nc=case["nc"], p=case["p"])

    # Verify value of means by accessor
    lda_means = lda_acc.get_mus()[0]
    assert np.allclose(
        lda_ref.mus, lda_means
    ), "Mean mismatch\nmu_ref:=\n{}\nmu_mean:=\n{}".format(lda_ref.mus, lda_means)

    s_b = lda_acc.get_sb()[0]
    s_w = lda_acc.get_sw()[0]
    assert np.allclose(s_b, lda_ref.s_b), f"SB mismatch\n{s_b=}\n{lda_ref.s_b=}"
    assert np.allclose(s_w, lda_ref.s_w), f"SB mismatch\n{s_w=}\n{lda_ref.s_w=}"

    # Solve the LdaAcc
    lda = Lda(lda_acc, p=case["p"])

    print(f"{pois=}")

    # Verify the projection
    id_pois = np.zeros((case["npois"], case["ns"]), dtype=np.int16)
    for i in range(case["npois"]):
        id_pois[i, pois[0][i]] = 1
    assert (id_pois[:, pois[0]] == np.eye(case["npois"])).all()
    proj_matrix = lda.project(id_pois)[0]
    assert all(
        [is_parallel(a, b) for a, b in zip(proj_matrix.T, lda_ref.projection.T)]
    ), "Projection mismatch"

    probas = lda.predict_proba(traces)
    ref_probas = lda_ref.predict_proba(traces[:, pois[0]])
    assert np.allclose(probas, ref_probas), f"{probas=}\n{ref_probas=}"


#### Pickle test

lda_pickle_unibatch_bc = LdaTestCase(
    ns=10,
    nc=2,
    nv=1,
    npois=4,
    n=500,
    n_batches=1,
    p=1,
    test_lp=True,
    maxl=2**15,
)
lda_pickle_unibatch_cases = [
    lda_ref_uni_bc,
    lda_ref_uni_bc.with_params(nc=4, p=2),
    lda_ref_uni_bc.with_params(nc=4, p=2, nv=4),
]


@pytest.mark.parametrize("case", lda_pickle_unibatch_cases)
def test_lda_pickle_unibatch(case):
    pois, traces, x = case.get_data()
    # Fetch the first batch only
    traces = traces[0]
    x = x[0]
    # LdaAc
    lda = LdaAcc(pois=pois, nc=case["nc"])
    lda.fit_u(traces, x)
    # Reference, check that the pickle is not modifiying the results
    lda_ref = LdaAcc(pois=pois, nc=case["nc"])
    lda_ref.fit_u(traces, x)

    dumped_lda = pickle.dumps(lda)
    lda = pickle.loads(dumped_lda)

    # Validate the matrices, MUST be strictly similar before solve
    for mus, mus_ref in zip(lda.get_mus(), lda_ref.get_mus()):
        assert (mus == mus_ref).all()
    for sb, sb_ref in zip(lda.get_sb(), lda_ref.get_sb()):
        assert (sb == sb_ref).all()
    for sw, sw_ref in zip(lda.get_sw(), lda_ref.get_sw()):
        assert np.allclose(sw, sw_ref)

    # Then, verify pickle writing with the Lda (solved Lda).
    lda_solved = Lda(lda, p=case["p"])
    lda_ref_solved = Lda(lda_ref, p=case["p"])

    dumped_lda_s = pickle.dumps(lda_solved)
    lda_solved = pickle.loads(dumped_lda_s)

    # Verify the projection
    ptraces = lda_solved.project(traces)
    ptraces_ref = lda_ref_solved.project(traces)
    for p, pref in zip(ptraces, ptraces_ref):
        projections_similar = all([is_parallel(a.T, b.T) for a, b in zip(p, pref)])
        assert projections_similar, (ptraces, ptraces_ref)

    # Verify the proba
    lda_prs = lda_solved.predict_proba(traces)
    lda_prs_ref = lda_ref_solved.predict_proba(traces)
    assert np.allclose(lda_prs, lda_prs_ref)


#### Multi vars vs indep single vars
mvars_vs_indepvars_bc = LdaTestCase(
    ns=2,
    nc=2,
    nv=1,
    npois=1,
    n=10,
    n_batches=1,
    p=1,
    test_lp=True,
    maxl=2**15,
)
mvars_vs_indepvars_cases = [
    mvars_vs_indepvars_bc,
    mvars_vs_indepvars_bc.with_params(n=5, n_batches=2),
    mvars_vs_indepvars_bc.with_params(ns=1000, nc=4, nv=10, n=500, n_batches=1),
    mvars_vs_indepvars_bc.with_params(ns=1000, nc=4, nv=10, n=500, n_batches=5),
    mvars_vs_indepvars_bc.with_params(
        ns=20, nc=4, nv=4, n=20, npois=5, n_batches=1, p=2
    ),
    mvars_vs_indepvars_bc.with_params(
        ns=20, nc=4, nv=4, n=20, npois=5, n_batches=3, p=2
    ),
    mvars_vs_indepvars_bc.with_params(
        ns=100, nc=256, nv=2, npois=20, n=5000, n_batches=1, p=4
    ),
    mvars_vs_indepvars_bc.with_params(
        ns=100, nc=256, nv=2, npois=20, n=5000, n_batches=5, p=4
    ),
]


@pytest.mark.parametrize("case", mvars_vs_indepvars_cases)
def test_mvars_lda_compare(case):
    pois, traces, x = case.get_data()
    print(80 * "#" + "\ncase:", case)
    mlda = [LdaAcc(pois=[pois[i]], nc=case["nc"]) for i in range(case["nv"])]
    mlda_acc = LdaAcc(pois=pois, nc=case["nc"])
    # Fit with batches
    for t, y in zip(traces, x):
        for i in range(case["nv"]):
            mlda[i].fit_u(t, y[:, i][:, np.newaxis])
        mlda_acc.fit_u(t, y)
    for mus, mus_acc in zip([lda.get_mus() for lda in mlda], mlda_acc.get_mus()):
        assert np.allclose(mus, mus_acc)
    for sb, sb_acc in zip([lda.get_sb() for lda in mlda], mlda_acc.get_sb()):
        assert np.allclose(sb, sb_acc)
    for sw, sw_acc in zip([lda.get_sw() for lda in mlda], mlda_acc.get_sw()):
        assert np.allclose(sw, sw_acc)
    mlda = [Lda(lda, p=case["p"]) for lda in mlda]
    mlda_acc = Lda(mlda_acc, p=case["p"])
    for t, y in zip(traces, x):
        mlda_acc_prs = mlda_acc.predict_proba(t)
        mlda_prs = [lda.predict_proba(t) for lda in mlda]
        for prs_acc, prs in zip(mlda_acc_prs, mlda_prs):
            assert np.allclose(prs_acc, prs)
        if case["test_lp"]:
            acc_lp = mlda_acc.predict_log2_proba_class(t, y)
            lp = np.log2(
                mlda_acc_prs[
                    np.arange(mlda_acc_prs.shape[0])[:, np.newaxis],
                    np.arange(mlda_acc_prs.shape[1])[np.newaxis, :],
                    y.T,
                ]
            )
            assert np.allclose(acc_lp, lp)


### Multivar with select
mvars_select_bc = LdaTestCase(
    ns=2,
    nc=2,
    nv=1,
    npois=1,
    n=10,
    n_batches=1,
    p=1,
    test_lp=True,
    maxl=2**15,
    n_sel=5,
    nv_sel=1,
)
mvars_select_cases = [
    mvars_select_bc,
    mvars_select_bc.with_params(n_batches=2, nv=10, nv_sel=3),
    mvars_select_bc.with_params(ns=5, n=20, maxl=100, nv=5, nv_sel=2),
    mvars_select_bc.with_params(n=100, nv=5, ns=25, npois=5, nv_sel=2, n_sel=2),
    mvars_select_bc.with_params(n=100, nv=5, ns=25, npois=15, nv_sel=3, n_sel=1),
    mvars_select_bc.with_params(n=100, nv=5, ns=25, npois=15, nv_sel=5, n_sel=1),
    mvars_select_bc.with_params(
        n=100, nv=5, ns=25, npois=15, nv_sel=5, n_sel=1, n_batches=3, p=3
    ),
]


@pytest.mark.parametrize("case", mvars_select_cases)
def test_mvars_select(case, select=None):
    pois, traces, x = case.get_data()
    ref = LdaAcc(pois=pois, nc=case["nc"])
    # Fit with batches
    for t, y in zip(traces, x):
        ref.fit_u(t, y)
    ref = Lda(ref, p=case["p"])
    # Compute the selection
    rng = get_rng()
    for _ in range(case["n_sel"]):
        if select is None:
            selection = list(rng.integers(0, case["nv"], (case["nv_sel"],)))
        else:
            selection = select
        smlda = ref.select_vars(selection)
        for t, y in zip(traces, x):
            # Reference probas
            ref_prs = ref.predict_proba(t)
            # Select probas
            s_prs = smlda.predict_proba(t)
            # Check
            for ref_prs, s_prs in zip(ref_prs[selection], s_prs):
                assert np.allclose(ref_prs, s_prs)
            if case["test_lp"]:
                ref_lp = ref.predict_log2_proba_class(t, y)
                acc_lp = smlda.predict_log2_proba_class(t, y[:, selection])
                assert np.allclose(acc_lp, ref_lp[selection])


### Hardcoded simplified case
def test_handmade_simplified_select():
    pois = [[0, 1], [1, 2]]
    selection = [1, 0]
    case = LdaTestCase(
        ns=3,
        nv=len(pois),
        n=20,
        nc=2,
        n_batches=1,
        npois=len(pois[0]),
        p=1,
        test_lp=True,
        maxl=2**15,
        n_sel=5,
        nv_sel=len(selection),
    ).with_pois(pois)
    test_mvars_select(case, select=selection)


def test_handmade_simplified_mvars():
    case = LdaTestCase(
        ns=5, nc=2, nv=1, npois=2, n=4, n_batches=1, p=1, test_lp=True, maxl=2**15
    )
    pois = [np.array([0, 1])]
    traces = [np.array([[1, 0], [0, 1], [0, 0], [4, 4]], dtype=np.int16)]
    x = [np.array([[0], [0], [1], [1]]).astype(np.uint16)]
    case_wparams = case.with_data(x=x, traces=traces).with_pois(pois=pois)
    test_mvars_lda_compare(case_wparams)


####### API related test (e.g., check errors, ...)
def try_with_expected_error(f, err_type, exp_err_msg):
    try:
        f()
    except err_type as e:
        assert f"{e}" == exp_err_msg, "MSG:\n{}\nINSTEAD OF\n{}\n".format(
            e, exp_err_msg
        )
    else:
        assert False, "Incorrect behavior not handled by scalib..."


def test_simple_format_check():
    maxt = 4092
    n = 1000
    nv = 2
    nc = 16
    npois = 5
    p = 3
    pois = [[i * npois + e for e in range(npois)] for i in range(nv)]
    ns = npois * nv

    # Wrong traces shape
    traces = np.random.randint(0, maxt, (n, npois - 1), dtype=np.int16)
    x = np.random.randint(0, nc, (n, nv), dtype=np.uint16)
    lda_acc = LdaAcc(nc=nc, pois=pois)
    try_with_expected_error(
        lambda: lda_acc.fit_u(traces, x), ScalibError, "POI out of bounds."
    )

    # Wrong labels shape [too much variables]
    traces = np.random.randint(0, maxt, (n, ns), dtype=np.int16)
    x = np.random.randint(0, nc, (n, nv + 1), dtype=np.uint16)
    lda_acc = LdaAcc(nc=nc, pois=pois)
    expected_error_msg = (
        "Number of variables {} does not match  previously-fitted classes ({}).".format(
            nv + 1, nv
        )
    )
    try_with_expected_error(
        lambda: lda_acc.fit_u(traces, x), ValueError, expected_error_msg
    )

    # Wrong labels values.
    # Not tested, would imply significant impact on performances

    # Validate matrices shape
    traces = np.random.randint(0, maxt, (n, ns), dtype=np.int16)
    x = np.random.randint(0, nc, (n, nv), dtype=np.uint16)
    lda_acc = LdaAcc(nc=nc, pois=pois)
    lda_acc.fit_u(traces, x)
    mus = lda_acc.get_mus()
    assert len(mus) == nv
    for mus_e, pois_e in zip(mus, pois):
        assert mus_e.shape == (nc, len(pois_e))

    sw = lda_acc.get_sw()
    assert len(sw) == nv
    for sw_e, pois_e in zip(sw, pois):
        assert sw_e.shape == (len(pois_e), len(pois_e))

    sb = lda_acc.get_sb()
    assert len(sb) == nv
    for sb_e, pois_e in zip(sb, pois):
        assert sb_e.shape == (len(pois_e), len(pois_e))

    # Solve the lda
    lda = Lda(lda_acc, p=p)

    # Wrong new traces for predictions
    new_traces = np.random.randint(0, 256, (20, ns + 1), dtype=np.int16)
    e_err_msg = "Traces length {} does not match previously-fitted traces ({}).".format(
        ns + 1, ns
    )
    try_with_expected_error(
        lambda: lda.predict_proba(new_traces), ValueError, e_err_msg
    )

    # Validate probas shape
    n_new = 20
    new_traces = np.random.randint(0, 256, (n_new, ns), dtype=np.int16)
    pr = lda.predict_proba(new_traces)
    assert len(pr) == nv
    for prs in pr:
        assert prs.shape == (n_new, nc)


################### Deprecated tests
@pytest.mark.parametrize("case", lda_ref_uni_cases)
def test_deprecated_ldaclassifier_sklearn(case):
    pois, btraces, bx = case.get_data()
    # LdaAc
    lda = LDAClassifier(case["nc"], case["p"])
    for bi, (traces, x) in enumerate(zip(btraces, bx)):
        lda.fit_u(traces[:, pois[0]], x[:, 0], 0)
    # LDARef
    traces = np.vstack(btraces)
    x = np.vstack(bx)
    lda_ref = RefLda(traces[:, pois[0]], x[:, 0], nc=case["nc"], p=case["p"])

    # Verify value of means by accessor
    lda_means = lda.get_mus()
    assert np.allclose(
        lda_ref.mus, lda_means
    ), "Mean mismatch\nmu_ref:=\n{}\nmu_mean:=\n{}".format(lda_ref.mus, lda_means)

    s_b = lda.get_sb()
    s_w = lda.get_sw()
    assert np.allclose(s_b, lda_ref.s_b), f"SB mismatch\n{s_b=}\n{lda_ref.s_b=}"
    assert np.allclose(s_w, lda_ref.s_w), f"SB mismatch\n{s_w=}\n{lda_ref.s_w=}"

    # Solve
    lda.solve(False)

    # Verify projection
    # Project traces with SCALib
    from scalib.config import get_config

    ptraces = lda.mlda.project(traces[:, pois[0]], get_config())
    # Project traces woth sklearn
    ptraces_sklearn = lda_ref.project(traces[:, pois[0]])
    projections_similar = all(
        [is_parallel(a, b) for a, b in zip(ptraces[0].T, ptraces_sklearn.T)]
    )
    assert projections_similar, "Projection mismatch"

    # Validate the probabilities
    probas = lda.predict_proba(traces[:, pois[0]])
    ref_probas = lda_ref.predict_proba(traces[:, pois[0]])
    assert np.allclose(probas, ref_probas), f"{probas=}\n{ref_probas=}"


@pytest.mark.parametrize("case", mvars_vs_indepvars_cases)
def test_deprecated_multi_lda_compare(case):
    pois, traces, x = case.get_data()
    ncs = [case["nc"] for _ in range(case["nv"])]
    ps = [case["p"] for _ in range(case["nv"])]
    multi_lda = MultiLDA(ncs, ps, pois=pois)
    multi_lda3 = LdaAcc(pois=pois, nc=case["nc"])
    for t, y in zip(traces, x):
        multi_lda.fit_u(t, y)
        multi_lda3.fit_u(t, y)
    for mus, mus3 in zip(multi_lda.get_mus(), multi_lda3.get_mus()):
        assert np.allclose(mus, mus3)
    for sb, sb3 in zip(multi_lda.get_sb(), multi_lda3.get_sb()):
        assert np.allclose(sb, sb3)
    for sw, sw3 in zip(multi_lda.get_sw(), multi_lda3.get_sw()):
        assert np.allclose(sw, sw3)
    multi_lda.solve(done=False)
    multi_lda3 = Lda(multi_lda3, p=case["p"])
    for t, y in zip(traces, x):
        probas = np.array(multi_lda.predict_proba(t))
        probas3 = multi_lda3.predict_proba(t)
        assert np.allclose(probas, probas3)
        if case["test_lp"]:
            lp = multi_lda3.predict_log2_proba_class(t, y)
            lp3 = np.log2(
                probas3[
                    np.arange(probas3.shape[0])[:, np.newaxis],
                    np.arange(probas3.shape[1])[np.newaxis, :],
                    y.T,
                ]
            )
            assert np.allclose(lp, lp3)
