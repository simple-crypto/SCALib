import pickle
import typing

import pytest
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA_sklearn
import numpy as np
import scipy.stats

from scalib import ScalibError
from scalib.modeling import LDAClassifier, MultiLDA, Lda, LdaAcc

from utils_test import get_rng

import copy

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


def is_parallel(x, y):
    z = np.abs(x.dot(y))
    z2 = np.linalg.norm(x) * np.linalg.norm(y)
    return np.allclose(z, z2, rtol=1e-3)


# 1. Test that univariate LDA results in similar results that the one from sklearn.
lda_sklearn_uni_bc = LdaTestCase(
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
lda_sklearn_uni_cases = [
    lda_sklearn_uni_bc,
    lda_sklearn_uni_bc.with_params(nc=4),
    lda_sklearn_uni_bc.with_params(nc=4, p=2),
]


@pytest.mark.parametrize("case", lda_sklearn_uni_cases)
def test_univariate_lda_sklearn(case):
    pois, traces, x = case.get_data()
    # LdaAc
    lda = LdaAcc(pois=pois, nc=case["nc"])
    lda.fit_u(traces[0], x[0])
    # LDARef
    lda_ref = LDA_sklearn(solver="eigen", n_components=case["p"])
    lda_ref.fit(traces[0][:, pois[0]], x[0])

    # Verify value of means by accessor
    ref_means = lda_ref.means_
    lda_means = lda.get_mus()[0]
    assert np.allclose(ref_means, lda_means, rtol=1e-10)

    # Verify scatter matrix by accessor
    lda_scat = lda.get_sw()[0] / case["n"]
    cov_ref = lda_ref.covariance_
    assert np.allclose(lda_scat, cov_ref, rtol=1e-5)

    # Not point of comparison with sklearn, but we here call
    # the accessor for the inter-class scatter matrix just to verify
    # that its working.
    smat = lda.get_sb()

    # Solve the LdaAcc
    lda = Lda(lda, p=case["p"])

    # Verify the projection
    ptraces = lda.project(traces[0])[0]
    ptraces_sklearn = lda_ref.transform(traces[0][:, pois[0]])
    projections_similar = all(
        [is_parallel(a, b) for a, b in zip(ptraces, ptraces_sklearn)]
    )
    assert projections_similar, (ptraces, ptraces_sklearn)

    # We can't do much more since sklearn has no way to reduce dimensionality of LDA.
    # e.g., comparing probas will fail -> TODO


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
    lda_sklearn_uni_bc,
    lda_sklearn_uni_bc.with_params(nc=4, p=2),
    lda_sklearn_uni_bc.with_params(nc=4, p=2, nv=4),
]


@pytest.mark.parametrize("case", lda_pickle_unibatch_cases)
def test_lda_pickle_unibatch(case):
    pois, traces, x = case.get_data()
    # LdaAc
    lda = LdaAcc(pois=pois, nc=case["nc"])
    lda.fit_u(traces[0], x[0])
    # Reference, check that the pickle is not modifiying the results
    lda_ref = LdaAcc(pois=pois, nc=case["nc"])
    lda_ref.fit_u(traces[0], x[0])

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
    ptraces = lda_solved.project(traces[0])
    ptraces_ref = lda_ref_solved.project(traces[0])
    for p, pref in zip(ptraces, ptraces_ref):
        projections_similar = all([is_parallel(a.T, b.T) for a, b in zip(p, pref)])
        assert projections_similar, (ptraces, ptraces_ref)

    # Verify the proba
    lda_prs = lda_solved.predict_proba(traces[0])
    lda_prs_ref = lda_ref_solved.predict_proba(traces[0])
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

######################## OLD
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
    mvars_select_bc.with_params(ns=5, n=20, maxl=100, n_batches=1, nv=5, nv_sel=2),
    mvars_select_bc.with_params(n=100, nv=5, ns=25, npois=5, nv_sel=2, n_sel=2),
    mvars_select_bc.with_params(n=100, nv=5, ns=25, npois=15, nv_sel=3, n_sel=1),
    mvars_select_bc.with_params(n=100, nv=5, ns=25, npois=15, nv_sel=5, n_sel=1),
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


####### Deprecated tests
@pytest.mark.parametrize("case", lda_sklearn_uni_cases)
def test_deprecated_ldaclassifier_sklearn(case):
    pois, traces, x = case.get_data()
    # LdaAc
    lda = LDAClassifier(case["nc"], case["p"])
    lda.fit_u(traces[0][:, pois[0]], x[0][:, 0], 0)
    lda.solve(False)
    # LDARef
    lda_ref = LDA_sklearn(solver="eigen", n_components=case["p"])
    lda_ref.fit(traces[0][:, pois[0]], x[0][:, 0])

    # Verify value of means by accessor
    ref_means = lda_ref.means_
    lda_means = lda.get_mus()
    assert np.allclose(ref_means, lda_means, rtol=1e-10), "Mean mismatch"

    # Verify scatter matrix by accessor
    lda_scat = lda.get_sw() / case["n"]
    cov_ref = lda_ref.covariance_
    assert np.allclose(lda_scat, cov_ref, rtol=1e-5), "Scatter mismatch"

    # Not point of comparison with sklearn, but we here call
    # the accessor for the inter-class scatter matrix just to verify
    # that its working.
    smat = lda.get_sb()

    # Verify projection
    # Project traces with SCALib
    from scalib.config import get_config

    ptraces = lda.mlda.project(traces[0][:, pois[0]], get_config())
    # Project traces woth sklearn
    ptraces_sklearn = lda_ref.transform(traces[0][:, pois[0]])
    projections_similar = all(
        [is_parallel(a, b) for a, b in zip(ptraces[0].T, ptraces_sklearn.T)]
    )
    assert projections_similar, (ptraces, ptraces_sklearn)

    # We can't do much more since sklearn has no way to reduce dimensionality of LDA.
    # e.g., comparing probas will fail -> TODO


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
