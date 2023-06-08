import numpy as np
import numpy.typing as npt

from scalib import _scalib_ext
from scalib.config import get_config


class RLDAClassifier:
    r"""Regression-based Linear Discriminant Analysis.

    Models the leakage using a regression-based linear discriminant analysis
    (RLDA) classifier :footcite:p:`RLDA`, which can efficiently handle long
    traces and large number of classes.

    In a nutshell, this model performs LDA with the class means modelled as
    linear regression based on the :math:`n_b` bits of the class value.
    Compared to the :class:`scalib.modeling.LDAClassifier`, this model will
    perform better when the number of classes is large and/or there are few
    profiling traces.

    Internally, it first estimates the coefficients of the regression then estimates
    the projection matrix to reduce the dimensionality of the gaussian template to
    :math:`p` dimensions. A second projection is applied such that the covariance
    matrix is identity.

    It is then able to predict the leakage likelihood

    .. math::
        \hat{\mathsf{f}}[\mathbf{l}|X=x] =
        \alpha
        \exp\left(
        -\frac{1}{2} \lVert\mathbf{W}^T\mathbf{l} - \mathbf{A}\mathbf\beta(x)\rVert^2
        \right).

    Where :math:`\mathbf{W}` is the projection matrix, :math:`\mathbf{A}` the projected
    regression coefficients, and :math:`\mathbf{\beta(x)}` the coefficients of :math:`x`.
    The parameter :math:`\alpha = 1/\sqrt{(2\pi)^p\lvert\hat\Sigma_\mathbf{W}}\rvert` does
    not need to be calculated as it will get canceled out when applying Bayes' law.

    :class:`RLDAClassifier` provides the probability for each of the :math:`2^{n_b}`
    classes with :meth:`predict_proba`.

    Examples
    --------

    >>> from scalib.modeling import RLDAClassifier
    >>> import numpy as np
    >>> traces_model = np.random.randint(0,256,(5000,10),dtype=np.int16)
    >>> labels_model = np.random.randint(0,256,(5000,1),dtype=np.uint64)
    >>> rlda = RLDAClassifier(8, 10, 1, 3)
    >>> rlda.fit_u(traces_model, labels_model)
    >>> rlda.solve()
    >>> traces_test = np.random.randint(0,256,(5000,10),dtype=np.int16)
    >>> prs = rlda.predict_proba(traces_test, 0)

    References
    ----------

    .. footbibliography::
    """

    def __init__(self, nb: int, ns: int, nv: int, p: int):
        """
        Parameters
        ----------
        nb:
            Number of bits of the profiled variables.
        ns:
            Number of dimensions in the leakage (trace length).
        nv:
            Number of variables to profile
        p:
            Number of dimensions in the linear subspace.
        """
        self._ns = ns
        self._nv = nv
        self._inner = _scalib_ext.RLDA(nb, ns, nv, p)
        self._solved = False

    def fit_u(self, l: npt.NDArray[np.int16], x: npt.NDArray[np.uint64], gemm_mode=1):
        """Update statistical model estimates with additional data.

        This can be called multiple times, the state is accumulated.

        Parameters
        ----------
        l : array_like, int16
            Array that contains the traces. Shape `(n,ns)`.
        x : array_like, uint64
            Labels for each trace. Shape `(n,nv)`.
        """
        assert l.shape[0] == x.shape[0]
        assert l.shape[1] == self._ns
        assert x.shape[1] == self._nv

        self._inner.update(l, x.T, gemm_mode, get_config())

    def solve(self):
        """Solve the RLDA equations.

        Notes
        -----
        Once this has been called, predictions can be performed.
        """
        self._inner.solve(get_config())
        self._solved = True

    def get_proj(self) -> npt.NDArray[np.float64]:
        """Returns the projection matrix.

        Returns
        -------
        array_like, float64
            The projection coefficients. Shape (nv,p,ns)."""
        return self._inner.get_norm_proj()

    def get_proj_coefs(
        self,
    ) -> npt.NDArray[np.float64]:
        """The projected regression coefficients.

        Returns
        -------
        array_like, float64
            The projection coefficients. Shape (nv,p,nb+1).
        """
        return self._inner.get_proj_coefs()

    def predict_proba(
        self, l: npt.NDArray[np.int16], var: int
    ) -> npt.NDArray[np.float64]:
        r"""Computes the probability for each of the classes for the traces
        contained in `l`.

        Parameters
        ----------
        l:
            Array that contains the traces. Shape ``(n,ns)``.
        var:
            Id (position in the ``x`` array) of the variable for which the
            probabilities are computed.

        Returns
        -------
        array_like, f64
            Probabilities. Shape `(n, nc)`.
        """
        assert self._solved, "Model not solved"
        return self._inner.predict_proba(l, var, get_config())

    class ClusteredModel:
        """Clustered RLDA model, see :func:`RLDAClassifier.get_clustered_model`."""

        pass

    def get_clustered_model(
        self,
        var: int,
        t: int,
        max_clusters: int = 10_000_000,
        store_associated_classes: bool = True,
    ) -> ClusteredModel:
        """Generate a simplified model for faster estimation of the information content in this model.

        This generates a model with clustered means that can be used to
        estimate the percevied or training information of the model. It applies
        a clustering method on the classes to regroup the closest ones up to a
        threshold distance :math:`t`.
        Internally, it uses a Kd-tree data structure to find the nearest cluster efficiently.
        Details on the clustering algorithm can be found in [1].

        The resulting model can be used with
        :class:`scalib.metrics.RLDAInformationEstimator` (see there for usage
        example).

        Parameters
        ----------
        var:
            Id (position in the ``x`` array) of the variable for which the
            probabilities are computed.
        t:
            Maximum distance between 2 cluster centers. This is a trade-off parameter between
            the tightness of the information bounds (lower value of t) and
            computation (time and memory) efficiency (higher value of t).
        max_clusters:
            The maximum number of clusters that can be generated. If during generation, this
            limit is exceeded, an exception is raised.
        store_associated_classes : bool
            If True, the generated model stores the classes associated to each cluster. This
            allows refining the information bounds by calculating using the exact class mean (and not the
            centroid it is associated to) for clusters that contribute the most to an untight bound.
            Note that this option requires significantly more RAM for high values of :math:`n_b`.

        Returns
        -------
        ClusteredModel
            A clustered model to be used in :class:`scalib.metrics.RLDAInformationEstimator`
        """
        res = self.ClusteredModel()
        res._inner = self._inner.get_clustered_model(
            var, store_associated_classes, t, max_clusters
        )
        return res
