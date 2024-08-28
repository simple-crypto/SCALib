from typing import Tuple

import numpy as np
import numpy.typing as npt

from scalib import _scalib_ext
from scalib.config import get_config
from scalib.modeling import RLDAClassifier


class RLDAInformationEstimator:
    r"""Amount of information that can be extracted from leakage with a RLDA model.

    This class can be used to compute Perceived Information (PI) or Training
    Information (TI) :footcite:p:`InfoBounds` bounds of a RLDA model
    :footcite:p:`RLDA` for a variable.

    It estimates the information using

    .. math::
        \hat{\mathrm{I}}(X,\mathbf{L}) = \mathrm{H}(X) + \frac{1}{|\mathcal{L}'|}
        \sum_{x \in \mathcal{X}} \sum_{\mathbf{l}\in \mathcal{L}'_x} \log_2 \hat{\mathsf{p}}[x|\mathbf{l}]

    PI or TI is obtained depending on the validation set used for the estimation.

    In order to obtain the PI : use a validation set independent of the training set of the model.

    In order to obtain the TI : use as validation set the full training set of the model.

    This object assumes that target variable is uniform
    and uses a clustered version of an RLDA model for faster computation,
    leading to bounds instead of exact values. If the bounds are not tight
    enough, change the parameters of the clustered model.

    The validation set is provided to this object by calling :meth:`fit_u`,
    and the estimated information is recovered with :meth:`get_information`.
    The estimator is further subject some variance due to the sampling of the
    validation set. The estimated standard deviation can be obtained with :meth:`get_deviation`.

    Examples
    --------

    >>> from scalib.modeling import RLDAClassifier
    >>> from scalib.metrics import RLDAInformationEstimator
    >>> import numpy as np
    >>> traces_model = np.random.randint(0,256,(5000,10),dtype=np.int16)
    >>> labels_model = np.random.randint(0,256,(5000,1),dtype=np.uint64)
    >>> rlda = RLDAClassifier(8, 3)
    >>> rlda.fit_u(traces_model, labels_model)
    >>> rlda.solve()
    >>> cl = rlda.get_clustered_model(0,0.5,100, False)
    >>> it = RLDAInformationEstimator(cl, 0)
    >>> traces_test = np.random.randint(0,256,(5000,10),dtype=np.int16)
    >>> labels_test = np.random.randint(0,256,(5000,),dtype=np.uint64)
    >>> it.fit_u(traces_test, labels_test)
    >>> pi_l,pi_u = it.get_information()

    See Also
    --------

    scalib.modeling.RLDAClassifier.get_clustered_model: Clustered model to use with ``RLDAInformationEstimator``.

    References
    ----------

    .. footbibliography::
    """

    def __init__(self, model: RLDAClassifier.ClusteredModel, max_popped_classes: int):
        r"""
        Parameters
        ----------
        model
            The clustered model, it is obtained by calling `get_clustered_model()` on an RLDA object.
        max_popped_classes
            Number of classes that are calculated without using the associated cluster. Corresponds to parameter :math:`\zeta` in :footcite:p:`RLDA`.
        """
        self._inner = _scalib_ext.ItEstimator(model._inner, max_popped_classes)

    def fit_u(self, traces: npt.NDArray[np.int16], x: npt.NDArray[np.uint64]):
        """Updates the estimatore with the given traces and the corresponding labels.

        This can be called multiple times with parts of the dataset: the state is accumulated.

        Parameters
        ----------
        traces : array_like, int16
            Array that contains the traces. Shape: ``(n,ns)``.
        x : array_like, uint64
            Label for each trace. Shape ``(n,)``.
        """
        assert traces.shape[0] == x.shape[0]
        self._inner.fit_u(traces, x, get_config())

    def get_information(
        self,
    ) -> Tuple[float, float]:
        """Returns the lower and upper bound information estimation."""
        return self._inner.get_information()

    def get_deviation(self) -> Tuple[float, float, int]:
        """Compute the approximate deviation of the information estimator for the lower and upper bound.
        Returns the deviation of the lower bound, the upper bound, and the number of traces used for the estimation.
        """
        raise self._inner.get_deviation()
