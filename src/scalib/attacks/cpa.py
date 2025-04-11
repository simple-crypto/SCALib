from typing import TypeAlias

import numpy as np
import numpy.typing as npt

from scalib import _scalib_ext
from scalib.config import get_config
import scalib.utils


class Cpa:
    r"""Performs a Correlation Power Attacks (CPA) by computing the pearson
    correlation between the traces associated to a specific intermediate state
    and an arbitrary leakage model [BrierCO04]_ . The correlation metric is computed
    over a range of key guesses, such that the key value maximising the
    correlation absolute value is considered as the correct key guess.

    The intermediate state :math:`y \in [0; nc[` is modelled as a function of the value :math:`x \in [0; nc[` and the key guess :math:`k_g \in [0; nc[` such that :math:`y=\text{intermediate}(x, k_g)`. Currently, two intermediate functions are supported: the bitwise xor and the addition modulo `nc`.

    The correlation metric between the leakages samples :math:`L_x` and their models :math:`M_y` is computed according to following equation:

    .. math::
        \mathrm{\hat{\rho}(L_x;M_y)} = \dfrac{\hat{\text{cov}}\left( L_x;M_y\right)}{\hat{\sigma}_{L_x}\sigma_{M_y}}

    where

    :math:`\hat{\text{cov}}\left( L_x;M_y\right)` :
        is the unbiased estimation of the covariance between the leakage and the models,
    :math:`\hat{\sigma}_{L_x}` :
        is the unbiased estimation of the leakages samples standard deviation.
    :math:`\sigma_{M_y}` :
        is the exact value of the model standard deviation, computed over the exhaustive model distribution provided.


    Parameters
    ----------
    nc : int
        Number of possible values for the random variable :math:`X` (e.g., 256 for 8-bit
        target). ``nc`` must be between :math:`2` and :math:`2^{16}` (included).
    kind:
        Addition function between key and label (``Cpa.Xor``, ``Cpa.Add``).
    use_64bit : bool (default False)
        Use 64 bits for intermediate sums instead of 32 bits.
        When using 64-bit sums, SNR can accumulate up to :math:`2^{32}` traces, while when
        32-bit sums are used, the bound is :math:`n_i < 2^{32}/b`, where b is the
        maximum absolute value of a sample rounded to the next power of 2, and
        :math:`n_i` is the maximum number of times a variable can take a given value.
        Concretely, the total number of traces `n` should be at most
        :math:`(nc \cdot 2^{32}/b) - k` , where :math:`k = O(\sqrt{n})`, typ.
        :math:`k>=3*\sqrt{n}`  (see https://mathoverflow.net/a/273060).

    Examples
    --------
    >>> from scalib.attacks import Cpa
    >>> import numpy as np
    >>> # 500 traces of 200 points, 8-bit samples
    >>> traces = np.random.randint(0,256,(500,200),dtype=np.int16)
    >>> # 10 variables on 8 bit (256 classes = 2^8)
    >>> x = np.random.randint(0,256,(500,10),dtype=np.uint16)
    >>> cpa = Cpa(nc=256, kind=Cpa.Xor)
    >>> cpa.fit_u(traces,x)
    >>> hamming_weights = np.bitwise_count(np.arange(256)).astype(np.float64)
    >>> models = np.tile(hamming_weights[np.newaxis,:,np.newaxis], (10, 1, 200))
    >>> cpa_val = cpa.get_correlation(models)

    Notes
    -----
    .. [BrierCO04] "Correlation Power Analysis with a Leakage Model", Eric Brier, Christophe Clavier, Francis Olivier, CHES 2004: 16-29

    """

    IntermediateKind: TypeAlias = _scalib_ext.CpaIntermediateKind
    Xor = IntermediateKind.Xor
    Add = IntermediateKind.Add

    def __init__(self, nc: int, kind: IntermediateKind, use_64bit: bool = False):
        if nc not in range(2, 2**16 + 1):
            raise ValueError(
                "CPA can be computed on max 16 bit variable (and at least 2 classes),"
                f" {nc=} given."
            )
        self._nc = nc
        assert isinstance(kind, self.IntermediateKind)
        self._kind = kind
        self._ns = None
        self._nv = None
        self._use_64bit = use_64bit
        self._init = False

    def fit_u(self, traces: npt.NDArray[np.int16], x: npt.NDArray[np.uint16]):
        r"""Updates the CPA with samples of `traces` for the classes `x`.
        This method may be called multiple times.

        Parameters
        ----------
        traces :
            Array that contains the leakage traces. The array must be of
            dimension ``(n, ns)``.
        x :
            Labels for each trace. Must be of shape ``(n, nv)``.
        """
        traces = scalib.utils.clean_traces(traces, self._ns)
        x = scalib.utils.clean_labels(x, self._nv)
        if not self._init:
            self._init = True
            self._ns = traces.shape[1]
            self._nv = x.shape[1]
            self._cpa = _scalib_ext.CPA(self._nc, self._ns, self._nv, self._use_64bit)
        if x.shape[0] != traces.shape[0]:
            raise ValueError(
                f"Number of traces {traces.shape[0]} does not match size of classes array {x.shape[0]}."
            )
        # _scalib_ext uses inverted axes for x.
        # we can copy when needed, as x should be small, so this should be cheap
        x = np.ascontiguousarray(x.transpose())
        with scalib.utils.interruptible():
            self._cpa.update(traces, x, get_config())

    def get_correlation(
        self,
        models: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        r"""
        Compute the correlation metric based on the fitted state, for a given model. More into the details, the later consists in an arbitrarily chosen value per leakage sample, associated to each class of every variable. The correlation is computed for every key guess assumptions.

        Parameters
        ----------
        models :
            Array that contains the leakage models. The array must be of shape ``(nv, nc, ns)`` and is formatted such that the element at location ``[i,j,k]`` is the leakage model for ``k``-th leakage sample associated to the ``j``-th class of the intermediate state for the ``i``-th variable.

        Returns
        -------
        Correlations as an array of shape ``(nv, nc, ns)``, such that the element at location ``[i,j,k]`` is the correlation computed for the ``k``-th leakage sample of the ``i``-th variable, under the assumption that the key guess ``j`` is used, .
        """
        if not self._init:
            raise ValueError("Need to call .fit_u at least once.")
        with scalib.utils.interruptible():
            return self._cpa.compute_cpa(models, self._kind, get_config())
