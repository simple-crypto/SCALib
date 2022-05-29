T-tests
-------
The Student's :math:`t`-test can be used to highlight a difference in the means
of two distributions. To do so, a `t` statistic is derived following the
expression:

.. math::
    t = \frac{\mu_0 - \mu_1}{\sqrt{\frac{v_0}{n_0} + \frac{v_1}{n_1}}}

where :math:`\mu_0` (resp. :math:`\mu_1`) is the estimated moment of the first
(resp.second) population and :math:`\frac{v_0}{n_0}` the variance of its
estimate from :math:`n_0` samples. In the context of side-channel analysis, many of
these statistical tests are performed independently for each point of the traces. 
See [1]_ for additional details.

In this module, the definition of :math:`\mu` and :math:`v` are adapted to perform
univariate and multivariate :math:`t`-test to compare higher-order moments of
two distributions.

**Warning**: Ttest should not be used alone as a standalone evaluation tool
because of its qualitative nature. See [2]_ and [3]_ for cautionary notes.


T-test Modules
^^^^^^^^^^^^^^

.. automodule:: scalib.metrics.ttest
   :members: Ttest, MTtest

Implementations Details
^^^^^^^^^^^^^^^^^^^^^^^
In order to enable both efficient one-core and parallelized performance of the
:math:`t`-test implementation, SCALib uses the one-pass formula for estimation
arbitrary order statistical moment from [4]_ and its application to
side-channel context in [1]_.

Concretely, the implementations first compute an estimation of the required
statistical moments using a two-passes algorithms (first pass to compute the
mean and the variances, and a second pass to compute the centered products).
This new estimation is then used to update the current estimation using the
merging rule from [4]_. To enable multi-threading, SCALib internally divides 
the fresh traces into smaller independent chunks and then merges the output of
each threads using [4]_. 

As a conclusion, the performance of the SCALib improves if the two-passes
algorithm can be used on large chunks. Hence, it is recommended to feed a large
enough amount of data for every call to `fit_u()`. 

References
^^^^^^^^^^
.. [1] "Leakage assessment methodology", Tobias Schneider, Amir Moradi, CHES
   2015
.. [2] "How (not) to Use Welch’s T-test in Side-Channel Security
   Evaluations", François-Xavier Standaert, CARDIS 2018
.. [3] "A Critical Analysis of ISO 17825 ('Testing Methods for the
   Mitigation of Non-invasive Attack Classes Against Cryptographic
   Modules')", Carolyn Whitnall, Elisabeth Oswald, ASIACRYPT 2019
.. [4] "Formulas for Robust, One-Pass Parallel Computation of Covariances and
    Arbitrary-Order Statistical Moments", Philippe Pébay, 2008

