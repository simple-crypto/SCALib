T-tests
=======
The Student's :math:`t`-test can be used to highlight a difference in the means
of two distributions. To do so, a `t` statistic is derived following the
expression:

.. math::
    t = \frac{\mu_0 - \mu_1}{\sqrt{\frac{v_0}{n_0} + \frac{v_1}{n_1}}}

where :math:`\mu_0` (resp. :math:`\mu_1`) is the estimated moment of the first
(resp.second) population and :math:`\frac{v_0}{n_0}` the variance of its
estimate from :math:`n_0` samples. In the context of side-channel analysis, many of
these statistical tests are performed independently. See [1]_ for additional
details.

In this module, the definition of :math:`\mu` and :math:`v` are adapted to perform
univariate and multivariate :math:`t`-test to compare higher-order moments of
two distributions.

Notes
-----
**Warning**: Ttest should not be used alone as a standalone evaluation tool
because of its qualitative nature. See [2]_ and [3]_ for cautionary notes.

.. [1] "Leakage assessment methodology", Tobias Schneider, Amir Moradi, CHES
   2015
.. [2] "How (not) to Use Welch’s T-test in Side-Channel Security
   Evaluations", François-Xavier Standaert, CARDIS 2018
.. [3] "A Critical Analysis of ISO 17825 ('Testing Methods for the
   Mitigation of Non-invasive Attack Classes Against Cryptographic
   Modules')", Carolyn Whitnall, Elisabeth Oswald, ASIACRYPT 2019


.. automodule:: scalib.metrics.ttest
   :members:

