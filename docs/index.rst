Welcome to SCALib
=================
The Side-Channel Analysis Library (SCALib) is a Python package that
contains state-of-the-art tools for side-channel evaluation. It focuses on
providing efficient implementations of analysis methods widely used by the
side-channel community and maintaining a flexible and simple interface.

SCALib is on `GitHub <https://github.com/simple-crypto/SCALib>`_!

Usability & Efficiency
----------------------
SCALib main characteristics are:

1. **High Performances**: Under its Python interface, most of SCALib
   functionality is implemented in optimized and highly parallel Rust code.
2. **Flexible & Simple Interface**: SCALib is a simple library. It provides a
   simple `numpy`-based interface, therefore it is simple to use (see `examples
   <https://github.com/simple-crypto/scalib/tree/main/examples>`_) while giving
   you freedom: you can simply call it in any Python workflow.
3. **Streaming APIs**: Most SCALib APIs allow for incremenal processing of chunks of data.
   This enables streaming implementations: with large datasets, no neeed to load everything at once of load multiple times.
   You don't even need to store datasets: you can compute on-the-fly.

Available features
------------------
SCALib contains various features for side-channel analysis:

- :mod:`scalib.metrics`:

  - Signal-to-noise ratio (:class:`scalib.metrics.SNR`).
  - Uni- and Multi-variate, arbitrary-order T-test estimation (:class:`scalib.metrics.Ttest` and :class:`scalib.metrics.MTtest`).

- :mod:`scalib.modeling`: 

  - Templates in linear subspaces (:class:`scalib.modeling.LDAClassifier`).

- :mod:`scalib.attacks`:

  - Generalization of "Divide & Conquer" with Soft Analytical Attacks (:class:`SASCA <scalib.attacks.FactorGraph>`).

- :mod:`scalib.postprocessing`:

  - Full key rank estimation.

Getting started
===============

Install
-------

See the `README <https://github.com/simple-crypto/SCALib>`_. TL;DR:

.. code-block::

   pip install scalib

Examples
--------

See our `examples <https://github.com/simple-crypto/scalib/tree/main/examples>`_
and a more complete `attack on ASCADv1 <https://github.com/cassiersg/ASCAD-5minutes>`_.

Where is SCALib used ?
======================

We strongly appreciate if you could mention to us your usage SCALib 
for concrete projects so that we can add you to the lists below. Please send 
an email to Olivier Bronchain and Gaëtan Cassiers, or directly
`open a pull request <https://github.com/simple-crypto/SCALib/edit/main/docs/index.rst>`_.

Scientific publications
-----------------------

SCALib has been used in various scientific publications. Here is a (non-exhaustive) list:

1. "Mode-Level vs. Implementation-Level Physical Security in Symmetric
   Cryptography: A Practical Guide Through the Leakage-Resistance Jungle", D.
   Bellizia, O. Bronchain, G. Cassiers, V. Grosso, Chun Guo, C. Momin, O.
   Pereira, T. Peters, F.-X. Standaert at CRYPTO2020.
2. "Exploring Crypto-Physical Dark Matter and Learning with Physical Rounding
   Towards Secure and Efficient Fresh Re-Keying", S. Duval, P. Méaux, C. Momin,
   F.-X. Standaert in TCHES2021 - Issue 1.
3. "Breaking Masked Implementations with Many Shares on 32-bit Software
   Platforms or When the Security Order Does Not Matter". O. Bronchain, F.-X.
   Standaert in TCHES2021 - Issue 3.
4. "Improved Leakage-Resistant Authenticated Encryption based on Hardware AES
   Coprocessors". O. Bronchain, C. Momin, T. Peters, F.-X. Standaert in
   TCHES2021 - Issue 3.
5. "Riding the Waves Towards Generic Single-Cycle Masking in Hardware". R.
   Nagpal, B. Girgel, R. Primas, S. Mangard, eprint 2022/505.
6. "Bitslice Masking and Improved Shuffling: How and When to Mix Them in
   Software?". M. Azouaoui, O. Bronchain, V. Grosso, K.  Papagiannopoulos,
   F.-X.  Standaert, TCHES2022 - Issue 2.
7. "A Second Look at the ASCAD Databases", M. Egger, T. Schamberger, L.
   Tebelmann, F. Lippert, G. Sigl, COSADE 2022. 
8. "Give Me 5 Minutes: Attacking ASCAD with a Single Side-Channel Trace". O.
   Bronchain, G. Cassiers, F.-X. Standaert, eprint 2021/817. 
9. "Towards a Better Understanding of Side-Channel Analysis Measurements
   Setups". D. Bellizia, B. Udvarhelyi, F.-X. Standaert, CARDIS 2021. 
10. "A Finer-Grain Analysis of the Leakage (Non) Resilience of OCB". F. Berti,
    S. Bhasin, J. Breier, X. Hou, R. Poussier, F.-X. Standaert, B. Udvarhelyi
    in TCHES2022 - Issue 1. 

Concrete evaluations
--------------------

1. `CHES 2020 CTF <https://github.com/obronchain/BS21_ches2020CTF>`_ published in TCHES2021.
2. `Attack against ASCAD <https://github.com/cassiersg/ASCAD-5minutes>`_ eprint 2021/817.
3. `TVLA On Selected NIST LWC Finalists <https://cryptography.gmu.edu/athena/LWC/Reports/TUGraz/TUGraz_Report_HW_5_candidates_RUB.pdf>`_.

About us
========
SCALib was initiated by Olivier Bronchain and Gaëtan Cassiers during their PhD
at UCLouvain. It is now developed as a project of
`SIMPLE-Crypto <https://www.simple-crypto.dev/>`_ and maintained by Gaëtan Cassiers.


.. toctree::
    :hidden:

    self

.. toctree::
    :caption: API Reference
    :hidden:

    source/api_ref.rst

.. toctree::
   :caption: Development
   :hidden:

   source/changelog.rst
   source/contributing.rst
   source/copyright.rst
