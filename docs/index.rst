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

See the :ref:`papers that use SCAlib<papers>`.

Complete security evaluations:

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
    source/papers.rst

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
