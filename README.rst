Welcome to SCALib
=================

.. image:: https://badge.fury.io/py/scalib.svg
    :target: https://pypi.org/project/scalib/
    :alt: PyPI
.. image:: https://readthedocs.org/projects/scalib/badge/?version=stable
    :target: https://scalib.readthedocs.io/en/stable/
    :alt: Documentation Status
.. image:: https://img.shields.io/matrix/scalib:matrix.org
    :target: https://matrix.to/#/#scalib:matrix.org
    :alt: Matrix room

The Side-Channel Analysis Library (SCALib) is a Python package that
contains state-of-the-art tools for side-channel security evaluation. It focuses on

- simple interface,
- state-of-the art algorithms,
- excellent performance (see `benchmarks <https://github.com/cassiersg/SCABench>`_).

SCALib should be useful for any side-channel practitionner who wants to
evaluate, but not necessarily attack, protected or non-protected
implementations.

See the documentation_ for the list of implemented tools.

We have a `matrix chat <https://matrix.to/#/#scalib:matrix.org>`_ for
announcements, questions, community support and discussions.

.. _documentation: https://scalib.readthedocs.io/en/stable

Install
=======

SCALib is on PyPi! Simple install:

.. code-block::

    pip install scalib

We provide pre-built wheels for any recent python on Linux and Windows (x86).
Be sure to use a **recent pip**.
For other plateforms, this will build SCALib (see below for dependencies).

Local build
-----------

To get **best performance**, you want to build locally (this will optimize
SCALib for your CPU).

**Depdendencies:** You need a C/C++ compiler and the latest stable
release of the `rust toolchain <https://rustup.rs/>`_.

To install from source:

.. code-block::

    git clone https://github.com/simple-crypto/SCALib
    cd SCALib
    pip install .


Usage
=====

See `API documentation <https://scalib.readthedocs.io/en/stable/#available-features>`_,
`example <https://github.com/simple-crypto/scalib/tree/main/examples>`_ and
`real-world usages <https://scalib.readthedocs.io/en/stable/#concrete-evaluations>`_.


Alternatives
============

If your needs are not covered by SCALib, you might be more lucky with 
`lascar <https://github.com/Ledger-Donjon/lascar>`_ or `scared <https://gitlab.com/eshard/scared>`_.

Please also let us know your needs by opening a 
`feature request <https://github.com/simple-crypto/SCALib/issues/new?assignees=&labels=&template=feature_request.md&title=>`_.

Versioning policy
=================

SCALib uses `semantic versioing <https://semver.org/>`_, see the `CHANGELOG
<CHANGELOG.rst>`_ for breaking changes and novelties.

About us
========
SCALib was initiated by Olivier Bronchain and Gaëtan Cassiers during their PhD
at UCLouvain. It is now developed as a project of
`SIMPLE-Crypto <https://www.simple-crypto.dev/>`_ and maintained by Gaëtan Cassiers (@cassiersg).

Contributions and Issues
========================

Contributions welcome !

Please file a **bug report** for any issue you encounter (even bad documentation is
a bug !), and let us know your **suggestions** (open a `github issue
<https://github.com/simple-crypto/SCALib/issues/new/choose>`_, email works too).
We also welcome code contributions, see `CONTRIBUTING.rst <CONTRIBUTING.rst>`_.

You can also come discuss on `matrix <https://matrix.to/#/#scalib:matrix.org>`_.

All code contributions are subject to the Contributor License Agreement (`CLA
<https://www.simple-crypto.dev/organization>`_) of SIMPLE-Crypto, which ensures
a thriving future for open-source hardware security.

License
=======
This project is licensed under GNU AFFERO GENERAL PUBLIC LICENSE, Version 3.
See `COPYRIGHT <COPYRIGHT>`_ and `COPYING <COPYING>`_ for more information.

For licensing-related matters, please contact info@simple-crypto.dev.

Acknowledgements
================

This work has been funded in part by the Belgian Fund for Scientific Research
(F.R.S.-FNRS) through the Equipment Project SCALAB, by the European Union (EU)
and the Walloon Region through the FEDER project USERMedia (convention number
501907-379156), and by the European Union (EU) through the ERC project 724725
(acronym SWORD).

