Welcome to SCALib
=================

.. image:: https://badge.fury.io/py/scalib.svg
    :target: https://badge.fury.io/py/scalib
.. image:: https://readthedocs.org/projects/scalib/badge/?version=stable
    :target: https://scalib.readthedocs.io/en/stable/
    :alt: Documentation Status

The Side-Channel Analysis Library (SCALib) is a Python package that
contains state-of-the-art tools for side-channel evaluation. It focuses on
providing efficient implementations of analysis methods widely used by the
side-channel community and maintaining a flexible and simple interface.

SCALib contains various features for side-channel analysis, see the documentation_.

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


About us
========
SCALib was initiated by Olivier Bronchain and Gaëtan Cassiers during their PhD
at UCLouvain. It is now developed as a project of
`SIMPLE-Crypto <https://www.simple-crypto.dev/>`_ and maintained by Gaëtan Cassiers (@cassiersg).

Contributions and Issues
========================

Contributions welcome !

Please file a bug report for any issue you encounter (even bad documentation is
a bug !), and let us know your suggestions (preferably through github, but
email works too).
We also welcome code contributions, see `CONTRIBUTING.rst <CONTRIBUTING.rst>`_.

All code contributions are subject to the Contributor License Agreement (`CLA
<https://www.simple-crypto.dev/organization>`_) of SIMPLE-Crypto, which ensures
a thriving future for open-source hardware security.

License
=======
This project is licensed under `GNU AFFERO GENERAL PUBLIC LICENSE, Version 3`.
See `COPYING <COPYING>`_ for more information.

Acknowledgements
================

This work has been funded in part by the Belgian Fund for Scientific Research
(F.R.S.-FNRS) through the Equipment Project SCALAB, by the European Union (EU)
and the Walloon Region through the FEDER project USERMedia (convention number
501907-379156), and by the European Union (EU) through the ERC project 724725
(acronym SWORD).

