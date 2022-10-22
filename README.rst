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

SCALib is on PyPi!

.. code-block::

    pip install scalib

See install_ for details.
 
.. _install: https://scalib.readthedocs.io/en/stable/index.html#install

Dependencies upgrade policy
===========================

- For python and python packages, we follow NEP29_.
- For rust: latest stable version.
- OS support:

  * Pre-built wheels for manylinux_ (supporting last two Ubuntu LTS) on x86_64.
  * Pre-built wheels for Windows 10 on x86_64.
  * Other: build yourself (CI configuration welcome).

.. _NEP29: https://numpy.org/neps/nep-0029-deprecation_policy.html
.. _manylinux: https://github.com/pypa/manylinux

About us
========
SCALib has been initiated by Olivier Bronchain during his PhD at Crypto Group,
UCLouvain, Belgium. His colleague Gaëtan Cassiers co-authored SCALib. The SCALib
project is part of `SIMPLE-Crypto <https://www.simple-crypto.dev/>`_ and is
maintained in that context.


Contributions and Issues
========================

Contributions welcome !
See contribution_ and `DEVELOP.rst <DEVELOP.rst>`_.

.. _contribution: https://scalib.readthedocs.io/en/stable/index.html#contributions-and-issues

License
=======
This project is licensed under `GNU AFFERO GENERAL PUBLIC LICENSE, Version 3`.
See `COPYING <COPYING>`_ for more information.

Acknowledgements
================

This work is supported by the Belgian Fund for Scientific Research
(F.R.S.-FNRS) through the Equipment Project SCALAB and by the ERC through the
Consolidator Grant SWORD.

