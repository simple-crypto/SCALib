======
SCALib
======

.. image:: https://badge.fury.io/py/scalib.svg
    :target: https://pypi.org/project/scalib/
    :alt: PyPI
.. image:: https://readthedocs.org/projects/scalib/badge/?version=stable
    :target: https://scalib.readthedocs.io/en/stable/
    :alt: Documentation Status
.. image:: https://img.shields.io/matrix/scalib:matrix.org
    :target: https://matrix.to/#/#scalib:matrix.org
    :alt: Matrix room
.. image:: https://joss.theoj.org/papers/10.21105/joss.05196/status.svg
   :target: https://doi.org/10.21105/joss.05196
   :alt: JOSS paper

The Side-Channel Analysis Library (SCALib) is a Python library that
contains state-of-the-art tools for side-channel security evaluation.

- **Documentation**: https://scalib.readthedocs.io/
- **Examples**: `examples/ <examples/>`_
- **Chat**: `https://matrix.to/#/#scalib:matrix.org <https://matrix.to/#/#scalib:matrix.org>`
- **Source code**: https://github.com/simple-crypto/SCALib
- **Bug reports/feature requests**: https://github.com/simple-crypto/SCALib/issues/new/choose
- **Contributing**: https://scalib.readthedocs.io/en/stable/source/contributing.html


SCALib focuses on

- simple interface,
- state-of-the art algorithms,
- excellent performance (see `benchmarks <https://github.com/cassiersg/SCABench>`_).

SCALib should be useful for any side-channel practitioner who wants to
evaluate, but not necessarily attack, protected or non-protected
implementations.
See the documentation_ for the list of implemented tools.

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

**Depdendencies:**

- ``python >= 3.10``,
- a C/C++ compiler for your platform,
- ``clang``,
- the latest stable release of the `rust toolchain <https://rustup.rs/>`_.

To install from source:

.. code-block::

    git clone https://github.com/simple-crypto/SCALib
    pip install ./SCALib

See `CONTRIBUTING.rst <CONTRIBUTING.rst>`__ for advanced build configuration.

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

SCALib uses `semantic versioning <https://semver.org/>`_, see the `CHANGELOG
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
<https://github.com/simple-crypto/SCALib/issues/new/choose>`_, `chat
<https://matrix.to/#/#scalib:matrix.org>`_ and `email
<mailto:gaetan.cassiers@uclouvain.be>`_ work too).
We also welcome code contributions, see `CONTRIBUTING.rst <CONTRIBUTING.rst>`_.

You can also come discuss on `matrix <https://matrix.to/#/#scalib:matrix.org>`_
(announcements, questions, community support, open discussion, etc.).

All code contributions are subject to the Contributor License Agreement (`CLA
<https://www.simple-crypto.dev/organization>`_) of SIMPLE-Crypto, which ensures
a thriving future for open-source hardware security.


Citation
========

If you use SCALib in your research, please cite our `software paper <https://doi.org/10.21105/joss.05196>`_:

.. code-block::

    Cassiers et al., (2023). SCALib: A Side-Channel Analysis Library. Journal of Open Source Software, 8(86), 5196, https://doi.org/10.21105/joss.05196

Bibtex:

.. code-block::

   @article{scalib,
       doi = {10.21105/joss.05196},
       url = {https://doi.org/10.21105/joss.05196},
       year = {2023},
       publisher = {The Open Journal},
       volume = {8},
       number = {86},
       pages = {5196},
       author = {Gaëtan Cassiers and Olivier Bronchain},
       title = {SCALib: A Side-Channel Analysis Library}, journal = {Journal of Open Source Software}
   }


License
=======
This project is licensed under GNU AFFERO GENERAL PUBLIC LICENSE, Version 3.
See `COPYRIGHT <COPYRIGHT>`_ and `COPYING <COPYING>`_ for more information.

For licensing-related matters, please contact info@simple-crypto.dev.

Acknowledgements
================

This work has been funded in part by the Belgian Fund for Scientific Research
(F.R.S.-FNRS) through the Equipment Project SCALAB and individual researchers'
grants, by the European Union (EU) and the Walloon Region through the FEDER
project USERMedia (convention number 501907-379156), and by the European Union
(EU) through the ERC project 724725 (acronym SWORD) and the ERC project
101096871 (acronym BRIDGE).
