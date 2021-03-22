.. Stella documentation master file, created by
   sphinx-quickstart on Fri Feb 26 15:50:43 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to SSCALE's documentation!
==================================
Simple Side-Channel Attack Leakage Evaluation (SSCALE) is a tool-box that
contains state-of-the-art tools for side-channel evaluation. Its focus is on
providing efficient implementations of analysis methods widely used by the
side-channel community.

For efficiency, Stella uses a custom Rust library which enables efficient
serialization and machine specific code while providing a userfriendly Python3
package. When applicable, it uses one-pass algorithms (e.g., `SNR`) which
allows to estimate metric / models directly when the data is collected without
requiring to store the traces.

The `SASCAGraph` is a central component of SSCALE. It allows to express in a
`.txt` what is the implementation to evaluate. It details what are the secrets
to recover (e.g., keys), what variables must be profiled and how they interact
with each other. 

Functionalities
===============

.. toctree::
   :maxdepth: 2
   
   source/stella.metrics.rst
   source/stella.modeling.rst
   source/stella.attacks.rst
   source/stella.ioutils.rst
   source/stella.postprocessing.rst

About us
========
Stella has been initiated by Olivier Bronchain during his PhD at Crypto Group,
UCLouvain, Belgium. It has already been used by many other researcher at
UCLouvain which contributed either directly or by constructive feedbacks. 

Contributions and Issues
========================
We are happy to take any suggestion for features that Stella should be useful
for side-channel evaluators. For such suggestion, contributions or issues,
please contact Olivier Bronchain at `olivier.bronchain@uclouvain.be
<olivier.bronchain@uclouvain.be>`_.

Publications
============
In order to cite SSCALE, please use the following bibtex.

.. code-block:: latex

    @misc{SSCALE,
      author = {Olivier Bronchain}
      title  = {{SSCALE: Simple Side-Channel Attacks Leakage Evaluation}},
      note   = {\url{github.com}},
      year   = {2021}
    }


SSCALE has been used in various publications, let us know if you used it:

1. "Mode-Level vs. Implementation-Level Physical Security in Symmetric
   Cryptography: A Practical Guide Through the Leakage-Resistance Jungle", D.
   Bellizia, O. Bronchain, G. Cassiers, V. Grosso, Chun Guo, C. Momin, O.
   Pereira, T. Peters, F.-X. Standaert at CRYPTO2020
2. "Breaking Masked Implementations with Many Shares on 32-bit Software
   Platforms or When the Security Order Does Not Matter". O. Bronchain, F.-X.
   Standaert in TCHES2021 - Issue 3
3. "Improved Leakage-Resistant Authenticated Encryption based on Hardware AES
   Coprocessors". O. Bronchain, C. Momin, T. Peters, F.-X. Standaert in
   TCHES2021 - Issue 3
