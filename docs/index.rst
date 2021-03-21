.. Stella documentation master file, created by
   sphinx-quickstart on Fri Feb 26 15:50:43 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to SSCALE's documentation!
==================================
Simple Side-Channel Attack Leakage Evaluation (SSCALE) is a tool-box that
contains state-of-the-art tools for side-channel evaluation. Its focus is on
providing efficient implementations of widely used functionalities by the
side-channel community. 

.. toctree::
   :maxdepth: 2
   :caption: Functionalities
   
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
SSCALE has been used in various publications:

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
