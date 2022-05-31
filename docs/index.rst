Welcome to SCALib
=================
The Side-Channel Analysis Library (SCALib) is a Python package that
contains state-of-the-art tools for side-channel evaluation. It focuses on
providing efficient implementations of analysis methods widely used by the
side-channel community and maintaining a flexible and simple interface.

The source code is available on `GitHub <https://github.com/simple-crypto/SCALib>`_!

Usability & Efficiency
----------------------
SCALib main advantages are:

1. **High Performances**: Despite SCALib is a Python package, it embeds a Rust
   back-end which enables a fine grain control of the implementation. Thanks to
   the Rust back-end, SCALib enables efficient parallelism. This is obtained
   thanks to `rayon <https://docs.rs/crate/rayon/latest>`_.
2. **Flexible & Simple Interface**: The interface provided by SCALib aims to be
   as simple and flexible as possible. As an example, the API is mostly based
   on numpy arrays. 
3. **Incremental APIs**: SCALib leverages as much as possible incremental APIs. That is, 
   it is possible to feed the data in multiple chunks to avoid loading multiple times
   the same data into RAM. 

Available features
------------------
SCALib contains various features for side-channel analysis. Please read `SCALib
workflow`_ for more details:

- :doc:`source/scalib.metrics`:

  - `SNR`: Signal-to-noise ratio.
  - `Ttest`: T-test estimation.

- :doc:`source/scalib.modeling`: 

  - `LDAClassifier`: Template in linear subspaces.
- :doc:`source/scalib.attacks`:

  - `SASCAGraph`: Generalization of `Divide & Conquer` with Soft Analytical Attacks.
- :doc:`source/scalib.postprocessing`:

  - `rankestimation`: Histogram based full key rank estimation.
- :doc:`source/scalib.configuration`:

  - `threading`: provides a fine control on the number of threads used by SCALib.  

Getting started
===============

Install
-------
You can install SCALib by using PyPi packages and running:

.. code-block::

   pip install scalib

Wheels for Windows and Linux are provided. More information about source
compilation, checkout `DEVELOP
<https://github.com/simple-crypto/SCALib/blob/main/DEVELOP.rst>`_ page.
Especially, we recommand to compile SCALib to get maximal performances.


SCALib workflow
---------------

The current version of SCALib contains modules for all the necessary steps for a
profiled side-channel attack. Even if modules of SCALib can be used
independently, a typical usage of SCALib for it goes in four steps:

1. **Metrics**: In this step, standard metrics are evaluated from the
   measurements. This helps to find point-of-interest (POIs), quantify avaiable information, etc. 
   When applicable, these metrics are implemented with a one-pass
   algorithm to save the cost of data load / store.

2. **Modeling**: In this step, models are built to extract information about a
   variable `y` from the leakage. Modeling methods work in two phases. The
   first one is to `fit(t,x)` with traces `t` and target values `x`. This creates the
   model parameters. In the second step, this model can be used to 
   `predict_proba(t)` that returns the probability of every possible target values based on the
   traces `t` and the model.

3. **Attacks**: This modules contains attack methodologies. It essentially uses
   the probabilities from the `modeling` step in order and recombine them to
   recover a key. The module `SASCAGraph` is an extension of Divide & Conquer attacks that leverage `soft analytical side-channel attacks`. It allows to define `PROPERTY` that link intermediate varibles within an implementation.
   By providing the `SASCAGraph` with the distributions of these variables, it propagates information on all the variables (e.g. secret keys).

4. **PostProcessing**: Once the attack has been performed, the postprocessing
   allows to evaluate the efficiency of the attack. Namely by estimating the
   rank of the correct key with `rank_accuracy`, it allows to quantify what is
   the remaining computational power that is needed by the adversary to recover
   the correct key.

Full example of SCALib is available `here <https://github.com/simple-crypto/scalib/tree/main/examples/aes_simulation>`_ for an unprotected simulated AES. 

Pseudo-example
--------------

Next, we detail a short pseudo example which illustrates the usage of SCALib. 
For a full running example, please visit `this example <https://github.com/simple-crypto/scalib/tree/main/examples/aes_simulation>`_. 

.. code-block::

     # compute snr
     snr = SNR(nc=256,ns=ns,p=1) 
     snr.fit(traces_p,x_p)
     
     # build model
     pois_x = np.argsort(snr.get_snr()[0][:-npoi])
     lda = LDAClassifier(nc=256,ns=npoi,p=1)
     lda.fit(traces_p[:,pois_x],x_p)

     # Describe and generate the SASCAGraph
     graph_desc = ´´´
        # Small unprotected Sbox example
        TABLE sbox   # The Sbox
        VAR SINGLE k # The key
        VAR MULTI p  # The plaintext
        VAR MULTI x  # Sbox input
        VAR MULTI y  # Sbox output
        PROPERTY x = k ^ p   # Key addition
        PROPERTY y = sbox[x] # Sbox lookup
        ´´´
     graph = SASCAGraph(graph_desc,256,len(traces_a))

     # Encode data into the graph
     graph.set_table("sbox",aes_sbox)
     graph.set_public("p",plaintexts)
     graph.set_distribution("x",lda.predict_proba(traces_a))

     # Solve graph
     graph.run_bp(it=3)

     # Get key distribution and derive key guess
     k_distri = graph.get_distribution("k")
     key_guess = np.argmax(k_distri[0,:])



Where is SCALib used ?
======================

We strongly appreciate if you could mention to us your usage SCALib 
for concrete projects so that we can add you to the lists below. Please send 
an email to Olivier Bronchain.

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

About us
========
SCALib has been initiated by Olivier Bronchain during his PhD at Crypto Group,
UCLouvain, Belgium. His colleague Gaëtan Cassiers co-authored
of SCALib starting from the first version. The SCALib project is part of `SIMPLE-Crypto
<https://www.simple-crypto.dev/>`_ and is now maintained in that context.


Contributions and issues
========================
We are happy to take any suggestion for features would be useful for
side-channel evaluators.  If you want to contribute code to the project, please
visit `DEVELOP
<https://github.com/simple-crypto/SCALib/blob/main/DEVELOP.rst>`_ for relevant
information as well as the Contributor License Agreement (`CLA
<https://www.simple-crypto.dev/organization>`_) of SIMPLE-Crypto. Please open
an `issue <https://github.com/simple-crypto/SCALib/issues>`_ on the GitHub repository for further questions, bug report or feature
requests. 

License
=======
This project is licensed under `GNU AFFERO GENERAL PUBLIC LICENSE, Version 3`.
See `COPYING <https://github.com/simple-crypto/scalib/blob/main/COPYING>`_ for
more information. 


.. toctree::
   :maxdepth: 2
   :hidden:

   source/scalib.metrics.rst
   source/scalib.modeling.rst
   source/scalib.attacks.rst
   source/scalib.postprocessing.rst
   source/scalib.configuration.rst
