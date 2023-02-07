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
SCALib contains various features for side-channel analysis. Please read `SCALib
workflow`_ for more details:

- :doc:`source/scalib.metrics`:

  - Signal-to-noise ratio (SNR).
  - Uni- and Multi-variate, arbitrary-order T-test estimation.

- :doc:`source/scalib.modeling`: 

  - Templates in linear subspaces (LDA).

- :doc:`source/scalib.attacks`:

  - Generalization of `Divide & Conquer` with Soft Analytical Attacks (SASCA).

- :doc:`source/scalib.postprocessing`:

  - Full key rank estimation.

Getting started
===============

Install
-------

See the `README <https://github.com/simple-crypto/SCALib>`_. TL;DR:

.. code-block::

   pip install scalib


SCALib workflow
---------------

The current version of SCALib contains algorithms for many steps of a
side-channel security evaluation.
These are grouped in four categories:

1. **Metrics**: Standard metrics leakage metrics.
   This helps to find point-of-interest (POIs), quantify avaiable information, etc. 

2. **Modeling**: Tools to mount profiled attacks first ``.fit(...)``, then ``.predict_proba(...)``.

3. **Attacks**: Once your have profiles, find the key. This currently contains
   an implementation the SASCA.

4. **PostProcessing**: Show the results of a trial attack.
   This currently contains key rank-estimation routines.

Pseudo-example
##############

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


See the `full examples <https://github.com/simple-crypto/scalib/tree/main/examples>`_. 

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

About us
========
SCALib was initiated by Olivier Bronchain and Gaëtan Cassiers during their PhD
at UCLouvain. It is now developed as a project of
`SIMPLE-Crypto <https://www.simple-crypto.dev/>`_ and maintained by Gaëtan Cassiers.

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
