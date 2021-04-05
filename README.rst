Welcome to SCALE
================
Side-Channel Attack & Leakage Evaluation (SCALE) is a tool-box that
contains state-of-the-art tools for side-channel evaluation. Its focus is on
providing efficient implementations of analysis methods widely used by the
side-channel community and maintaining a flexible and simple interface.

SCALE contains various features for side-channel analysis:

- Metrics for side-channel analysis:

  - `SNR <scale/metrics/snr.py>`_: Signal-to-noise ratio.
- Modeling leakage distribution:

  - `LDClassifier <scale/modeling/ldaclassifier.py>`_: Template in linear subspaces.
- Attacks to recover secret keys

  - `SASCAGraph <scale/attacks/sascagraph.py>`_: Generalization of divide and conquer with Soft Analytical Attacks.
- Postprocessing to analyse attacks results

  - `rankestimation <scale/postprocessing/rankestimation.py>`_: Histogram based full key rank estimation 


Install
=======
You can simply install SCALE by using PyPi and running:

.. code-block::

   pip install scale

Wheels for Windows and Linux are provided. More information about source
compilation, checkout `develop <DEVELOP.rst>`_ informations.

Pseudo-Example
==============
Next, we detail a short pseudo example which illustrates the usage of SCALE. 
For a full running example, please visit `this example <examples/aes_simulation/>`_. 

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


SCALE workflow
==============

The current version of SCALE contains modules for all the necessary steps for a
profiled side-channel attack. Even if modules of SCALE can be used
independently, a typical usage of SCALE for it goes in four steps:

1. **Metrics**: In this step, standard metrics are evaluated for the
   measurements. This could be helpful to find point-of-interest (POIs),
   leakage, etc. When applicable, these metrics are implemented with a one-pass
   algorithm. This allows either to load one the traces from the disk and
   evaluate the metric on the complete dataset. It also allows to directly
   compute the metrics once the traces are captured. In the case where only the
   metric must be evaluated, this remove the need to store the data. The
   standard `SNR` metric is available and allows to find point of interest of a
   given variable (first order).

2. **Modeling**: In this step, models are built to extract information about a
   variable `y` from the leakage. Modeling methods works in two phases. The
   first one is to `fit()` the model with the random value `x` is the training
   data and `y` is the target value. The will build a model. The second one is
   to return probabilities for each of the classes based on leakage `xt` by
   using the function `predict_proba(xt)`. Only modeling based on `LDA` and
   Gaussian templates is available.

3. **Attacks**: This modules contains attack methodologies. It essentially uses
   the probabilities from the `modeling` step in order and recombine them to
   recover a key. The module `SASCAGraph` represent how the probabilities on
   variables can be recombined. It can be used to model a standard template
   attack on unprotected implementations. The same module can also be used to
   run advanced SASCA attacks for any circuit that contains boolean operations
   and table lookups.

4. **PostProcessing**: Once the attack has been performed, the postprocessing
   allows to evaluation the efficiency of the attack. Namely by estimating the
   rank of the correct key with `rank_accuracy`, it allows to quantify what is
   the remaining computational power that is needed by the adversary to recover
   the correct key.

For details about of the usage of SCALE in a complete analysis, please visit
the examples against protected and unprotected in  `examples <examples/>`.  We
note that the modules of SCALE can easily be replaced by other libraries. As an
example, the `modeling` methods have an interface similar to `scikit-learn`.
The modeling could also be done with any other tools (e.g., deep learning) as
soon as the modeling is able to return probabilities.

About us
========
SCALE has been initiated by Olivier Bronchain during his PhD at Crypto Group,
UCLouvain, Belgium. His colleague Gaëtan Cassiers co-authored SCALE. It has
already been used by many other researcher at UCLouvain which contributed
either directly or by constructive feedbacks. 

Contributions and Issues
========================
We are happy to take any suggestion for features would be useful for
side-channel evaluators. For such suggestion, contributions or issues, please
contact Olivier Bronchain at `olivier.bronchain@uclouvain.be
<olivier.bronchain@uclouvain.be>`_.

License
=======

Publications
============

SCALE has been used in various publications, let us know if you used it:

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

