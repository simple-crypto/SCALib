SCALib examples
===============

**For more detailed documentation and parameters, see the
API documentation at https://scalib.readthedocs.io/.**

The examples in this directory attack simlated leakage from an unprotected AES
implementation leaking the Hamming weight of the state bytes, with additive Gaussian noise.

Key recovery
------------

This example demonstrates a key-recovery attack on the simulated leakage.

.. code-block::

    python3 aes_attack.py


The attack goes in multiple steps:

1. Generate the simulated traces
    1. One set of profiling traces with random keys.
    2. One set of attack traces with a fixed key.

2. Select POIs for attacking the Sbox outputs :math:`x_i`.
    1. Compute the ``scalib.metrics.SNR`` for all the 16 :math:`x`.
    2. Keep as POI the 2 points with the largest SNR.

3. Profile the variables at the outputs of the Sboxes.
    1. Fit a ``scalib.models.LDAClassifier`` to model the leakage PDF.
    *Note: We could also use the convenience wrapper ``scalib.models.MultiLDA`` here: it provides a concise API (with automatic parallelization), but it is less flexible.*
    2. Extract the probabilities of :math:`x_i` using the attack traces and the models.

4. Recover the key :math:`k`.
    1. Create a factor graph describing the inference problem with ``scalib.attacks.FactorGraph``.
    2. Create a belief propagation object (``scalib.attacks.BPState``) with the prior distributions of :math:`x` and the values of the public variables :math:`p`.
    3. Run belief propagation to map the information from :math:`x` to :math:`k`.
    
    *Our factor graph here is acyclic, so we can to exact inference. SCALib also supports approximate inference with loopy belief propagation for more complex cases.*
    
5. Evaluate the attack results
    1. Show the rank for each key bytes.
    2. Show the overall key rank with ``scalib.postprocessing.rank_accuracy``.


TVLA
----

This example runs a fixed-vs-random first- and second-order unvariate TVLA using ``scalib.metrics.Ttest``.

.. code-block::

    python3 aes_tvla.py

*SCALib also supports multivariate T-test in ``scalib.metrics.MTtest``, and
allows you to arbitrarily choose your sets of points of interest.*


Information metrics
-------------------

The quality of a model can be quantified using the Perceived Information (PI) and Training Information (TI) (see https://eprint.iacr.org/2022/490).
In the example, we do this for pooled Gaussian Templates

.. code-block::

    python3 aes_info.py

Protected AES (ASCAD)
---------------------

See https://github.com/cassiersg/ASCAD-5minutes

This attack is fairly similar to ``aes_attack,py``, but attacks a real-world
protected implementation with first-order masking.

