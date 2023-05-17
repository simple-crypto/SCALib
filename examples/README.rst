SCALib examples
===============

Simple usage examples of SCALib algorithms are also provided in the
`API documentation<https://scalib.readthedocs.io/>`_.

Simulated unprotected AES
-------------------------

This is an example of attack against a simple AES implementation leaking with
an Hamming weight model and with Gaussian noise addition. This example
demonstrates basic use of most of the algorithms implemented in SCALib. The
example can be executed by running ``aes_simulation.py`` as

.. code-block::

    python3 aes_simulation.py


The example goes in multiple steps: 

1. Generate the simulated traces
    1. One set of profiling traces with random keys.
    2. One set of attack traces with a fixed key.
2. TVLA
    1. Run fixed-vs-random first- and second-order unvariate TVLA using ``scalib.metrics.Ttest``
       (we also support multivariate tests in ``scalib.metrics.MTtest``).
3. Profile the variables at the output of the Sbox :math:`x`.
    1. Compute the ``scalib.metrics.SNR`` for all the 16 :math:`x`.
    2. Keep as POI the 2 points with the largest SNR.
    3. Use a ``scalib.models.LDAClassifier`` to model the leakage PDF. 
4. Attack to recover the key :math:`k`.
    1. Insert ``plaintexts`` and estimated distributions of :math:`x` in a ``scalib.attacks.FactorGraph``.
    2. Run belief propagation to map the information from :math:`x` to :math:`k`.
5. Evaluate the attack results
    1. Show the rank for each key bytes.
    2. Show the overall key rank with ``scalib.postprocessing.rank_accuracy``.

Protected AES (ASCAD)
---------------------

`Here <https://github.com/cassiersg/ASCAD-5minutes>`_.

This attack is fairly similar to the example, but attacks a real-world
protected implementation with first-order masking.


