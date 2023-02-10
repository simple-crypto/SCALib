SCALib examples
===============

Simulated unprotected AES
-------------------------

This directory contains an example of attack against an simple AES
implementation leaking with an HW model and with Gaussian noise addition. We
note that this code can simply be reused for a real implementation by replacing
the simulated traces by real measurements. The example can simply be executed
by running `aes_simulation.py` with:

.. code-block::

    python3 aes_simulation.py


The simulations goes in multiple steps: 

1. Generate the simulated traces
    1. One set of profiling traces with random keys.
    2. One set of attack traces with a fixed key.
2. Profile the variables at the output of the Sbox `x`.
    1. Compute the SNR for all the 16 `x`.
    2. Keep as POI the 2 points with the largest SNR.
    3. Use a `LDAClassifier` to model the leakage PDF. 

3. Attack to recover the key `k`.
    1. Insert `plaintexts` and estimated distributions of `x` within graph.
    2. Run belief propagation for map information from `x` to `k`.
    3. Evaluate the attack with `rank_accuracy`.
