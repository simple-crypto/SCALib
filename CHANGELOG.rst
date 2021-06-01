=========
Changelog
=========

Not released
------------

v0.3.0
------

* Rename `num_threads` parameter of `modeling.MultiLDA` to `num_cpus`.
* Fix rank estimation when there is only one key chunk.
* Improve performance of `SNR.get_snr`.

v0.2.0
------

* Remove OpenBLAS and LAPACK, use Spectra and nalgebra instead.
* Use BLIS for matrix multiplications (Linux-only for now).
* Make `modeling.LDAClassifier` incremental (breaking change).
* Add `modeling.MultiLDA`.

v0.1.1
------

* Fix "invalid instruction" bug for CI wheel on windows.

v0.1.0
------

* Initial release, with the following features:
  * LDA and Gaussian templates modelling
  * SNR
  * T-test any order (for TLVA)
  * Soft Analytical Side-Channel Attack (SASCA)
  * Rank Estimation
