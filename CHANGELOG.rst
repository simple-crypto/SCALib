=========
Changelog
=========

Not released
------------

* SASCA: support modular ADD and MUL operations (#18)
* TTest: Performance improvement by using a mix of 2 passes and 1 pass algorithms 
* MTTest: First implementation of multivariate T-test.
* Improved documentation and README.rst
* SNR: use pooled formulas for better correctness then there are few traces,
  saves RAM (up to 75% reduction) and improves perf (about 2x single-threaded).
* Bump python minimum version to 3.7
* Revamp multi-threading handling thanks to new `scalib.threading` module.

v0.3.4
------

* Release GC in SASCA's `run_bp` .
* Release GC in `rank_accurary` and `rank_nbin`.
* `LDA.predict_proba` is marked thread-safe.
* Hide by default the progress bar of `SASCAGraph.run_bp` (can be re-enable
  with the `progress` parameter).

v0.3.3
------

* Solving minor issues in `MultiLDA` and `LDAClassifier`. Allowing multiple
  threads in `predict_proba()` and add a `done` flag to `solve()`.

v0.3.2
------

* Chunk `SNR.fit_u` to maintain similar performances with long traces and
  adding a progress bar 

v0.3.1
------

* Add `max_nb_bin` parameter to `postprocessing.rank_accuracy` (that was
  previously hard-coded).

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
