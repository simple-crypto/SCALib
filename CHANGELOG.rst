=========
Changelog
=========

Not released
------------

v0.4.3 (2022/10/27)
-------------------

* Upgrade CI to 3.11.
* Update dependancies and add python 3.10 to CI (#49)

v0.4.2 (2022/05/31)
-------------------

* Fix AVX2 not used when building rust dependencies.

v0.4.1 (2022/05/31)
-------------------

* Fix docs not building

v0.4.0 (2022/05/31)
-------------------

* SASCA: support modular ADD and MUL operations (#18)
* TTest: Performance improvement by using a mix of 2 passes and 1 pass algorithms 
* MTTest: First implementation of multivariate T-test.
* Improved documentation and README.rst
* SNR: use pooled formulas for better correctness then there are few traces,
  saves RAM (up to 75% reduction) and improves perf (about 2x single-threaded).
* Bump python minimum version to 3.7
* Revamp multi-threading handling thanks to new `scalib.threading` module.
* AVX2: Wheels on PyPi are built with AVX2 feature. 

v0.3.4 (2021/12/27)
-------------------

* Release GC in SASCA's `run_bp` .
* Release GC in `rank_accurary` and `rank_nbin`.
* `LDA.predict_proba` is marked thread-safe.
* Hide by default the progress bar of `SASCAGraph.run_bp` (can be re-enable
  with the `progress` parameter).

v0.3.3 (2021/07/13)
-------------------

* Solving minor issues in `MultiLDA` and `LDAClassifier`. Allowing multiple
  threads in `predict_proba()` and add a `done` flag to `solve()`.

v0.3.2 (2021/07/12)
-------------------

* Chunk `SNR.fit_u` to maintain similar performances with long traces and
  adding a progress bar 

v0.3.1 (2021/06/03)
-------------------

* Add `max_nb_bin` parameter to `postprocessing.rank_accuracy` (that was
  previously hard-coded).

v0.3.0 (2021/06/01)
-------------------

* Rename `num_threads` parameter of `modeling.MultiLDA` to `num_cpus`.
* Fix rank estimation when there is only one key chunk.
* Improve performance of `SNR.get_snr`.

v0.2.0 (2021/05/20)
-------------------

* Remove OpenBLAS and LAPACK, use Spectra and nalgebra instead.
* Use BLIS for matrix multiplications (Linux-only for now).
* Make `modeling.LDAClassifier` incremental (breaking change).
* Add `modeling.MultiLDA`.

v0.1.1 (2021/04/26)
-------------------

* Fix "invalid instruction" bug for CI wheel on windows.

v0.1.0 (2021/04/16)
-------------------

* Initial release, with the following features:
  * LDA and Gaussian templates modelling
  * SNR
  * T-test any order (for TLVA)
  * Soft Analytical Side-Channel Attack (SASCA)
  * Rank Estimation
