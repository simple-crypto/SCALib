Contributing
============

SCALib is developped on github_.

.. _github: https://github.com/simple-crypto/SCALib

Dependencies
------------

The minimum dependencies for building SCALib are

- A C/C++ compiler.
- The latest stable release of the `rust toolchain <https://rustup.rs/>`_.
- Python (see ``setup.cfg`` for supported versions).
- :code:`pip`

Moreover, for development, we use

- `tox <https://pypi.org/project/tox>`_
- :code:`make` (optional)

Development commands
--------------------

The ``Makefile`` contains aliases for frequently-used commands, including
building (with various compile optioms), running tests, etc.
See

.. code-block::

    make help

Build wheels
------------

If you need to get wheels (e.g. to install elsewhere), you can simply
use ``make wheel_local``, ``make wheel_avx2`` or ``make wheel_portable``.


Development flow
----------------

Multiple builds with misc. trade-offs can be used, depending on what you are working on:

- Develop Rust code:

.. code-block::

   # Edit code...
   make fmt # Nicely formats your code
   make test # Build SCALib and run tests
   # `make dev` can also be used: better debuginfo, but tests are slower

- When developping Python code, you don't have to wait for the build:

.. code-block::

   make dev
   source .tox/dev/bin/activate # To be adapted when using powershell
   # Do the above only once, and iterate the following:
   # Edit code...
   # Run the tests:
   pytest
   # or, to focus on one feature test:
   pytest tests/test_X.py

- When developping tests, you can do the same as for Python development, but
  using the ``test`` environment to run the test faster!



Before committing or pull request
---------------------------------

1. Ensure that your new code is poperly **tested and documented**.

2. Ensure that your changes are documented in ``CHANGELOG.rst``.

3. Nicely format the code:

.. code-block::
 
    make fmt

4. Run tests with

.. code-block::
 
    make test

5. Build the documentation and check it

.. code-block::
 
    make docs

More development commands
-------------------------

Report test code coverage as html:

.. code-block::

    make coverage


Performance measurements
------------------------

Do not guess, Measure !

Py-spy is a nice tool to show execution flamegraphs. It can even report the profile of rust code, but not the one executed in native thread pools (such as rayon's):

.. code-block::

    py-spy record --native -- python my_bench_code.py

In order to benchmark directly the Rust crate (without using Python), you can
leverage the `criterion
<https://bheisler.github.io/criterion.rs/book/criterion_rs.html>`_ cargo's
utilities. A command line example is:

.. code-block::

    cargo bench --  

You may also use ``perf`` on linux, running a test case, for general metrics, or instruction-oriented data.

.. code-block::

    perf stat python3 scalib_benchcase.py # you may want to look at option -e of perf stat

.. code-block::

    # This might generate a lot of data, probably a small example (a few
    # seconds) on a single thread is enough statistical evidence.
    SCALIB_NUM_THREADS=1 perf record python3 scalib_benchcase.py
    perf report -g folded

Also, when you develop, looking at generated assembly may help

.. code-block::

    RUSTFLAGS="-C target-feature=+avx2" cargo asm scalib::module::function --rust
    # or
    RUSTFLAGS="-C target-cpu=native" cargo asm scalib::module::function --rust

Dependencies upgrade policy
---------------------------

- For python and python packages, we follow NEP29_.
- For rust: latest stable version.
- OS support:

  * Pre-built wheels for manylinux_ (supporting last two Ubuntu LTS) on x86_64.
  * Pre-built wheels for Windows 10 on x86_64.
  * Other: build yourself (CI configuration welcome).

.. _NEP29: https://numpy.org/neps/nep-0029-deprecation_policy.html
.. _manylinux: https://github.com/pypa/manylinux


Make a release
--------------

0. Start from main branch.
1. Review ``git log`` and add any missing element in ``CHANGELOG.rst``.
2. Add the new release with the release date in ``CHANGELOG.rst``.
3. Commit, create pull request and merge it (after CI succeeds).
4. Create and push release tag: ``git tag vX.Y.Z && git push origin vX.Y.Z``.
5. Check that CI build, PyPI upload and ReadTheDocs all worked. Otherwise fix and make a patch release.

