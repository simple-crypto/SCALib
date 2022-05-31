For developers
==============

SCALib is developped on github_.

.. _github: https://github.com/simple-crypto/SCALib

Local build
-----------

For best adaptation of SCALib to your CPU (to get best performance or if you
CPU does not support AVX2), you may want to build your own version of SCALib.

To do so, run

.. code-block::
    
    make wheel

in the source tree, which will create a wheel file.
You can then install it (in any python environment) with 

.. code-block::

    pip install XXX.whl --force-reinstall 

**NOTE**: The wheel on PyPi is built with AVX2 feature. If our CPU does not
support AVX2 instructions, you have to install SCALib from source. 

Development setup
-----------------

In order to use the developer environment, you will need `make`, `tox` and the
latest stable release of the `rust toolchain <https://rustup.rs/>`_. Once this is installed, you can
create a `virtualenv` and enter it with

.. code-block::

    make dev
    source .tox/dev/bin/activate

It will compile the Rust library and enter a python virtual environment where
`import scalib` imports the files from the repo.
You can then run your code that uses SCALib. If you edit SCALib's python code,
you have nothing to do to get the updated version. If you edit rust code,
re-run `make dev` to compile it.

Alternatively, you can develop SCALib and test your changes with the integrated
testsuite.
You can build the code and run the test in a single command:

.. code-block::
    
    make devtest
    
You can also run the same tests with 

.. code-block::

    make test

This will create the wheel and compile the Rust library with optimizations,
therefore it takes longer to build.

**Warning**: the `make dev` command builds the native code in debug mode, which
makes it very slow. If you want to really use your current version of SCALib,
it is recommended to follow the "Local build" procedure.

Before committing or pull request
---------------------------------

Ensure that your code passes tests (`make test`) and that the new code is
poperly tested.

Then ensure that the codestyle is followed: reformat the code automatically with

.. code-block::
 
    make fmt

Finally, ensure that you documented the new features (if relevant), and check
building the documentation with

.. code-block::
 
    make docs

More development commands
-------------------------

To measure test code coverage:

.. code-block::

    make coverage

To check code formatting

.. code-block::

    make codestyle

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

You may also use `perf` on linux, running a test case, for general metrics, or instruction-oriented data.

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

Make a release
--------------

0. Start from main branch
1. Add any missing element in CHANGELOG.rst
2. Add the new release with the release date in CHANGELOG.rst
3. Commit, create pull request and merge it (after CI succeeds).
4. Create and push release tag: `git tag vX.Y.Z && git push origin vX.Y.Z`

