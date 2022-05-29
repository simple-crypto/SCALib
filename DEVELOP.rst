For developers
===============

SCALib is developped on github at github_.

.. _github: https://github.com/simple-crypto/SCALib

Local build
-----------

If you want to build your own version of SCALib, you can run

.. code-block::
    
    make wheel

in the source tree, which will create a wheel file.
You can then install it (in any python environment) with 

.. code-block::

    pip install XXX.whl --force-reinstall 


Development setup
-----------------

In order to use the developer environment, you will need `make`, `tox` and the
latest stable release of the `rust toolchain`. Once this is installed, you can
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


Measure !

Py-spy is a nice tool to show execution flamegraphs. It can even report the profile of rust code, but not the one executed in native thread pools (such as rayon's):

.. code-block::

    py-spy record --native -- python my_bench_code.py

