For developers
===============

In order to use the developer environment, you will need to use `tox`. Once it
is installed, you can create a `virtualenv` and enter it with

.. code-block::

    make dev
    source .tox/dev/bin/activate

It will compile the Rust library and import SCALib python files
from `src/scalib`. You can then edit directly these files. To run tests in that
virtualenv, you can run (no need to activate the virtualenv before)

.. code-block::

    make devtest


**Warning**: this builds the native code in debug mode, which makes it very
slow. If you want to compile your current version of SCALib with efficient
code, you can run

.. code-block::

    make test

The following create the wheel and compile the Rust library with optimization

.. code-block::
 
    CARGO_TARGET_DIR=.cargo_build pip wheel .

which will create the wheel. You can the simply install it with 

.. code-block::
 
    pip install XXX.whl --force-reinstall 

You can run the test with python coverage information with

.. code-block::
 
    make test-cov

then open the result in `htmlcov`.

Finally, to build the docs in `docs/_build/html`

.. code-block::
 
    make docs
