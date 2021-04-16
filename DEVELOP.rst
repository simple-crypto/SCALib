For developers
===============

In order to use the developer environment, you will need to use `tox` and the
latest stable release of the `rust toolchain`. Once it is installed, you can
create a `virtualenv` and enter it with

.. code-block::

    make dev
    source .tox/dev/bin/activate

It will compile the Rust library and import SCALib python files
from `src/scalib`. You can then edit directly these files. To run tests in that
virtualenv, you can run 

.. code-block::
    
    make devtest
    
You can also run the same tests with 

.. code-block::

    make test

This will create the wheel and compile the Rust library with optimization
It must be done before deployment.



**Warning**: this builds the native code in debug mode, which makes it very slow. If you want to compile your current version of SCALib with efficient code, you can run

.. code-block::
    
    CARGO_TARGET_DIR=.cargo_build pip wheel .

which will create the wheel. You can the simply install it with 

.. code-block::

    pip install XXX.whl --force-reinstall 
