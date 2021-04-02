For developpers
===============
Install the `pipenv` tool from PyPI, then run 

.. code-block::

    pipenv install
    
to initialize the development environment. Running 

.. code-block::

    pipenv run python setup.py develop

builds the native code and makes SCALE importable in the environment.

**Warning**: this builds the native code in debug mode, which makes it very slow.
For production usage, build and install the wheel using ``pipenv run setup.py
bdist_wheel``, then ``pip install path/to/the/wheel``.

Tests
-----
In the environment, the tests can be exacted with `pytest`. Running 
.. code-block:: 
    
    pipenv run pytest

will test functionality of SCALE. Please run the tests before pushing new code.


Documentation
-------------
The documentations can be built by running 

.. code-block::
    
    pipenv run make -C docs html

The documentation are available in `docs/_build/html/`.
