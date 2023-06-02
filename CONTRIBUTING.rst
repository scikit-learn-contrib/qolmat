=======================
Contribution guidelines
=======================

What to work on?
----------------

You are welcome to propose and contribute new ideas.
We encourage you to `open an issue <https://github.com/quantmetry/qolmat/issues>`_ so that we can align on the work to be done.
It is generally a good idea to have a quick discussion before opening a pull request that is potentially out-of-scope.

Fork/clone/pull
---------------

The typical workflow for contributing to `Qolmat` is:

1. Fork the `main` branch from the `GitHub repository <https://github.com/quantmetry/qolmat>`_.
2. Clone your fork locally.
3. Commit changes.
4. Push the changes to your fork.
5. Send a pull request from your fork back to the original `main` branch.

Local setup
-----------

We encourage you to use a virtual environment. You'll want to activate it every time you want to work on `Qolmat`.

You can create a virtual environment via `conda`:

.. code:: sh

    $ conda env create -f environment.dev.yml
    $ conda activate env_qolmat_dev

If you need to use tensorflow, enter the command:

.. code:: sh

    $ pip install -e .[tensorflow]

Once the environment is installed, pre-commit is installed, but need to be activated using the following command:

.. code:: sh

    $ pre-commit install

Documenting your change
-----------------------

If you're adding a class or a function, then you'll need to add a docstring with a doctest. We follow the `numpy docstring convention <https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_numpy.html>`_, so please do too.
Any estimator should follow the [scikit-learn API](https://scikit-learn.org/stable/developers/develop.html), so please follow these guidelines.

Updating changelog
------------------

You can make your contribution visible by :

1. adding your name to the Contributors sections of `AUTHORS.rst <https://github.com/quantmetry/qolmat/blob/main/AUTHORS.rst>`_
2. adding a line describing your change into `HISTORY.rst <https://github.com/quantmetry/qolmat/blob/main/HISTORY.rst>`_

Testing
-------

Pre-commit
^^^^^^^^^^

These tests absolutely have to pass.

.. code:: sh

    $ pre-commit run --all-files

Static typing
^^^^^^^^^^^^^

These tests absolutely have to pass.

.. code:: sh

    $ mypy qolmat

Unit test
^^^^^^^^^

These tests absolutely have to pass.
The coverage should absolutely be 100%.

.. code:: sh

    $ pytest -vs --cov-branch --cov=qolmat --pyargs tests --cov-report term-missing
