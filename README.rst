theano_models
-------------
This is a helper package for working with generalized abstract theano functions, here called models.


Example
-------
Go through the examples within subdirectory ``examples/``. There also jupyter notebooks are included.


Install
-------
To install a developemntal version of the project execute the following.
(Note, pip might need be exectued with sudo, or add the ``--user`` flag)::

    git clone https://github.com/schlichtanders/theano_models
    pip install --process-dependency-links -e theano_models

For non-development version skip the ``-e`` flag.
The ``--process-dependency-links`` flag is needed, as direct dependencies are not yet supported by pip/setuptools.
