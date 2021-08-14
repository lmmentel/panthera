Installation
============

Dependencies
------------

    - `numpy <http://www.numpy.org/>`_
    - `scipy <https://www.scipy.org/>`_
    - `pandas <http://pandas.pydata.org/>`_
    - `matplotlib <http://matplotlib.org/>`_
    - `seaborn <https://stanford.edu/~mwaskom/software/seaborn/>`_
    - `future <https://pypi.python.org/pypi/future>`_
    - `six <https://pypi.python.org/pypi/six>`_
    - `bmatrix <https://github.com/lmmentel/bmatrix>`_ based on the approach presented in [1]_
    - `lxml <http://lxml.de/>`_

The recommended installation method is with pip_. The latest version
can be installed directly from `github repository`_:

.. code-block:: bash

   pip install https://bitbucket.org/lukaszmentel/panther/get/tip.tar.gz

or cloned first

.. code-block:: bash

   hg clone https://lukaszmentel@bitbucket.org/lukaszmentel/panther

and installed via

.. code-block:: bash

   pip install -U [--user] ./panther

.. _github repository: https://github.com/lmmentel/panther
.. _pip: https://pip.pypa.io/en/stable/

.. [1] Bučko, T., Hafner, J., & Ángyán, J. G. (2005). Geometry optimization of
   periodic systems using internal coordinates. The Journal of Chemical Physics,
   122(12), 124508. doi:10.1063/1.1864932
