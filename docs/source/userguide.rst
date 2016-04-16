User guide
==========

Available CLI programs:

    - :ref:`panther-label` format conversion, calcualtion of harmonic and anharmonic frequencies
    - :ref:`plotmode-label` visualiztion of vibrational potential per mode
    - :ref:`writemodes-label` conversion of geometry files to collections of modes


.. _panther-label:

panther
-------

The panther script takes two command line argument

.. code-block:: bash

   $ panther

   usage: panther [-h] {convert,harmonic,anharmonic} config

   positional arguments:
     {convert,harmonic,anharmonic}
                           choose what to do
     config                file with the configuration parameters for thermo

   optional arguments:
     -h, --help            show this help message and exit



The input file is in the standard condif file format and contains three sections
``conditions``, ``job`` and ``system`` defining the parameters. 

conditions
^^^^^^^^^^
    
    pressure : float
        Pressure in MPa
    Tinitial : float
        Smallest Temperature in K
    Tfinal : float
        Largest temperatue in K
    Tstep : float
        Temperature step for temperature grid (in K)

.. code-block:: python

   [conditions]
   Tinitial = 303.15
   Tfinal = 403.15
   Tstep = 10.0
   pressure = 0.1

job
^^^
    
    translations : bool
        If True the translational degrees of freedom will be projected out from the hessian
    rotations : bool
        If True the translational degrees of freedom will be projected out from the hessian
    code : str
        Program to use for single point calcualtions

.. code-block:: python

   [job]
   translations = true
   rotations = false
   code = vasp


system
^^^^^^
    
    pointgroup : str
        Point group symbol of the system
    phase : str
        Phase of the system, either ``gas`` or ``solid``

.. code-block:: python

   [system]
   pointgroup = Dooh
   phase = gas

.. _plotmode-label:

plotmode
--------

.. code-block:: bash

   $ plotmode -h

   usage: plotmode [-h] [-s SIXTH] [-f FOURTH] [-p PES] mode

   positional arguments:
     mode                  number of the mode to be printed

   optional arguments:
     -h, --help            show this help message and exit
     -s SIXTH, --sixth SIXTH
                           file with sixth order polynomial fit,
                           default="em_freq"
     -f FOURTH, --fourth FOURTH
                           file with fourth order polynomial fit,
                           default="em_freq_4th"
     -p PES, --pes PES     file with the potential energy surface (PES),
                           default="test_anharm"



Example
^^^^^^^

Provided that the default files ``em_freq``, ``em_freq_4th`` and ``test_anharm`` are present
to plot the first mode only requires the argument ``1``

.. code-block:: bash

   plotmode 1

.. image:: gfx/mode1.png
    :width: 800px
    :align: center
    :alt: Plot of the mode potential


.. _writemodes-label:

writemodes
----------

This program takes the single file with continuous geometries in VASP POSCAR format as input
and writes separate file in ASE trajectory format per node to a specified directory.

.. code-block:: bash

   $ writemodes -h

   usage: writemodes [-h] [-d DIR] filename

   positional arguments:
     filename           name of the file with geometries, default="POSCARs"

   optional arguments:
     -h, --help         show this help message and exit
     -d DIR, --dir DIR  directory to put the modes, default="modes"

