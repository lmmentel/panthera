Tutorials
=========

Anharmonic Thermochemistry
--------------------------

Internal coordinate displacements
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

First of all you'll need a relaxed structure and a `hessian matrix`_, it is
recommended to converge the forces below 1.0e-5 eV/A.

In the code below methanol molecule is used as an example and the relaxed
structure is stored in a trajectory_ file. Moreover the `hessian matrix`_
is available as an `numpy array`_ with the shape N x N, where N = 18 (number
of the degrees of freedom). The hessian is in `atomic units`_.

Next step is to generate a grid of displacements for each normal mode using
internal coordinates. This can be done using
:py:func:`calculate_displacements <panther.displacements.calculate_displacements>`
function. The function returns a nested :py:class:`OrderedDict <collections.OrderedDict>` of structures
as `ase.Atoms`_ objects with mode number and displacement sample number as keys.
For example if ``npoints=4`` is given as an argument there will be 8 structures
per mode labeled with numbers 1, 2, 3, 4, -1, -2, -3, -4 signifying the direction
and the magnitude of the displacement. 

The function also returns ``modeinfo`` DataFrame_ with additional characteristics
of the mode such as ``displacement``, ``is_stretch`` and ``effective_mass``.

It also produces a file with vibrational population analysis ``vib_pop.log``.

.. code-block:: python

   import ase.io
   import numpy as np
   from panther.displacements import calculate_displacements

   meoh = ase.io.read('meoh.traj')
   hessian = np.load('hessian.npy')

   images, modeinfo = calculate_displacements(meoh, hessian, npoints=4)


Calculating energies for the displaced structures
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Now per each structure we can calculate the energy, in this example
using the VASP calculator interface

.. code-block:: python

   from ase.calculators.vasp import Vasp
   from panther.pes import calculate_energies


   calc = Vasp(
           prec='Accurate',
           gga='PE',
           lreal=False,
           ediff=1.0e-8,
           encut=600.0,
           nelmin=5,
           nsw=1,
           nelm=100,
           ediffg=-0.001,
           ismear=0,
           ibrion=-1,
           nfree=2,
           isym=0,
           lvdw=True,
           lcharg=False,
           lwave=False,
           istart=0,
           npar=2,
           ialgo=48,
           lplane=True,
           ispin=1,
   )

   energies = calculate_energies(images, calc, modes='all')

This will return a DataFrame_ with ``npoints * 2`` energies per mode.

The ``energies`` are missing the equilibrium structure energy which can be
easily set through

.. code-block:: python

   energies['E_0'] = meoh.get_potential_energy()

   energies
            E_-4       E_-3       E_-2       E_-1        E_0        E_1        E_2        E_3        E_4
   0  -29.837971 -29.999032 -30.129785 -30.219157 -30.252818 -30.212085 -30.072419 -29.801395 -29.356168
   1  -29.596340 -29.913292 -30.113599 -30.220609 -30.252818 -30.224942 -30.148733 -30.033567 -29.886922
   2  -29.766539 -29.984240 -30.134987 -30.223584 -30.252818 -30.223564 -30.134827 -29.983692 -29.765209
   3  -29.880605 -30.032762 -30.149874 -30.225677 -30.252818 -30.222672 -30.125184 -29.948648 -29.679404
   4  -30.194525 -30.220204 -30.238397 -30.249231 -30.252818 -30.249266 -30.238675 -30.221134 -30.196737
   5  -30.196200 -30.220992 -30.238680 -30.249284 -30.252818 -30.249285 -30.238675 -30.220973 -30.196157
   6  -30.199365 -30.222449 -30.239173 -30.249365 -30.252818 -30.249283 -30.238467 -30.220021 -30.193522
   7  -30.198879 -30.222722 -30.239552 -30.249531 -30.252818 -30.249581 -30.239995 -30.224237 -30.202484
   8  -30.208404 -30.227854 -30.241729 -30.250046 -30.252818 -30.250048 -30.241733 -30.227861 -30.208420
   9  -30.209174 -14.455888 -30.242217 -30.250206 -30.252818 -30.250275 -30.242783 -30.230543 -30.213739
   10 -30.210682 -30.229501 -30.242619 -30.250308 -30.252818 -30.250389 -30.243244 -30.231599 -30.215658
   11 -30.242001 -30.246562 -30.249985 -30.252102 -30.252818 -30.252093 -30.249970 -30.246538 -30.241966


Calculating the frequencies
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Frequencies can now be calculated using the finite difference method implemented in
:py:func:`panther.pes.differentiate` function and appended as a ``frequency`` column
to the ``modeinfo`` 

.. code-block:: python

   from panther.pes import differentiate
   from scipy.constants import value

   # calculate the frequencies
   dsp = modeinfo['displacement'][:12].astype(float).values
   vibs = differentiate(dsp, energies, order=2)
   
   # update the modeinfo with frequencies
   au2invcm = 0.01 * value('hartree-inverse meter relationship')
   mmodeinfo.loc[:11, 'frequency'] = (np.sqrt(vibs)*au2invcm)[:, 3]


Fitting the potentials
~~~~~~~~~~~~~~~~~~~~~~

The last this is to fit the potentials into 6th and 4th order polynomials

.. code-block:: python

   from panther.pes import fit_potentials

   # fit the potentials on 6th and 4th order polynomials
   c6o, c4o = fit_potentials(mi, energies)

The two DataFrame_ object ``c6o`` and ``c4o`` contain fitted polynomial coefficients for each
mode.

.. _OUTCAR: http://cms.mpi.univie.ac.at/vasp/guide/node50.html#SECTION00070000000000000000
.. _hessian matrix: https://en.wikipedia.org/wiki/Hessian_matrix
.. _trajectory: https://wiki.fysik.dtu.dk/ase/ase/io/trajectory.html
.. _numpy array: http://docs.scipy.org/doc/numpy-1.10.0/reference/generated/numpy.array.html
.. _atomic units: https://en.wikipedia.org/wiki/Atomic_units
.. _ase.Atoms: https://wiki.fysik.dtu.dk/ase/ase/atoms.html#module-ase.atoms
.. _DataFrame: http://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.html