Tutorials
=========

Internal coordinate displacements
---------------------------------

Assuming that there is and OUTCAR_ the following script will calculate the displaced
structures that are returned into ``images`` as a list of ``ase.Atoms`` objects

.. code-block:: python

   from ase.io.vasp import read_vasp_out

   from panther.inputreader import read_vasp_hessian
   from panther.displacements import calculate_displacements

   atoms = read_vasp_out('OUTCAR')
   hessian = read_vasp_hessian('OUTCAR', symmetrize=True, convert2au=True)

   images = calculate_displacements(meoh, hessian, npoints=4)


.. _OUTCAR: http://cms.mpi.univie.ac.at/vasp/guide/node50.html#SECTION00070000000000000000
