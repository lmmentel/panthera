
from __future__ import print_function, absolute_import, division

import numpy as np
from scipy.constants import angstrom, pi, value
from numpy.linalg import inv as npinvert

import ase.io

from writeBmat import get_internals

from .vibrations import project

# TODO: this should generate a list/dict of Atoms objects
#       having displaced coordiantes assigned to atoms, this
#       will enable then to generate input files and run the
#       jobs,
#       the dict key might be a tuple of (mode, point)


THR = 1.0e-6


def calculate_displacements(atoms, hessian, npoints, mode_min=None,
                            mode_max=None):
    '''
    Calculate displacements in internal coordinates

    Args:
        atoms : ase.Atoms
        hessian : numpy.array
        npoints : int
            Number of points to displace structure, the code will calculate
            ``2*npoints`` displacements since + and - directions are taken
        mode_min : int
            Smallest mode number
        mode_max : int
            Largest mode number
    '''

    prm = 1.0 / value('electron mass in u')
    ang2bohr = angstrom / value('atomic unit of length')
    invcm2au = 100.0 * value('inverse meter-hartree relationship')

    natoms = atoms.get_number_of_atoms()
    ndof = 3 * natoms
    masses = atoms.get_masses()
    pos = atoms.get_positions()

    if mode_min is None:
        mode_min = 0

    if mode_max is None:
        mode_max = ndof

    coords = pos.ravel() * ang2bohr

    # write equilibrium POSCAR
    ase.io.write('POSCAR.eq', atoms, vasp5=True)

    # internals is a numpy record array with 'type' and 'value' records
    # bmatrix is a numpy array n_int x n_cart
    internals, Bmatrix = get_internals(atoms, return_bmatrix=True)

    # matrix with square root of masses
    B_mass_inv = np.zeros((ndof, ndof), dtype=float)
    np.fill_diagonal(B_mass_inv, np.repeat(1.0 / np.sqrt(masses * prm), 3))

    # matrix with inverse masses
    M_inv = np.zeros((ndof, ndof), dtype=float)
    np.fill_diagonal(M_inv, np.repeat(1.0 / (masses * prm), 3))

    Gmatrix = np.dot(Bmatrix, np.dot(M_inv, Bmatrix.T))

    Gmatrix_inv = npinvert(Gmatrix)

    # symmetrize the hessian
    hessian = (hessian + hessian.T) * 0.5e0

    mwhessian = -np.dot(B_mass_inv, np.dot(hessian, B_mass_inv))

    # TODO: projection needs to be updated for the case of mass weighted coords
    #       see: projection.f90, trvmatr.f90 from eigenhess
    prmwhessian = project(mwhessian)

    evals, evecs = np.linalg.eigh(prmwhessian)

    mwevecs = np.dot(B_mass_inv, evecs)

    Bmatrix_inv = np.dot(M_inv, np.dot(Bmatrix.T, Gmatrix_inv))

    Dmatrix = np.dot(Bmatrix, mwevecs)

    is_stretch = vib_population(atoms, hessian, evecs, Bmatrix_inv, Dmatrix,
                                internals)

    displ = np.zeros(len(internals), dtype=float)
    displ[is_stretch] = 8.0 / np.sqrt(2.0 * pi * np.sqrt(np.abs(evals[is_stretch])))
    displ[~is_stretch] = 4.0 / np.sqrt(2.0 * pi * np.sqrt(np.abs(evals[~is_stretch])))
    displ = displ / (npoints * 2.0)

    eff_mass = 1.0 / np.diagonal(np.dot(mwevecs.T, mwevecs))

    for mode in range(mode_min, mode_max):

        # eff_mass = 1.0 / np.dot(mwevecs[:, mode], mwevecs[:, mode])

        nu = np.sqrt(np.abs(evals[mode])) / invcm2au
        if nu < THR and nu > 0.0:

            for sign in [1, -1]:

                for point in range(npoints):
                    print(' Point : {} '.format(point).center(80, '='))

                    coords = pos.ravel() * ang2bohr

                    internal_coord_disp = sign * Dmatrix[:, mode] * disp * point

                    cart_coord_disp = np.dot(Bmatrix_inv, internal_coord_disp)

                    coords += cart_coord_disp

                    coords_init = coords.copy()

                    iteration = 0
                    not_converged = True
                    while not_converged:
                        # update atoms with new coords
                        internals_new, _ = getbmatrix(atoms)

                        delta_int = internal_coord_disp - (internals_new - internals)

                        disp_norm = np.sqrt(np.dot(delta_int, delta_int))



def vib_population(atoms, hessian, h_evecs, Bmatrix_inv, Dmatrix, internals, output):
    '''
    Calculate the vibrational population analysis

    Args:
        atoms : ase.Atoms
        hessian : numpy.array
        output : str

    Returns:
        is_stretch : numpy array
            A boolean mask array, with ``True`` value for stretching frequencies
    '''

    # Wilson F matrix
    Fmatrix = np.dot(Bmatrix_inv.T, np.dot(hessian, Bmatrix_inv))

    # norm of |Fmatrix - Fbcktr| can be used to check the accuracy of the transformation
    # Fbcktr = np.dot(Bmatrix.T, np.dot(Fmatrix, Bmatrix))

    is_stretch = np.zeros(3 * atoms.get_number_of_atoms(), dtype=bool)





    # print a table in the format
    #
    # | mode  | omega [cm^-1] | stretch | bend | torsion | long-range |
    # |------:|--------------:|--------:|-----:|--------:|-----------:|
    # |     1 |   3136.932765 |     100 |    0 |       0 |          0 |
    # ...

    return is_stretch
