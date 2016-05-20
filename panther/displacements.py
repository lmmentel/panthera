
from __future__ import print_function, absolute_import, division

import numpy as np
from scipy.constants import angstrom, pi, value

from writeBmat import get_internals

from .vibrations import project

# TODO: this should generate a list/dict of Atoms objects
#       having displaced coordiantes assigned to atoms, this
#       will enable then to generate input files and run the
#       jobs,
#       the dict key might be a tuple of (mode, point)


THR = 1.0e-6
THRESH = 1.0e-14
prm = 1.0 / value('electron mass in u')
ang2bohr = angstrom / value('atomic unit of length')
ev2hartree = value('electron volt-hartree relationship')
invcm2au = 100.0 * value('inverse meter-hartree relationship')


def calculate_displacements(atoms, hessian, npoints, mode_min=None,
                            mode_max=None):
    '''
    Calculate displacements in internal coordinates

    Args:
        atoms : ase.Atoms
            Atoms object
        hessian : numpy.array
            Symmetric hessian matrix in atomic units [hartree/bohr**2]
        npoints : int
            Number of points to displace structure, the code will calculate
            ``2*npoints`` displacements since + and - directions are taken
        mode_min : int
            Smallest mode number
        mode_max : int
            Largest mode number
    '''

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
    # ase.io.write('POSCAR.eq', atoms, vasp5=True)

    # internals is a numpy record array with 'type' and 'value' records
    # bmatrix is a numpy array n_int x n_cart
    internals, Bmatrix = get_internals(atoms, sort=False, return_bmatrix=True)

    # matrix with square root of masses
    B_mass_inv = np.zeros((ndof, ndof), dtype=float)
    np.fill_diagonal(B_mass_inv, np.repeat(1.0 / np.sqrt(masses * prm), 3))

    # matrix with inverse masses
    M_inv = np.zeros((ndof, ndof), dtype=float)
    np.fill_diagonal(M_inv, np.repeat(1.0 / (masses * prm), 3))

    Gmatrix = np.dot(Bmatrix, np.dot(M_inv, Bmatrix.T))

    Gmatrix_inv = np.linalg.pinv(Gmatrix)

    # calculate hessian eigenvalues and eigenvectors
    prhessian = project(atoms, hessian, ndof, proj_translations=True,
                        proj_rotations=True)
    prmwhessian = np.dot(B_mass_inv, np.dot(prhessian, B_mass_inv))
    evals, evecs = np.linalg.eigh(prmwhessian)

    # sort eigenvalues and corresponding eiegenvectors in descending order
    evals[evals < THRESH] = 0.0
    vibdof = np.count_nonzero(evals)

    evals = evals[::-1]
    evecs = evecs[:, ::-1]

    mwevecs = np.dot(B_mass_inv, evecs)

    Bmatrix_inv = np.dot(M_inv, np.dot(Bmatrix.T, Gmatrix_inv))

    Dmatrix = np.dot(Bmatrix, mwevecs)

    eff_mass = 1.0 / np.einsum('ij,ji->i', mwevecs.T, mwevecs)

    vibpop = vib_population(hessian, evals, Bmatrix_inv, Dmatrix, internals,
                            vibdof)
    is_stretch = vibpop['R'] > 0.9

    return internals, Bmatrix

    nint = len(internals)

    displ = np.zeros(nint, dtype=float)
    displ[is_stretch] = 8.0 / np.sqrt(2.0 * pi * np.sqrt(np.abs(evals[is_stretch])))
    displ[~is_stretch] = 4.0 / np.sqrt(2.0 * pi * np.sqrt(np.abs(evals[~is_stretch])))
    displ = displ / (npoints * 2.0)

    for mode in range(mode_min, mode_max):

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
                        internals_new = get_internals(atoms)

                        delta_int = internal_coord_disp - (internals_new - internals)

                        disp_norm = np.sqrt(np.dot(delta_int, delta_int))


def vib_population(hessian, h_evals, Bmatrix_inv, Dmatrix, internals, vibdof,
                   output='vib_pop.log'):
    '''
    Calculate the vibrational population analysis

    Args:
        hessian : numpy.array
            Hessian matrix
        h_evals : numpy.array
            A vector of hessian eigenvalues
        vibdof : int
            Number of vibrational degrees of freedom
        output : str
            Name of the file to store the results

    Returns:
        vibpop : numpy array
            Numpy structured array with the vibrational populations for
            stretches, bends, and torsions, per mode
    '''

    nint, ndof = Dmatrix.shape

    # Wilson F matrix
    Fmatrix = np.dot(Bmatrix_inv.T, np.dot(hessian, Bmatrix_inv))

    # norm of |Fmatrix - Fbcktr| can be used to check the accuracy of
    # the transformation Fbcktr = np.dot(Bmatrix.T, np.dot(Fmatrix, Bmatrix))

    # construct the nu matrix with poppulation contributions from
    # internal coordinates to cartesians
    nu = np.multiply(Dmatrix.T, np.dot(Dmatrix.T, Fmatrix))
    nu[nu < 0.0] = 0.0
    # divide each column of nu by the hessian eigenvalues
    nu = nu / h_evals[:, np.newaxis]

    # sum over rows of nu to get the vibrational populations
    internal_types = np.unique(internals['type']).tolist()
    vibpop = np.zeros(ndof, dtype=list(zip(internal_types,
                                [float] * len(internal_types))))

    for inttype in internal_types:
        mask = internals['type'] == inttype
        vibpop[inttype] = np.sum(nu[:, mask], axis=1)

    print_vib_pop(vibpop, h_evals, vibdof, output=output)

    return vibpop


def print_vib_pop(vibpop, evals, vibdof, output='vib_pop.log'):
    '''
    Print the vibrational population data

    Args:
        vibpop : numpy.recarray
            Numpy structured array with the vibrational populations for
            stretches, bends, and torsions, per mode
        h_evals : numpy.array
            A vector of hessian eigenvalues
        vibdof : int
            Number of vibrational degrees of freedom
        output : str
            Name of the file to store the printout, if ``None`` stdout will be
            used
    '''

    if output is not None:
        fobj = open(output, 'w')

    print('{0:^5s} {1:^10s} {2:^8s} {3:^8s} {4:^8s}'.format('mode', 'omega', 'stretch', 'bend', 'torsion'), file=fobj)
    for mode, row in enumerate(vibpop[:vibdof]):
        print('{0:5d} {1:>10.4f} {2:>8.2%} {3:>8.2%} {4:>8.2%}'.format(
            mode + 1, np.sqrt(evals[mode]) / invcm2au, row['R'], row['A'],
            row['T']), file=fobj)

    if output is not None:
        fobj.close()
