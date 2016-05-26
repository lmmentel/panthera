
from __future__ import print_function, absolute_import, division

import copy
import numpy as np
from scipy.constants import angstrom, pi, value
from collections import OrderedDict

from writeBmat import get_internals

from .vibrations import get_harmonic_vibrations

FREQ_THRESH = 1.0e6
DISPNORM_THRESH = 1.0e-2
CARTDISP_THRESH = 1.0e-6


def calculate_displacements(atoms, hessian, npoints, mode_min=None,
                            mode_max=None, verbose=False):
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
        verbose : bool
            If ``True`` additional debug information is printed to stdout

    Returns:
        images : dict
            A dictionary with the structures with tuples of (mode, point) as
            keys, where point is a number from -4, -3, -2, -1, 1, 2, 3, 4
    '''

    ang2bohr = angstrom / value('atomic unit of length')
    prm = 1.0 / value('electron mass in u')
    au2invcm = 0.01 * value('hartree-inverse meter relationship')

    images = OrderedDict()

    natoms = atoms.get_number_of_atoms()
    ndof = 3 * natoms
    masses = atoms.get_masses()
    pos = atoms.get_positions()

    if mode_min is None:
        mode_min = 0

    if mode_max is None:
        mode_max = ndof

    coords = pos.ravel() * ang2bohr

    # internals is a numpy record array with 'type' and 'value' records
    # bmatrix is a numpy array n_int x n_cart
    internals, Bmatrix = get_internals(atoms, return_bmatrix=True)

    mask = internals['value'] < 0.0
    internals['value'][mask] = 2 * pi + internals['value'][mask]

    # matrix with inverse square roots of masses on diagonal
    M_invsqrt = np.zeros((ndof, ndof), dtype=float)
    np.fill_diagonal(M_invsqrt, np.repeat(1.0 / np.sqrt(masses * prm), 3))

    # matrix with inverse masses
    M_inv = np.zeros((ndof, ndof), dtype=float)
    np.fill_diagonal(M_inv, np.repeat(1.0 / (masses * prm), 3))

    Gmatrix = np.dot(Bmatrix, np.dot(M_inv, Bmatrix.T))
    Gmatrix_inv = np.linalg.pinv(Gmatrix)

    # calculate hessian eigenvalues and eigenvectors
    evals, evecs = get_harmonic_vibrations(hessian, atoms,
                                           proj_translations=True,
                                           proj_rotations=True)
    vibdof = np.count_nonzero(evals)

    mwevecs = np.dot(M_invsqrt, evecs)

    Bmatrix_inv = np.dot(M_inv, np.dot(Bmatrix.T, Gmatrix_inv))

    Dmatrix = np.dot(Bmatrix, mwevecs)

    eff_mass = 1.0 / np.einsum('ij,ji->i', mwevecs.T, mwevecs)
    np.save('effective_masses', eff_mass)

    vibpop = vib_population(hessian, evals, Bmatrix_inv, Dmatrix, internals,
                            vibdof)
    is_stretch = vibpop['R'] > 0.9

    # calculate the megnitude of the displacement for all the modes
    displ = np.zeros(ndof, dtype=float)
    displ[is_stretch] = 8.0 / np.sqrt(2.0 * pi * np.sqrt(np.abs(evals[is_stretch])))
    displ[~is_stretch] = 4.0 / np.sqrt(2.0 * pi * np.sqrt(np.abs(evals[~is_stretch])))
    displ = displ / (npoints * 2.0)

    for mode in range(mode_min, mode_max):
        nu = np.sqrt(np.abs(evals[mode])) * au2invcm
        if nu < FREQ_THRESH and nu > 0.0:

            for sign in [1, -1]:
                for point in range(1, npoints + 1):
                    if verbose:
                        line = ' mode : {0:d} '.format(mode) +\
                               ' nu : {1:.4f} '.format(nu) +\
                               ' point : {1:d} '.format(point * sign)
                        print(line.center(80, '*'))

                    # equilibrium structure
                    coords = pos.ravel().copy() * ang2bohr

                    internal_coord_disp = sign * Dmatrix[:, mode] * displ[mode] * point
                    cart_coord_disp = np.dot(Bmatrix_inv, internal_coord_disp)

                    coords += cart_coord_disp
                    coords_init = coords.copy()

                    iteration = 1
                    not_converged = True
                    while not_converged:
                        # update atoms with new coords
                        newatoms = atoms.copy()
                        newatoms.set_positions(coords.reshape(natoms, 3) / ang2bohr)
                        internals_new, Bmatrix = get_internals(newatoms,
                                                        return_bmatrix=True)

                        mask = internals_new['value'] < 0.0
                        internals_new['value'][mask] = 2 * pi + internals_new['value'][mask]

                        if verbose:
                            print('internals'.center(80, '-'))
                            for row in internals_new:
                                print('{0:5s} {1:20.10f}'.format(row['type'], row['value']))

                        delta_int = internal_coord_disp - (internals_new['value'] - internals['value'])

                        disp_norm = np.sqrt(np.dot(delta_int, delta_int))
                        print('disp_norm: ', disp_norm)

                        if iteration == 1:
                            disp_norm_init = copy.copy(disp_norm)
                        elif iteration > 1:
                            if disp_norm - disp_norm_init > DISPNORM_THRESH:
                                print('### Back iteration not convergerd after', iteration, 'iterations')
                                print('### disp_norm - disp_norm_init: ', disp_norm - disp_norm_init)
                                coords = coords_init.copy()
                                break

                        for internal_type in ['A', 'T']:
                            mask = np.logical_and(internals['type'] == internal_type, delta_int > pi)
                            delta_int[mask] = 2 * pi - np.abs(delta_int[mask])

                        cart_coord_disp = np.dot(Bmatrix_inv, delta_int)

                        coords += cart_coord_disp

                        if np.max(np.abs(cart_coord_disp)) < CARTDISP_THRESH:
                            print('### convergence achieved after ', iteration, ' iterations')
                            break
                        elif iteration > 25:
                            print('### convergence NOT achieved after ', iteration, ' iterations')
                            break
                        else:
                            iteration += 1

                    newatoms.set_positions(coords.reshape(natoms, 3) / ang2bohr)
                    images[tuple([mode, sign * point])] = newatoms

    return images


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

    au2invcm = 0.01 * value('hartree-inverse meter relationship')

    if output is not None:
        fobj = open(output, 'w')

    print('{0:^5s} {1:^10s} {2:^8s} {3:^8s} {4:^8s}'.format('mode', 'omega',
        'stretch', 'bend', 'torsion'), file=fobj)
    for mode, row in enumerate(vibpop[:vibdof]):
        print('{0:5d} {1:>10.4f} {2:>8.2%} {3:>8.2%} {4:>8.2%}'.format(
            mode + 1, np.sqrt(evals[mode]) * au2invcm, row['R'], row['A'],
            row['T']), file=fobj)

    if output is not None:
        fobj.close()
