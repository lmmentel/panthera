
from __future__ import print_function, absolute_import, division

import pickle
import copy
import numpy as np
import pandas as pd
from scipy.constants import angstrom, pi, value
from collections import OrderedDict
from six import string_types

from writeBmat import get_internals

from .inputreader import print_modeinfo
from .pes import expandrange

FREQ_THRESH = 1.0e6
DISPNORM_THRESH = 1.0e-2
CARTDISP_THRESH = 1.0e-6


def get_nvibdof(atoms, proj_rotations, proj_translations, phase,
                include_constr=False):
    '''
    Calculate the number of vibrational degrees of freedom

    Parameters
    ----------
    atoms : ase.Atoms
    proj_translations : bool
        If ``True`` translational degrees of freedom will be projected from
        the hessian
    proj_rotations : bool
        If ``True`` rotational degrees of freedom will be projected from
        the hessian
    include_constr : bool
        If ``True`` the constraints will be included

    Returns
    -------
    nvibdof : float
        Number of vibrational degrees of freedom
    '''

    # get the total number of degrees of freedom
    if include_constr:
        ndof = 3 * (len(atoms) - len(atoms.constraints))
    else:
        ndof = 3 * len(atoms)

    extradof = 0
    if phase.lower() == 'gas':
        if proj_rotations & proj_translations:
            if ndof > 6:
                extradof = 6
            elif ndof == 6:
                extradof = 5
    elif phase.lower() == 'solid':
        if proj_rotations | proj_translations:
            extradof = 3
    else:
        raise ValueError('Wrong phase specification: {}, expecting either '
                         '"gas" or "solid"'.format(phase))

    return ndof - extradof


def calculate_displacements(atoms, hessian, freqs, normal_modes, npoints=4,
                            modes='all', verbose=False):
    '''
    Calculate displacements in internal coordinates

    Parameters
    ----------
    atoms : ase.Atoms
        Atoms object
    hessian : array_like
        Hessian matrix
    freqs : array_like
        Frequencies (square root of the hessian eigenvalues) in atomic units
    normal_modes : array_like
        Normal modes in atomic units
    npoints : int
        Number of points to displace structure, the code will calculate
        ``2*npoints`` displacements since + and - directions are taken
    modes : str or list/tuple of ints, default 'all'
        Range of the modes for which the displacements will be calculated
    verbose : bool
        If ``True`` additional debug information is printed to stdout

    Returns
    -------
    images : dict of dicts
        A nested (ordred) dictionary with the structures with mode, point as
        keys, where point is a number from -4, -3, -2, -1, 1, 2, 3, 4
    mi : pandas.DataFrame
        DataFrame with per mode characteristics, displacements, masses
        and vibrational population analysis
    '''

    ang2bohr = angstrom / value('atomic unit of length')
    prm = 1.0 / value('electron mass in u')
    au2invcm = 0.01 * value('hartree-inverse meter relationship')

    natoms = atoms.get_number_of_atoms()
    ndof = 3 * natoms
    masses = atoms.get_masses()
    pos = atoms.get_positions()

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

    mwevecs = np.dot(M_invsqrt, normal_modes)

    Bmatrix_inv = np.dot(M_inv, np.dot(Bmatrix.T, Gmatrix_inv))

    Dmatrix = np.dot(Bmatrix, mwevecs)

    # DataFrame with mode data
    mi = pd.DataFrame(index=pd.Index(data=range(ndof), name='mode'),
                      columns=['HOfreq', 'effective_mass', 'displacement',
                               'is_stretch', 'vibration'])

    mi['HOfreq'] = freqs * au2invcm
    mi['vibration'] = (mi.HOfreq != 0.0) & (mi.HOfreq.notnull())

    mi = vib_population(hessian, freqs, Bmatrix_inv, Dmatrix, internals, mi)

    mi['is_stretch'] = mi['P_stretch'] > 0.9
    mi['effective_mass'] = 1.0 / np.einsum('ij,ji->i', mwevecs.T, mwevecs)

    # calculate the megnitude of the displacement for all the modes
    mi.loc[mi['is_stretch'], 'displacement'] = 8.0 / np.sqrt(2.0 * pi *
                        np.abs(freqs[mi['is_stretch'].values]))
    mi.loc[~mi['is_stretch'], 'displacement'] = 4.0 / np.sqrt(2.0 * pi *
                        np.abs(freqs[~mi['is_stretch'].values]))
    mi['displacement'] = mi['displacement'] / (npoints * 2.0)
    mi.to_pickle('modeinfo.pkl')

    if verbose:
        print_modeinfo(mi)

    if isinstance(modes, string_types):
        if modes.lower() in ['all', ':']:
            modes = mi[mi['vibration']].index.values
        else:
            modes = expandrange(modes)
    elif isinstance(modes, (list, tuple)):
        pass
    else:
        ValueError('<modes> should be a str, list or tuple '
                   'got: {}'.format(type('modes')))

    images = OrderedDict()

    for mode in modes:
        nu = np.abs(freqs[mode]) * au2invcm
        images[mode] = OrderedDict()

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

                    internal_coord_disp = sign * Dmatrix[:, mode] *\
                                          mi.loc[mode, 'displacement'] * point

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
                    images[mode][sign * point] = newatoms

    with open('images.pkl', 'w') as fpkl:
        pickle.dump(images, fpkl)

    return images, mi


def vib_population(hessian, freqs, Bmatrix_inv, Dmatrix, internals, mi):
    '''
    Calculate the vibrational population analysis

    Parameters
    ----------
    hessian : array_like
        Hessian matrix
    freqs : array_like
        A vector of frequencies (square roots fof hessian eigenvalues)
    Bmatrix_inv : array_like
        Inverse of the B matrix
    Dmatrix : array_like
        D matrix
    internals : array_like
        Structured array with internal coordinates
    mi : pandas.DataFrame
        Modeinfo

    Returns
    -------
    mi : pandas.DataFrame
        Modeinfo DataFrame updated with columns with vibrational
        population analysis results
    '''

    # Wilson F matrix
    Fmatrix = np.dot(Bmatrix_inv.T, np.dot(hessian, Bmatrix_inv))

    # norm of |Fmatrix - Fbcktr| can be used to check the accuracy of
    # the transformation Fbcktr = np.dot(Bmatrix.T, np.dot(Fmatrix, Bmatrix))

    # construct the nu matrix with population contributions from
    # internal coordinates to cartesians
    nu = np.multiply(Dmatrix.T, np.dot(Dmatrix.T, Fmatrix))
    nu[nu < 0.0] = 0.0
    # divide each column of nu by the hessian eigenvalues
    nu = nu / np.power(freqs[:, np.newaxis], 2.0)

    icnames = [('R', 'P_stretch'), ('A', 'P_bend'), ('T', 'P_torsion'),
               ('IR1', 'P_longrange')]

    for iname, cname in icnames:
        mi.loc[:, cname] = 0.0
        if iname in internals['type'].tolist():
            mask = internals['type'] == iname
            # sum over rows of nu to get the vibrational populations
            mi.loc[:, cname] = np.sum(nu[:, mask], axis=1)

    return mi
