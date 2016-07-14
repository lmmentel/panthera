
from __future__ import print_function

from datetime import datetime

import numpy as np
from .displacements import get_nvibdof
from .vibrations import harmonic_vibrational_analysis

THREVAL = 1.0e-12


def nmoptimize(atoms, hessian, calc, phase, proj_translations=True,
               proj_rotations=True, gtol=1.0e-5, verbose=False,
               hessian_update='BFGS', maxiter=50):
    '''
    Relax the strcture using normal mode displacements

    Parameters
    ----------
    atoms : ase.Atoms
    hessian : array_like
        Hessian matrix in eV/Angstrom^2
    calc : ase.Calculator
        ASE Calcualtor instance to be used to calculate forces
    phase : str
        Phase, 'solid' or 'gas'
    gtol : float, default=1.0e-5
        Energy gradient threshold
    hessian_update : str
        Approximate formula to update hessian, possible values are 'BFGS',
        'SR1' and 'DFP'
    maxiter : int
        Maximal number of iteration to be performed
    verbose : bool
        If ``True`` additional debug information will be printed

    Notes
    -----

    Internally eV and Angstroms are used.

    .. seealso::

       Bour, P., & Keiderling, T. A. (2002). Partial optimization of molecular geometry
       in normal coordinates and use as a tool for simulation of vibrational spectra.
       The Journal of Chemical Physics, 117(9), 4126.
       `doi:10.1063/1.1498468 <http://dx.doi.org/10.1063/1.1498468>`_

    '''

    natoms = atoms.get_number_of_atoms()
    ndof = 3 * natoms
    masses = atoms.get_masses()
    coords = atoms.get_positions().ravel()
    nvibdof = get_nvibdof(atoms, proj_rotations, proj_translations, phase)

    # matrix with inverse square roots of masses on diagonal
    M_invsqrt = np.zeros((ndof, ndof), dtype=float)
    np.fill_diagonal(M_invsqrt, np.repeat(1.0 / np.sqrt(masses), 3))

    # calculate hessian eigenvalues and eigenvectors
    evals, evecs = harmonic_vibrational_analysis(hessian, atoms,
                                           proj_translations=proj_translations,
                                           proj_rotations=proj_rotations,
                                           ascomplex=False, massau=False)

    evals = np.power(evals, 2)

    mwevecs = np.dot(M_invsqrt, evecs)

    coords_old = coords.copy()

    # run the job for the initial structure
    atoms.set_calculator(calc)

    # get forces after run
    grad = -1.0 * atoms.get_forces().ravel()

    grad_old = grad.copy()

    grad_nm = np.dot(mwevecs.T, grad)

    step_nm = np.zeros_like(grad_nm)
    step_nm[:nvibdof] = -2.0 * grad_nm[:nvibdof] / (evals[:nvibdof]
        + np.sqrt(evals[:nvibdof]**2 + 4.0 * grad_nm[:nvibdof]**2))

    step_cart = np.dot(mwevecs, step_nm)
    coords = coords_old + step_cart

    if verbose:
        print(' eigenvalues '.center(50, '-'))
        print(evals)

        print(' cart gradient '.center(50, '-'))
        print(grad)

        print(' nm gradient '.center(50, '-'))
        print(grad_nm)

        print(' nm step '.center(50, '-'))
        print(step_nm)

        print(' cart step '.center(50, '-'))
        print(step_cart)

        print(' new coordinates '.center(50, '-'))
        print(coords)

    not_converged = True
    iteration = 0

    # header for the convergence information
    print('{0:<6s} {1:^15s} {2:^15s} {3:^15s} {4:^15s} {5:^20s}'.format(
        'iter', 'G(NM) max', 'G(NM) norm', 'NM step norm', 'energy [eV]', 'time'))
    print('=' * 91)
    print('{0:>6d} {1:>15.8f} {2:>15.8f} {3:>15.8f} {4:>15.8f} {5:>20s}'.format(
        iteration, np.max(np.abs(grad_nm)), np.sqrt(np.dot(grad_nm, grad_nm)),
        np.sqrt(np.dot(step_nm, step_nm)), atoms.get_potential_energy(),
        datetime.now().strftime('%H:%M:%S %d-%m-%Y')))

    while not_converged:
        iteration += 1

        if iteration > maxiter:
            print('### convergence NOT achieved after ', iteration, ' iterations')
            break

        # delta_coord = coords - coords_old

        atoms.set_positions(coords.reshape(natoms, 3))

        coords_old = coords.copy()
        grad = -1.0 * atoms.get_forces().ravel()

        # delta_grad = grad - grad_old

        hessian = update_hessian(grad, grad_old, step_cart, hessian,
                                 update=hessian_update)

        grad_old = grad.copy()

        # calculate hessian eigenvalues and eigenvectors
        evals, evecs = harmonic_vibrational_analysis(hessian, atoms,
                                               proj_translations=proj_translations,
                                               proj_rotations=proj_rotations,
                                               ascomplex=False, massau=False)
        evals = np.power(evals, 2)

        mwevecs = np.dot(M_invsqrt, evecs)
        grad_nm = np.dot(mwevecs.T, grad)

        gmax = np.max(np.abs(grad_nm))

        # print the convergence info
        print('{0:>6d} {1:>15.8f} {2:>15.8f} {3:>15.8f} {4:>15.8f} {5:>20s}'.format(
            iteration, gmax, np.sqrt(np.dot(grad_nm, grad_nm)),
            np.sqrt(np.dot(step_nm, step_nm)), atoms.get_potential_energy(),
            datetime.now().strftime('%H:%M:%S %d-%m-%Y')))

        if gmax < gtol:
            evals, evecs = harmonic_vibrational_analysis(hessian, atoms,
                                               proj_translations=proj_translations,
                                               proj_rotations=proj_rotations,
                                               ascomplex=False, massau=False)
            np.save('hessian_evalues', evals)
            np.save('hessian_evectors', evecs)

        step_nm[:nvibdof] = -2.0 * grad_nm[:nvibdof] / (evals[:nvibdof]
            + np.sqrt(evals[:nvibdof]**2 + 4.0 * grad_nm[:nvibdof]**2))

        step_cart = np.dot(mwevecs, step_nm)
        coords = coords_old + step_cart

        if verbose:
            print(' eigenvalues '.center(50, '-'))
            print(evals)

            print(' cart gradient '.center(50, '-'))
            print(grad)

            print(' nm gradient '.center(50, '-'))
            print(grad_nm)

            print(' nm step '.center(50, '-'))
            print(step_nm)

            print(' cart step '.center(50, '-'))
            print(step_cart)

            print(' new coordinates '.center(50, '-'))
            print(coords)


def update_hessian(grad, grad_old, dx, hessian, update='BFGS'):
    '''
    Perform hessian update

    Parameters
    ----------
    grad : array_like (N,)
        Current gradient
    grad_old : array_like (N,)
        Previous gradient
    dx : array_like (N,)
        Step size
    hessian : array_like (N, N)
        Hessian matrix
    update : str
        Name of the hessian update to perform, possible values are 'BFGS',
        'SR1' and 'DFP'

    Returns
    -------
    uhessian : array_like
        Update hessian matrix
    '''

    macheps = np.finfo(np.float64).eps

    dg = grad - grad_old

    if update == 'BFGS':
        dxdg = np.dot(dx, dg)
        hdx = np.dot(hessian, dx)
        b = np.dot(dx, hdx)

        #print(' BFGS '.center(80, '='))
        #print('dxdg : ', dxdg)
        #print('b    : ', b)
        #print('dg   : ', dg)
        #print('dx   : ', dx)
        #print('hdx  : ', hdx)

        if np.abs(dxdg) < macheps or np.abs(b) < macheps:
            return hessian
        else:
            return hessian + np.outer(dg, dg) / dxdg - np.outer(hdx, hdx) / b

    elif update == 'DFP':
        dgdxT = np.outer(dg, dx)
        dgTdx = np.dot(dg, dx)
        uleft = np.eye(hessian.shape[0]) - dgdxT / dgTdx
        uright = np.eye(hessian.shape[0]) - dgdxT.T / dgTdx
        bk = np.dot(uleft, np.dot(hessian, uright))

        return bk + np.outer(dg, dg) / dgTdx

    elif update.upper() == 'SR1':
        hdx = np.dot(hessian, dx)
        dghdx = dg - hdx
        return hessian + np.outer(dghdx, dghdx) / np.dot(dghdx, dx)

    else:
        raise NotImplementedError
