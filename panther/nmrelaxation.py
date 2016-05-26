
import numpy as np

from .vibrations import get_harmonic_vibrations


def nmoptimize(atoms, hessian):

    natoms = atoms.get_number_of_atoms()
    ndof = 3 * natoms
    masses = atoms.get_masses()
    pos = atoms.get_positions()

    # matrix with inverse square roots of masses on diagonal
    M_invsqrt = np.zeros((ndof, ndof), dtype=float)
    np.fill_diagonal(M_invsqrt, np.repeat(1.0 / np.sqrt(masses * prm), 3))


    # calculate hessian eigenvalues and eigenvectors
    evals, evecs = get_harmonic_vibrations(hessian, atoms,
                                           proj_translations=True,
                                           proj_rotations=True)

    mwevecs = np.dot(M_invsqrt, evecs)


    # run the job for the initial structure

    # get forces after run
    grad = atoms.get_forces()

    grad_nm = np.dot(mwevecs.T, grad)

    step_nm = -2.0 * grad_nm / (evals * (1.0 + np.sqrt(1.0 + (4.0 * grad_nm**2) / evals**2)))

    step_cart = np.dot(mwevecs, step_nm)

    coords = coords_old + step_cart

    not_converged = True
    iteration = 0
    while not_converged:
        print(' iteration {0:d} '.format(iteration).center(80, '='))

        delta_coords = coords - coords_old


        update_hessian()


def update_hessian():

    pass
