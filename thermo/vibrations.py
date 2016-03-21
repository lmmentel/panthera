
from __future__ import print_function, division

from scipy.constants import hbar, angstrom, value, elementary_charge, value, pi, speed_of_light
import numpy as np

# conversion fator from eV/A^2 to cm^-1
vasp2invcm = 1.0e8*np.sqrt(elementary_charge)/(np.sqrt(value('atomic mass constant'))*2.0*pi*speed_of_light)

def get_levicivita():
    'Get the Levi_civita symemtric tensor'

    eijk = np.zeros((3, 3, 3))
    eijk[0, 1, 2] = eijk[1, 2, 0] = eijk[2, 0, 1] = 1
    eijk[0, 2, 1] = eijk[2, 1, 0] = eijk[1, 0, 2] = -1

    return eijk

def get_harmonic_vibrations(atoms, hessian, args):
    '''
    Given a force constant matrix (hessian) decide whether or not to project the translational and
    rotational degrees of freedom based on ``proj_translations`` and ``proj_rotations`` argsuments
    in ``args`` respectively.

    Args:
        atoms : Atoms
            ASE atoms object
        hessian : np.array
            Force constant (Hessian) matrix, should be square
        args : Namespace
            Namespace with the arguments parsed from the input/config
    '''

    ndof = hessian.shape[0]

    # crreate the mass vector, with the masses for each atom repeated 3(dof) times
    masses = np.repeat(atoms.get_masses(), 3)

    # symmetrize the hessian
    hessian = (hessian + hessian.T)*0.5e0

    # calculate the mass weighted hessian
    masssqrt = np.diag(np.sqrt(1.0/masses))
    mwhessian = np.dot(masssqrt, np.dot(hessian, masssqrt))

    # calculate the center of mass in mass weighted cartesian coordinates
    com = atoms.get_center_of_mass()
    xyzcom = (atoms.get_positions() - com)*np.sqrt(atoms.get_masses())[:, np.newaxis]


    if args.proj_translations:
        print('INFO: Projecting translations')

        # part of the translation projection, see equation 4.10b in Miller, W.
        # H., Handy, N. C., & Adams, J. E. (1980). Reaction path Hamiltonian
        # for polyatomic molecules. The Journal of Chemical Physics, 72(1), 99.
        # doi:10.1063/1.438959
        DTmat = np.zeros((ndof, 3), dtype=float)
        DTmat[np.arange(ndof), np.tile(np.arange(3), ndof//3)] = np.sqrt(masses/np.sum(atoms.get_masses()))

    if args.proj_rotations:
        print('INFO: Projecting rotations')

        # calcualate the inverse square root of the inertia tensor
        Ival, Ivec = atoms.get_moments_of_inertia(vectors=True)
        Imhalf = np.dot(Ivec, np.dot(np.diag(1.0/np.sqrt(Ival)), np.linalg.inv(Ivec)))

        # part of the rotation projection, see equation 4.10a in Miller, W.
        # H., Handy, N. C., & Adams, J. E. (1980). Reaction path Hamiltonian
        # for polyatomic molecules. The Journal of Chemical Physics, 72(1), 99.
        # doi:10.1063/1.438959
        DRmat = np.zeros((ndof, 3), dtype=float)
        DRmat[::3, 0] = np.cross(Imhalf[0, :], xyzcom)
        DRmat[1::3, 1] = np.cross(Imhalf[1, :], xyzcom)
        DRmat[2::3, 2] = np.cross(Imhalf[2, :], xyzcom)

    if args.proj_translations & args.proj_rotations:
        Dmat = np.hstack((DTmat, DRmat))
    elif args.proj_translations | args.proj_rotations:
        if args.proj_translations:
            Dmat = DTmat
        else:
            Dmat = DRmat

    # compose the projection matrix P = D * D.T
    Pmat = np.dot(Dmat, Dmat.T)

    # project the mass weighted hessian (MWH) using (1 - P) * MWH * (1 - P)
    eye = np.eye(Pmat.shape[0])
    proj_hessian = np.dot(eye - Pmat, np.dot(mwhessian, eye - Pmat))

    # diagonalize the projected hessian to the the squared frequencies and normal modes
    w, v = np.linalg.eigh(proj_hessian)

    freq = vasp2invcm *(-1 * w).astype(complex)**0.5

    # save the result
    print('INFO: Saving vibrational frequencies to: frequencies.npy')
    np.save('frequencies', freq)
    print('INFO: Saving vibrational normal modes to: normal_modes.npy')
    np.save('normal_modes', v)

    print('Vibrational frequencies in cm^-1')
    for i, v in enumerate(freq.real):
        print('{0:5d} : '.format(i), '{0:20.10f}'.format(v))

    return freq
