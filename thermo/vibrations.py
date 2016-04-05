
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

def project_massweighted(args, atoms, ndof, hessian, verbose=False):
    
    Dmat = np.zeros((ndof, 6), dtype=float)

    # part of the translation projection, see equation 4.10b in Miller, W.
    # H., Handy, N. C., & Adams, J. E. (1980). Reaction path Hamiltonian
    # for polyatomic molecules. The Journal of Chemical Physics, 72(1), 99.
    # doi:10.1063/1.438959

    DTmat[np.arange(ndof), np.tile(np.arange(3), ndof//3)] = np.sqrt(masses/np.sum(atoms.get_masses()))

    # part of the rotation projection, see equation 4.10a in Miller, W.
    # H., Handy, N. C., & Adams, J. E. (1980). Reaction path Hamiltonian
    # for polyatomic molecules. The Journal of Chemical Physics, 72(1), 99.
    # doi:10.1063/1.438959

    # calculate the inverse square root of the inertia tensor
    Ival, Ivec = atoms.get_moments_of_inertia(vectors=True)
    Imhalf = np.dot(Ivec, np.dot(np.diag(1.0/np.sqrt(Ival)), Ivec.T))
    #print(Imhalf)


    #DRmat = np.zeros((ndof, 3), dtype=float)
    
    # from Gaussian
    #Pmat = np.dot(xyzcom, Ivec.T)

    #Dmat[ ::3, 3] = Pmat[:, 1]*Ivec[0, 2] - Pmat[:, 2]*Ivec[0, 1]/masses[::3]
    #Dmat[1::3, 3] = Pmat[:, 1]*Ivec[1, 2] - Pmat[:, 2]*Ivec[1, 1]/masses[1::3]
    #Dmat[2::3, 3] = Pmat[:, 1]*Ivec[2, 2] - Pmat[:, 2]*Ivec[2, 1]/masses[2::3]


    #Dmat[ ::3, 4] = Pmat[:, 2]*Ivec[0, 0] - Pmat[:, 0]*Ivec[0, 2]/masses[::3]
    #Dmat[1::3, 4] = Pmat[:, 2]*Ivec[1, 0] - Pmat[:, 0]*Ivec[1, 2]/masses[1::3]
    #Dmat[2::3, 4] = Pmat[:, 2]*Ivec[2, 0] - Pmat[:, 0]*Ivec[2, 2]/masses[2::3]

    #Dmat[ ::3, 5] = Pmat[:, 0]*Ivec[0, 1] - Pmat[:, 1]*Ivec[0, 0]/masses[::3]
    #Dmat[1::3, 5] = Pmat[:, 0]*Ivec[1, 1] - Pmat[:, 1]*Ivec[1, 0]/masses[1::3]
    #Dmat[2::3, 5] = Pmat[:, 0]*Ivec[2, 1] - Pmat[:, 1]*Ivec[2, 0]/masses[2::3]

    # compose the projection matrix P = D * D.T
    #Pmat = np.dot(Dmat, Dmat.T)

    #print('INFO: Orthogonalizing Pmat: ', Dmat.shape)
    #U, s, V = np.linalg.svd(Pmat) 
    #Pmat = np.dot(U, V)

    # project the mass weighted hessian (MWH) using (1 - P) * MWH * (1 - P)
    #eye = np.eye(Pmat.shape[0])
    #proj_hessian = np.dot(eye - Pmat, np.dot(mwhessian, eye - Pmat))

    #U, s, V = np.linalg.svd(Dmat)
    #Dmat = np.dot(U, V)
    #print('Final Dmat: \n', Dmat)
    #proj_hessian = np.dot(Dmat.T, np.dot(mwhessian, Dmat))

def project(args, atoms, ndof, hessian, verbose=False):

    if verbose:
        print('INFO: CARTESIAN coordinates')
        for row in atoms.get_positions():
            print("".join(['{0:15.8f}'.format(x) for x in row]))


    uatoms = atoms.copy()
    uatoms.set_masses(np.ones(len(atoms), dtype=float))

    # calculate the center of mass in cartesian coordinates
    com = uatoms.get_center_of_mass()
    xyzcom = (uatoms.get_positions() - com)

    if verbose:
        print('INFO: center of mass coordinates')
        for row in xyzcom:
            print("".join(['{0:15.8f}'.format(x) for x in row]))

    Dmat = np.zeros((ndof, 6), dtype=float)
    umasses = np.ones(ndof, dtype=float)
    Dmat[np.arange(ndof), np.tile(np.arange(3), ndof//3)] = np.sqrt(umasses)

    if args.proj_translations:

        Dmat[np.arange(ndof), np.tile(np.arange(3), ndof//3)] = np.sqrt(umasses)
    
        if verbose:
            print('INFO: Projecting out translations')
            for row in Dmat:
                print("".join(['{0:15.8f}'.format(x) for x in row]))

    if args.proj_rotations:

        Dmat[ ::3, 3] = 0.0
        Dmat[1::3, 3] = -xyzcom[:, 2]
        Dmat[2::3, 3] = xyzcom[:, 1]

        Dmat[ ::3, 4] = xyzcom[:, 2]
        Dmat[1::3, 4] = 0.0
        Dmat[2::3, 4] = -xyzcom[:, 0]

        Dmat[ ::3, 5] = -xyzcom[:, 1]
        Dmat[1::3, 5] = xyzcom[:, 0]
        Dmat[2::3, 5] = 0.0

        if verbose:
            print('INFO: Projecting out rotations')
            for row in Dmat:
                print("".join(['{0:15.8f}'.format(x) for x in row]))

    # orthogonalize
    q, r = np.linalg.qr(Dmat)
    if verbose:
        print('INFO: ORTHOGONALIZED Dmat')
        for row in q:
            print("".join(['{0:15.8f}'.format(x) for x in row]))
    
    I = np.eye(q.shape[0])

    qqp = np.dot(q, q.T)

    # project the force constant matrix (1 - P)*H*(1 - P)
    return np.dot(I - qqp, np.dot(hessian, I - qqp))

def get_harmonic_vibrations(args, atoms, hessian):
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

    # threshold for keeping the small eigenvalues of the hamiltonian
    # THRESH = np.finfo(np.float32).eps
    THRESH = 1.0e-10

    ndof = hessian.shape[0]

    # symmetrize the hessian
    hessian = (hessian + hessian.T)*0.5e0 

    if args.proj_translations | args.proj_rotations:

        hessian = project(args, atoms, ndof, hessian, verbose=False)

    # create the mass vector, with the masses for each atom repeated 3 times
    masses = np.repeat(atoms.get_masses(), 3)

    # calculate the mass weighted hessian
    masssqrt = np.diag(np.sqrt(1.0/masses))
    mwhessian = -np.dot(masssqrt, np.dot(hessian, masssqrt))

    # diagonalize the projected hessian to get the squared frequencies and normal modes
    w, v = np.linalg.eigh(mwhessian)

    w[w < THRESH] = 0.0

    freq = vasp2invcm * w.astype(complex)**0.5

    # save the result
    print('INFO: Saving vibrational frequencies to: frequencies.npy')
    np.save('frequencies', freq)
    print('INFO: Saving vibrational normal modes to: normal_modes.npy')
    np.save('normal_modes', v)

    freq = np.sort(freq)[::-1]

    print('Vibrational frequencies in [cm^-1]')
    for i, v in enumerate(freq, start=1):
        print('{0:5d} : '.format(i), '{0:20.10f} {1:20.10f}'.format(v.real, v.imag))

    return freq.real