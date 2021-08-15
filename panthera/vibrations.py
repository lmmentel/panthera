from __future__ import print_function, division

import numpy as np
from ase import units
from .thermochemistry import constraints2mask


def get_levicivita():
    "Get the Levi_civita symemtric tensor"

    eijk = np.zeros((3, 3, 3))
    eijk[0, 1, 2] = eijk[1, 2, 0] = eijk[2, 0, 1] = 1
    eijk[0, 2, 1] = eijk[2, 1, 0] = eijk[1, 0, 2] = -1

    return eijk


def project_massweighted(args, atoms, ndof, hessian, verbose=False):
    """
    Project translational and or rotatioanl dgrees of freedom from
    mass weighted hessian
    """

    Dmat = np.zeros((ndof, 6), dtype=float)

    # part of the translation projection, see equation 4.10b in Miller, W.
    # H., Handy, N. C., & Adams, J. E. (1980). Reaction path Hamiltonian
    # for polyatomic molecules. The Journal of Chemical Physics, 72(1), 99.
    # doi:10.1063/1.438959

    DTmat[np.arange(ndof), np.tile(np.arange(3), ndof // 3)] = np.sqrt(
        masses / np.sum(atoms.get_masses())
    )

    # part of the rotation projection, see equation 4.10a in Miller, W.
    # H., Handy, N. C., & Adams, J. E. (1980). Reaction path Hamiltonian
    # for polyatomic molecules. The Journal of Chemical Physics, 72(1), 99.
    # doi:10.1063/1.438959

    # calculate the inverse square root of the inertia tensor
    Ival, Ivec = atoms.get_moments_of_inertia(vectors=True)
    Imhalf = np.dot(Ivec, np.dot(np.diag(1.0 / np.sqrt(Ival)), Ivec.T))
    # print(Imhalf)

    # DRmat = np.zeros((ndof, 3), dtype=float)

    # from Gaussian
    # Pmat = np.dot(xyzcom, Ivec.T)

    # Dmat[ ::3, 3] = Pmat[:, 1]*Ivec[0, 2] - Pmat[:, 2]*Ivec[0, 1]/masses[::3]
    # Dmat[1::3, 3] = Pmat[:, 1]*Ivec[1, 2] - Pmat[:, 2]*Ivec[1, 1]/masses[1::3]
    # Dmat[2::3, 3] = Pmat[:, 1]*Ivec[2, 2] - Pmat[:, 2]*Ivec[2, 1]/masses[2::3]

    # Dmat[ ::3, 4] = Pmat[:, 2]*Ivec[0, 0] - Pmat[:, 0]*Ivec[0, 2]/masses[::3]
    # Dmat[1::3, 4] = Pmat[:, 2]*Ivec[1, 0] - Pmat[:, 0]*Ivec[1, 2]/masses[1::3]
    # Dmat[2::3, 4] = Pmat[:, 2]*Ivec[2, 0] - Pmat[:, 0]*Ivec[2, 2]/masses[2::3]

    # Dmat[ ::3, 5] = Pmat[:, 0]*Ivec[0, 1] - Pmat[:, 1]*Ivec[0, 0]/masses[::3]
    # Dmat[1::3, 5] = Pmat[:, 0]*Ivec[1, 1] - Pmat[:, 1]*Ivec[1, 0]/masses[1::3]
    # Dmat[2::3, 5] = Pmat[:, 0]*Ivec[2, 1] - Pmat[:, 1]*Ivec[2, 0]/masses[2::3]

    # compose the projection matrix P = D * D.T
    # Pmat = np.dot(Dmat, Dmat.T)

    # print('INFO: Orthogonalizing Pmat: ', Dmat.shape)
    # U, s, V = np.linalg.svd(Pmat)
    # Pmat = np.dot(U, V)

    # project the mass weighted hessian (MWH) using (1 - P) * MWH * (1 - P)
    # eye = np.eye(Pmat.shape[0])
    # proj_hessian = np.dot(eye - Pmat, np.dot(mwhessian, eye - Pmat))

    # U, s, V = np.linalg.svd(Dmat)
    # Dmat = np.dot(U, V)
    # print('Final Dmat: \n', Dmat)
    # proj_hessian = np.dot(Dmat.T, np.dot(mwhessian, Dmat))


def project(
    atoms, hessian, ndof, proj_translations=True, proj_rotations=False, verbose=False
):
    """
    Project out the translational and/or rotational degrees of freedom
    from the hessian.

    Parameters
    ----------
    atoms : ase.Atoms
        Atoms object
    ndof : int
        Number of degrees of freedom
    hessian : array_like
        Hessian/force constant matrix
    proj_translations : bool
        If ``True`` translational degrees of freedom will be projected from
        the hessian
    proj_rotations : bool
        If ``True`` rotational degrees of freedom will be projected from
        the hessian

    Returns
    -------
    proj_hessian : array_like
        Hessian matrix with translational and/or rotational degrees of
        freedom projected out
    """

    if verbose:
        print("INFO: CARTESIAN coordinates")
        for row in atoms.get_positions():
            print("".join("{0:15.8f}".format(x) for x in row))

    uatoms = atoms.copy()
    uatoms.set_masses(np.ones(len(atoms), dtype=float))

    # calculate the center of mass in cartesian coordinates
    com = uatoms.get_center_of_mass()
    xyzcom = uatoms.get_positions() - com

    if verbose:
        print("INFO: center of mass coordinates")
        for row in xyzcom:
            print("".join(["{0:15.8f}".format(x) for x in row]))

    Dmat = np.zeros((ndof, 6), dtype=float)
    umasses = np.ones(ndof, dtype=float)
    Dmat[np.arange(ndof), np.tile(np.arange(3), ndof // 3)] = np.sqrt(umasses)

    if proj_translations:

        Dmat[np.arange(ndof), np.tile(np.arange(3), ndof // 3)] = np.sqrt(umasses)

        if verbose:
            print("INFO: Projecting out translations")
            for row in Dmat:
                print("".join("{0:15.8f}".format(x) for x in row))

    if proj_rotations:

        Dmat[::3, 3] = 0.0
        Dmat[1::3, 3] = -xyzcom[:, 2]
        Dmat[2::3, 3] = xyzcom[:, 1]

        Dmat[::3, 4] = xyzcom[:, 2]
        Dmat[1::3, 4] = 0.0
        Dmat[2::3, 4] = -xyzcom[:, 0]

        Dmat[::3, 5] = -xyzcom[:, 1]
        Dmat[1::3, 5] = xyzcom[:, 0]
        Dmat[2::3, 5] = 0.0

        if verbose:
            print("INFO: Projecting out rotations")
            for row in Dmat:
                print("".join(["{0:15.8f}".format(x) for x in row]))

    # orthogonalize
    if proj_translations and proj_rotations:
        q, _ = np.linalg.qr(Dmat)
    elif proj_translations:
        q, _ = np.linalg.qr(Dmat[:, :3])
        q = np.hstack((q, Dmat[:, 3:]))
    elif proj_rotations:
        q, _ = np.linalg.qr(Dmat[:, 3:])
        q = np.hstack((Dmat[:, :3], q))

    if verbose:
        print("INFO: ORTHOGONALIZED Dmat")
        for row in q:
            print("".join("{0:15.8f}".format(x) for x in row))

    I = np.eye(q.shape[0])

    qqp = np.dot(q, q.T)

    # project the force constant matrix (1 - P)*H*(1 - P)
    return np.dot(I - qqp, np.dot(hessian, I - qqp))


def harmonic_vibrational_analysis(
    hessian,
    atoms,
    proj_translations=True,
    proj_rotations=False,
    ascomplex=True,
    massau=True,
):
    """
    Given a force constant matrix (hessian) perform the harmonic vibrational
    analysis, by calculating the eigevalues and eigenvectors of the mass
    weighted hessian. Additionally projection of the translational and
    rotational degrees of freedom can be performed by specifying
    ``proj_translations`` and ``proj_rotations`` argsuments.

    Parameters
    ----------
    hessian : array_like
        Force constant (Hessian) matrix in atomic units, should be
        square and symmetric

    atoms : Atoms
        ASE atoms object

    proj_translations : bool
        If ``True`` translational degrees of freedom will be projected from
        the hessian

    proj_rotations : bool
        If ``True`` rotational degrees of freedom will be projected from
        the hessian

    massau : bool
        If ``True`` atomic units of mass will be used

    ascomplex : bool
        If there are complex eigenvalues return the array as complex type
        otherwise make the complex values negative and return array of reals

    Returns
    -------
    out : (w, v)
        Tuple of numpy arrays with hessian square roots of the eigevalues
        (frequencies) and eiegenvectors in atomic units, both sorted in
        descending order of eigenvalues
    """

    # threshold for keeping the small eigenvalues of the hamiltonian
    THRESH = 1.0e-10

    ndof = hessian.shape[0]

    if proj_translations | proj_rotations:

        hessian = project(
            atoms,
            hessian,
            ndof,
            proj_translations=proj_translations,
            proj_rotations=proj_rotations,
            verbose=False,
        )

    # create the mass vector, with the masses for each atom repeated 3 times
    # and convert to atomic units

    masses = np.repeat(atoms.get_masses(), 3)
    if massau:
        masses *= units._amu / units._me

    massvec = 1.0 / np.sqrt(masses)
    mwhessian = np.multiply(hessian, np.outer(massvec, massvec))

    # diagonalize the projected hessian to get the squared frequencies and
    # normal modes
    wals, vecs = np.linalg.eigh(mwhessian)

    wals[wals < THRESH] = 0.0

    wals = wals[::-1]
    vecs = vecs[:, ::-1]

    wals = wals.astype(complex) ** 0.5

    if ascomplex:
        return wals, vecs
    mask = np.iscomplex(wals)
    wreal = np.zeros_like(wals, dtype=float)
    wreal[~mask] = wals[~mask].real
    wreal[mask] = -1.0 * np.abs(wals[mask].imag)
    return wreal, vecs
