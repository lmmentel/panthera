
import os
import numpy as np

import ase.io

from panther.vibrations import harmonic_vibrational_analysis

from scipy.constants import angstrom, value

ang2bohr = angstrom / value('atomic unit of length')
ev2hartree = value('electron volt-hartree relationship')

cwd = os.path.abspath(os.path.dirname(__file__))


def test_harmonic_methanol():

    # the hessian is in the same form as in the OUCAR file
    hessian = np.load(os.path.join(cwd, 'data/meoh_hessian.npy'))
    hessian = (hessian + hessian.T) * 0.5
    hessian = hessian * ev2hartree / (ang2bohr**2)
    hessian = -1 * hessian

    meoh = ase.io.read(os.path.join(cwd, 'data/meoh.traj'))

    ewals, evecs = harmonic_vibrational_analysis(hessian, meoh,
                                                 proj_translations=True,
                                                 proj_rotations=True)

    eigenvalues = np.load(os.path.join(cwd, 'data/meoh_evalues.npy'))
    eigenvectors = np.load(os.path.join(cwd, 'data/meoh_evectors.npy'))

    assert np.allclose(ewals, eigenvalues)
    assert np.allclose(evecs, eigenvectors)


def test_harmonic_hcha():

    # the hessian is in the same form as in the OUCAR file
    hessian = np.load(os.path.join(cwd, 'data/hcha_hessian.npy'))
    hessian = (hessian + hessian.T) * 0.5
    hessian = hessian * ev2hartree / (ang2bohr**2)
    hessian = -1 * hessian

    hcha = ase.io.read(os.path.join(cwd, 'data/hcha.traj'))

    ewals, evecs = harmonic_vibrational_analysis(hessian, hcha,
                                                 proj_translations=True,
                                                 proj_rotations=False)

    eigenvalues = np.load(os.path.join(cwd, 'data/hcha_evalues.npy'))
    eigenvectors = np.load(os.path.join(cwd, 'data/hcha_evectors.npy'))

    assert np.allclose(ewals, eigenvalues)
    assert np.allclose(evecs, eigenvectors)
