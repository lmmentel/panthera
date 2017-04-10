import os
import numpy as np
from scipy.constants import angstrom, value

from panther.io import read_vasp_hessian, read_vasp_hessian_xml


ang2bohr = angstrom / value('atomic unit of length')
ev2hartree = value('electron volt-hartree relationship')
cwd = os.path.abspath(os.path.dirname(__file__))


def test_read_vasp_hessian_meoh():

    # the hessian is in the same form as in the OUCAR file
    refhessian = np.load(os.path.join(cwd, 'data/meoh_hessian_raw.npy'))

    outcar = os.path.join(cwd, 'data/meoh_hessian.OUTCAR')

    # as stored in OUTCAR
    hessian = read_vasp_hessian(outcar, symmetrize=False, convert_to_au=False)

    assert np.allclose(-1.0 * refhessian, hessian)

    # symmetrized
    hessian = read_vasp_hessian(outcar, symmetrize=True, convert_to_au=False)
    assert np.allclose(-1.0 * (refhessian.T + refhessian) * 0.5, hessian)

    # converted to au
    hessian = read_vasp_hessian(outcar, symmetrize=False, convert_to_au=True)
    assert np.allclose(-1.0 * (refhessian * ev2hartree / (ang2bohr**2)),
                       hessian)


def test_read_vasp_hessian_hcha():

    # the hessian is in the same form as in the OUCAR file
    refhessian = np.load(os.path.join(cwd, 'data/hcha_hessian.npy'))

    outcar = os.path.join(cwd, 'data/hcha.OUTCAR')

    # as stored in OUTCAR
    hessian = read_vasp_hessian(outcar, symmetrize=False, convert_to_au=False)

    assert np.allclose(-1.0 * refhessian, hessian)

    # symmetrized
    hessian = read_vasp_hessian(outcar, symmetrize=True, convert_to_au=False)
    assert np.allclose(-1.0 * (refhessian.T + refhessian) * 0.5, hessian)

    # converted to au
    hessian = read_vasp_hessian(outcar, symmetrize=False, convert_to_au=True)
    assert np.allclose(-1.0 * (refhessian * ev2hartree / (ang2bohr**2)),
                       hessian)


def test_read_vasp_hessian_xml():

    vaspxml = os.path.join(cwd, 'data/hcha.vasprun.xml')
    hessian = read_vasp_hessian_xml(vaspxml, convert_to_au=False,
                                    stripmass=True)

    refhessian = np.load(os.path.join(cwd, 'data/hcha_hessian.npy'))

    refhessian = (refhessian + refhessian.T) * -0.5

    assert np.allclose(refhessian, hessian, atol=1.0e-6)
