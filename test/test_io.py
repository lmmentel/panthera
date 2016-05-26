import os
import numpy as np

from panther.inputreader import read_vasp_hessian

from scipy.constants import angstrom, value

ang2bohr = angstrom / value('atomic unit of length')
ev2hartree = value('electron volt-hartree relationship')
cwd = os.path.abspath(os.path.dirname(__file__))


def test_read_vasp_hessian_meoh():

    # the hessian is in the same form as in the OUCAR file
    refhessian = np.load(os.path.join(cwd, 'data/meoh_hessian.npy'))

    outcar = os.path.join(cwd, 'data/meoh.OUTCAR')

    # as stored in OUTCAR
    hessian = read_vasp_hessian(outcar, symmetrize=False,
                                convert2au=False, negative=False)

    assert np.allclose(refhessian, hessian)

    # symmetrized
    hessian = read_vasp_hessian(outcar, symmetrize=True,
                                convert2au=False, negative=False)
    assert np.allclose((refhessian.T + refhessian) * 0.5, hessian)

    # converted to au
    hessian = read_vasp_hessian(outcar, symmetrize=False,
                                convert2au=True, negative=False)
    assert np.allclose((refhessian * ev2hartree / (ang2bohr**2)), hessian)

    # negative
    hessian = read_vasp_hessian(outcar, symmetrize=False,
                                convert2au=False, negative=True)
    assert np.allclose(-1.0 * refhessian, hessian)


def test_read_vasp_hessian_hcha():

    # the hessian is in the same form as in the OUCAR file
    refhessian = np.load(os.path.join(cwd, 'data/hcha_hessian.npy'))

    outcar = os.path.join(cwd, 'data/hcha.OUTCAR')

    # as stored in OUTCAR
    hessian = read_vasp_hessian(outcar, symmetrize=False,
                                convert2au=False, negative=False)

    assert np.allclose(refhessian, hessian)

    # symmetrized
    hessian = read_vasp_hessian(outcar, symmetrize=True,
                                convert2au=False, negative=False)
    assert np.allclose((refhessian.T + refhessian) * 0.5, hessian)

    # converted to au
    hessian = read_vasp_hessian(outcar, symmetrize=False,
                                convert2au=True, negative=False)
    assert np.allclose((refhessian * ev2hartree / (ang2bohr**2)), hessian)

    # negative
    hessian = read_vasp_hessian(outcar, symmetrize=False,
                                convert2au=False, negative=True)
    assert np.allclose(-1.0 * refhessian, hessian)