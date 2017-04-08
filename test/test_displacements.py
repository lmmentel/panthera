
import pickle
import os
import numpy as np

import ase.io

from panther.vibrations import harmonic_vibrational_analysis
from panther.displacements import calculate_displacements

from scipy.constants import angstrom, value

ang2bohr = angstrom / value('atomic unit of length')
ev2hartree = value('electron volt-hartree relationship')

cwd = os.path.abspath(os.path.dirname(__file__))


def test_displacements_meoh(tmpdir):

    tmpdir.chdir()

    meoh = ase.io.read(os.path.join(cwd, 'data', 'meoh.traj'))

    # the hessian is in the same form as in the OUCAR file
    hessian = np.load(os.path.join(cwd, 'data/meoh_hessian_raw.npy'))
    hessian = (hessian + hessian.T) * 0.5
    hessian = hessian * ev2hartree / (ang2bohr**2)
    hessian = -1 * hessian

    freqs, normal_modes = harmonic_vibrational_analysis(hessian, meoh,
                                                        proj_translations=True,
                                                        proj_rotations=True,
                                                        ascomplex=False)

    with open(os.path.join(cwd, 'data', 'meoh_images.pkl'), 'rb') as fpkl:
        refimages = pickle.load(fpkl)

    images, modeinfo = calculate_displacements(meoh, hessian, freqs,
                                               normal_modes, npoints=4)

    assert len(refimages) == len(images)

    for mode in images.keys():
        for point in images[mode].keys():
            assert np.allclose(images[mode][point].positions, refimages[mode][point].positions)
