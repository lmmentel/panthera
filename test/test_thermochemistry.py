
import numpy as np
from scipy.constants import Planck, gas_constant, value
from ase import Atoms
from panther.thermochemistry import Thermochemistry


def test_thermochemistry_hf():

    hf = Atoms('HF', [[0.0, 0.0, 0.0], [0.0, 0.0, 0.917]])

    viben = np.array([3983.0])

    viben = viben * Planck * 100.0 * value('inverse meter-hertz relationship')

    thermo = Thermochemistry(viben, hf, phase='gas', pointgroup='Coov')

    assert np.isclose(thermo.get_translational_heat_capacity(), 2.5 * gas_constant * 1.0e-3)
    assert np.isclose(thermo.get_rotational_heat_capacity(), gas_constant * 1.0e-3)
    assert np.isclose(thermo.get_vibrational_heat_capacity(298.15), 1.3801632451676039e-08)
