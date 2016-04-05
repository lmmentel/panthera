

import numpy as np
from scipy.constants import value, pi, Avogadro, Planck, hbar, Boltzmann, gas_constant

def get_total_mass(atoms):
    '''
    Calculate the total mass of unconstrained atoms.

    The masses are assumed to be in the ``masses`` (np.array) attribute and the flags for coordinates
    are in ``free`` attribute
    '''

    atmass = value('atomic mass unit-kilogram relationship')
    if len(atoms.constraints) == 0:
        return np.sum(atoms.get_masses()) * atmass
    else:
        raise NotImplementerError('not yet')

def qtranslational(atoms, T, p):
    '''
    Calculate the translational partition function for a mole of ideal gas at temperature ``T`` and
    pressure ``p``.

    Args:
        atoms : ase.Atoms
            Atoms
        T : float
            Temperature in K
        p : float
            Pressure in MPa
    '''

    vol = gas_constant*T/(p*1.0e6)
    totmass = get_total_mass(atoms)

    return vol*np.power(2.0*pi*totmass*Boltzmann*T/Planck**2, 1.5)/Avogadro

def qrotational(atoms, T, point_group, sigma):
    '''
    Calculate the rotational partition function for a rigid rotor

    Args:
        atoms : ase.Atoms
            Atoms object
        T : float
            Temperature in K
        point_group : str
            Symbol of the point group of the structure
        sigma : int
            Rotational symmetry number
    '''

    # calcualte the moments of ineria and convert them to [kg*m^2]
    atmass = value('atomic mass unit-kilogram relationship')
    I = atoms.get_moments_of_inertia(vectors=False)*atmass*np.power(10.0, -20)

    if point_group in ['Coov', 'Dooh']:
        return np.sqrt(pi*I[0]*I[1]*np.power(2.0*Boltzmann*T/hbar**2, 2))/sigma
    else:
        return np.sqrt(pi*np.product(I)*np.power(2.0*Boltzmann*T/hbar**2, 3))/sigma

class Thermochemistry(object):

	def __init__(self, vibenergies):

		self.vibenergies = vibenergies

		# check if the energies are real/positive

	def get_ZPVE(self, T):
		'''
		Calculate the Zero Point Vibrational Energy (ZPVE)
		'''

		return 0.5*np.sum(self.vibenergies)

	def get_qvibrational(self, T, uselog=True):
		'''
		Calculate the vibrational partition function at temperature `T`

		Args:
			T : float
				Temperature in `K`
			uselog : bool
				When ``True`` return the natural logarithm of the partition function
		'''

		kT = Boltzmann * T
		if uselog:
			return np.sum(-np.log(1.0 - np.exp(-self.vibenergies/kT)))
		else:
			return np.prod(1.0/(1.0 - np.exp(-self.vibenergies/kT)))

class AnharmonicThermo(object):

    def __init__(self, vibenergies, potentialenergy):

        self.vibenergies = vibenergies

    def get_entropy(self):

        pass

    def get_internal_energy(self):

        pass

    def get_helmholtz_energy(self):

        pass

    def summary(self):

        pass
