

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
        raise NotImplementedError('not yet')

def qtranslational(atoms, T, p):
    '''
    Calculate the translational partition function for a mole of ideal gas at temperature ``T`` and
    pressure ``p``.

	.. math::

	   q_{vib}(V, T) = \left( \\frac{ 2\pi M k_{B} T }{h^{2}} \\right)^{3/2} V


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

def qrotational(atoms, system, T):
    '''
    Calculate the rotational partition function in a rigid rotor approximation

	.. math::

	   q_{rot}(T) = \\frac{\sqrt{\pi I_{A}I_{B}I_{C}} }{\sigma} \left( \\frac{ 2 k_{B} T }{\hbar^{2}} \\right)^{3/2}


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

    sigma = system['symmetrynumber']
    if system['pointgroup'] in ['Coov', 'Dooh']:
        return np.sqrt(pi*I[0]*I[1]*np.power(2.0*Boltzmann*T/hbar**2, 2))/sigma
    else:
        return np.sqrt(pi*np.product(I)*np.power(2.0*Boltzmann*T/hbar**2, 3))/sigma

class Thermochemistry(object):

	'''
	Thermochemistry the results will be in kJ/mol

	Args:
		vibenergies : numpy.array
			A vector of `3N-6` vibrational energies in Joules
	'''

	def __init__(self, vibenergies, potentialenergy):

		self.vibenergies = vibenergies
		self.potentialenergy = potentialenergy

		# check if the energies are real/positive

	def get_ZPVE(self, T):
		'''
		Calculate the Zero Point Vibrational Energy (ZPVE)
		'''

		return 0.5*np.sum(self.vibenergies)*Avogadro*1.0e-3

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
