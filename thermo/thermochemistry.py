

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

class Thermochemistry(object):
	'''
	Thermochemistry the results will be in kJ/mol

	Args:
		atoms : ase.Atoms
			Atoms obect
		vibenergies : numpy.array
			A vector of `3N-6` vibrational energies in Joules
	'''

	def __init__(self, atoms, vibenergies, conditions, system):

		self.atoms = atoms
		self.vibenergies = vibenergies
		self.conditions = conditions
		self.system = system

		# check if the energies are real/positive

	def get_ZPVE(self):
		'''
		Calculate the Zero Point Vibrational Energy (ZPVE) in kJ/mol

		.. math::
		
		   ZPVE(T) = \\frac{1}{2}\sum^{3N-6}_{i=1} h\omega_{i}

		'''

		return 0.5*np.sum(self.vibenergies)*Avogadro*1.0e-3

	def get_qrotational(self, T):
	    '''
	    Calculate the rotational partition function in a rigid rotor approximation

		.. math::

		   q_{rot}(T) = \\frac{\sqrt{\pi I_{A}I_{B}I_{C}} }{\sigma} \left( \\frac{ 2 k_{B} T }{\hbar^{2}} \\right)^{3/2}


	    Args:
	        T : float
	            Temperature in K
	    '''

	    # calcualte the moments of ineria and convert them to [kg*m^2]
	    atmass = value('atomic mass unit-kilogram relationship')
	    I = self.atoms.get_moments_of_inertia(vectors=False)*atmass*np.power(10.0, -20)

	    sigma = self.system['symmetrynumber']
	    if self.system['pointgroup'] in ['Coov', 'Dooh']:
	        return 2.0*np.max(I)*Boltzmann*T/(sigma*hbar**2)
	    else:
	        return np.sqrt(pi*np.product(I)*np.power(2.0*Boltzmann*T/hbar**2, 3))/sigma

	def get_qtranslational(self, T):
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
	    '''

	    vol = gas_constant*T/(self.conditions['pressure']*1.0e6)
	    totmass = get_total_mass(self.atoms)

	    return vol*np.power(2.0*pi*totmass*Boltzmann*T/Planck**2, 1.5)/Avogadro

	def get_qvibrational(self, T, uselog=True):
		'''
		Calculate the vibrational partition function at temperature `T` in kJ/mol

		.. math::

		   q_{vib}(T) = \prod^{3N-6}_{i=1}\\frac{1}{1 - \exp(-h\omega_{i}/k_{B}T)}		

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

	def get_vib_energy(self, T):
		'''
		Calculate the vibational energy correction at temperature `T` in kJ/mol

		.. math::

		   U_{vib}(T) = \\frac{R}{k_{B}}\sum^{3N-6}_{i=1} \\frac{h\omega_{i}}{\exp(h\omega_{i}/k_{B}T) - 1}

		Args:
			T : float
				Temaperature in `K`
		'''

		kT = Boltzmann * T
		frac = np.sum(self.vibenergies/(np.exp(self.vibenergies/kT) -1.0))
		return 1.0e-3*gas_constant*frac/Boltzmann

	def get_vib_entropy(self, T):
		'''
		Calculate the vibrational entropy at temperature `T` in kJ/mol

		.. math::

		   S_{vib}(T) = R\sum^{3N-6}_{i=1}\left[ \\frac{h\omega_{i}}{k_{B}T(\exp(h\omega_{i}/k_{B}T) - 1)}
			          - \ln(1 - \exp(-h\omega_{i}/k_{B}T)) \\right]

		Args:
			T : float
				Temperature in `K`
		'''

		kT = Boltzmann * T

		frac = np.sum(self.vibenergies/(np.exp(self.vibenergies/kT) -1.0))/kT
		nlog = np.sum(np.log(1.0 - np.exp(-self.vibenergies/kT)))
		return 1.0e-3*gas_constant*(frac - nlog)

	def summary(self, T):
		'''
		Print summary with the thermochemical data at temperature `T` in kJ/mol

		Args:
			T : float
				Temperature in `K`
		'''

		if self.system['phase'] == 'gas':
			qtrans = self.get_qtranslational(T)
			energy_trans = 1.5*gas_constant*T*1.0e-3
			entropy_trans = gas_constant*(np.log(qtrans) + 2.5)*1.0e-3

			qrot = self.get_qrotational(T)
			if self.system['pointgroup'] in ['Coov', 'Dooh']:
				energy_rot = gas_constant*T*1.0e-3
				entropy_rot = gas_constant*(np.log(qrot) + 1)*1.0e-3
			else:
				energy_rot = 1.5*gas_constant*T*1.0e-3
				entropy_rot = gas_constant*(np.log(qrot) + 1.5)*1.0e-3
		else:
			qtrans = qrot = energy_trans = entropy_trans = energy_rot = entropy_rot = 0.0


		print('\n' + 'THERMOCHEMISTRY'.center(45, '='))
		print('\n\t @ T = {0:6.2f} K\t p = {1:6.2f} MPa\n'.format(T, self.conditions['pressure']))
		print('-'*45)

		print('{0:<25s} : {1:10.3f}'.format('ln qtranslational', np.log(qtrans)))
		print('{0:<25s} : {1:10.3f}'.format('ln qrotational', np.log(qrot)))
		print('{0:<25s} : {1:10.3f}'.format('ln qvibrational', self.get_qvibrational(T, uselog=True)))

		print('{0:<25s} : {1:10.3f}  kJ/mol'.format('ZPVE', self.get_ZPVE()))
		print('{0:<25s} : {1:10.3f}  kJ/mol'.format('U translational', energy_trans))
		print('{0:<25s} : {1:10.3f}  kJ/mol'.format('U rotational', energy_rot))
		print('{0:<25s} : {1:10.3f}  kJ/mol'.format('U vibrational', self.get_vib_energy(T)))
		print('{0:<25s} : {1:11.4f} kJ/mol'.format('S translational', entropy_trans))
		print('{0:<25s} : {1:11.4f} kJ/mol'.format('S rotational', entropy_rot))
		print('{0:<25s} : {1:11.4f} kJ/mol'.format('S vibrational', self.get_vib_entropy(T)))


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
