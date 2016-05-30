
'Methods for calculating thermochemistry'

from __future__ import print_function, division, absolute_import
from builtins import (bytes, str, open, super, range,
                      zip, round, input, int, pow, object)

import numpy as np
from scipy.constants import value, pi, Avogadro, Planck, hbar, Boltzmann, gas_constant

def constraints2mask(atoms):
    '''
    Convert constraints from default ase objects to VASP compatible numpy array
    of boolean triples describing whether atomic degress of freedom are fixed in
    x, y, z dimensions

    Parameters
    ----------
    atoms : ase.Atoms
        ASE atoms object

    Returns
    -------
    slfags : numpy.array
        Boolean array with the size `N` x 3 where `N` is the number of atoms
    '''

    from ase.constraints import FixAtoms, FixScaled, FixedPlane, FixedLine

    if atoms.constraints:
        sflags = np.zeros((len(atoms), 3), dtype=bool)
        for constr in atoms.constraints:
            if isinstance(constr, FixScaled):
                sflags[constr.a] = constr.mask
            elif isinstance(constr, FixAtoms):
                sflags[constr.index] = [True, True, True]
            elif isinstance(constr, FixedPlane):
                mask = np.all(np.abs(np.cross(constr.dir, atoms.cell)) < 1e-5, axis=1)
                if sum(mask) != 1:
                    raise RuntimeError(
                        'VASP requires that the direction of FixedPlane '
                        'constraints is parallel with one of the cell axis')
                sflags[constr.a] = mask
            elif isinstance(constr, FixedLine):
                mask = np.all(np.abs(np.cross(constr.dir, atoms.cell)) < 1e-5, axis=1)
                if sum(mask) != 1:
                    raise RuntimeError(
                        'VASP requires that the direction of FixedLine '
                        'constraints is parallel with one of the cell axis')
                sflags[constr.a] = ~mask
    else:
        sflags = np.ones((len(atoms), 3), dtype=bool)
    return sflags


def get_total_mass(atoms):
    '''
    Calculate the total mass of unconstrained atoms in kg.

    The masses are assumed to be in the ``masses`` (np.array) attribute and
    the flags for coordinates are in ``free`` attribute

    Parameters
    ----------
    atoms : ase.Atoms
        ASE Atoms object

    Returns
    -------
    mass : float
        Mass of the unconstrained atoms in kg
    '''

    atmass = value('atomic mass unit-kilogram relationship')
    if len(atoms.constraints) == 0:
        return np.sum(atoms.get_masses()) * atmass
    else:
        mask = constraints2mask(atoms)
        return np.sum(atoms.get_masses()[np.any(mask, axis=1)]) * atmass


class Thermochemistry(object):

    '''
    Thermochemistry the results will be in kJ/mol

    Parameters
    ----------
    atoms : ase.Atoms
        Atoms obect
    conditions : dict
        Dictionary with the conditions specicification
    system : dict
        Dicitonary with the system specification
    '''

    def __init__(self, atoms, conditions, system):

        self.atoms = atoms
        self.conditions = conditions
        self.system = system

    def get_qrotational(self, T):
        '''
        Calculate the rotational partition function in a rigid rotor
        approximation

        Parameters
        ----------
        T : float
            Temperature in `K`

        Notes
        -----

        .. math::

           q_{rot}(T) = \\frac{\sqrt{\pi I_{A}I_{B}I_{C}} }{\sigma}
                        \left( \\frac{ 2 k_{B} T }{\hbar^{2}} \\right)^{3/2}

        '''

        if self.system['phase'] == 'gas':
            # calculate the moments of ineria and convert them to [kg*m^2]
            atmass = value('atomic mass unit-kilogram relationship')
            I = self.atoms.get_moments_of_inertia(vectors=False)*atmass*np.power(10.0, -20)

            sigma = self.system['symmetrynumber']
            if self.system['pointgroup'] in ['Coov', 'Dooh']:
                return 2.0*np.max(I)*Boltzmann*T/(sigma*hbar**2)
            else:
                return np.sqrt(pi*np.product(I)*np.power(2.0*Boltzmann*T/hbar**2, 3))/sigma
        else:
            return 0.0

    def get_qtranslational(self, T):
        '''
        Calculate the translational partition function for a mole of ideal gas
        at temperature `T`

        Parameters
        ----------
        atoms : ase.Atoms
            Atoms
        T : float
            Temperature in K

        Notes
        -----

        .. math::

           q_{vib}(V, T) = \left( \\frac{ 2\pi M k_{B} T }{h^{2}} \\right)^{3/2} V

        '''

        if self.system['phase'] == 'gas':

            vol = gas_constant * T / (self.conditions['pressure'] * 1.0e6)
            totmass = get_total_mass(self.atoms)

            return vol*np.power(2.0*pi*totmass*Boltzmann*T/Planck**2, 1.5)/Avogadro
        else:
            return 0.0

    def get_translational_energy(self, T):
        '''
        Calculate the translational energy

        Parameters
        ----------
        T : float
            Temperature in `K`
        '''

        if self.system['phase'] == 'gas':
            return 1.5 * gas_constant * T * 1.0e-3
        else:
            return 0.0

    def get_translational_entropy(self, T):
        '''
        Calculate the translational entropy

        Parameters
        ----------
        T : float
            Temperature in `K`
        '''

        if self.system['phase'] == 'gas':
            qtrans = self.get_qtranslational(T)
            return gas_constant * (np.log(qtrans) + 2.5) * 1.0e-3
        else:
            return 0.0

    def get_rotational_energy(self, T):
        '''
        Calculate the rotational energy

        Parameters
        ----------
        T : float
            Temperature in `K`
        '''

        if self.system['phase'] == 'gas':
            if self.system['pointgroup'] in ['Coov', 'Dooh']:
                energy_rot = gas_constant * T * 1.0e-3
            else:
                energy_rot = 1.5 * gas_constant * T * 1.0e-3
        else:
            energy_rot = 0.0

        return energy_rot

    def get_rotational_entropy(self, T):
        '''
        Calculate the rotational entropy

        Parameters
        ----------
        T : float
            Temperature in `K`
        '''

        if self.system['phase'] == 'gas':
            qrot = self.get_qrotational(T)
            if self.system['pointgroup'] in ['Coov', 'Dooh']:
                entropy_rot = gas_constant * (np.log(qrot) + 1) * 1.0e-3
            else:
                entropy_rot = gas_constant * (np.log(qrot) + 1.5) * 1.0e-3
        else:
            entropy_rot = 0.0

        return entropy_rot

    def summary(self, T):
        '''
        Print summary with the thermochemical data at temperature `T` in kJ/mol

        Parameters
        ----------
        T : float
            Temperature in `K`
        '''

        print('\n' + 'THERMOCHEMISTRY'.center(50, '='))
        print('\n\t @ T = {0:6.2f} K\t p = {1:6.2f} MPa\n'.format(T, self.conditions['pressure']))
        print('-'*50)

        print('{0:<25s} : {1:15.3f}'.format('ln qtranslational', np.log(self.get_qtranslational(T))))
        print('{0:<25s} : {1:15.3f}'.format('ln qrotational', np.log(self.get_qrotational(T))))

        print('{0:<25s} : {1:15.3f}  kJ/mol'.format('U translational', self.get_translational_energy(T)))
        print('{0:<25s} : {1:15.3f}  kJ/mol'.format('U rotational', self.get_rotational_energy(T)))

        print('{0:<25s} : {1:11.4f} kJ/mol'.format('S translational', self.get_translational_entropy(T)))
        print('{0:<25s} : {1:11.4f} kJ/mol'.format('S rotational', self.get_rotational_entropy(T)))


class HarmonicThermo(Thermochemistry):
    '''
    Calculate thermochemistry in harmonic approximation

    Parameters
    ----------
    vibenergies : numpy.array
        Vibrational energies in Joules
    atoms : ase.Atoms
        Atoms obect
    conditions : dict
        Dictionary with the conditions specicification
    system : dict
        Dicitonary with the system specification
    '''

    def __init__(self, vibenergies, atoms, conditions, system):

        super().__init__(atoms, conditions, system)
        self.vibenergies = vibenergies

    def get_zpve(self):
        '''
        Calculate the Zero Point Vibrational Energy (ZPVE) in kJ/mol

        Notes
        -----

        .. math::

           ZPVE(T) = \\frac{1}{2}\sum^{3N-6}_{i=1} h\omega_{i}

        '''

        return 0.5 * np.sum(self.vibenergies) * Avogadro * 1.0e-3

    def get_qvibrational(self, T, uselog=True):
        '''
        Calculate the vibrational partition function at temperature `T` in
        kJ/mol

        Parameters
        ----------
        T : float
            Temperature in `K`
        uselog : bool
            When `True` return the natural logarithm of the partition function

        Notes
        -----

        .. math::

           q_{vib}(T) = \prod^{3N-6}_{i=1}\\frac{1}{1 - \exp(-h\omega_{i}/k_{B}T)}

        '''

        kT = Boltzmann * T
        if uselog:
            return np.sum(-np.log(1.0 - np.exp(-self.vibenergies/kT)))
        else:
            return np.prod(1.0/(1.0 - np.exp(-self.vibenergies/kT)))

    def get_vibrational_energy(self, T):
        '''
        Calculate the vibational energy correction at temperature `T` in kJ/mol

        Parameters
        ----------
        T : float
            Temperature in `K`

        Notes
        -----

        .. math::

           U_{vib}(T) = \\frac{R}{k_{B}}\sum^{3N-6}_{i=1} \\frac{h\omega_{i}}{\exp(h\omega_{i}/k_{B}T) - 1}

        '''

        kT = Boltzmann * T
        frac = np.sum(self.vibenergies/(np.exp(self.vibenergies/kT) -1.0))
        return 1.0e-3*gas_constant*frac/Boltzmann

    def get_vibrational_entropy(self, T):
        '''
        Calculate the vibrational entropy at temperature `T` in kJ/mol

        Parameters
        ----------
        T : float
            Temperature in `K`

        Notes
        -----

        .. math::

           S_{vib}(T) = R\sum^{3N-6}_{i=1}\left[ \\frac{h\omega_{i}}{k_{B}T(\exp(h\omega_{i}/k_{B}T) - 1)}
                      - \ln(1 - \exp(-h\omega_{i}/k_{B}T)) \\right]

        '''

        kT = Boltzmann * T

        frac = np.sum(self.vibenergies/(np.exp(self.vibenergies/kT) -1.0))/kT
        nlog = np.sum(np.log(1.0 - np.exp(-self.vibenergies/kT)))
        return 1.0e-3*gas_constant*(frac - nlog)

    def summary(self, T):
        '''
        Print summary with the thermochemical data at temperature `T` in kJ/mol

        Parameters
        ----------
        T : float
            Temperature in `K`
        '''

        if self.system['phase'] == 'solid':
            lnqtrans = lnqrot = 0.0
        else:
            lnqtrans = np.log(self.get_qtranslational(T))
            lnqrot = np.log(self.get_qrotational(T))

        print('\n' + ' THERMOCHEMISTRY '.center(50, '='), end='\n\n')
        print('\t @ T = {0:6.2f} K\t p = {1:6.2f} MPa'.format(T, self.conditions['pressure']), end='\n\n')
        print('-' * 50)

        print('Partition functions:')
        print('    {0:<20s} : {1:15.3f}'.format('ln qtranslational', lnqtrans))
        print('    {0:<20s} : {1:15.3f}'.format('ln qrotational', lnqrot))
        print('    {0:<20s} : {1:15.3f}'.format('ln qvibrational', self.get_qvibrational(T, uselog=True)))
        print('-' * 50)
        internal = self.get_translational_energy(T) + self.get_rotational_energy(T) + self.get_zpve() + self.get_vibrational_energy(T)
        print('{0:<24s} : {1:15.3f}  kJ/mol'.format('Internal energy (U)', internal))
        print('    {0:<20s} : {1:15.3f}  kJ/mol'.format('U translational', self.get_translational_energy(T)))
        print('    {0:<20s} : {1:15.3f}  kJ/mol'.format('U rotational', self.get_rotational_energy(T)))
        print('    {0:<20s} : {1:15.3f}  kJ/mol'.format('U vibrational', self.get_zpve() + self.get_vibrational_energy(T)))
        print('        {0:<16s} : {1:15.3f}  kJ/mol'.format('@ 0 K (ZPVE)', self.get_zpve()))
        print('        {0:<16s} : {1:15.3f}  kJ/mol'.format('@ {0:6.2f} K'.format(T), self.get_vibrational_energy(T)))
        print('-' * 74)
        entropy = self.get_translational_entropy(T) + self.get_rotational_entropy(T) + self.get_vibrational_entropy(T)
        St = self.get_translational_entropy(T)
        Sr = self.get_rotational_entropy(T)
        Sv = self.get_vibrational_entropy(T)
        print('*T'.rjust(65))
        print('{0:<24s} : {1:16.4f} kJ/mol*K    {2:11.4f} kJ/mol'.format('Entropy (S)', entropy, T * entropy))
        print('    {0:<20s} : {1:16.4f} kJ/mol*K    {2:11.4f} kJ/mol'.format('S translational', St, T * St))
        print('    {0:<20s} : {1:16.4f} kJ/mol*K    {2:11.4f} kJ/mol'.format('S rotational', Sr, T * Sr))
        print('    {0:<20s} : {1:16.4f} kJ/mol*K    {2:11.4f} kJ/mol'.format('S vibrational', Sv, T * Sv))
        print('-' * 74)
        print('{0:<24s} : {1:16.4f} kJ/mol'.format('U - T*S', internal - T * entropy))
        print('-' * 50)
        elenergy = self.atoms.get_potential_energy() * value('electron volt') * 1.0e-3 * Avogadro
        print('{0:<24s} : {1:16.4f} kJ/mol'.format('Electronic energy', elenergy))


class AnharmonicThermo(Thermochemistry):
    '''
    Calculate thermochemistry in harmonic approximation

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame with the anharmonic data at a given temperature, must have the following columns:
            - ``freq`` frequency of a given mode in cm^-1
            - ``zpve`` zero point vibrational energy contribution from a given mode in kJ/mol
            - ``qvib`` vibrational partition function contribution from a given mode
            - ``U`` internal energy contribution from a given mode
            - ``S`` entropy contribution from a given mode
    atoms : ase.Atoms
        Atoms obect
    conditions : dict
        Dictionary with the conditions specicification
    system : dict
        Dicitonary with the system specification
    '''

    def __init__(self, df, atoms, conditions, system):

        super().__init__(atoms, conditions, system)
        self.df = df

    def get_zpve(self):
        '''
        Calculate the Zero Point Vibrational Energy (ZPVE) in kJ/mol
        '''

        return self.df.zpve.sum()

    def get_qvibrational(self, uselog=True):
        '''
        Calculate the vibrational partition function in kJ/mol

        Parameters
        ----------
        uselog : bool
            When `True` return the natural logarithm of the partition function
        '''

        if uselog:
            return np.sum(np.log(self.df.qvib.astype('float64')))
        else:
            return self.df.qvib.sum()

    def get_vibrational_energy(self):
        '''
        Calculate the vibational energy correction in kJ/mol
        '''

        return self.df.U.sum() * 1.0e-3

    def get_vibrational_entropy(self):
        '''
        Calculate the vibrational entropy in kJ/mol
        '''

        return self.df.S.sum()

    def summary(self, T):
        '''
        Print summary with the thermochemical data at temperature `T` in kJ/mol

        Parameters
        ----------
        T : float
            Temperature in `K`
        '''

        if self.system['phase'] == 'solid':
            lnqtrans = lnqrot = 0.0
        else:
            lnqtrans = np.log(self.get_qtranslational(T))
            lnqrot = np.log(self.get_qrotational(T))

        print('\n' + ' THERMOCHEMISTRY '.center(50, '='), end='\n\n')
        print('\t @ T = {0:6.2f} K\t p = {1:6.2f} MPa'.format(T, self.conditions['pressure']), end='\n\n')
        print('-' * 50)

        print('Partition functions:')
        print('    {0:<20s} : {1:15.3f}'.format('ln qtranslational', lnqtrans))
        print('    {0:<20s} : {1:15.3f}'.format('ln qrotational', lnqrot))
        print('    {0:<20s} : {1:15.3f}'.format('ln qvibrational', self.get_qvibrational(uselog=True)))
        print('-' * 50)
        internal = self.get_translational_energy(T) + self.get_rotational_energy(T)\
                 + self.get_zpve() + self.get_vibrational_energy()
        print('{0:<24s} : {1:15.3f}  kJ/mol'.format('Internal energy (U)', internal))
        print('    {0:<20s} : {1:15.3f}  kJ/mol'.format('U translational', self.get_translational_energy(T)))
        print('    {0:<20s} : {1:15.3f}  kJ/mol'.format('U rotational', self.get_rotational_energy(T)))
        print('    {0:<20s} : {1:15.3f}  kJ/mol'.format('U vibrational', self.get_zpve() + self.get_vibrational_energy()))
        print('        {0:<16s} : {1:15.3f}  kJ/mol'.format('@ 0 K (ZPVE)', self.get_zpve()))
        print('        {0:<16s} : {1:15.3f}  kJ/mol'.format('@ {0:6.2f} K'.format(T), self.get_vibrational_energy()))
        print('-' * 74)
        entropy = self.get_translational_entropy(T) + self.get_rotational_entropy(T) + self.get_vibrational_entropy()
        St = self.get_translational_entropy(T)
        Sr = self.get_rotational_entropy(T)
        Sv = self.get_vibrational_entropy()
        print('*T'.rjust(65))
        print('{0:<24s} : {1:16.4f} kJ/mol*K    {2:11.4f} kJ/mol'.format('Entropy (S)', entropy, T * entropy))
        print('    {0:<20s} : {1:16.4f} kJ/mol*K    {2:11.4f} kJ/mol'.format('S translational', St, T * St))
        print('    {0:<20s} : {1:16.4f} kJ/mol*K    {2:11.4f} kJ/mol'.format('S rotational', Sr, T * Sr))
        print('    {0:<20s} : {1:16.4f} kJ/mol*K    {2:11.4f} kJ/mol'.format('S vibrational', Sv, T * Sv))
        print('-' * 74)
        print('{0:<24s} : {1:16.4f} kJ/mol'.format('U - T*S', internal - T * entropy))
        print('-' * 50)
        elenergy = self.atoms.get_potential_energy() * value('electron volt') * 1.0e-3 * Avogadro
        print('{0:<24s} : {1:16.4f} kJ/mol'.format('Electronic energy', elenergy))
