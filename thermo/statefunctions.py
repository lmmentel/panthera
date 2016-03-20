import numpy as np

from scipy.constants import value, pi, Avogadro, Planck, Boltzmann, gas_constant

# TODO this should be a method from the Structure class
def get_total_mass(structure):
    '''
    Calculate the total mass of all atoms that have at least one coordinate that can be relaxed.

    The masses are assumed to be in the ``masses`` (np.array) attribute and the flags for coordinates
    are in ``free`` attribute
    '''

    return np.sum(structure.masses[np.any(structure.free, axis=1)]) * atmass

def qtranlational(structure, T, p):
    '''
    Calculate the translational partition function for a mole of ideal gas at temperature ``T`` and
    pressure ``p``.
    '''

    vol = gas_constant*T/p
    totmass = structure.get_total_mass()

    return vol*np.power(2.0*pi*totmass*Boltzmann*T/Planck**2, 1.5)/Avogadro

def qrotational(structure, point_group):
    '''
    Calculate the rotational partition function for a rigid rotor
    '''

    I = rotationalintertia(structure.atoms)

    k_hartree = Boltzmann*value('joule-hartree relationship')

    if point_group in ['Coov', 'Dooh']:
        return np.sqrt(I[0]*I[1]*np.power(2.0*pi*k_hartree*T, 2))/(pi*sigma)
    else:
        return np.sqrt(np.product(I)*np.power(2.0*pi*k_hartree*T, 3))/(pi*sigma)



def read_em_freq(fname):

    pass


