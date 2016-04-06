
'Main package file for the themopy package'

from __future__ import print_function, division

from pprint import pprint

from scipy.constants import Planck, value

import numpy as np

from ase.io.vasp import read_vasp_out
#from ase.thermochemistry import HarmonicThermo

from .inputreader import parse_arguments, read_vasp_hessian
from .vibrations import get_harmonic_vibrations
from .anharmonic_hamiltonian import anharmonic_frequencies
from .thermochemistry import Thermochemistry

def temperature_range(conditions):
    '''
    Calculate the temperature grid from the input values and return them as numpy array

    Args:
        conditions : dict
            Variable for conditions read from the input/config

    Returns:
        temps : numpy.array
            Array with the temperature grid
    '''

    epsilon = np.finfo(np.float).eps
    if np.abs(conditions['Tinitial'] - conditions['Tfinal']) > epsilon:
        if np.abs(conditions['Tstep']) > epsilon:
            num = int((conditions['Tfinal'] - conditions['Tinitial'])/conditions['Tstep']) + 1
            temps = np.linspace(conditions['Tinitial'], conditions['Tfinal'], num)
        else:
            temps = np.array([conditions['Tinitial'], conditions['Tfinal']])
    else:
        temps = np.array([conditions['Tinitial']])

    return temps


def main():
    '''The main Thermo program'''

    conditions, job, system = parse_arguments()

    pprint(conditions)
    pprint(job)
    pprint(system)

    if job['code'] == 'VASP':
        atoms = read_vasp_out('OUTCAR', index=0)
        hessian = read_vasp_hessian('OUTCAR')

        freqs = get_harmonic_vibrations(job, atoms, hessian)

        # convert frequencies from [cm^-1] to [Hz]
        vibenergies = Planck * freqs * 100.0*value('inverse meter-hertz relationship')
        vibenergies = vibenergies[vibenergies > 0.0]

        thermo = Thermochemistry(atoms, vibenergies, conditions, system)

        for temp in temperature_range(conditions):

            thermo.summary(temp)


        #cm1_to_eV = 100.0*value('inverse meter-electron volt relationship')

        #thermo = HarmonicThermo(freqs.real[:-3]*cm1_to_eV, atoms.get_potential_energy())
        #print('Internal : ', thermo.get_internal_energy(303.))
        #print('Entropy  : ', thermo.get_entropy(303.))
        #print('Helmholtz: ', thermo.get_gibbs_energy(303.))

        #anharmonic_frequencies(atoms, args)

    else:
        raise NotImplementedError('Code {} is not supported yet.'.format(job['code']))

if __name__ == '__main__':

    main()
