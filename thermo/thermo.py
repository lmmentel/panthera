
'Main package file for the themopy package'

from __future__ import print_function, division

from pprint import pprint

from scipy.constants import Planck, value

import numpy as np
import pandas as pd

from ase.io.vasp import read_vasp_out
#from ase.thermochemistry import HarmonicThermo

from .inputreader import parse_arguments, read_vasp_hessian, write_internal
from .vibrations import get_harmonic_vibrations
from .anharmonicity import anharmonic_frequencies, merge_vibs
from .thermochemistry import HarmonicThermo, AnharmonicThermo

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

    args, conditions, job, system = parse_arguments()

    pprint(conditions)
    pprint(job)
    pprint(system)

    if args.command == 'writeinp':
        atoms = read_vasp_out('OUTCAR', index=-1)
        hessian = read_vasp_hessian('OUTCAR')

        write_internal(atoms, hessian, filename='default')

    elif args.command == 'harmonic':

        if job['code'] == 'VASP':
            atoms = read_vasp_out('OUTCAR', index=0)
            hessian = read_vasp_hessian('OUTCAR')

            freqs = get_harmonic_vibrations(job, atoms, hessian)

            # convert frequencies from [cm^-1] to [Hz] and get vib. energies in Joules
            vibenergies = Planck * freqs * 100.0*value('inverse meter-hertz relationship')
            vibenergies = vibenergies[vibenergies > 0.0]

            thermo = HarmonicThermo(vibenergies, atoms, conditions, system)

            for temp in temperature_range(conditions):

                thermo.summary(temp)

        else:
            raise NotImplementedError('Code {} is not supported yet.'.format(job['code']))

    elif args.command == 'anharmonic':
        atoms = read_vasp_out('OUTCAR', index=-1)

        pd.set_option("display.width", 120)

        for temp in temperature_range(conditions):
            print(' 6th order T = {} '.format(temp).center(80, '='))
            df6 = anharmonic_frequencies(atoms, temp, job, system, fname='em_freq')
            print(df6)
            df6.to_pickle('anh6.pkl')
            print(' 4th order T = {} '.format(temp).center(80, '='))
            df4 = anharmonic_frequencies(atoms, temp, job, system, fname='em_freq_4th')
            print(df4)
            df4.to_pickle('anh4.pkl')

            df = merge_vibs(df6, df4)

            thermo = AnharmonicThermo(df, atoms, conditions, system)
            thermo.summary(temp)

if __name__ == '__main__':

    main()
