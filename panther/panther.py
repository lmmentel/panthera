
'Python package for Anharmonic Thermochemistry'

from __future__ import print_function, division

from pprint import pprint

from scipy.constants import Planck, value

import numpy as np

from ase.io.vasp import read_vasp_out

from .inputreader import parse_arguments, read_vasp_hessian, write_internal
from .vibrations import get_harmonic_vibrations
from .anharmonicity import anharmonic_frequencies, merge_vibs
from .thermochemistry import HarmonicThermo, AnharmonicThermo


def temperature_range(conditions):
    '''
    Calculate the temperature grid from the input values and return them as
    numpy array

    Parameters
    ----------
    conditions : dict
        Variable for conditions read from the input/config

    Returns
    -------
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

    if args.command == 'convert':
        atoms = read_vasp_out('OUTCAR', index=0)
        hessian = read_vasp_hessian('OUTCAR', symmetrize=False, convert2au=False,
                                    negative=False)

        write_internal(atoms, hessian, job)

    elif args.command == 'harmonic':

        if job['code'] == 'VASP':
            atoms = read_vasp_out('OUTCAR', index=0)
            hessian = read_vasp_hessian('OUTCAR', symmetrize=True,
                                        convert2au=True, negative=True)
        else:
            raise NotImplementedError('Code {} is not supported yet.'.format(job['code']))

        freqs, normal_modes = get_harmonic_vibrations(hessian, atoms,
                                                      job['proj_translations'],
                                                      job['proj_rotations'])

        freqs = 0.01 * value('hartree-inverse meter relationship') * freqs.astype(complex)**0.5

        # save the result
        print('INFO: Saving vibrational frequencies to: frequencies.npy')
        np.save('frequencies', freqs)
        print('INFO: Saving vibrational normal modes to: normal_modes.npy')
        np.save('normal_modes', normal_modes)

        print('\n' + ' Vibrational frequencies in [cm^-1] '.center(50, '='), end='\n\n')
        print('        {0:^20s} {1:^20s}'.format('real', 'imag'))
        for i, v in enumerate(freqs, start=1):
            print('{0:5d} : '.format(i), '{0:20.10f} {1:20.10f}'.format(v.real, v.imag))

        # convert frequencies from [cm^-1] to [Hz] and get vib. energies in Joules
        vibenergies = Planck * freqs.real * 100.0 * value('inverse meter-hertz relationship')
        vibenergies = vibenergies[vibenergies > 0.0]

        thermo = HarmonicThermo(vibenergies, atoms, conditions, system)

        for temp in temperature_range(conditions):

            thermo.summary(temp)

    elif args.command == 'anharmonic':
        atoms = read_vasp_out('OUTCAR', index=-1)

        for temp in temperature_range(conditions):

            df6 = anharmonic_frequencies(atoms, temp, job, system, fname='em_freq')
            df4 = anharmonic_frequencies(atoms, temp, job, system, fname='em_freq_4th')

            df = merge_vibs(df6, df4, temp)

            thermo = AnharmonicThermo(df, atoms, conditions, system)
            thermo.summary(temp)

if __name__ == '__main__':

    main()
