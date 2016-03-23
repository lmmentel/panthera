
from __future__ import print_function, division

from pprint import pprint

from scipy.constants import value

from ase.io.vasp import read_vasp_out
from ase.thermochemistry import HarmonicThermo

from inputreader import parse_arguments, read_vasp_hessian
from vibrations import get_harmonic_vibrations
from anharmonic_hamiltonian import anharmonic_frequencies

def main():
    '''The main Thermo program'''

    args = parse_arguments()

    pprint(args)

    if args.code == 'VASP':
        atoms = read_vasp_out('OUTCAR')
        #hessian = read_vasp_hessian('OUTCAR')

        #freqs = get_harmonic_vibrations(atoms, hessian, args)

        #cm1_to_eV = 100.0*value('inverse meter-electron volt relationship')

        #thermo = HarmonicThermo(freqs.real[:-3]*cm1_to_eV, atoms.get_potential_energy())
        #print('Internal : ', thermo.get_internal_energy(303.))
        #print('Entropy  : ', thermo.get_entropy(303.))
        #print('Helmholtz: ', thermo.get_gibbs_energy(303.))

        anharmonic_frequencies(atoms, args)

    else:
        raise NotImplementedError('Code {} is not supported yet.'.format(args.code))



if __name__ == '__main__':

    main()
