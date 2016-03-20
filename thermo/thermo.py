

from pprint import pprint

from ase.io.vasp import read_vasp_out

from inputreader import parse_arguments, read_vasp_hessian
from vibrations import get_harmonic_vibrations

def main():
    '''The main Thermo program'''

    args = parse_arguments()

    pprint(args)

    if args.code == 'VASP':
        atoms = read_vasp_out('OUTCAR')
        hessian = read_vasp_hessian('OUTCAR')

        get_harmonic_vibrations(atoms, hessian, args)

    else:
        raise NotImplementedError('Code {} is not supported yet.'.format(args.code))


if __name__ == '__main__':

    main()
