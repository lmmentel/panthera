
import re
import argparse
import os
import sys

if sys.version_info.major == 3:
    import configparser as cp
else:
    import ConfigParser as cp

import numpy as np

def parse_arguments():
    '''
    Parse the input/config file name from the command line, parse the config and return the
    parameters.
    '''

    parser = argparse.ArgumentParser()
    parser.add_argument('config', help='file with the configuration parameters for thermo')
    args = parser.parse_args()

    if not os.path.exists(args.config):
        raise ValueError('Specified file <{}> does not exist'.format(args.config))

    defaults = {
        'Tinitial'     : '298.15',
        'Tfinal'       : '298.15',
        'Tstep'        : '0.0',
        'pressure'     : '1.0',
        'translations' : 'true',
        'rotations'    : 'true',
        'pointgroup'   : 'C1',
        'phase'        : 'gas',
        'code'         : None,
    }

    config = cp.ConfigParser(defaults=defaults, allow_no_value=True)
    config.read(args.config)

    symmetrynumbers = {
        'C1'   : 1,
        'Cs'   : 1,
        'C2'   : 2,
        'C2v'  : 2,
        'C3v'  : 3,
        'C2h'  : 2,
        'Coov' : 1,
        'D2h'  : 4,
        'D3h'  : 6,
        'D5h'  : 10,
        'Dooh' : 2,
        'D3d'  : 6,
        'Td'   : 12,
        'Oh'   : 24,
        }

    args.Tinitial = config.getfloat('thermo', 'Tinitial')
    args.Tfinal = config.getfloat('thermo', 'Tfinal')
    args.step = config.getfloat('thermo', 'Tstep')
    args.pressure = config.getfloat('thermo', 'pressure')
    args.proj_translations = config.getboolean('thermo', 'translations')
    args.proj_rotations = config.getboolean('thermo', 'rotations')
    args.code = config.get('thermo', 'code')
    args.phase = config.get('thermo', 'phase')
    args.pointgroup = config.get('thermo', 'pointgroup')
    if args.pointgroup in symmetrynumbers.keys():
        args.symmetrynumber = symmetrynumbers[args.pointgroup]
    else:
        raise ValueError('Point group label <{}> unknown, cannot assign a rotational symmetry '
                         'number'.format(args.pointgroup))

    return args

def read_vasp_hessian(outcar='OUTCAR'):
    '''
    Parse the hessian from the VASP ``OUTCAR`` file into a numpy array
    '''

    if os.path.exists(outcar):
        with open(outcar, 'r') as foutcar:
            lines = foutcar.readlines()
    else:
        raise OSError("File {} doesn't exist".format(outcar))

    for n, line in enumerate(lines):

        if 'SECOND DERIVATIVES ' in line:
            dofpatt = re.compile(r'Degrees of freedom DOF\s*=\s*(\d+)')
            match = next(dofpatt.search(line) for line in lines if dofpatt.search(line))
            dof = int(match.group(1))

            hessian = np.zeros((dof, dof), dtype=float)

            for i, row in enumerate(lines[n + 3 : n + 3 + dof]):
                hessian[i] = [float(x) for x in row.split()[1:]]

            return hessian
    else:
        raise ValueError('No hessian found in file: {}'.format(outcar))

def read_em_freq(fname):
    '''
    Read the file ``fname`` with the frequencies, reduced masses and fitted
    fitted coefficients for the potential  into a pandas DataFrame.
    
    Args:
        fname : str
            Name of the file
    '''

    cols = ['type', 'freq', 'mass', 'a0', 'a1', 'a2', 'a3', 'a4', 'a5', 'a6']
    data = pd.read_csv(fname, sep='\s+', engine='python', names=cols)
    return data