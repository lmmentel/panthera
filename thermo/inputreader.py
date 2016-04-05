
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
        'Tinitial'     : '303.15',
        'Tfinal'       : '303.15',
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

    conditions = {}
    conditions['Tinitial'] = config.getfloat('conditions', 'Tinitial')
    conditions['Tfinal'] = config.getfloat('conditions', 'Tfinal')
    conditions['Tstep'] = config.getfloat('conditions', 'Tstep')
    conditions['pressure'] = config.getfloat('conditions', 'pressure')
    
    job = {}
    job['proj_translations'] = config.getboolean('job', 'translations')
    job['proj_rotations'] = config.getboolean('job', 'rotations')
    job['code'] = config.get('job', 'code')
    
    system = {}
    system['phase'] = config.get('system', 'phase')
    system['pointgroup'] = config.get('system', 'pointgroup')
    system['symmetrynumber'] = get_symmetry_number(system['pointgroup'])
    
    return conditions, job, system

def get_symmetry_number(pointgroup):
    '''
    Return the symmetry number for a given point group

    .. see::
       C. J. Cramer, `Essentials of Computational Chemistry, Theories and Models`, 
       second edition, p. 363 

    Args:
        pointgroup : str
            Symbol of the point group
    '''

    symmetrynumbers = {'Ci' :  1, 'Cs' :  1, 'Coov' :  1, 'Dooh' : 2,
                       'T'  : 12, 'Td' : 12, 'Oh'   : 24, 'Ih'   : 60}

    cpatt = re.compile(r'C(\d+)[vh]?')    
    dpatt = re.compile(r'D(\d+)[dh]?')
    spatt = re.compile(r'S(\d+)')

    if pointgroup in symmetrynumbers.keys():
        return symmetrynumbers[pointgroup]
    else:
        mc = cpatt.match(pointgroup)
        md = dpatt.match(pointgroup)
        ms = spatt.match(pointgroup)

        if mc:
            return int(mc.group(1))
        elif md:
            return 2*int(md.group(1))
        elif ms:
            return int(ms.group(1))//2
        else:
            raise ValueError('Point group label "{}" unknown, cannot assign a rotational symmetry '
                             'number'.format(pointgroup))

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