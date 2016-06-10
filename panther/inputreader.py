
'''
Module providing functions for reading the input and other related files
'''

from __future__ import print_function, absolute_import, division

import re
import argparse
import os
import sys
import io
import six

from collections import defaultdict, OrderedDict

import numpy as np
import pandas as pd

from ase.io.vasp import read_vasp
from ase.io.trajectory import Trajectory

if sys.version_info.major == 3:
    import configparser as cp
else:
    import ConfigParser as cp


def parse_arguments():
    '''
    Parse the input/config file name from the command line, parse the config
    and return the parameters.
    '''

    parser = argparse.ArgumentParser()
    parser.add_argument('command',
                        choices=('convert', 'harmonic', 'anharmonic'),
                        help='choose what to do')
    parser.add_argument('config',
                        help='file with the configuration parameters for thermo')
    args = parser.parse_args()

    if not os.path.exists(args.config):
        raise ValueError('Specified file <{}> does not exist'.format(args.config))

    defaults = {
        'Tinitial'       : '303.15',
        'Tfinal'         : '303.15',
        'Tstep'          : '0.0',
        'pressure'       : '0.1',
        'translations'   : 'true',
        'rotations'      : 'true',
        'pointgroup'     : 'C1',
        'phase'          : 'gas',
        'code'           : None,
        'internal_fname' : 'default',
        'eigenhess_fname': 'userinstr',
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
    job['internal_fname'] = config.get('job', 'internal_fname')
    job['eigenhess_fname'] = config.get('job', 'eigenhess_fname')
    job['code'] = config.get('job', 'code')

    system = {}
    system['phase'] = config.get('system', 'phase')
    system['pointgroup'] = config.get('system', 'pointgroup')
    system['symmetrynumber'] = get_symmetry_number(system['pointgroup'])

    return args, conditions, job, system


def get_symmetry_number(pointgroup):
    '''
    Return the symmetry number for a given point group

    .. seealso::
       C. J. Cramer, `Essentials of Computational Chemistry, Theories and Models`,
       2nd Edition, p. 363

    Parameters
    ----------
    pointgroup : str
        Symbol of the point group
    '''

    symmetrynumbers = {'Ci':  1, 'Cs':  1, 'Coov':  1, 'Dooh': 2,
                       'T' : 12, 'Td': 12, 'Oh'  : 24, 'Ih'  : 60}

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
            return 2 * int(md.group(1))
        elif ms:
            return int(ms.group(1)) // 2
        else:
            raise ValueError('Point group label "{}" unknown, cannot assign '
                             'a rotational symmetry number'.format(pointgroup))


def read_vasp_hessian(outcar='OUTCAR', symmetrize=True, convert2au=True,
                      negative=True):
    '''
    Parse the hessian from the VASP ``OUTCAR`` file into a numpy array

    Parameters
    ----------
    outcar : str
        Name of the VASP output, default is ``OUTCAR``
    symmetrize : bool
        If ``True`` the hessian will be symmetrized
    covert2au : bool
        If ``True`` convert the hessian to atomic units, in the other
        case hessian is returned in [eV/Angstrom**2]
    negative : bool
        If ``True`` the hessian will be multiplied by -1 on return

    Returns
    -------
    hessian : numpy.array
        Hessian matrix

    .. note::
       By default VASP prints negative hessian so the elements are
       multiplied by -1 to restore the original hessian

    '''

    from scipy.constants import angstrom, value

    ang2bohr = angstrom / value('atomic unit of length')
    ev2hartree = value('electron volt-hartree relationship')

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

            for i, row in enumerate(lines[n + 3: n + 3 + dof]):
                hessian[i] = [float(x) for x in row.split()[1:]]

            if symmetrize:
                hessian = (hessian + hessian.T) * 0.5

            if convert2au:
                hessian = hessian * ev2hartree / (ang2bohr**2)

            if negative:
                hessian = -1 * hessian

            return hessian
    else:
        raise ValueError('No hessian found in file: {}'.format(outcar))


def print_modeinfo(mi, output=None):
    '''
    Print the vibrational population data

    Parameters
    ----------
    mi : pandas.DataFrame
    output : str
        Name of the file to store the printout, if ``None`` stdout will be used
    '''

    if output is not None:
        fobj = open(output, 'w')
    else:
        fobj = None

    fmts = {
        'HOfreq': '{:12.6f}'.format,
        'effective_mass': '{:12.6f}'.format,
        'displacement': '{:12.6f}'.format,
        'P_stretch': '{:>8.2%}'.format,
        'P_bend': '{:>8.2%}'.format,
        'P_torsion': '{:>8.2%}'.format,
        'P_longrange': '{:>8.2%}'.format,
    }

    print(mi[mi['vibration']].to_string(formatters=fmts), file=fobj)

    if output is not None:
        fobj.close()


def print_mode_thermo(df, info=False):
    '''
    After calculating all the anharmonic modes print the per mode themochemical
    functions
    '''

    fmts = {
        'freq': '{:12.6f}'.format,
        'zpve': '{:12.6f}'.format,
        'qvib': '{:14.6e}'.format,
        'U'   : '{:12.6f}'.format,
        'S'   : '{:14.6e}'.format,
    }

    # header with the units
    header = '     {0:>12s} {1:>12s} {2:>14s} {3:>12s} {4:>14s}'
    print(header.format('[cm^-1]', '[kJ/mol]', ' ', '[kJ/mol]', '[kJ/mol*K]'))

    print(df.to_string(formatters=fmts))

    if info:
        print('INFO codes')
        print('-' * 10)
        print('OK      : Succesfully converged the anharmonic eigenproblem')
        print('AGTH    : Anharmonic frequency greater than the harmonic')
        print('MAXITER : Maximum number of iterations exhausted')


# parsers and write methods for legacy EIGEN_HESS files

def read_bmatdat():
    '''
    Read the ``bmat.dat`` file with internal coordiantes and the B matrix
    produced by the original ``writeBmat`` code

    Returns
    -------
    internals, Bmatrix : tuple
        Internal coordiantes and B matrix
    '''

    with open('bmat.dat', 'r') as fdat:
        lines = fdat.readlines()

    nint, ndof = (int(x) for x in lines[1].split())
    coordlno = next(i for i, line in enumerate(lines) if 'Coordinates (au):' in line)
    bmatlno = next(i for i, line in enumerate(lines) if 'Bmatrix(ij):' in line)

    internals = np.array([tuple(row.split()[:2]) for row in lines[coordlno + 1:coordlno + nint + 1]],
                         dtype=[('type', 'S4'), ('value', np.float32)])

    Bmatrix = np.zeros((nint, ndof), dtype=float)

    for line in lines[bmatlno + 1:]:
        i, j, val = line.split()
        Bmatrix[int(i) - 1, int(j) - 1] = float(val)

    return internals, Bmatrix


def read_em_freq(fname):
    '''
    Read the file ``fname`` with the frequencies, reduced masses and fitted
    fitted coefficients for the potential  into a pandas DataFrame.

    Parameters
    ----------
    fname : str
        Name of the file with PES
    '''

    cols = ['type', 'freq', 'mass', 'a0', 'a1', 'a2', 'a3', 'a4', 'a5', 'a6']
    data = pd.read_csv(fname, sep=r'\s+', engine='python', names=cols)
    data.set_index(np.arange(1, data.shape[0] + 1), inplace=True)
    for col in cols[1:]:
        data[col] = data[col].astype(float)
    return data


def read_pes(fname):
    '''
    Parse the file with the potential energy surface (PES) into a dict of
    numpy arrays with mode numbers as keys

    Parameters
    ----------
    fname : str
        Name of the file with PES
    '''

    with open(fname, 'r') as fobj:
        data = fobj.read()

    pat = re.compile(' Scan along mode # =\s*(\d+)')
    parsed = [x for x in pat.split(data) if x != '']
    it = iter(parsed)
    parsed = {int(mode): np.loadtxt(io.StringIO(six.text_type(pes))) for mode, pes in zip(it, it)}
    return parsed


def read_poscars(filename):
    '''
    Read POSCARs file with the displaced structures and return
    an OrderedDict with the Atoms objects
    '''

    pat = re.compile(r'Mode\s*=\s*(\d+)\s*point\s*=\s*(-?\d+)')

    if os.path.exists(filename):
        with open(filename, 'r') as fdata:
            poscars = fdata.read()
    else:
        raise OSError('File "{}" does not exist'.format(filename))

    parsed = [x for x in pat.split(poscars) if x != ' ']
    images = OrderedDict()

    it = iter(parsed)
    for mode, point, geometry in zip(it, it, it):
        images[tuple([int(mode) -1, int(point)])] = read_vasp(io.StringIO(six.text_type(geometry)))
    return images


def write_internal(atoms, hessian, job):
    '''
    Write a file with the system details
    '''

    with open(job['internal_fname'], 'w') as fout:

        fout.write('start title:\n')
        fout.write('End of frequencies calculation\n')

        fout.write('start lattice\n')
        for row in atoms.get_cell():
            fout.write('{0:15.9f} {1:15.9f} {2:15.9f}\n'.format(*tuple(row)))
        fout.write('end lattice\n')

        fout.write('start atoms\n')
        for atom in atoms:
            fout.write('{0:2s} {1:15.5f} {2:15.5f} {3:15.5f} T T T\n'.format(
                atom.symbol, *tuple(atom.position)))
        fout.write('end atoms\n')

        fout.write('start energy\n')
        fout.write('{0:15.8f}\n'.format(atoms.get_potential_energy()))
        fout.write('end energy\n')

        if len(atoms) * 3 == hessian.shape[0]:
            hesstype = 'full'
        else:
            hesstype = 'partial'

        fout.write('input hessian\n')
        fout.write('{0:s}\n'.format(hesstype))

        fout.write('start hessian matrix\n')
        for row in hessian:
            fout.write(' '.join(['{0:15.8f}'.format(x) for x in row]) + '\n')
        fout.write('end hessian matrix\n')

    print('wrote file: "{}" with the data in internal format'.format(
        job['internal_fname']))


def write_modes(filename='POSCARs'):
    '''
    Convert a file with multiple geometries representing vibrational modes
    in ``POSCAR``/``CONTCAR`` format into trajectory files with modes.
    '''

    pat = re.compile(r'Mode\s*=\s*(\d+)\s*point\s*=\s*(-?\d+)')

    if os.path.exists(filename):
        with open(filename, 'r') as fdata:
            poscars = fdata.read()
    else:
        raise OSError('File "{}" does not exist'.format(filename))

    parsed = [x for x in pat.split(poscars) if x != ' ']

    it = iter(parsed)
    dd = defaultdict(list)
    for i, j, item in zip(it, it, it):
        dd[i].append(item)

    for mode, geometries in dd.items():
        traj = Trajectory('mode.{}.traj'.format(mode), 'w')
        for geometry in geometries:
            atoms = read_vasp(io.StringIO(six.text_type(geometry)))
            traj.write(atoms)
        traj.close()
