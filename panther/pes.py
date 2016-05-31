
import numpy as np
import pandas as pd
from scipy.constants import value


def calculate_energies(images, calc, modes='all'):
    '''
    Given a set of images as a nested OrderedDict of Atoms objects and a
    calculator, calculate the energy for each displaced structure

    Parameters
    ----------
    images : OrderedDict
        A nested OrderedDict of displaced Atoms objects
    calc : calculator instance
    modes : str or list

    Returns
    -------
    energies : pandas.DataFrame
        DataFrame with the energies per displacement

    '''

    if modes == 'all':
        nmodes = len(images)
        modes = range(nmodes)
    elif isinstance(modes, (list, tuple)):
        nmodes = len(modes)
    else:
        ValueError('<modes> should be a str, list or tuple '
                   'got: {}'.format(type('modes')))

    ecols = ['E_' + str(i) for i in range(-4, 5)]
    energies = pd.DataFrame(0.0, columns=ecols, index=range(12))

    for mode in modes:
        for point in images[mode].keys():

            atoms = images[mode][point]
            atoms.set_calculator(calc)
            energy = atoms.get_potential_energy()
            energies.set_value(mode, 'E_' + str(point), energy)

    return energies


def fit_potentials(modeinfo, energies):
    '''
    Fit the potentials with 6th and 4th order polynomials

    Parameters
    ----------
    modeinfo : pandas.DataFrame
        DataFrame with per mode characteristics, displacements, masses
        and a flag to mark it a mode is a stretching mode or not
    energies : pd.DataFrame
        Energies per displacement

    Returns
    -------
    out : (coeffs6o, coeffs4o)
        DataFrames with 6th and 4th polynomial coefficients fitted to the
        potential
    '''

    energies = energies.subtract(energies['E_0'], axis=0)
    energies = energies * value('electron volt-hartree relationship')
    E = energies.as_matrix()

    D = np.dot(modeinfo['displacement'].reshape(-1, 1), np.arange(-4, 5).reshape(1, -1))
    D = D / np.sqrt(modeinfo['effective_mass'].values).reshape(-1, 1)
    D = D.astype(float)

    coeffs6o = pd.DataFrame(index=energies.index,
                            columns=['c_' + str(x) for x in np.arange(6, -1, -1)])
    coeffs4o = pd.DataFrame(index=energies.index,
                            columns=['c_' + str(x) for x in np.arange(4, -1, -1)])

    for i in energies.index:
        coeffs4o.loc[i] = np.polyfit(D[i], E[i], deg=4)
        coeffs6o.loc[i] = np.polyfit(D[i], E[i], deg=6)

    return coeffs6o, coeffs4o


def differentiate(displacements, energies, order=2):
    '''
    Calculate numerical detivatives using the central difference formula

    Parameters
    ----------
    displacements : array_like
    energies : DataFrame
    order : int
        Order of the derivative

    Notes
    -----
    Central difference coefficients taken from [1]_

    .. [1] Fornberg, B. (1988). Generation of finite difference
       formulas on arbitrarily spaced grids. Mathematics of Computation,
       51(184), 699-699. doi:10.1090/S0025-5718-1988-0935077-0

    '''

    energies = energies.subtract(energies['E_0'], axis=0)
    energies = energies * value('electron volt-hartree relationship')
    E = energies.as_matrix()

    cols = ['p2', 'p4', 'p6', 'p8']
    dt = [('idx', int)] + list(zip(cols, [float] * 4))

    coeffs1d = np.zeros(9, dtype=dt)
    coeffs1d['idx'] = range(-4, 5)
    coeffs1d['p2'] = [0.0, 0.0, 0.0, -0.5, 0.0, 0.5, 0.0, 0.0, 0.0]
    coeffs1d['p4'] = [0.0, 0.0, 1.0/12.0, -2.0/3.0, 0.0, 2.0/3.0, -1.0/12.0, 0.0, 0.0]
    coeffs1d['p6'] = [0.0, -1.0/60.0, 3.0/20.0 , -3.0/4.0, 0.0, 3.0/4.0, -3.0/20.0, 1.0/60.0, 0.0]
    coeffs1d['p8'] = [1.0/280.0, -4.0/105.0, 1.0/5.0, -4.0/5.0, 0.0, 4.0/5.0, -1.0/5.0, 4.0/105.0, -1.0/280.0]

    coeffs2d = np.zeros(9, dtype=dt)
    coeffs2d['idx'] = range(-4, 5)
    coeffs2d['p2'] = [0.0, 0.0, 0.0, 1.0, -2.0, 1.0, 0.0, 0.0, 0.0]
    coeffs2d['p4'] = [0.0, 0.0, -1.0/12.0, 4.0/3.0, -5.0/2.0, 4.0/3.0, -1.0/12.0, 0.0, 0.0]
    coeffs2d['p6'] = [0.0, 1.0/90.0, -3.0/20.0, 3.0/2.0, -49.0/18.0, 3.0/2.0, -3.0/20.0, 1.0/90.0, 0.0]
    coeffs2d['p8'] = [-1.0/560.0, 8.0/315.0, -1.0/5.0, 8.0/5.0, -205.0/72.0, 8.0/5.0, -1.0/5.0, 8.0/315.0, -1.0/560.0]

    if order == 1:
        C = coeffs1d[cols].view(np.float64).reshape(coeffs1d.shape + (-1,))
        return np.dot(E, C) / displacements[:, np.newaxis]
    elif order == 2:
        C = coeffs2d[cols].view(np.float64).reshape(coeffs2d.shape + (-1,))
        return np.dot(E, C) / np.power(displacements, 2)[:, np.newaxis]
    else:
        raise NotImplementedError('{} order derivatives not available'.format(order))