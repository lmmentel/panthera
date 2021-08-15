from __future__ import print_function

import numpy as np
import pandas as pd
from six import string_types

from ase import units


def expandrange(modestr):
    """
    Convert a comma separated string of indices and dash separated ranges into
    a list of integer indices

    Parameters
    ----------
    modestr : str

    Returns
    -------
    indices : list of ints

    Examples
    --------

    >>> from panthera.pes import expandrange
    >>> s = "2,3,5-10,20,25-30"
    >>> expandrange(s)
    [2, 3, 5, 6, 7, 8, 9, 10, 20, 25, 26, 27, 28, 29, 30]
    """

    ranges = (x.split("-") for x in modestr.split(","))
    return [i for r in ranges for i in range(int(r[0]), int(r[-1]) + 1)]


def calculate_energies(images, calc, modes="all"):
    """
    Given a set of images as a nested OrderedDict of Atoms objects and a
    calculator, calculate the energy for each displaced structure

    Parameters
    ----------
    images : OrderedDict
        A nested OrderedDict of displaced Atoms objects
    calc : calculator instance
        ASE calculator
    modes : str or list
        Mode for which the PES will be calculated

    Returns
    -------
    energies : pandas.DataFrame
        DataFrame with the energies per displacement
    """

    if isinstance(modes, string_types):
        if modes.lower() in ["all", ":"]:
            modes = range(len(images))
        else:
            modes = expandrange(modes)
    elif not isinstance(modes, (list, tuple)):
        ValueError(
            "<modes> should be a str, list or tuple " "got: {}".format(type("modes"))
        )

    ecols = ["E_" + str(i) for i in range(-4, 5)]
    energies = pd.DataFrame(0.0, columns=ecols, index=pd.Index(modes, name="mode"))

    for mode in modes:
        for point in images[mode].keys():
            print("# calculating energy for mode: {} point: {}".format(mode, point))
            atoms = images[mode][point]
            atoms.set_calculator(calc)
            energy = atoms.get_potential_energy()
            print("E[{0:d}, {1:d}] : {2:25.12f}".format(mode, point, energy))
            energies.set_value(mode, "E_" + str(point), energy)

    return energies


def fit_potentials(modeinfo, energies):
    """
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
    """

    energies = energies.subtract(energies["E_0"], axis=0)
    energies = energies / units.Hartree
    E = energies.as_matrix()

    D = np.dot(
        modeinfo["displacement"].values.reshape(-1, 1), np.arange(-4, 5).reshape(1, -1)
    )
    D = D / np.sqrt(modeinfo["effective_mass"].values).reshape(-1, 1)
    D = D.astype(float)

    coeffs6o = pd.DataFrame(
        index=energies.index, columns=["c_" + str(x) for x in np.arange(6, -1, -1)]
    )
    coeffs4o = pd.DataFrame(
        index=energies.index, columns=["c_" + str(x) for x in np.arange(4, -1, -1)]
    )

    for i in energies.index:
        coeffs4o.loc[i] = np.polyfit(D[i], E[i], deg=4)
        coeffs6o.loc[i] = np.polyfit(D[i], E[i], deg=6)

    return coeffs6o, coeffs4o


def differentiate(displacements, energies, order=2):
    """
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

    """

    energies = energies.subtract(energies["E_0"], axis=0)
    energies = energies / units.Hartree
    E = energies.as_matrix()

    cols = ["p2", "p4", "p6", "p8"]
    dt = [("idx", int)] + list(zip(cols, [float] * 4))

    coeffs1d = np.zeros(9, dtype=dt)
    coeffs1d["idx"] = range(-4, 5)
    coeffs1d["p2"] = [0.0, 0.0, 0.0, -0.5, 0.0, 0.5, 0.0, 0.0, 0.0]
    coeffs1d["p4"] = [
        0.0,
        0.0,
        1.0 / 12.0,
        -2.0 / 3.0,
        0.0,
        2.0 / 3.0,
        -1.0 / 12.0,
        0.0,
        0.0,
    ]
    coeffs1d["p6"] = [
        0.0,
        -1.0 / 60.0,
        3.0 / 20.0,
        -3.0 / 4.0,
        0.0,
        3.0 / 4.0,
        -3.0 / 20.0,
        1.0 / 60.0,
        0.0,
    ]
    coeffs1d["p8"] = [
        1.0 / 280.0,
        -4.0 / 105.0,
        1.0 / 5.0,
        -4.0 / 5.0,
        0.0,
        4.0 / 5.0,
        -1.0 / 5.0,
        4.0 / 105.0,
        -1.0 / 280.0,
    ]

    coeffs2d = np.zeros(9, dtype=dt)
    coeffs2d["idx"] = range(-4, 5)
    coeffs2d["p2"] = [0.0, 0.0, 0.0, 1.0, -2.0, 1.0, 0.0, 0.0, 0.0]
    coeffs2d["p4"] = [
        0.0,
        0.0,
        -1.0 / 12.0,
        4.0 / 3.0,
        -5.0 / 2.0,
        4.0 / 3.0,
        -1.0 / 12.0,
        0.0,
        0.0,
    ]
    coeffs2d["p6"] = [
        0.0,
        1.0 / 90.0,
        -3.0 / 20.0,
        3.0 / 2.0,
        -49.0 / 18.0,
        3.0 / 2.0,
        -3.0 / 20.0,
        1.0 / 90.0,
        0.0,
    ]
    coeffs2d["p8"] = [
        -1.0 / 560.0,
        8.0 / 315.0,
        -1.0 / 5.0,
        8.0 / 5.0,
        -205.0 / 72.0,
        8.0 / 5.0,
        -1.0 / 5.0,
        8.0 / 315.0,
        -1.0 / 560.0,
    ]

    if order == 1:
        C = coeffs1d[cols].view(np.float64).reshape(coeffs1d.shape + (-1,))
        return np.dot(E, C) / displacements[:, np.newaxis]
    elif order == 2:
        C = coeffs2d[cols].view(np.float64).reshape(coeffs2d.shape + (-1,))
        return np.dot(E, C) / np.power(displacements, 2)[:, np.newaxis]
    else:
        raise NotImplementedError("{} order derivatives not available".format(order))


def harmonic_potential(x, freq, mu):
    """
    Calculate the harmonic potential

    Parameters
    ----------
    x : float of numpy.array
        Coordinate

    mu : float
        Reduced mass

    freq : float
        Frequency in cm^-1
    """

    # conversion factor from cm^-1 to hartrees
    conv = 100.0 * units.J * units._hplanck * units._c / units.Hartree

    kconst = mu * (freq * conv) ** 2
    return 0.5 * kconst * x ** 2
