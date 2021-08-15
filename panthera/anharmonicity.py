"Methods for solving the one dimentional vibrational eigenproblem"

from __future__ import print_function, division, absolute_import

import numpy as np
import pandas as pd

from scipy.constants import value, Boltzmann, Avogadro, Planck, gas_constant

from .io import print_mode_thermo


def factsqrt(m, n):
    """
    Return a factorial like constant

    Parameters
    ----------
    m : int
        Argument of the series
    n : int
        Length of the series

    Notes
    -----

    .. math::

        f(m, n) = \prod^{n - 1}_{i = 0} \sqrt{m - i}

    """

    return np.sqrt(np.prod([m - i for i in range(n)]))


def get_hamiltonian(rank, freq, mass, coeffs):
    """
    Compose the Hamiltonian matrix for the anharmonic oscillator with the
    potential described by the sixth order polynomial.

    Parameters
    ----------
    rank : int
        Rank of the Hamiltonian matrix
    freq : float
        Fundamental frequency in hartrees
    mass : float
        Reduced mass of the mode
    coeffs : array
        A one dimensional array with polynomial coeffients

    Notes
    -----

    .. math::

       H_{ij} = \left\langle \Psi_{i} \left| \hat{H} \\right| \Psi_{j} \\right\\rangle

    where

    .. math::

       \hat{H} = -\\frac{\hbar^2}{2}\\frac{\partial^2}{\partial \\boldsymbol{Q}^2}
               + \sum_{\mu=0}^{6}c_{\mu}\\boldsymbol{Q}^{\mu}

    and :math:`\Psi_{i}` are the standard harmonic oscillator functions.

    """

    Hamil = np.zeros((rank, rank), dtype=float)

    # change this to proper value
    vk = np.sqrt(1.0 / (mass * freq))
    uk = -0.5 * freq

    # main diagonal i == j
    idx = np.arange(rank)
    Hamil[idx, idx] = [
        -0.5 * uk * (2 * i + 1.0)
        + coeffs[0]
        + 0.5 * coeffs[2] * vk ** 2 * (2 * i + 1.0)
        + 0.25 * coeffs[4] * vk ** 4 * (6.0 * i ** 2 + 6.0 * i + 3)
        + 0.125
        * coeffs[6]
        * vk ** 6
        * (20.0 * i ** 3 + 30.0 * i ** 2 + 40.0 * i + 15.0)
        for i in idx
    ]

    # diagonal wih offset 1, i == j + 1
    k = rank - 1
    idx = np.arange(k)
    Hamil[idx + 1, idx] = [
        np.sqrt(2.0 * i)
        * (
            0.5 * coeffs[1] * vk
            + 0.75 * coeffs[3] * vk ** 3 * i
            + 0.125 * coeffs[5] * vk ** 5 * (10.0 * i ** 2 + 5.0)
        )
        for i in idx + 1
    ]

    # diagonal wih offset 2, i == j + 2
    k = rank - 2
    if k > 0:
        idx = np.arange(k)
        Hamil[idx + 2, idx] = [
            factsqrt(i, 2)
            * (
                0.5 * (uk + coeffs[2] * vk ** 2)
                + 0.25 * coeffs[4] * vk ** 4 * (4.0 * i - 2.0)
                + 15.0 * coeffs[6] * vk ** 6 * (i ** 2 - i + 1.0) / 8.0
            )
            for i in idx + 2
        ]

    # diagonal wih offset 3, i == j + 3
    k = rank - 3
    if k > 0:
        idx = np.arange(k)
        Hamil[idx + 3, idx] = [
            factsqrt(i, 3)
            * (
                coeffs[3] * vk ** 3 / np.sqrt(8.0)
                + np.sqrt(2.0) * coeffs[5] * vk ** 5 * (5.0 * i - 5) / 8.0
            )
            for i in idx + 3
        ]

    # diagonal wih offset 4, i == j + 4
    k = rank - 4
    if k > 0:
        idx = np.arange(k)
        Hamil[idx + 4, idx] = [
            factsqrt(i, 4)
            * (
                0.25e0 * coeffs[4] * vk ** 4
                + 0.125e0 * coeffs[6] * vk ** 6 * (6.0 * i - 9)
            )
            for i in idx + 4
        ]

    # diagonal wih offset 5, i == j + 5
    k = rank - 5
    if k > 0:
        idx = np.arange(k)
        Hamil[idx + 5, idx] = [
            np.sqrt(1.0 / 320) * factsqrt(i, 5) * coeffs[5] * vk ** 5 for i in idx + 5
        ]

    # diagonal wih offset 6, i == j + 6
    k = rank - 6
    if k > 0:
        idx = np.arange(k)
        Hamil[idx + 6, idx] = [
            0.125e0 * factsqrt(i, 6) * coeffs[6] * vk ** 6 for i in idx + 6
        ]

    # make the ful symmetrix matrix
    Hamil = Hamil + Hamil.T - np.diag(Hamil.diagonal())

    return Hamil


def anharmonic_frequencies(atoms, T, coeffs, modeinfo):
    """
    Calculate the anharmonic frequencies

    Parameters
    ----------
    atoms : ase.Atoms
        Atoms object
    T : float
        Temperature in `K`
    coeffs : pandas.DataFrame
    modeinfo : pandas.DataFrame
    """

    MAXITER = 100
    QVIB_THRESH = 1.0e-8
    FREQ_THRESH = 1.0e-7

    au2joule = value("hartree-joule relationship")
    invcm2au = 100 * value("inverse meter-hartree relationship")
    kT = Boltzmann * T

    df = pd.DataFrame(
        columns=[
            "freq",
            "zpve",
            "qvib",
            "U",
            "S",
            "converged",
            "info",
            "rank",
            "type",
            "d_qvib",
            "d_nu",
        ],
        index=modeinfo[modeinfo["vibration"]].index,
        dtype=float,
    )

    for mode, row in coeffs.iterrows():

        terminate = False
        rank = 4
        niter = 0
        qvib_last = 0.0
        freq_last = 0.0

        while not terminate:

            # get polynomial coefficients
            if row.shape[0] == 7:
                pc = row.values[::-1].astype(float)
            elif row.shape[0] == 5:
                pc = np.append(row.values[::-1], np.zeros(2)).astype(float)

            hamil = get_hamiltonian(
                rank,
                modeinfo.loc[mode, "frequency"] * invcm2au,
                modeinfo.loc[mode, "effective_mass"],
                pc,
            )

            w, v = np.linalg.eig(hamil)
            w = np.sort(w)
            qvib = np.sum(np.exp(-w * au2joule / kT))

            anhfreq = (w[1] - w[0]) / invcm2au
            zpve = w[0] * au2joule * 1.0e-3 * Avogadro
            U, S = get_anh_state_functions(w * au2joule, T)

            d_qvib = np.abs(qvib - qvib_last)
            d_nu = np.abs(w[0] - freq_last)

            terminate = (d_qvib < QVIB_THRESH) & (d_nu < FREQ_THRESH)

            if terminate:
                if anhfreq < modeinfo.loc[mode, "frequency"]:
                    anh = (
                        anhfreq,
                        zpve,
                        qvib,
                        U,
                        S,
                        True,
                        "OK",
                        rank,
                        "A",
                        d_qvib,
                        d_nu,
                    )
                else:
                    anh = (
                        anhfreq,
                        zpve,
                        qvib,
                        U,
                        S,
                        True,
                        "AGTH",
                        rank,
                        "A",
                        d_qvib,
                        d_nu,
                    )
            else:
                rank += 1
                qvib_last = qvib
                freq_last = w[0]

                if niter >= MAXITER:
                    anh = (
                        anhfreq,
                        zpve,
                        qvib,
                        U,
                        S,
                        False,
                        "MAXITER",
                        rank,
                        "A",
                        d_qvib,
                        d_nu,
                    )
                    break

            niter += 1

        df.loc[mode] = anh

    df["rank"] = df["rank"].fillna(0).astype(int)
    return df


def merge_vibs(anh6, anh4, harmonic, verbose=False):
    """
    Form a DataFrame with the per mode thermochemical
    contributions from three separate dataframes with sixth order
    polynomial fitted potentia, fourth order fitted potential and
    harmonic frequencies.

    Parameters
    ----------
    anh6 : pandas.DataFrame
    anh4 : pandas.DataFrame
    harmonic : pandas.DataFrame

    Returns
    -------
    df : pandas.DataFrame
    """
    anh6["order"] = 6
    anh4["order"] = 4

    if verbose:
        print("\n" + " Thermochemistry per mode harmonic ".center(80, "="), end="\n\n")
        print_mode_thermo(harmonic)
        print("\n" + " Thermochemistry per mode 6th order ".center(80, "="), end="\n\n")
        print_mode_thermo(anh6)
        print("\n" + " Thermochemistry per mode 4th order ".center(80, "="), end="\n\n")
        print_mode_thermo(anh4)

    df = pd.DataFrame(columns=anh6.columns, index=anh6.index)

    df.update(anh4[anh4["converged"]])
    df.update(anh6[anh6["info"] == "OK"])

    for col in ["freq", "zpve", "qvib", "U", "S"]:
        df[col] = df[col].astype(float)

    if df.isnull().any(axis=1).any():
        df.fillna(harmonic, inplace=True)

    if verbose:
        print(
            "\n" + " Final data to be used for thermochemistry ".center(80, "="),
            end="\n\n",
        )
        print_mode_thermo(df)

    return df


def harmonic_df(modeinfo, T):
    """
    Calculate per mode contributions to the thermodynamic functions in the
    harmonic approximation

    Parameters
    ----------
    modeinfo : pandas.DataFrame
    T : float
        Temperature in `K`

    Returns
    -------
    df : pandas.DataFrame
    """

    df = pd.DataFrame(
        columns=["freq", "zpve", "qvib", "U", "S", "energy", "type"],
        index=modeinfo.index,
        dtype=float,
    )

    kT = Boltzmann * T
    df["type"] = "H"
    df["freq"] = modeinfo["frequency"]
    df["energy"] = (
        Planck * df["freq"] * 100.0 * value("inverse meter-hertz relationship")
    )
    df = df[df["freq"] > 0.0]
    df["zpve"] = 0.5 * df["energy"] * 1.0e-3 * Avogadro
    df["qvib"] = 1.0 / (1.0 - np.exp(-df["energy"] / kT))
    df["U"] = (
        df["zpve"]
        + 1.0e-3
        * gas_constant
        * df["energy"]
        / (np.exp(df["energy"] / kT) - 1.0)
        / Boltzmann
    )
    df["S"] = (
        1.0e-3
        * gas_constant
        * (
            df["energy"] / (np.exp(df["energy"] / kT) - 1.0) / kT
            - np.log(1.0 - np.exp(-df["energy"] / kT))
        )
    )

    return df


def get_anh_state_functions(eigenvals, T):
    """
    Calculate the internal energy ``U`` and entropy ``S`` for an anharmonic
    vibrational mode with eigenvalues ``eigvals`` at temperature ``T`` in
    kJ/mol

    .. math::

       U &= N_{A}\\frac{\sum^{n}_{i=1} \epsilon_{i}\exp(\epsilon_{i}/k_{B}T) }{\sum^{n}_{i=1} \exp(\epsilon_{i}/k_{B}T)}

       S &= N_{A}k_{B}\log(\sum^{n}_{i=1} \exp(\epsilon_{i}/k_{B}T))
       + \\frac{N_{A}}{T}\\frac{\sum^{n}_{i=1} \epsilon_{i}\exp(\epsilon_{i}/k_{B}T) }{\sum^{n}_{i=1} \exp(\epsilon_{i}/k_{B}T)}

    Parameters
    ----------
    eigenvals : numpy.array
        Eigenvalues of the anharmonic 1D Hamiltonian in Joules
    T : float
        Temperature in `K`

    Returns
    -------
    (U, S) : tuple of floats
        Tuple with the internal energy and entropy in kJ/mol
    """

    kT = Boltzmann * T
    sum1 = np.sum(eigenvals * np.exp(-eigenvals / kT))
    sum2 = np.sum(np.exp(-eigenvals / kT))

    U = Avogadro * sum1 / sum2
    S = Boltzmann * Avogadro * np.log(sum2) + Avogadro * sum1 / (sum2 * T)
    # convert J/mol to kJ/mol
    return (U * 1.0e-3, S * 1.0e-3)
