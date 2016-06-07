
'Functions for plotting the each mode and PES fits'

from __future__ import print_function, division

from functools import partial

import numpy as np
from scipy.constants import value

import matplotlib.pyplot as plt
import seaborn as sns

from .pes import harmonic_potential


def plot_mode_legacy(mode, pes, coeff6, coeff4):
    'Plot a given mode'

    cp = sns.color_palette('muted')
    sns.set(font_scale=1.5, style='whitegrid')
    plt.figure(figsize=(14, 10))

    poly6 = np.poly1d(coeff6.loc[mode, 'a0': 'a6'].values[::-1])
    poly4 = np.poly1d(coeff4.loc[mode, 'a0': 'a4'].values[::-1])
    harm = partial(harmonic_potential, freq=coeff6.loc[mode, 'freq'],
                   mu=coeff6.loc[mode, 'mass'])

    lw = 1.2   # line width
    ms = 13    # markersize
    mew = 2.0  # markeredgewidth

    plt.plot(pes[mode][:, 0], pes[mode][:, 1], marker='x', color='k', linewidth=lw,
             markersize=ms, markerfacecolor='none', markeredgecolor='k', markeredgewidth=mew, label='PES')
    plt.plot(pes[mode][:, 0], poly6(pes[mode][:, 0]), marker='s', color=cp[0], linewidth=lw,
             markersize=ms, markerfacecolor='none', markeredgecolor=cp[0], markeredgewidth=mew, label='6th order')
    plt.plot(pes[mode][:, 0], poly4(pes[mode][:, 0]), marker='D', color=cp[1], linewidth=lw,
             markersize=ms, markerfacecolor='none', markeredgecolor=cp[1], markeredgewidth=mew, label='4th order')
    plt.plot(pes[mode][:, 0], harm(pes[mode][:, 0]), marker='o', color=cp[2], linewidth=lw,
             markersize=ms, markerfacecolor='none', markeredgecolor=cp[2], markeredgewidth=mew, label='harmonic')

    plt.title(r'Mode # {0:d}, $\nu$ = {1:6.2f} [cm$^{{-1}}$]'.format(mode, coeff6.loc[mode, 'freq']))
    plt.xlabel('$\Delta x$')
    plt.ylabel('$\Delta E$')
    plt.legend(loc='best', frameon=False)
    plt.show()


def plotmode(mode, energies, mi, c6o, c4o, output=None):
    '''
    Plot a given mode

    Parameters
    ----------
    mode : int
        Mode number (indexed from 0)
    energies : pandas.DataFrame
    mi : pandas.DataFrame
        Modeinfo
    c6o : pandas.DataFrame
    c4o : pandas.DataFrame
    '''

    cp = sns.color_palette('muted')
    sns.set(font_scale=1.5, style='white')
    plt.figure(figsize=(14, 10))

    poly6 = np.poly1d(c6o.loc[mode].values)
    poly4 = np.poly1d(c4o.loc[mode].values)
    harm = partial(harmonic_potential, freq=mi.loc[mode, 'frequency'],
                   mu=mi.loc[mode, 'effective_mass'])

    dx = np.dot(mi.loc[mode, 'displacement'], np.arange(-4, 5))
    dx = dx / np.sqrt(mi.loc[mode, 'effective_mass'])
    dx = dx.astype(float)

    x = np.linspace(dx.min(), dx.max(), 100)

    de = energies.loc[mode] - energies.loc[mode, 'E_0']
    de = de.values * value('electron volt-hartree relationship')

    lw = 3.0   # line width

    plt.plot(dx, de, marker='o', color='k', linewidth=lw,
             markersize=13, markerfacecolor='none', markeredgecolor='k',
             markeredgewidth=2.0, label='PES')
    plt.plot(x, harm(x), color=cp[2], linewidth=lw, label='harmonic')
    plt.plot(x, poly6(x), '--', color=cp[0], linewidth=lw, label='6th order')
    plt.plot(x, poly4(x), '-.', color=cp[1], linewidth=lw, label='4th order')

    plt.title(r'Mode # {0:d}, $\nu$ = {1:6.2f} [cm$^{{-1}}$]'.format(mode,
        mi.loc[mode, 'frequency']))
    plt.xlabel('$\Delta x$')
    plt.ylabel('$\Delta E$')
    plt.legend(loc='best', frameon=False)

    if output is not None:
        plt.savefig(output)
    else:
        plt.show()
