
'Functions for plotting the each mode and PES fits'

from __future__ import print_function, division

import argparse
import os
import re
from functools import partial
from io import StringIO

import numpy as np
import pandas as pd
from scipy.constants import value

import matplotlib.pyplot as plt
import seaborn as sns

from .inputreader import read_em_freq

def parse_pes(fname):
    '''
    Parse the file with the potential energy surface (PES) into a dict of numpy arrays with
    mode numbers as keys
    
    Args:
        fname : str
            Name of the file with PES
    '''
    
    with open(fname, 'r') as fobj:
        data = fobj.read()
        
    pat = re.compile(' Scan along mode # =\s*(\d+)')
    parsed = [x for x in pat.split(data) if x != '']
    it = iter(parsed)
    parsed = {int(mode): np.loadtxt(StringIO(pes)) for mode, pes in zip(it, it)}
    return parsed

def harmonic(x, freq, mu):
    '''
    Calculate the harmonic potential

    Args:
        x : float of numpy.array
            Coordinate
        mu : float
            Reduced mass
        freq : float
            Frequency in cm^-1
    '''
    
    kconst = 0.5*mu*(freq*100*value('inverse meter-hartree relationship'))**2
    return kconst*x**2

def plot_mode(mode, pes, coeff6, coeff4):
    'Plot a given mode'

    cp = sns.color_palette('muted')
    sns.set(font_scale=1.5, style='whitegrid')
    plt.figure(figsize=(14, 10))

    poly6 = np.poly1d(coeff6.loc[mode, 'a0' : 'a6'].values[::-1])
    poly4 = np.poly1d(coeff4.loc[mode, 'a0' : 'a4'].values[::-1])
    harm = partial(harmonic, freq=coeff6.loc[mode, 'freq'], mu=coeff6.loc[mode, 'mass'])

    lw = 1.2  # line width
    ms = 13   # markersize
    mew = 2.0 # markeredgewidth
        
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

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('mode', type=int, help='number of the mode to be printed')
    parser.add_argument('-s', '--sixth', default='em_freq', help='file with sixth order polynomial fit')
    parser.add_argument('-f', '--fourth', default='em_freq_4th', help='file with fourth order polynomial fit')
    parser.add_argument('-p', '--pes', default='test_anharm', help='file with the potential energy surface (PES)')
    args = parser.parse_args()

    if os.path.exists(args.sixth):
        coeff6 = read_em_freq(args.sixth)
    else:
        raise OSError('File {} does not exist'.format(args.sixth))
    if os.path.exists(args.fourth):
        coeff4 = read_em_freq(args.fourth)
    else:
        raise OSError('File {} does not exist'.format(args.fourth))
    if os.path.exists(args.sixth):
        pes = parse_pes(args.pes)
    else:
        raise OSError('File {} does not exist'.format(args.pes))

    if args.mode > max(pes.keys()):
        raise ValueError('Mode number {} unavailable, max mode number is: {}'.format(args.mode, max(pes.keys())))

    plot_mode(args.mode, pes, coeff6, coeff4)
