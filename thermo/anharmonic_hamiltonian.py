import os
import numpy as np
import pandas as pd

from scipy.constants import value, Boltzmann

def factsqrt(m, n):
    '''
    Return a constant, factorial like constant
    '''

    return np.sqrt(np.prod([m - i for i in range(n)] ))

def get_vibdof(atoms, args):
    'Calculate the number of vibrational degrees of freedom'

    # get the total number of degrees of freedom
    ndof = 3*(len(atoms) - len(atoms.constraints))

    extradof = 0
    if args.phase.lower() == 'gas':
        if args.proj_rotations & args.proj_translations:
            if ndof > 6:
                extradof = 6
            elif ndof == 6:
                extradof = 5
    elif args.phase.lower() == 'solid':
        if args.proj_rotations | args.proj_translations:
            extradof = 3
    else:
        raise ValueError('Wrong phase specification: {}, expecting one of: "gas", "solid"'.format(args.phase))

    return ndof - extradof

def get_hamiltonian(rank, freq, mass, coeffs):
    '''
    Compose the Hamiltonian matrix for the anharmonic oscillator with the potential described by
    the sixth order polynomial.

    Args:
        rank : int
            Rank of the Hamiltonian matrix
        freq : float
            Fundamental frequency in hartrees
        mass : float
            Reduced mass of the mode
        coeffss : array
            A one dimensional array with polynomial coeffients
    '''

    Hamil = np.zeros((rank, rank), dtype=float)

    # change this to proper value
    vk = np.sqrt(1.0/(mass*freq))
    uk = -0.5*freq

    print('mass: ', mass)
    print('freq: ', freq*value('hartree-inverse meter relationship')/100.0)
    print('vk: ', vk)
    print('uk: ', uk)
    print('coeffs: ', coeffs)

    # main diagonal i == j
    idx = np.arange(rank)
    Hamil[idx, idx] = [-0.5*uk*(2*i + 1.0) + coeffs[0] + 0.5*coeffs[2]*vk**2*(2*i + 1.0) \
                    + 0.25*coeffs[4]*vk**4*(6.0*i**2 + 6.0*i + 3) \
                    + 0.125*coeffs[6]*vk**6*(20.0*i**3 + 30.0*i**2 + 40.0*i + 15.0) for i in idx]

    # diagonal wih offset 1, i == j + 1
    k = rank - 1
    idx = np.arange(k)
    Hamil[idx + 1, idx] = [np.sqrt(2.0*i)*(0.5*coeffs[1]*vk + 0.75*coeffs[3]*vk**3*i \
                        + 0.125*coeffs[5]*vk**5*(10.0*i**2 + 5.0)) for i in idx + 1]

    # diagonal wih offset 2, i == j + 2
    k = rank - 2
    if k > 0:
        idx = np.arange(k)
        Hamil[idx + 2, idx] = [factsqrt(i, 2)*(0.5*(uk + coeffs[2]*vk**2) \
                            + 0.25*coeffs[4]*vk**4*(4.0*i - 2.0) \
                            + 15.0*coeffs[6]*vk**6*(i**2 - i + 1.0)/8.0) for i in idx + 2]

    # diagonal wih offset 3, i == j + 3
    k = rank - 3
    if k > 0:
        idx = np.arange(k)
        Hamil[idx + 3, idx] = [factsqrt(i, 3)*(coeffs[3]*vk**3/np.sqrt(8.0) \
                            + np.sqrt(2.0)*coeffs[5]*vk**5*(5.0*i - 5)/8.0) for i in idx + 3]

    # diagonal wih offset 4, i == j + 4
    k = rank - 4
    if k > 0:
        idx = np.arange(k)
        Hamil[idx + 4, idx] = [factsqrt(i, 4)*(0.25e0*coeffs[4]*vk**4 \
                            + 0.125e0*coeffs[6]*vk**6*(6.0*i - 9)) for i in idx + 4]

    # diagonal wih offset 5, i == j + 5
    k = rank - 5
    if k > 0:
        idx = np.arange(k)
        Hamil[idx + 5, idx] = [np.sqrt(1.0/320)*factsqrt(i, 5)*coeffs[5]*vk**5 for i in idx + 5]

    # diagonal wih offset 6, i == j + 6
    k = rank - 6
    if k > 0:
        idx = np.arange(k)
        Hamil[idx + 6, idx] = [0.125e0*factsqrt(i, 6)*coeffs[6]*vk**6 for i in idx + 6]

    # make the ful symmetrix matrix
    Hamil = Hamil + Hamil.T - np.diag(Hamil.diagonal())

    return Hamil

def anharmonic_frequencies(atoms, args, fname='em_freq'):
    'Calculate the anharmonic frequencies'

    if not os.path.exists(fname):
        raise OSError('File "{}" does not exist'.format(fname))

    MAXITER = 50
    QVIB_THRESH = 1.0e-8
    FREQ_THRESH = 1.0e-6

    cols = ['type', 'freq', 'mass', 'a0', 'a1', 'a2', 'a3', 'a4', 'a5', 'a6']
    data = pd.read_csv(fname, sep='\s+', engine='python', names=cols)

    nvibdof = get_vibdof(atoms, args)

    print('Number of vibrational DOF : {0:5d}'.format(nvibdof))
    print('Number of read frequencies: {0:5d}'.format(data.shape[0]))

    au2joule = value('hartree-joule relationship')
    invcm2au = 100*value('inverse meter-hartree relationship')

    for i, row in data.iterrows():

        if row.type.strip() == 'A':

            converged = False
            niters = 0
            rank = 4
            niter = 0
            qvib_last = 1.0
            freq_last = 0.0
            print('mode: {0:d}'.format(i).center(80, '='))
            while not converged:

                print('iteration: {0:d}'.format(niter).center(40, '-'))
                hamil = get_hamiltonian(rank, row.freq*invcm2au, row.mass, row[cols[-7:]].values)
                print('hamiltonian'.center(60, '*'))
                print(hamil)
                print('hamiltonian'.center(60, '*'))
                w, v = np.linalg.eig(hamil)
                w = np.sort(w)
                qvib = np.sum(np.exp(-w*au2joule/(Boltzmann*args.Tfinal)))
                print(args.Tfinal)
                print('mode: ',i , ' qvib: {0:15.8e}'.format(qvib),  ' w: ', w)

                converged = (np.abs(qvib - qvib_last) < QVIB_THRESH) | (np.abs(w[0] - freq_last) < FREQ_THRESH)

                if not converged:
                    rank += 1
                    qvib_last = qvib
                    freq_last = w[0]
                    if niter >= MAXITER:
                        converged = True
                niter += 1

        else:
            print(row.type, row.freq)
    return
