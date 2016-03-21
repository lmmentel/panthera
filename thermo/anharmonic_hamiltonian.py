import os
import numpy as np
import pandas as pd

def factsqrt(m, n):
    '''
    Return a constant, factorial like constant
    '''

    return np.sqrt(np.prod([m - i for i in range(n)] ))

def get_hamiltonian(n, freq, mass, acoeffs):
    '''
    Compose the Hamiltonian matrix for the anharmonic oscillator with the potential described by
    the sixth order polynomial.

    Args:
        n : int
            Rank of the Hamiltonian matrix
        freq : float
            Fundamental frequency in hartrees
        mass : float
            Reduced mass of the mode
        acoeffs : array
            A one dimensional array with polynomial coeffients
    '''

    Hamil = np.zeros((n, n), dtype=float)

    # change this to proper value
    vk = np.sqrt(1.0/(mass*freq))
    uk = -0.5*freq

    # main diagonal i == j
    idx = np.arange(n)
    Hamil[idx, idx] = [-0.5*uk*(2*i + 1.0) + acoeff[0] + 0.5*acoeff[2]*vk**2*(2*i + 1.0) \
                    + 0.25*acoeff[4]*vk**4*(6.0*i**2 + 6.0*i + 3) \
                    + 0.125*acoeff[6]*vk**6*(20.0*i**3 + 30.0*i**2 + 40.0*i + 15.0) for i in idx]

    # diagonal wih offset 1, i == j + 1
    k = n - 1
    idx = np.arange(k)
    Hamil[idx + 1, idx] = [np.sqrt(2*i)*(0.5*acoeff[1]*vk + 0.25*acoeff[3]*vk**3*i \
                        + 0.125*acoeff[5]*vk**5*(10.0*i**2 + 5.0)) for i in idx + 1]

    # diagonal wih offset 2, i == j + 2
    k = n - 2
    if k > 0:
        idx = np.arange(k)
        Hamil[idx + 2, idx] = [factsqrt(i, 2)*(0.5*(uk + acoeff[2]*vk**2) \
                            + 0.25*acoeff[4]*(4.0*i - 2.0) \
                            + 15.0*acoeff[6]*vk**6*(i**2 - i + 1.0)/8.0) for i in idx + 2]

    # diagonal wih offset 3, i == j + 3
    k = n - 3
    if k > 0:
        idx = np.arange(k)
        Hamil[idx + 3, idx] = [factsqrt(i, 3)*(acoeff[3]*vk**3/np.sqrt(8.0) \
                            + np.sqrt(2.0)*acoeff[5]*vk**5*(5.0*i - 5)/8.0 ) for i in idx + 3]

    # diagonal wih offset 4, i == j + 4
    k = n - 4
    if k > 0:
        idx = np.arange(k)
        Hamil[idx + 4, idx] = [factsqrt(i, 4)*(0.25e0*acoeff[4]*vk**4 \
                            + 0.125e0*acoeff[6]*vk**6*(6.0*i - 9)) for i in idx + 4]

    # diagonal wih offset 5, i == j + 5
    k = n - 5
    if k > 0:
        idx = np.arange(k)
        Hamil[idx + 5, idx] = [np.sqrt(1.0/320)*factsqrt(i, 5)*acoeff[5]*vk**5 for i in idx + 5]

    # diagonal wih offset 6, i == j + 6
    k = n - 6
    if k > 0:
        idx = np.arange(k)
        Hamil[idx + 6, idx] = [0.125e0*factsqrt(i, 6)*acoeff[6]*vk**6 for i in idx + 6]

    # make the ful symmetrix matrix
    Hamil = np.maximum(Hamil, Hamil.T)

    return Hamil

def anharmonic_frequencies(fname='em_freq'):

    if not os.path.exists(fname):
        raise OSError('File "{}" does not exist'.format(fname))

    cols = ['type', 'freq', 'mass', 'a0', 'a1', 'a2', 'a3', 'a4', 'a5', 'a6']
    data = pd.read_csv(fname, sep='\s+', engine='python', names=cols)

    for i, row in data.iterrows():
        hamil = get_hamiltonian(i, row.freq, row.mass, row[cols[-7:]].values)

        w, v = numpy.linag.eig(hamil)

