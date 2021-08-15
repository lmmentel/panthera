from __future__ import print_function
from builtins import super

from datetime import datetime

import numpy as np

from ase.optimize.optimize import Optimizer

from .displacements import get_nvibdof
from .vibrations import harmonic_vibrational_analysis


class NormalModeBFGS(Optimizer, object):
    """
    Normal mode optimizer with approximate hessian update

    Parameters
    ----------
    atoms : ase.Atoms
        Atoms object with the structure to optimize

    phase : str
        Phase, should be either `gas` or `solid`

    hessian : array_like (N, N)
        Initial hessian matrix in eV/Angstrom^2

    hessian_update : str
        Name of the approximate formula to udpate hessian, one of: `BFGS`,
        `SR1`, `DFP`

    proj_translations : bool
        If ``True`` translational degrees of freedom will be projected from
        the hessian

    proj_rotations : bool
        If ``True`` rotational degrees of freedom will be projected from
        the hessian

    logfile : str
        Name log the log file

    trajectory : str
        Name of the trajectory file
    """

    def __init__(
        self,
        atoms,
        phase,
        hessian=None,
        hessian_update="BFGS",
        logfile="-",
        trajectory=None,
        restart=None,
        proj_translations=True,
        proj_rotations=True,
        master=None,
        verbose=False,
    ):

        super().__init__(atoms, restart, logfile, trajectory, master)

        self.trajectory = trajectory
        self.hessian = hessian
        self.phase = phase
        self.hessian_update = hessian_update
        self.verbose = verbose
        self.restart = restart
        self.nsteps = 0

        self.proj_translations = proj_translations
        self.proj_rotations = proj_rotations

        # initialize other necessary variables

        self.coords_0 = None
        self.grad_0 = None

        self.natoms = atoms.get_number_of_atoms()
        self.ndof = 3 * self.natoms
        self.nvibdof = get_nvibdof(atoms, proj_rotations, proj_translations, self.phase)

        # matrix with inverse square roots of masses on diagonal
        self.M_invsqrt = np.zeros((self.ndof, self.ndof), dtype=float)
        np.fill_diagonal(
            self.M_invsqrt, np.repeat(1.0 / np.sqrt(atoms.get_masses()), 3)
        )

        # write the header to the logfile
        self.log_header()

        @property
        def hessian(self):
            return self._hessian

        @hessian.setter
        def hessian(self, value):
            "Initialize the hessian matrix"

            if hessian is None:
                self._hessian = np.eye(3 * len(self.atoms)) * 70.0

    def log_header(self):
        "Header for the log with convergence information"

        print(
            "{0:<6s} {1:^15s} {2:^15s} {3:^15s} {4:^15s} {5:^15s} {6:^20s}".format(
                "iter",
                "G(C) max",
                "G(NM) max",
                "G(NM) norm",
                "NM step norm",
                "energy [eV]",
                "time",
            ),
            file=self.logfile,
        )
        print("=" * 107, file=self.logfile)
        self.logfile.flush()

    def log(self, grad, grad_nm, step_nm):
        "Print a line with convergence information"

        gmax = np.sqrt((grad.reshape(self.natoms, 3)) ** 2).sum(axis=1).max()
        gnnmorm = np.sqrt(np.dot(grad_nm, grad_nm))

        print(
            "{0:>6d} {1:>15.8f} {2:>15.8f} {3:>15.8f} {4:>15.8f} {5:>15.8f} {6:>20s}".format(
                self.nsteps,
                gmax,
                np.max(np.abs(grad_nm)),
                gnnmorm,
                np.sqrt(np.dot(step_nm, step_nm)),
                self.atoms.get_potential_energy(),
                datetime.now().strftime("%H:%M:%S %d-%m-%Y"),
            ),
            file=self.logfile,
        )
        self.logfile.flush()

    def update_hessian(self, coords, grad):
        """
        Perform hessian update

        Parameters
        ----------
        coords : array_like (N,)
            Current coordinates as vector

        grad : array_like (N,)
            Current gradient
        """

        # on first step there's no previous grad and coords so not updated is
        # performed
        if self.grad_0 is None:
            return

        macheps = np.finfo(np.float64).eps

        dx = coords - self.coords_0
        dg = grad - self.grad_0

        # same configuration again
        if np.abs(dx).max() < 1.0e-7:
            return

        if self.hessian_update == "BFGS":
            dxdg = np.dot(dx, dg)
            hdx = np.dot(self.hessian, dx)
            b = np.dot(dx, hdx)

            if np.abs(dxdg) > macheps and np.abs(b) > macheps:
                self.hessian += np.outer(dg, dg) / dxdg - np.outer(hdx, hdx) / b

        elif self.hessian_update == "DFP":
            dgdxT = np.outer(dg, dx)
            dgTdx = np.dot(dg, dx)
            uleft = np.eye(self.hessian.shape[0]) - dgdxT / dgTdx
            uright = np.eye(self.hessian.shape[0]) - dgdxT.T / dgTdx
            bk = np.dot(uleft, np.dot(self.hessian, uright))

            self.hessian = bk + np.outer(dg, dg) / dgTdx

        elif self.hessian_update.upper() == "SR1":
            hdx = np.dot(self.hessian, dx)
            dghdx = dg - hdx
            self.hessian += np.outer(dghdx, dghdx) / np.dot(dghdx, dx)

        else:
            raise NotImplementedError(
                "update <{}> not available".format(self.hessian_update)
            )

    def step(self, grad):
        """
        Calculate the step in cartesian coordinates based on the step in
        normal modes in the rational function approximation (RFO)

        Args:
            grad : array_like (N,)
                Current gradient
        """

        nv = self.nvibdof
        coords = self.atoms.get_positions().ravel()
        grad = -1.0 * self.atoms.get_forces().ravel()

        self.update_hessian(coords, grad)

        # calculate hessian eigenvalues and eigenvectors
        evals, evecs = harmonic_vibrational_analysis(
            self.hessian,
            self.atoms,
            proj_translations=self.proj_translations,
            proj_rotations=self.proj_rotations,
            ascomplex=False,
            massau=False,
        )

        evals = np.power(evals, 2)
        mwevecs = np.dot(self.M_invsqrt, evecs)

        grad_nm = np.dot(mwevecs.T, grad)
        step_nm = np.zeros_like(grad_nm)
        step_nm[:nv] = (
            -2.0
            * grad_nm[:nv]
            / (evals[:nv] + np.sqrt(evals[:nv] ** 2 + 4.0 * grad_nm[:nv] ** 2))
        )

        step_cart = np.dot(mwevecs, step_nm)
        new_coords = coords + step_cart
        self.atoms.set_positions(new_coords.reshape(self.natoms, 3))

        self.coords_0 = new_coords.copy()
        self.grad_0 = grad.copy()
        self.dump((self.hessian, self.coords_0, self.grad_0))

        self.log(grad, grad_nm, step_nm)

    def run(self, fmax=0.05, steps=100000000):
        """Run structure optimization algorithm.

        This method will return when the forces on all individual
        atoms are less than *fmax* or when the number of steps exceeds
        *steps*."""

        self.fmax = fmax
        for _ in range(steps):
            f = self.atoms.get_forces()
            self.call_observers()
            if self.converged(f):
                return
            self.step(f)
            self.nsteps += 1

    def read(self):
        self.hessian, self.coords_0, self.grad_0 = self.load()


def nmoptimize(
    atoms,
    hessian,
    calc,
    phase,
    proj_translations=True,
    proj_rotations=True,
    gtol=1.0e-5,
    verbose=False,
    hessian_update="BFGS",
    steps=100000,
):
    """
    Relax the strcture using normal mode displacements

    Parameters
    ----------
    atoms : ase.Atoms
        Atoms object with the structure to optimize

    hessian : array_like
        Hessian matrix in eV/Angstrom^2

    calc : ase.Calculator
        ASE Calcualtor instance to be used to calculate forces

    phase : str
        Phase, 'solid' or 'gas'

    gtol : float, default=1.0e-5
        Energy gradient threshold

    hessian_update : str
        Approximate formula to update hessian, possible values are 'BFGS',
        'SR1' and 'DFP'

    steps : int
        Maximal number of iteration to be performed

    verbose : bool
        If ``True`` additional debug information will be printed

    Notes
    -----

    Internally eV and Angstroms are used.

    .. seealso::

       Bour, P., & Keiderling, T. A. (2002). Partial optimization of molecular geometry
       in normal coordinates and use as a tool for simulation of vibrational spectra.
       The Journal of Chemical Physics, 117(9), 4126.
       `doi:10.1063/1.1498468 <http://dx.doi.org/10.1063/1.1498468>`_

    """

    natoms = atoms.get_number_of_atoms()
    ndof = 3 * natoms
    masses = atoms.get_masses()
    coords = atoms.get_positions().ravel()
    nvibdof = get_nvibdof(atoms, proj_rotations, proj_translations, phase)

    # matrix with inverse square roots of masses on diagonal
    M_invsqrt = np.zeros((ndof, ndof), dtype=float)
    np.fill_diagonal(M_invsqrt, np.repeat(1.0 / np.sqrt(masses), 3))

    # calculate hessian eigenvalues and eigenvectors
    evals, evecs = harmonic_vibrational_analysis(
        hessian,
        atoms,
        proj_translations=proj_translations,
        proj_rotations=proj_rotations,
        ascomplex=False,
        massau=False,
    )

    evals = np.power(evals, 2)

    mwevecs = np.dot(M_invsqrt, evecs)

    coords_old = coords.copy()

    # run the job for the initial structure
    atoms.set_calculator(calc)

    # get forces after run
    grad = -1.0 * atoms.get_forces().ravel()

    grad_old = grad.copy()

    grad_nm = np.dot(mwevecs.T, grad)

    step_nm = np.zeros_like(grad_nm)
    step_nm[:nvibdof] = (
        -2.0
        * grad_nm[:nvibdof]
        / (
            evals[:nvibdof]
            + np.sqrt(evals[:nvibdof] ** 2 + 4.0 * grad_nm[:nvibdof] ** 2)
        )
    )

    step_cart = np.dot(mwevecs, step_nm)
    coords = coords_old + step_cart

    if verbose:
        print(" eigenvalues ".center(50, "-"))
        print(evals)

        print(" cart gradient ".center(50, "-"))
        print(grad)

        print(" nm gradient ".center(50, "-"))
        print(grad_nm)

        print(" nm step ".center(50, "-"))
        print(step_nm)

        print(" cart step ".center(50, "-"))
        print(step_cart)

        print(" new coordinates ".center(50, "-"))
        print(coords)

    iteration = 0

    # header for the convergence information
    print(
        "{0:<6s} {1:^15s} {2:^15s} {3:^15s} {4:^15s} {5:^20s}".format(
            "iter", "G(NM) max", "G(NM) norm", "NM step norm", "energy [eV]", "time"
        )
    )
    print("=" * 91)
    print(
        "{0:>6d} {1:>15.8f} {2:>15.8f} {3:>15.8f} {4:>15.8f} {5:>20s}".format(
            iteration,
            np.max(np.abs(grad_nm)),
            np.sqrt(np.dot(grad_nm, grad_nm)),
            np.sqrt(np.dot(step_nm, step_nm)),
            atoms.get_potential_energy(),
            datetime.now().strftime("%H:%M:%S %d-%m-%Y"),
        )
    )

    while iteration <= steps:
        iteration += 1

        # delta_coord = coords - coords_old

        atoms.set_positions(coords.reshape(natoms, 3))

        coords_old = coords.copy()
        grad = -1.0 * atoms.get_forces().ravel()

        # delta_grad = grad - grad_old

        hessian = update_hessian(
            grad, grad_old, step_cart, hessian, update=hessian_update
        )

        grad_old = grad.copy()

        # calculate hessian eigenvalues and eigenvectors
        evals, evecs = harmonic_vibrational_analysis(
            hessian,
            atoms,
            proj_translations=proj_translations,
            proj_rotations=proj_rotations,
            ascomplex=False,
            massau=False,
        )
        evals = np.power(evals, 2)

        mwevecs = np.dot(M_invsqrt, evecs)
        grad_nm = np.dot(mwevecs.T, grad)

        gmax = np.max(np.abs(grad_nm))

        # print the convergence info
        print(
            "{0:>6d} {1:>15.8f} {2:>15.8f} {3:>15.8f} {4:>15.8f} {5:>20s}".format(
                iteration,
                gmax,
                np.sqrt(np.dot(grad_nm, grad_nm)),
                np.sqrt(np.dot(step_nm, step_nm)),
                atoms.get_potential_energy(),
                datetime.now().strftime("%H:%M:%S %d-%m-%Y"),
            )
        )

        if gmax < gtol:
            evals, evecs = harmonic_vibrational_analysis(
                hessian,
                atoms,
                proj_translations=proj_translations,
                proj_rotations=proj_rotations,
                ascomplex=False,
                massau=False,
            )
            np.save("hessian_evalues", evals)
            np.save("hessian_evectors", evecs)
            print("# convergence achieved after {} iterations".format(iteration))
            break

        step_nm[:nvibdof] = (
            -2.0
            * grad_nm[:nvibdof]
            / (
                evals[:nvibdof]
                + np.sqrt(evals[:nvibdof] ** 2 + 4.0 * grad_nm[:nvibdof] ** 2)
            )
        )

        step_cart = np.dot(mwevecs, step_nm)
        coords = coords_old + step_cart

        if verbose:
            print(" eigenvalues ".center(50, "-"))
            print(evals)

            print(" cart gradient ".center(50, "-"))
            print(grad)

            print(" nm gradient ".center(50, "-"))
            print(grad_nm)

            print(" nm step ".center(50, "-"))
            print(step_nm)

            print(" cart step ".center(50, "-"))
            print(step_cart)

            print(" new coordinates ".center(50, "-"))
            print(coords)
    else:
        print("### convergence NOT achieved after ", iteration, " iterations")


def update_hessian(grad, grad_old, dx, hessian, update="BFGS"):
    """
    Perform hessian update

    Parameters
    ----------
    grad : array_like (N,)
        Current gradient
    grad_old : array_like (N,)
        Previous gradient
    dx : array_like (N,)
        Step vector x_n - x_(n-1)
    hessian : array_like (N, N)
        Hessian matrix
    update : str
        Name of the hessian update to perform, possible values are 'BFGS',
        'SR1' and 'DFP'

    Returns
    -------
    hessian : array_like
        Update hessian matrix
    """

    macheps = np.finfo(np.float64).eps

    dg = grad - grad_old

    if update == "BFGS":
        dxdg = np.dot(dx, dg)
        hdx = np.dot(hessian, dx)
        b = np.dot(dx, hdx)

        if np.abs(dxdg) < macheps or np.abs(b) < macheps:
            return hessian
        else:
            return hessian + np.outer(dg, dg) / dxdg - np.outer(hdx, hdx) / b

    elif update == "DFP":
        dgdxT = np.outer(dg, dx)
        dgTdx = np.dot(dg, dx)
        uleft = np.eye(hessian.shape[0]) - dgdxT / dgTdx
        uright = np.eye(hessian.shape[0]) - dgdxT.T / dgTdx
        bk = np.dot(uleft, np.dot(hessian, uright))

        return bk + np.outer(dg, dg) / dgTdx

    elif update.upper() == "SR1":
        hdx = np.dot(hessian, dx)
        dghdx = dg - hdx
        return hessian + np.outer(dghdx, dghdx) / np.dot(dghdx, dx)

    else:
        raise NotImplementedError
