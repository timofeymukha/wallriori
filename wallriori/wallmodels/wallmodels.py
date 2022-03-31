# This file is part of wallriori
# (c) Timofey Mukha
# The code is released under the MIT Licence.
# See LICENCE.txt and the Legal section in the README for more information

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
from functools import partial
from ..lawsofthewall import Spalding

__all__ = ["WallModel", "LinearWallModel","LOTWWallModel",
           "IntegratedLOTWWallModel", "LSQRWallModel"]


class WallModel:

    def __init__(self, h, nu):
        self.h = h
        self.nu = nu

    @property
    def h(self):
        return self.__h

    @property
    def nu(self):
        return self.__nu

    @h.setter
    def h(self, val):
        self.__h = val

    @nu.setter
    def nu(self, val):
        self.__nu = val


class LinearWallModel(WallModel):

    def __init__(self, h, nu):
        WallModel.__init__(self, h, nu)

    def utau(self, sampledU):
        return np.sqrt(self.nu*sampledU/self.h)


class LOTWWallModel(WallModel):

    def __init__(self, h, nu, law, rootFinder):
        WallModel.__init__(self, h, nu)

        self.law = law
        self.rootFinder = rootFinder

    @property
    def law(self):
        return self.__law

    @property
    def rootFinder(self):
        return self.__rootFinder

    @law.setter
    def law(self, value):
        self.__law = value

    @rootFinder.setter
    def rootFinder(self, value):
        self.__rootFinder = value

    def nut(self, guess, sampledU, wallGradU):
        magGradU = np.abs(wallGradU)
        uTau = self.utau(guess, sampledU)
        return np.max([0.0, uTau**2/magGradU - self.nu])

    def utau(self, guess, sampledU):
        f = partial(self.law.value,  sampledU, self.h, self.nu)
        d = partial(self.law.derivative, sampledU, self.h, self.nu)

        self.rootFinder.f = f
        self.rootFinder.d = d
        return self.rootFinder.solve(guess)


class IntegratedLOTWWallModel(WallModel):

    def __init__(self, h1, h2, nu, law, rootFinder):
        WallModel.__init__(self, (h1 + h2)/2, nu)

        self.h1 = h1
        self.h2 = h2
        self.law = law
        self.rootFinder = rootFinder

    @property
    def h1(self):
        return self.__h1

    @property
    def h2(self):
        return self.__h2

    @property
    def law(self):
        return self.__law

    @property
    def rootFinder(self):
        return self.__rootFinder

    @h1.setter
    def h1(self, value):
        self.__h1 = value

    @h2.setter
    def h2(self, value):
        self.__h2 = value

    @law.setter
    def law(self, value):
        self.__law = value

    @rootFinder.setter
    def rootFinder(self, value):
        self.__rootFinder = value

    def nut(self, guess, sampledU, wallGradU):

        magGradU = np.abs(wallGradU)
        uTau = self.utau(guess, sampledU)
        return np.max([0.0, uTau**2/magGradU - self.nu])

    def utau(self, guess, sampledU):
        f = partial(self.law.value,  sampledU, self.h1, self.h2, self.nu)
        d = partial(self.law.derivative, sampledU, self.h1, self.h2, self.nu)

        self.rootFinder.f = f
        self.rootFinder.d = d
        return np.max([0, self.rootFinder.solve(guess)])


class LSQRWallModel(WallModel):

    def __init__(self, h, nu, rootFinder):
        """
        Model with dynamic kappa and B coefficients for the log-law.

        Parameters
        ----------
        h : ndarray
            Array of wall-normal distances, where the velocity values
            are sampled.
        nu : float
            Kinematic viscosity.

        """
        WallModel.__init__(self, h, nu)

        self.rootFinder = rootFinder

    @property
    def rootFinder(self):
        return self.__rootFinder

    @rootFinder.setter
    def rootFinder(self, value):
        self.__rootFinder = value

    def nut(self, guess, sampledU, wallGradU):
        """
        Compute the nut needed to enforce the correct shear stress.
        Parameters
        ----------
        guess : float
            Initial guess for the friction velocity.
        sampledU : 1d array
            Sampled velocity values.
        wallGradU
            The wall-normal velocity gradient.

        Returns
        -------
        float
            The value of turbulent viscosity

        """
        magGradU = np.abs(wallGradU)
        uTau = self.utau(guess, sampledU)
        return np.max([0.0, uTau**2/magGradU - self.nu])

    def kappa_and_b(self, uTau, sampledU):
        """
        Compute kappa and B using the formula in the paper.

        Parameters
        ----------
        uTau : float
            A guess for the friction velocity
        sampledU : 1d array
            The sampled values of velocity

        Returns
        -------
        (scalar, scalar)
            The values of kappa and b

        """

        from numpy import sum, log

        # u* and y* in the paper
        u = sampledU/uTau
        y = self.h*uTau/self.nu

        n = self.h.size

        kappaNom = n*sum(log(y)**2) - sum(log(y))**2
        kappaDenom = n*sum(u*log(y)) - sum(u)*sum(log(y))

        kappa = kappaNom/(kappaDenom + 1e-12)
        b = 1/n*np.sum(u - 1/kappa*np.log(y))

        return kappa, b

    def kappa_and_b_builtin(self, uTau, sampledU):
        """
        Compute kappa and B using np.polyfit.

        Can be used to check the manual implementation.

        Parameters
        ----------
        uTau : float
            A guess for the friction velocity
        sampledU : 1d array
            The sampled values of velocity

        Returns
        -------
        (scalar, scalar)
            The values of kappa and b

        """
        # u* and y* in the paper
        u = sampledU/uTau
        y = self.h*uTau/self.nu

        kappaInv, b = np.polyfit(np.log(y), u, deg=1)

        return 1/kappaInv, b

    def utau_iteration(self, uTau, sampledU, index):
        """
        Perform one iteration of computing the friction velocity.

        Parameters
        ----------
        uTau : float
            A guess for the friction velocity
        sampledU : 1d array
            The sampled values of velocity
        index : int
            Index of the velocity  and y value to use in Spalding's law.

        Returns
        -------

        """
        kappa, b = self.kappa_and_b(uTau, sampledU)
        law = Spalding(kappa, b)

        f = partial(law.value,  sampledU[index], self.h[index], self.nu)
        d = partial(law.derivative, sampledU[index], self.h[index], self.nu)

        self.rootFinder.f = f
        self.rootFinder.d = d
        return np.max([0, self.rootFinder.solve(uTau)])

    def utau(self, guess, sampledU, index, nIter, eps=1, verbose=True):
        """
        Compute the friction velocity.

        Parameters
        ----------
        guess : float
            Initial guess for the friction velocity
        sampledU : 1d array
            The sampled velocity values
        index : int
            The index of the velocity and y value to use in Spalding's law.
        nIter : int
            The amount of iterations to compute the friction velocity.
        eps : float
            Under-relaxation factor, defaults 1, i.e. no under-relaxation.
        verbose : bool
            Wether to print results from each iteration, defaults to True

        Returns
        -------
        float
            The friction velocity.
        """

        uTau = guess
        for i in range(nIter):
            uTauNew = self.utau_iteration(uTau, sampledU, index)

            uTau = eps*uTauNew + (1- eps)*uTau

            if verbose:
                print("Iteration", i, "uTau", uTau)

        return np.max([0, self.rootFinder.solve(guess)])
