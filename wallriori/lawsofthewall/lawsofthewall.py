# This file is part of wallriori
# (c) Timofey Mukha
# The code is released under the MIT Licence.
# See LICENCE.txt and the Legal section in the README for more information

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np

__all__ = ["LawOfTheWall", "Spalding"]


class LawOfTheWall:

    def __init__(self):
        pass


class Spalding(LawOfTheWall):

    def __init__(self, kappa=0.4, B=5.5):
        LawOfTheWall.__init__(self)
        self.kappa = kappa
        self.B = B

    @property
    def kappa(self):
        return self.__kappa

    @property
    def B(self):
        return self.__B

    @kappa.setter
    def kappa(self, value):
        self.__kappa = value

    @B.setter
    def B(self, value):
        self.__B = value

    def value(self, u, y, uTau, nu):
        """Return the value of the implicit function defined by the
         law"""

        kappa = self.kappa
        B = self.B

        uPlus = u/uTau
        return (uPlus + np.exp(-kappa*B)*(np.exp(kappa*uPlus) - 1 -
                kappa*uPlus - 0.5*(kappa*uPlus)**2 - 1./6*(kappa*uPlus)**3) -
                y*uTau/nu)


    def derivative(self, u, y, uTau, nu):
        """Return the value of the derivative of the implicit function
         defined by the law"""

        uPlus = u/uTau
        kappa = self.kappa
        B = self.B

        return (-y/nu - u/uTau**2 - kappa*uPlus/uTau*np.exp(-kappa*B)*
               (np.exp(kappa*uPlus) - 1 - kappa*uPlus - 0.5*(kappa*uPlus)**2))
