# This file is part of wallriori
# (c) 2019 Timofey Mukha
# The code is released under the MIT Licence.
# See LICENCE.txt and the Legal section in the README for more information

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np

__all__ = ["EddyViscosity", "VanDriestEddyViscosity", "DupratEddyViscosity"]


class EddyViscosity:

    def __init__(self):
        pass


class VanDriestEddyViscosity(EddyViscosity):

    def __init__(self, kappa=0.4, Aplus=18):
        EddyViscosity.__init__(self)
        self.kappa = kappa
        self.Aplus = Aplus

    @property
    def kappa(self):
        return self.__kappa

    @property
    def Aplus(self):
        return self.__Aplus

    @kappa.setter
    def kappa(self, value):
        self.__kappa = value

    @Aplus.setter
    def Aplus(self, value):
        self.__Aplus = value

    def value(self, y, nu, uTau, **kwargs):
        yPlus = y*uTau/nu
        return self.kappa*uTau*y*(1 - np.exp(-yPlus/self.Aplus))**2


class DupratEddyViscosity(EddyViscosity):

    def __init__(self, kappa=0.4, Aplus=18, beta=0.78):
        EddyViscosity.__init__(self)
        EddyViscosity.__init__(self)
        self.kappa = kappa
        self.Aplus = Aplus
        self.beta = beta

    @property
    def kappa(self):
        return self.__kappa

    @property
    def Aplus(self):
        return self.__Aplus

    @property
    def beta(self):
        return self.__beta

    @kappa.setter
    def kappa(self, value):
        self.__kappa = value

    @Aplus.setter
    def Aplus(self, value):
        self.__Aplus = value

    @beta.setter
    def beta(self, value):
        self.__beta = value

    def value(self, y, nu, uTau, magPGrad):
        uP = (nu*magPGrad)**(1./3)
        uTauP = np.sqrt(uTau**2 + uP**2)
        alpha = uTau**2/uTauP**2

        yStar = y*uTauP/nu
        return nu*self.kappa*yStar*(alpha + yStar*(1 - alpha)**1.5)**self.beta * \
               (1 - np.exp(-yStar/(1 + self.Aplus*alpha**3)))**2

