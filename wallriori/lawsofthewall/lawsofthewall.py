# This file is part of wallriori
# (c) Timofey Mukha
# The code is released under the MIT Licence.
# See LICENCE.txt and the Legal section in the README for more information

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
from functools import partial
from ..rootfinders.rootfinders import Newton

__all__ = ["LawOfTheWall", "Spalding", "WernerWengle", "Reichardt",
           "IntegratedWernerWengle", "IntegratedReichardt", "CaiSagaut"]


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

    def explicit_value(self, y, nu, uTau):
        """Return the value velocity."""

        newton = Newton(maxIter=500, eps=1e-3)

        def val(y, nu, uTau, u):
            return self.value(u, y, nu, uTau)

        def deriv(y, nu, uTau, u):
            B = 5.5
            kappa = 0.4

            uPlus = u/uTau
            x = kappa*uPlus
            y = kappa/uTau
            return (1/uTau +
                   np.exp(-kappa*B)*(y*np.exp(x) - y - y*x -0.5*y*x**2))

        f = partial(val, y, nu, uTau)
        d = partial(deriv, y, nu, uTau)

        newton.f = f
        newton.d = d

        return newton.solve(guess=1)

    def value(self, u, y, nu, uTau):
        """Return the value of the implicit function defined by the
         law."""

        kappa = self.kappa
        B = self.B

        uPlus = u/uTau
        yPlus = y*uTau/nu
        return (uPlus + np.exp(-kappa*B)*(np.exp(kappa*uPlus) - 1 -
                kappa*uPlus - 0.5*(kappa*uPlus)**2 - 1./6*(kappa*uPlus)**3) -
                yPlus)

    def derivative(self, u, y, nu, uTau):
        """Return the value of the derivative of the implicit function
         defined by the law."""

        uPlus = u/uTau
        kappa = self.kappa
        B = self.B

        return (-y/nu - u/uTau**2 - kappa*uPlus/uTau*np.exp(-kappa*B) *
                (np.exp(kappa*uPlus) - 1 - kappa*uPlus - 0.5*(kappa*uPlus)**2))


class WernerWengle(LawOfTheWall):

    def __init__(self, A=8.3, B=1/7):
        LawOfTheWall.__init__(self)
        self.A = A
        self.B = B

    @property
    def A(self):
        return self.__A

    @property
    def B(self):
        return self.__B

    @A.setter
    def A(self, value):
        self.__A = value

    @B.setter
    def B(self, value):
        self.__B = value

    def explicit_value(self, y, nu, uTau):
        """Return the value of velocity."""

        A = self.A
        B = self.B

        yPlus = y*uTau/nu
        if yPlus <= A**(1/(1 - B)):
            return uTau*yPlus
        else:
            return uTau*A*yPlus**B

    def value(self, u, y, nu, uTau):
        """Return the value of the implicit function defined by the
         law"""

        A = self.A
        B = self.B

        uPlus = u/uTau
        yPlus = y*uTau/nu
        if yPlus <= A**(1/(1 - B)):
            return uPlus - yPlus
        else:
            return uPlus - A*yPlus**B

    def derivative(self, u, y, nu, uTau):
        """Return the value of the derivative of the implicit function
         defined by the law"""

        A = self.A
        B = self.B

        yPlus = y*uTau/nu

        if yPlus <= A**(1/(1 - B)):
            return -u/uTau**2 - y/nu
        else:
            return -u/uTau**2 - A*B*(y/nu)**B*uTau**(B - 1)


class Reichardt(LawOfTheWall):

    def __init__(self, kappa=0.4, B1=11, B2=3, C=7.8):
        LawOfTheWall.__init__(self)
        self.kappa = kappa
        self.B1 = B1
        self.B2 = B2
        self.C = C

    @property
    def kappa(self):
        return self.__kappa

    @property
    def B1(self):
        return self.__B1

    @property
    def B2(self):
        return self.__B2

    @property
    def C(self):
        return self.__C

    @kappa.setter
    def kappa(self, value):
        self.__kappa = value

    @B1.setter
    def B1(self, value):
        self.__B1 = value

    @B2.setter
    def B2(self, value):
        self.__B2 = value

    @C.setter
    def C(self, value):
        self.__C = value

    def explicit_value(self, y, nu, uTau):
        """Return the value of velocity."""

        kappa = self.kappa
        B1 = self.B1
        B2 = self.B2
        C = self.C

        yPlus = y*uTau/nu

        return uTau*(1/kappa*np.log(1 + kappa*yPlus) +
                     C*(1 - np.exp(-yPlus/B1) - yPlus/B1*np.exp(-yPlus/B2)))

    def value(self, u, y, nu, uTau):
        """Return the value of the implicit function defined by the
         law"""

        kappa = self.kappa
        B1 = self.B1
        B2 = self.B2
        C = self.C

        uPlus = u/uTau
        yPlus = y*uTau/nu

        return (uPlus - 1/kappa*np.log(1 + kappa*yPlus) -
                C*(1 - np.exp(-yPlus/B1) - yPlus/B1*np.exp(-yPlus/B2)))

    def derivative(self, u, y, nu, uTau):
        """Return the value of the derivative of the implicit function
         defined by the law"""

        kappa = self.kappa
        B1 = self.B1
        B2 = self.B2
        C = self.C

        uPlus = u/uTau
        yPlus = y*uTau/nu

        return (-uPlus/uTau - y/nu/(1 + kappa*yPlus) -
                C*y/(nu*B1)*(np.exp(-yPlus/B1) - np.exp(-yPlus/B2) +
                             yPlus/B2*np.exp(-yPlus/B2)))


class IntegratedWernerWengle(LawOfTheWall):

    def __init__(self, A=8.3, B=1/7):
        LawOfTheWall.__init__(self)
        self.A = A
        self.B = B

    @property
    def A(self):
        return self.__A

    @property
    def B(self):
        return self.__B

    @A.setter
    def A(self, value):
        self.__A = value

    @B.setter
    def B(self, value):
        self.__B = value

    def value(self, u, h1, h2, nu, uTau):
        """Return the value of the implicit function defined by the
         law"""

        A = self.A
        B = self.B

        return (uTau - ((1 + B)/A*(nu/h2)**B*u +
                (1 - B)/2*A**((1 + B)/(1 - B))*(nu/h2)**(B + 1))**(1/(B + 1)))

    def derivative(self, u, h1, h2, nu, uTau):
        """Return the value of the derivative of the implicit function
         defined by the law"""

        return 1


class IntegratedReichardt(LawOfTheWall):

    def __init__(self, kappa=0.4, B1=11, B2=3, C=7.8):
        LawOfTheWall.__init__(self)
        self.kappa = kappa
        self.B1 = B1
        self.B2 = B2
        self.C = C

    @property
    def kappa(self):
        return self.__kappa

    @property
    def B1(self):
        return self.__B1

    @property
    def B2(self):
        return self.__B2

    @property
    def C(self):
        return self.__C

    @kappa.setter
    def kappa(self, value):
        self.__kappa = value

    @B1.setter
    def B1(self, value):
        self.__B1 = value

    @B2.setter
    def B2(self, value):
        self.__B2 = value

    @C.setter
    def C(self, value):
        self.__C = value

    def explicit_value(self, y, nu, uTau):
        """Return the value of velocity."""

        kappa = self.kappa
        B1 = self.B1
        B2 = self.B2
        C = self.C

        yPlus = y*uTau/nu

        return uTau*(1/kappa*np.log(1 + kappa*yPlus) +
                     C*(1 - np.exp(-yPlus/B1) - yPlus/B1*np.exp(-yPlus/B2)))

    def value(self, u, h1, h2, nu, uTau):
        """Return the value of the implicit function defined by the
         law

         Parameters
         ----------

         u : float
            The value of the velocity integrated across the h1-h2 interval

         """

        return u - (self.logterm(h2, uTau, nu) -
                              self.logterm(h1, uTau, nu) +
                              self.expterm(h2, uTau, nu) -
                              self.expterm(h1, uTau, nu))

    def derivative(self, u, h1, h2, nu, uTau):
        """Return the value of the derivative of the implicit function
         defined by the law"""

        return -(self.logterm_derivative(h2, uTau, nu) -
                 self.logterm_derivative(h1, uTau, nu) +
                 self.expterm_derivative(h2, uTau, nu) -
                 self.expterm_derivative(h1, uTau, nu))

    def logterm(self, y, uTau, nu):
        kappa = self.kappa
        yPlus = y*uTau/nu
        return nu/kappa*(-yPlus + np.log(1 + kappa*yPlus)*(yPlus + 1/kappa))

    def expterm(self, y, uTau, nu):
        B1 = self.B1
        B2 = self.B2
        C = self.C
        yPlus = y*uTau/nu

        term1 = yPlus
        term2 = B1*np.exp(-yPlus/B1)
        term3 = B2*(B2 + yPlus)/B1*np.exp(-yPlus/B2)
        return C*nu*(term1 + term2 + term3)

    def logterm_derivative(self, y, uTau, nu):
        kappa = self.kappa
        yPlus = y*uTau/nu

        return y*(yPlus/(kappa*yPlus + 1) -
                  1/kappa +
                  1/kappa*np.log(kappa*yPlus + 1) +
                  1/(kappa*(kappa*yPlus + 1)))

    def expterm_derivative(self, y, uTau, nu):
        B1 = self.B1
        B2 = self.B2
        C = self.C
        yPlus = y*uTau/nu

        return C*(y - y*np.exp(-yPlus/B1) - y*yPlus/B1*np.exp(-yPlus/B2))


class CaiSagaut(LawOfTheWall):

    def __init__(self, kappa=0.4, B=5.5):
        LawOfTheWall.__init__(self)
        self.kappa = kappa
        self.B = B

        self.p = 1.138
        self.s = 217.8

    @property
    def kappa(self):
        return self._kappa

    @property
    def B(self):
        return self._B

    @property
    def E(self):
        return self.E

    @kappa.setter
    def kappa(self, value):
        self._kappa = value

    @B.setter
    def B(self, value):
        self._B = value


    def lambert(self, x, n=4):
        from numpy import log
        W = log(x) - log(log(x))

        for _ in range(n):
            W = W / (1 + W) * (1 + log(x / W))
        return W


    def explicit_value(self, y, nu, uTau):
        """Return the value of velocity."""
        pass

    def value(self, u, y, nu, uTau):
        """Return the value of the implicit function defined by the
         law."""
        from scipy.special import lambertw as W
        kappa = self.kappa
        E = np.exp(kappa*self.B)
        re = u*y/nu

        f = np.exp(-re/self.s)

        uplus = f**self.p*np.sqrt(re)
        uplus += (1 - f)**self.p/kappa*np.real(self.lambert(np.maximum(kappa*E*re, np.e)))

        return uTau - u/uplus

    def derivative(self, u, y, nu, uTau):
        """Return the value of the derivative of the implicit function
         defined by the law."""

        return 1
