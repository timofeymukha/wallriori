# This file is part of wallriori
# (c) Timofey Mukha
# The code is released under the MIT Licence.
# See LICENCE.txt and the Legal section in the README for more information

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
from functools import partial

__all__ = ["WallModel", "LOTWWallModel"]


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
        return np.max([0, self.rootFinder.solve(guess)])

