# This file is part of wallriori
# (c) Timofey Mukha
# The code is released under the MIT Licence.
# See LICENCE.txt and the Legal section in the README for more information

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np

__all__ = ["WallModel", "LOTWWallModel"]


class WallModel:

    def __init__(self, h):
        self.h = h

    @property
    def h(self):
        return self.__h

    @h.setter
    def h(self, val):
        self.__h = val

class LOTWWallModel(WallModel):

    def __init__(self, h, law, rootFinder):
        WallModel.__init__(self, h)

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
