# This file is part of wallriori
# (c) Timofey Mukha
# The code is released under the MIT Licence.
# See LICENCE.txt and the Legal section in the README for more information

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np

__all__ = ["Problem"]


class Problem:

    def __init__(self, mesh, ic):
        self.mesh = mesh
        self.ic = ic

    @property
    def mesh(self):
        """The mesh."""
        return self.__mesh

    @property
    def ic(self):
        """Initial conditions"""
        return self.__ic

    @mesh.setter
    def mesh(self, mesh):
        self.__mesh = mesh

    @c.setter
    def ic(self, vals):
        if vals.size == self.mesh.centres.size:
            self.__ic = vals
        else:
            raise Error

