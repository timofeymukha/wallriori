# This file is part of wallriori
# (c) Timofey Mukha
# The code is released under the MIT Licence.
# See LICENCE.txt and the Legal section in the README for more information

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np

__all__ = ["RootFinder", "Newton"]

def f(x):
    return x

class RootFinder:

    def __init__(self, function=f, derivative=f, maxIter=50, eps=0.001, debug=False):
        self.f = function
        self.d = derivative
        self.maxIter = maxIter
        self.eps = eps
        self.debug = debug


    @property
    def f(self):
        """The function representing the equation to be solved"""
        return self.__f

    @property
    def d(self):
        """
        The derivative of function representing the equation to be
        solved
        """
        return self.__d

    @property
    def maxIter(self):
        """Maximum amount of iterations"""
        return self.__maxIter

    @property
    def eps(self):
        """The tolerance of the solver"""
        return self.__eps

    @property
    def debug(self):
        """Whether to provide additional output"""
        return self.__debug

    @f.setter
    def f(self, function):
        self.__f = function

    @d.setter
    def d(self, function):
        self.__d = function

    @maxIter.setter
    def maxIter(self, n):
        self.__maxIter = n

    @eps.setter
    def eps(self, eps):
        self.__eps = eps

    @debug.setter
    def debug(self, debug):
        self.__debug = debug


class Newton(RootFinder):

    def __init__(self,function=f, derivative=f, maxIter=100, eps=0.001, debug=False):
        RootFinder.__init__(self, function, derivative, maxIter, eps,
                            debug=debug)

    def solve(self, guess):

        for i in range(self.maxIter):
            f = self.f(guess)
            d = self.d(guess)

            newGuess = guess - f/d

            error = np.abs(newGuess - guess)/np.abs(guess)

            if np.ndim(guess) == 0:
                if error <= self.eps:
                    return newGuess
            elif np.all(error <= self.eps):
                return newGuess

            guess = newGuess

        if (self.debug):
            print("Newton: not converged", "error", error)
        return guess




