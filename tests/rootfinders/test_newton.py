# This file is part of wallriori
# (c) Timofey Mukha
# The code is released under the MIT Licence.
# See LICENCE.txt and the Legal section in the README for more information

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from wallriori.rootfinders import Newton
from numpy.testing import assert_allclose


def f(x):
    return x**2


def d(x):
    return 2*x


def test_newton_init_default():
    newton = Newton()


def test_newton_init():
    newton = Newton(f, f, 10, 0.01)


def test_newton_solve():
    newton = Newton(f, d, 100, 0.01)
    root = newton.solve(1)
    assert_allclose(root, 0, rtol=0.01, atol=1e-2)
