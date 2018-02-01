# This file is part of wallriori
# (c) Timofey Mukha
# The code is released under the MIT Licence.
# See LICENCE.txt and the Legal section in the README for more information

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from wallriori import rootfinders


def f(x):
    return x


def test_rootfinder_init():
    rootFinder = rootfinders.RootFinder(f, f, 10, 0.01)
    assert rootFinder.eps == 0.01
    assert rootFinder.maxIter == 10
    assert rootFinder.f == f
    assert rootFinder.d == f
    assert rootFinder.debug is False


def test_rootfinder_mutable():
    rootFinder = rootfinders.RootFinder(f, f, 10, 0.01)
    rootFinder.eps = 0.001
    rootFinder.maxIter = 1
    rootFinder.debug = True
    assert rootFinder.eps == 0.001
    assert rootFinder.maxIter == 1
    assert rootFinder.debug is True
