# This file is part of wallriori
# (c) Timofey Mukha
# The code is released under the MIT Licence.
# See LICENCE.txt and the Legal section in the README for more information

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from wallriori.wallmodels import LOTWWallModel
from wallriori.rootfinders import Newton
from wallriori.lawsofthewall import Spalding


def test_lotw_init():
    w = LOTWWallModel(0.01, 1e-4, 1, 1)
    assert w.h == 0.01
    assert w.nu == 1e-4
    assert w.law == 1
    assert w.rootFinder == 1


def test_lotw_mutable():
    w = LOTWWallModel(0.01, 1e-4, 1, 1)
    w.h = 2
    w.nu = 2
    w.law = 2
    w.rootFinder = 2
    assert w.h == 2
    assert w.nu == 2
    assert w.law == 2
    assert w.rootFinder == 2


def test_lotw_utau():
    rf = Newton()
    law = Spalding()
    nu = 8e-6
    w = LOTWWallModel(0.1, nu, law, rf)
    w.utau(5200.*nu, 0.8)


def test_lotw_nut():
    rf = Newton()
    law = Spalding()
    nu = 8e-6
    w = LOTWWallModel(0.1, nu, law, rf)
    w.nut(5200.*nu, 0.8, 40)
