# This file is part of wallriori
# (c) Timofey Mukha
# The code is released under the MIT Licence.
# See LICENCE.txt and the Legal section in the README for more information

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from wallriori.wallmodels import IntegratedLOTWWallModel
from wallriori.rootfinders import Newton
from wallriori.lawsofthewall import IntegratedReichardt


def test_integratedlotw_init():
    w = IntegratedLOTWWallModel(0.01, 0.02, 1e-4, 1, 1)
    assert w.h == 0.015
    assert w.h1 == 0.01
    assert w.h2 == 0.02
    assert w.nu == 1e-4
    assert w.law == 1
    assert w.rootFinder == 1


def test_integratedlotw_mutable():
    w = IntegratedLOTWWallModel(0.01, 0.02, 1e-4, 1, 1)
    w.h1 = 2
    w.h2 = 3
    w.nu = 2
    w.law = 2
    w.rootFinder = 2
    assert w.h1 == 2
    assert w.h2 == 3
    assert w.nu == 2
    assert w.law == 2
    assert w.rootFinder == 2


def test_integratedlotw_utau():
    rf = Newton()
    law = IntegratedReichardt()
    nu = 8e-6
    w = IntegratedLOTWWallModel(0.1, 0.2, nu, law, rf)
    w.utau(5200.*nu, 0.8)


def test_integratedlotw_nut():
    rf = Newton()
    law = IntegratedReichardt()
    nu = 8e-6
    w = IntegratedLOTWWallModel(0.1, 0.2, nu, law, rf)
    w.nut(5200.*nu, 0.8, 40)
