# This file is part of wallriori
# (c) 2019 Timofey Mukha
# The code is released under the MIT Licence.
# See LICENCE.txt and the Legal section in the README for more information

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from wallriori.eddyviscosities import VanDriestEddyViscosity


def test_vandriest_init_default():
    eddy = VanDriestEddyViscosity()
    assert eddy.kappa == 0.4
    assert eddy.Aplus == 18


def test_vandriest_init():
    eddy = VanDriestEddyViscosity(0.395, 17)
    assert eddy.kappa == 0.395
    assert eddy.Aplus == 17


def test_vandriest_setters():
    eddy = VanDriestEddyViscosity(0.395, 17)
    eddy.kappa = 0.1
    eddy.Aplus = 1
    assert eddy.kappa == 0.1
    assert eddy.Aplus == 1


def test_vandriest_values():
    eddy = VanDriestEddyViscosity()
    from numpy import linspace
    eddy.value(y=linspace(0, 0.01, 20), nu=8e-6, uTau=0.04)


