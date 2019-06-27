# This file is part of wallriori
# (c) 2019 Timofey Mukha
# The code is released under the MIT Licence.
# See LICENCE.txt and the Legal section in the README for more information

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from wallriori.eddyviscosities import DupratEddyViscosity


def test_duprat_init_default():
    eddy = DupratEddyViscosity()
    assert eddy.kappa == 0.4
    assert eddy.Aplus == 18
    assert eddy.beta == 0.78


def test_duprat_init():
    eddy = DupratEddyViscosity(0.395, 17, 0.5)
    assert eddy.kappa == 0.395
    assert eddy.Aplus == 17
    assert eddy.beta == 0.5


def test_duprat_setters():
    eddy = DupratEddyViscosity(0.395, 17)
    eddy.kappa = 0.1
    eddy.Aplus = 1
    eddy.beta = 2
    assert eddy.kappa == 0.1
    assert eddy.Aplus == 1
    assert eddy.beta == 2


def test_duprat_values():
    eddy = DupratEddyViscosity()
    from numpy import linspace
    eddy.value(y=linspace(0, 0.01, 20), magPGrad=0.1, nu=8e-6, uTau=0.04)


