# This file is part of wallriori
# (c) Timofey Mukha
# The code is released under the MIT Licence.
# See LICENCE.txt and the Legal section in the README for more information

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from wallriori.lawsofthewall import Spalding
from numpy.testing import assert_allclose


def test_spalding_init_defaults():
    law = Spalding()
    assert law.kappa == 0.4
    assert law.B == 5.5


def test_spalding_init():
    law = Spalding(0.3, 5)
    assert law.kappa == 0.3
    assert law.B == 5.0


def test_spalding_mutate():
    law = Spalding()
    law.kappa = 0.3
    law.B = 5
    assert law.kappa == 0.3
    assert law.B == 5.0


def test_spalding_value_call():
    law = Spalding()
    law.value(0.8, 0.1, 8e-6, 5200*8e-6)


def test_spalding_derivative_call():
    law = Spalding()
    law.derivative(0.8, 0.1, 8e-6, 5200*8e-6)


def test_spalding_value():
    law = Spalding()
    val = law.value(1, 1, 1, 1)
    assert_allclose([val], [0.00012831348946749514], atol=1e-10)


def test_spalding_derivative():
    law = Spalding()
    val = law.derivative(1, 1, 1, 1)
    assert_allclose([val], [-2.000524085538133], atol=1e-10)


def test_spalding_explicit_value_call():
    law = Spalding()
    law.explicit_value(0.1, 8e-6, 5200*8e-6)
