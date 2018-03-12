# This file is part of wallriori
# (c) Timofey Mukha
# The code is released under the MIT Licence.
# See LICENCE.txt and the Legal section in the README for more information

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from wallriori.lawsofthewall import Reichardt
from numpy.testing import assert_allclose


def test_reichardt_init_defaults():
    law = Reichardt()
    assert law.kappa == 0.4
    assert law.B1 == 11
    assert law.B2 == 3
    assert law.C == 7.8


def test_reichardt_init():
    law = Reichardt(0.3, 5, 1, 1)
    assert law.kappa == 0.3
    assert law.B1 == 5
    assert law.B2 == 1
    assert law.C == 1


def test_reichardt_mutate():
    law = Reichardt()
    law.kappa = 0.3
    law.B1 = 5
    law.B2 = 1
    law.C = 1

    assert law.kappa == 0.3
    assert law.B1 == 5
    assert law.B2 == 1
    assert law.C == 1


def test_reichardt_value_call():
    law = Reichardt()
    law.value(0.8, 0.1, 8e-6, 5200*8e-6)


def test_reichardt_derivative_call():
    law = Reichardt()
    law.derivative(0.8, 0.1, 8e-6, 5200*8e-6)


def test_reichardt_explicit_value_call():
    law = Reichardt()
    law.explicit_value(0.1, 8e-6, 5200*8e-6)
