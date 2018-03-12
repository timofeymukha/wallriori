# This file is part of wallriori
# (c) Timofey Mukha
# The code is released under the MIT Licence.
# See LICENCE.txt and the Legal section in the README for more information

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from wallriori.lawsofthewall import IntegratedWernerWengle
from numpy.testing import assert_allclose


def test_iww_init_defaults():
    law = IntegratedWernerWengle()
    assert law.A == 8.3
    assert law.B == 1/7


def test_iww_init():
    law = IntegratedWernerWengle(0.3, 5)
    assert law.A == 0.3
    assert law.B == 5.0


def test_iww_mutate():
    law = IntegratedWernerWengle()
    law.A = 0.3
    law.B = 5
    assert law.A == 0.3
    assert law.B == 5.0


def test_iww_value_call():
    law = IntegratedWernerWengle()
    law.value(0.8, 0.1, 8e-6, 5200*8e-6)


def test_iww_derivative_call():
    law = IntegratedWernerWengle()
    law.derivative(0.8, 0.1, 8e-6, 5200*8e-6)
