# This file is part of wallriori
# (c) Timofey Mukha
# The code is released under the MIT Licence.
# See LICENCE.txt and the Legal section in the README for more information

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from wallriori.lawsofthewall import CaiSagaut
import numpy as np


def test_caisagaut_init_defaults():
    law = CaiSagaut()
    assert law.kappa == 0.4
    assert law.B == 5.5


def test_caisagaut_init():
    law = CaiSagaut(0.3, 5)
    assert law.kappa == 0.3
    assert law.B == 5


def test_caisagaut_mutate():
    law = CaiSagaut()
    law.kappa = 0.3
    law.B = 5

    assert law.kappa == 0.3
    assert law.B == 5


def test_caisagaut_value_call():
    law = CaiSagaut()
    law.value(0.8, 0.1, 8e-6, 5200*8e-6)


def test_caisagaut_derivative_call():
    law = CaiSagaut()
    a = law.derivative(0.8, 0.1, 8e-6, 5200*8e-6)
    assert a == 1.0
