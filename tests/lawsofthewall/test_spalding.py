# This file is part of wallriori
# (c) Timofey Mukha
# The code is released under the MIT Licence.
# See LICENCE.txt and the Legal section in the README for more information

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from wallriori.lawsofthewall import Spalding


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
