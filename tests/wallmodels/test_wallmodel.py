# This file is part of wallriori
# (c) Timofey Mukha
# The code is released under the MIT Licence.
# See LICENCE.txt and the Legal section in the README for more information

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from wallriori.wallmodels import WallModel


def test_wallmodel_init():
    w = WallModel(0.01)
    assert w.h == 0.01


def test_wallmodel_mutable():
    w = WallModel(0.01)
    w.h = 2
    assert w.h == 2
