# This file is part of wallriori
# (c) 2019 Timofey Mukha
# The code is released under the MIT Licence.
# See LICENCE.txt and the Legal section in the README for more information

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from wallriori.eddyviscosities import EddyViscosity


def test_eddyviscosity_init():
    eddy = EddyViscosity()
