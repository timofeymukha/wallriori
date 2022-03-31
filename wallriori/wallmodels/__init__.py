# This file is part of wallriori
# (c) Timofey Mukha
# The code is released under the MIT Licence.
# See LICENCE.txt and the Legal section in the README for more information

from .wallmodels import *
from .odewallmodels import *

__all__ = ["wallmodels", "odewallmodels"]
__all__.extend(wallmodels.__all__)
__all__.extend(odewallmodels.__all__)
