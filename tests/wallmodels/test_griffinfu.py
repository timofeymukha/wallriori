# This file is part of wallriori
# (c) Timofey Mukha
# The code is released under the MIT Licence.
# See LICENCE.txt and the Legal section in the README for more information

from wallriori import GriffinFuWallModel
from wallriori import Mesh
import numpy as np


def test_griffinfu_init():
    mesh = Mesh.from_faces(np.linspace(0, 0.1, 10))
    m = GriffinFuWallModel(h=0.1, nu=8e-6, mesh=mesh, maxiter=10, tol=0.01)

def test_griffinfu_lplus():
    mesh = Mesh.from_faces(np.linspace(0, 0.1, 10))
    m = GriffinFuWallModel(h=0.1, nu=8e-6, mesh=mesh, maxiter=10, tol=0.01)
    m.l_plus(0.38, 5e-3, 17)

def test_griffinfu_utau():
    mesh = Mesh.from_faces(np.linspace(0, 0.1, 10))
    m = GriffinFuWallModel(h=0.1, nu=8e-6, mesh=mesh, maxiter=10, tol=0.01)
    m.utau(5e-3, 1, 0.5, 1000)


