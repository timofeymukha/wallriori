# This file is part of wallriori
# (c) Timofey Mukha
# The code is released under the MIT Licence.
# See LICENCE.txt and the Legal section in the README for more information

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from wallriori.mesh import Mesh
from wallriori.problems import SteadyDiffusion
import numpy as np
from numpy.testing import assert_array_equal
from numpy.testing import assert_allclose


def test_mesh_init():
    mesh = Mesh.from_faces(np.linspace(0, 1, 11))
    nu = np.ones(mesh.centres.size)

    ic = np.zeros(mesh.centres.size)
    problem = SteadyDiffusion(mesh, ic, nu, [100, 2], [10, 2], 100)


def test_solve_linear():
    mesh = Mesh.from_faces(np.linspace(0, 1, 11))
    nu = np.ones(mesh.centres.size)
    ic = np.zeros(mesh.centres.size)
    source = 0
    problem = SteadyDiffusion(mesh, ic, nu, [0, 1], [1, 1], source)
    problem.solve()

    assert_allclose(problem.solution, mesh.centres, atol=1e-15)
