# This file is part of wallriori
# (c) Timofey Mukha
# The code is released under the MIT Licence.
# See LICENCE.txt and the Legal section in the README for more information

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from wallriori.mesh import Mesh
import numpy as np
from numpy.testing import assert_array_equal


def test_mesh_init():
    m = Mesh.from_faces(np.linspace(0, 1, 11))
    assert_array_equal(m.faces, np.linspace(0, 1, 11))


def test_mesh_mutable():
    m = Mesh.from_faces(np.linspace(0, 1, 11))
    m.nCells = 12
    m.dim2 = 12
    m.dim3 = 12
    assert m.nCells == 12
    assert m.dim2 == 12
    assert m.dim3 == 12


def test_mesh_volumes():
    m = Mesh.from_faces(np.linspace(0, 3, 4))
    volumes = np.array([1, 1, 1])
    assert_array_equal(m.volumes, volumes)


def test_mesh_centres():
    m = Mesh.from_faces(np.linspace(0, 1, 3))
    centres = np.array([0.25, 0.75])
    assert_array_equal(m.centres, centres)


def test_mesh_from_centres_two_cells():
    m = Mesh.from_centres_and_edges(np.array([0, 0.25, 0.75, 1]))
    assert_array_equal(m.faces, np.array([0, 0.5, 1]))


def test_mesh_from_centres_one_cell():
    m = Mesh.from_centres_and_edges(np.array([0, 0.5, 1]))
    assert_array_equal(m.faces, np.array([0, 1]))
