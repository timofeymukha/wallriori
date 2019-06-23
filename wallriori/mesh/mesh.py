# This file is part of wallriori
# (c) Timofey Mukha
# The code is released under the MIT Licence.
# See LICENCE.txt and the Legal section in the README for more information

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np

__all__ = ["Mesh"]


class Mesh:

    @classmethod
    def from_faces(cls, faces, dim2=1, dim3=1):
        obj = cls()
        obj.faces = faces
        obj.dim2 = dim2
        obj.dim3 = dim3

        obj.volumes = obj._compute_volumes()
        obj.centres = obj._compute_centres()
        obj.nCells = obj.centres.size
        return obj

    @classmethod
    def from_centres_and_edges(cls, centresAndEdges, dim2=1, dim3=1):
        obj = cls()

        obj.centres = centresAndEdges[1:-1]
        obj.dim2 = dim2
        obj.dim3 = dim3

        obj.nCells = obj.centres.size
        obj.faces = obj._compute_faces(centresAndEdges)
        obj.volumes = obj._compute_volumes()
        return obj


    @property
    def faces(self):
        """The edges of the cells."""
        return self.__faces

    @property
    def dim2(self):
        """
        The size in the second dimension.
        """
        return self.__dim2

    @property
    def dim3(self):
        """
        The size in the third dimension.
        """
        return self.__dim3

    @property
    def volumes(self):
        """
        The volumes of the cells.
        """
        return self.__volumes

    @property
    def centres(self):
        """
        The cell centres.
        """
        return self.__centres

    @property
    def nCells(self):
        """
        The number of cells.
        """
        return self.__nCells

    @faces.setter
    def faces(self, vals):
        self.__faces = vals

    @dim2.setter
    def dim2(self, vals):
        self.__dim2 = vals

    @dim3.setter
    def dim3(self, vals):
        self.__dim3 = vals

    @volumes.setter
    def volumes(self, vals):
        self.__volumes = vals

    @centres.setter
    def centres(self, vals):
        self.__centres = vals

    @nCells.setter
    def nCells(self, vals):
        self.__nCells = vals

    def _compute_volumes(self):
        """Cumpute the volumes of each cell"""
        faces = self.faces

        volumes = np.zeros(faces.size - 1)
        for i in range(volumes.size):
            volumes[i] = (faces[i+1] - faces[i])*self.dim2*self.dim3
        return volumes

    def _compute_centres(self):
        """Compute the cell centre of each cell"""
        faces = self.faces

        centres = np.zeros(faces.size - 1)
        for i in range(centres.size):
            centres[i] = 0.5*(faces[i+1] + faces[i])
        return centres

    def _compute_faces(self, centresAndEdges):
        """Compute the faces from the cell centres and edges"""
        faces = np.zeros(centresAndEdges.size-1)
        for i in range(faces.size):
            if i == 0:
                faces[i] = centresAndEdges[i]
            elif i == faces.size:
                faces[i] = centresAndEdges[-1]

            faces[i] = faces[i-1] + 2*(centresAndEdges[i] - faces[i-1])
        return faces