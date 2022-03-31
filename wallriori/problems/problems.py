# This file is part of wallriori
# (c) Timofey Mukha
# The code is released under the MIT Licence.
# See LICENCE.txt and the Legal section in the README for more information

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
from scipy.interpolate import interp1d
from scipy.sparse.linalg import spsolve
from scipy.sparse import csr_matrix

__all__ = ["Problem", "SteadyDiffusion", "UnsteadyDiffusion"]


class Problem:

    def __init__(self, mesh, ic):
        self.mesh = mesh
        self.ic = ic
        self.solution = ic

    @property
    def mesh(self):
        """The mesh."""
        return self.__mesh

    @property
    def ic(self):
        """Initial conditions"""
        return self.__ic

    @property
    def solution(self):
        """The solution"""
        return self.__solution

    @mesh.setter
    def mesh(self, mesh):
        self.__mesh = mesh

    @ic.setter
    def ic(self, vals):
        if vals.size == self.mesh.centres.size:
            self.__ic = vals
        else:
            raise ValueError("The size of the data and the mesh mush match")

    @solution.setter
    def solution(self, vals):
        if vals.size == self.mesh.centres.size:
            self.__solution = vals
        else:
            raise ValueError("The size of the data and the mesh mush match")


class SteadyDiffusion(Problem):

    def __init__(self, mesh, ic, nu, bcLeft, bcRight, source):
        Problem.__init__(self, mesh, ic)
        self.nu = nu
        self.bcLeft = bcLeft
        self.bcRight = bcRight
        if np.ndim(source) == 0:
            self.source = source*np.ones(mesh.nCells)
        elif source.size == mesh.nCells:
            self.source = source
        else:
            raise ValueError("Source has incorrect dimensions", source.size, mesh.nCells)

    @property
    def nu(self):
        return self.__nu

    @property
    def bcLeft(self):
        """The boundary conidtions on the left side, for unknown and nu."""
        return self.__bcLeft

    @property
    def bcRight(self):
        """The boundary conidtions on the right side, for unknown and nu."""
        return self.__bcRight

    @property
    def source(self):
        """The source term."""
        return self.__source

    @nu.setter
    def nu(self, vals):
        if vals.size == self.mesh.centres.size:
            self.__nu = vals
        else:
            raise ValueError("The size of the data and the mesh mush match")

    @bcLeft.setter
    def bcLeft(self, vals):
        self.__bcLeft = vals

    @bcRight.setter
    def bcRight(self, vals):
        self.__bcRight = vals

    @source.setter
    def source(self, val):
        if np.ndim(val) == 0:
            self.__source = val*np.ones(self.mesh.nCells)
        else:
            self.__source = val

    def assemble(self):
        nCells = self.mesh.nCells
        centres = self.mesh.centres
        volumes = self.mesh.volumes
        faces = self.mesh.faces
        nu = self.nu
        A = self.mesh.dim2*self.mesh.dim3
        bcLeft = self.bcLeft
        bcRight = self.bcRight

        nuF = interp1d(centres, nu, kind='linear')(faces[1:-1])
        nuF = np.insert(nuF, 0, self.bcLeft[1])
        nuF = np.append(nuF, self.bcRight[1])

        system = np.zeros([nCells, nCells])
        rhs = np.zeros(nCells)

        for i in range(nCells):
            dx = volumes[i]/A
            if i == 0:
                rhs[i] = A*nuF[i]/(centres[i] - faces[0])*bcLeft[0]
                system[i, i+1] = -A*nuF[i+1]/(centres[i+1] - centres[i])
                system[i, i] = -system[i, i+1] + A*nuF[i]/(centres[i] - faces[0])
            elif i == nCells - 1:
                rhs[i] = A*nuF[i+1]/(faces[-1] - centres[i])*bcRight[0]
                system[i, i-1] = -A*nuF[i]/(centres[i] - centres[i-1])
                system[i, i] = -system[i, i-1] + A*nuF[i+1]/(faces[-1] - centres[i])
            else:
                system[i, i-1] = -A*nuF[i]/(centres[i] - centres[i-1])
                system[i, i+1] = -A*nuF[i+1]/(centres[i+1] - centres[i])
                system[i, i] = -(system[i, i-1] + system[i, i+1])
                rhs[i] = self.source[i]*volumes[i]

        return system, rhs

    def solve(self):
        system, rhs = self.assemble()
        self.solution = np.linalg.solve(system, rhs)

class  UnsteadyDiffusion(Problem):

    def __init__(self, mesh, ic, nu, dt, T, bcLeft, bcRight, source):
        Problem.__init__(self, mesh, ic)
        self.nu = nu
        self.dt = dt
        self.T = T
        self.time = 0
        self.bcLeft = bcLeft
        self.bcRight = bcRight
        self.source = source

    @property
    def nu(self):
        return self.__nu

    @property
    def dt(self):
        """The time-step"""
        return self.__dt

    @property
    def T(self):
        """The total simulation time"""
        return self.__T

    @property
    def time(self):
        """The current time"""
        return self.__time

    @property
    def bcLeft(self):
        """The boundary conidtions on the left side, for unknown and nu."""
        return self.__bcLeft

    @property
    def bcRight(self):
        """The boundary conidtions on the right side, for unknown and nu."""
        return self.__bcRight

    @property
    def source(self):
        """The source term."""
        return self.__source

    @nu.setter
    def nu(self, vals):
        if vals.size == self.mesh.centres.size:
            self.__nu = vals
        else:
            raise ValueError("The size of the data and the mesh mush match")

    @dt.setter
    def dt(self, val):
        self.__dt = val

    @T.setter
    def T(self, val):
        self.__T = val

    @time.setter
    def time(self, val):
        self.__time = val

    @bcLeft.setter
    def bcLeft(self, vals):
        self.__bcLeft = vals

    @bcRight.setter
    def bcRight(self, vals):
        self.__bcRight = vals

    @source.setter
    def source(self, val):
        self.__source = val

    def assemble(self):
        nCells = self.mesh.nCells
        centres = self.mesh.centres
        volumes = self.mesh.volumes
        faces = self.mesh.faces
        nu = self.nu
        A = self.mesh.dim2*self.mesh.dim3
        sol = self.solution
        dt = self.dt
        bcLeft = self.bcLeft
        bcRight = self.bcRight

        nuF = interp1d(centres, nu, kind='linear')(faces[1:-1])
        nuF = np.insert(nuF, 0, self.bcLeft[1])
        nuF = np.append(nuF, self.bcRight[1])


        system = np.zeros([nCells, nCells])

        rhs = np.zeros(nCells)

        for i in range(nCells):
            dx = volumes[i]/A
            if i == 0:
                rhs[i] = dx/dt*sol[i] + A*nuF[i]/(centres[i] - faces[0])*bcLeft[0]
                system[i, i+1] = -A*nuF[i+1]/(centres[i+1] - centres[i])
                system[i, i] = -system[i, i+1]  +  A*nuF[i]/(centres[i] - faces[0]) + dx/dt
            elif i == nCells - 1:
                rhs[i] = dx/dt*sol[i] + A*nuF[i+1]/(faces[-1] - centres[i])*bcRight[0]
                system[i, i-1] = -A*nuF[i]/(centres[i] - centres[i-1])
                system[i, i] = -system[i, i-1]  +  A*nuF[i+1]/(faces[-1] - centres[i]) + dx/dt
            else:
                system[i, i-1] = -A*nuF[i]/(centres[i] - centres[i-1])
                system[i, i+1] = -A*nuF[i+1]/(centres[i+1] - centres[i])
                system[i, i] = -(system[i, i-1] + system[i, i+1]) + dx/dt
                rhs[i] = dx/dt*sol[i] + self.source*volumes[i]

        return csr_matrix(system), rhs

    def advance(self):
        system, rhs = self.assemble()
        self.solution = spsolve(system, rhs)
        self.time += self.dt

    def solve(self):

        while self.time < self.T:
            self.advance()