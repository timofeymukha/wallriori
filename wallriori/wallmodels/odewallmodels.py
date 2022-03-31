# This file is part of wallriori
# (c) Timofey Mukha
# The code is released under the MIT Licence.
# See LICENCE.txt and the Legal section in the README for more information

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from .wallmodels import WallModel
import numpy as np
from ..mesh import Mesh
from ..problems import SteadyDiffusion

__all__ = ["IntegratedODEWallModel", "ODEWallModel", "GriffinFuWallModel"]

class IntegratedODEWallModel(WallModel):
    def __init__(self, h, nu, eddyViscosity, n, maxiter, tol):
        WallModel.__init__(self, h, nu)

        self.eddyViscosity = eddyViscosity
        self.mesh = np.linspace(0, h, n)
        self.maxiter = maxiter
        self.tol = tol

    def utau(self, guess, sampledU, sourceField, **kwargs):
        from scipy.integrate import simps
        y = self.mesh

        tau = guess**2
        for _ in range(self.maxiter):
            nut = self.eddyViscosity.value(y, self.nu, np.sqrt(tau), **kwargs)

            integral = simps(1/(self.nu + nut), x=y)
            integral2 = simps(y/(self.nu + nut), x=y)

            magU = np.linalg.norm(sampledU)
            newTau = (magU**2 + (np.linalg.norm(sourceField) * integral2)**2 -
                      2 * np.dot(sampledU, sourceField) * integral2)

            newTau = np.sqrt(newTau) / integral

            error = np.abs(tau - newTau) / tau
            tau = newTau

            if error < self.tol:
                break
        print(error)
        return np.sqrt(tau)


class ODEWallModel(WallModel):
    def __init__(self, nu, eddyViscosity, mesh, maxiter, tol):
        WallModel.__init__(self, mesh.faces[-1], nu)

        self.h = mesh.faces[-1]
        self.mesh = mesh
        self.y = self.mesh.centres
        self.yPlus = np.zeros(self.y.size)
        self.u = np.zeros(self.y.size)

        self.eddyViscosity = eddyViscosity
        self.maxiter = maxiter
        self.tol = tol

    @classmethod
    def from_cell_number(cls, h, nu, eddyViscosity, n, maxiter, tol):
        faces = np.linspace(0, h, n+1)
        mesh = Mesh.from_faces(faces)
        return ODEWallModel(nu, eddyViscosity, mesh, maxiter, tol)

    def utau(self, guess, sampledU, sourceField, **kwargs):
        ic = np.zeros(self.y.size)

        uTau = guess

        for i in range(self.maxiter):
            nut = self.eddyViscosity.value(np.append(self.y, self.mesh.faces[-1]), self.nu, uTau)
            bcLeft = [0, self.nu]
            bcRight = [sampledU, self.nu + nut[-1]]
            ode = SteadyDiffusion(self.mesh, ic, self.nu + nut[:-1], bcLeft, bcRight, sourceField)
            ode.solve()
            sol = ode.solution
            uTauOld = uTau
            uTau = np.sqrt(self.nu*sol[0]/self.y[0])

            if np.abs(uTau - uTauOld)/uTauOld < self.tol:
                break

        self.yPlus = self.y*uTau/self.nu
        self.u = sol

        return uTau


class GriffinFuWallModel(WallModel):
    def __init__(self, h, nu, mesh, maxiter, tol):
        WallModel.__init__(self, mesh.faces[-1], nu)

        self.h = h
        self.nu = nu
        self.mesh = mesh
        self.maxiter = maxiter
        self.tol = tol

    def a_plus(self, H, reTau):
        return 45.2 - 11.8*H - 0.993*np.log(reTau)

    def l_plus(self, kappa, uTau, aPlus):
        y = self.mesh.centres
        yPlus = y*uTau/self.nu
        return kappa*yPlus*(1 - np.exp(-(yPlus/aPlus)**2))

    def utau(self, guess, sampledU, H, reTau=None, kappa=0.38):
        from scipy.integrate import simps

        for _ in range(self.maxiter):
            if reTau is None:
                # CHANNEL WITH DELTA=1 ONLY
                aPlus = self.a_plus(H, guess/self.nu)
            else:
                aPlus = self.a_plus(H, reTau)
            lPlus = self.l_plus(kappa, guess, aPlus)
            yPlus = self.mesh.centres*guess/self.nu

            rhs = simps(1/(1 + lPlus), x=yPlus)

            new = sampledU/rhs
            if (guess - new)/guess < self.tol:
                return new
            else:
                guess = new

        return guess







