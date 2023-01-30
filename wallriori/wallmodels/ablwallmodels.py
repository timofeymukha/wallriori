# This file is part of wallriori
# (c) Timofey Mukha
# The code is released under the MIT Licence.
# See LICENCE.txt and the Legal section in the README for more information

import numpy as np
from functools import partial
from ..wallmodels import WallModel

__all__ = ["MOSTWallModel"]


class MOSTWallModel(WallModel):
    def __init__(self, h: float, nu: float, z0u: float, z0t: float,
                 theta0: float, kappa=0.4, l_obukhov=None, g=9.81):
        WallModel.__init__(self, h, nu)

        self.z0u = z0u
        self.z0t = z0t
        self.theta0 = theta0
        self.kappa = kappa
        self.g = g
        self.utau_val = 0
        self.q_val = 0
        self.l_obukhov = l_obukhov

    def similarity_law(self, l_obukhov: float, z0: float, corr_func):
        #return np.log(self.h/z0) - corr_func(self.h/l_obukhov)\
        #                         - corr_func(z0/l_obukhov)
        return np.log(self.h/z0)

    def rib(self, u: float, theta: float, q=None, theta_s=None):
        if q is None:
            return self.g*self.h/theta*(theta - theta_s)/u**2
        else:
            return -self.g*self.h/theta*q/(u**3*self.kappa**2)

    def compute_length_scale(self, u: float, theta: float, q=None, theta_s=None):

        self.l_obukhov = self.length_scale(u, theta, q, theta_s)

        h = self.h

        for i in range(20):
            print(self.l_obukhov)
            l_upper = self.l_obukhov + 1e-3*self.l_obukhov
            l_lower = self.l_obukhov - 1e-3*self.l_obukhov

            if q is None:
                rib = self.rib(u, theta, theta_s=theta_s)
#                print(theta, theta_s, rib)
                f = (rib - self.h/self.l_obukhov*self.similarity_law(self.l_obukhov, self.z0t, self.correction_t)/
                                                 self.similarity_law(self.l_obukhov, self.z0u, self.correction_u)**2)

                # coeffs in front of linear correction term for the stable case
                a = 7.8
                b = 4.8
                dfdl = ((h*np.log(h/self.z0u)*(2*a*h - b*h + self.l_obukhov*np.log(h/self.z0u)))/
                        (b*h + self.l_obukhov*np.log(h/self.z0u))**3)

#                dfdl = (-self.h/l_upper*self.similarity_law(l_upper, self.z0t, self.correction_t)/
#                                        self.similarity_law(l_upper, self.z0u, self.correction_u)**2)
#                dfdl += (self.h/l_lower*self.similarity_law(l_lower, self.z0t, self.correction_t)/
#                                        self.similarity_law(l_lower, self.z0u, self.correction_u)**2)
            else:
                rib = self.rib(u, theta, q)
                f = (rib - self.h/self.l_obukhov/self.similarity_law(self.l_obukhov, self.z0u, self.correction_u)**3)

                dfdl = (-self.h/l_upper/self.similarity_law(l_upper, self.z0u, self.correction_u)**3)
                dfdl += (self.h/l_lower/self.similarity_law(l_lower, self.z0u, self.correction_u)**3)

            dfdl /= (l_upper - l_lower)
            self.l_obukhov -= f/dfdl

    def length_scale(self, u: float, theta: float, q=None, theta_s=None):
        if self.l_obukhov is None:
            utau = self.kappa*u/(np.log(self.h/self.z0u))
            if q is None:
                q = utau * self.kappa * (theta_s - theta) / np.log(self.h / self.z0t)

            print(utau, q)
            return -(utau**3 * self.theta0) / (self.kappa * self.g * q)
        else:
            return self.l_obukhov

    def correction_u(self, zeta: float):
        if zeta < 0:
            xi = (1 - 16*zeta)**0.25
            return 2*np.log(0.5*(1 + xi)) + np.log(0.5*(1 + xi**2)) - 2*np.arctan(xi) + np.pi/2
        else:
            print("zeta", zeta)
            a = 1.0
            b = 0.66666666666
            c = 5.0
            d = 0.35
            c_d_d = c/d
            bc_d_d = b*c/d
            return - b*(zeta - c_d_d)*np.exp(-d*zeta) - a*zeta - bc_d_d

    def correction_t(self, zeta: float):
        if zeta <0:
            xi = (1 - 16*zeta)**0.25
            return 2*np.log(0.5*(1 + xi**2))
        else:
            return -7.8*zeta

    def utau(self, sampled_u):
        psi_m = self.correction_u(self.h/self.l_obukhov)
        self.utau_val = self.kappa * sampled_u / (np.log(self.h / self.z0u) - psi_m)
        return self.utau_val

    def q(self, theta_s, theta):
        psi_q = self.correction_t(self.h/self.l_obukhov)
        self.q_val = self.utau_val*self.kappa*(theta_s - theta)/(np.log(self.h/self.z0t)
                                                                 - psi_q)
        return self.q_val

    def u_explicit(self, y, utau, l_obukhov, stable=False):
        print("hi")
        if l_obukhov != 0:
            xi = (1 - 16*y/l_obukhov)**0.25
            if stable:
                a = 1.0
                b = 0.66666666666
                c = 5.0
                d = 0.35
                c_d_d = c / d
                bc_d_d = b * c / d
                zeta = y/l_obukhov
                psi_m = - b * (zeta - c_d_d) * np.exp(-d * zeta) - a * zeta - bc_d_d
                zeta = self.z0u/l_obukhov
                psi_m -= - b * (zeta - c_d_d) * np.exp(-d * zeta) - a * zeta - bc_d_d

                psi_m = -4.8*y/l_obukhov + 4.8*self.z0u/l_obukhov
                print("h2")
            else:
                psi_m = 2*np.log(0.5*(1 + xi)) + np.log(0.5*(1 + xi**2))\
                        - 2*np.arctan(xi) + np.pi/2
        else:
            psi_m = 0
        print(psi_m)
        return utau/self.kappa * (np.log(y / self.z0u) - psi_m)

    def t_explicit(self, y, utau, q, l_obukhov, stable=False):
        if l_obukhov != 0:
            xi = (1 - 16*y/l_obukhov)**0.25
            if stable:
                psi_h = -7.8*y/l_obukhov + 7.8*0.1/l_obukhov
            else:
                psi_h = 2*np.log(0.5*(1 + xi**2))
        else:
            psi_h = 0
        return -q/(utau*self.kappa) * (np.log(y / self.z0t) - psi_h)
