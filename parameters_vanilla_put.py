"""
parameters_vanilla_put.py: European put under the uncertain-Heston
worst-case model. The operator, control set and solver are identical to
the call, only the payoff and the boundary data differ:
  1. payoff Lambda(S) = max(K - S, 0);
  2. the low-S Dirichlet value is K (as S -> 0 the put is worth the
     discounted strike);
  3. the pinned edges carry the put payoff.
"""
import numpy as np
from dolfin import Expression, Constant, SubDomain, near, PETScKrylovSolver
from PDE_Solver.tools import ParametersBase

K_STRIKE = 1.60
ALPHA_RANGE = [-2.4, -1.6]
N_CONTROLS = 9

class Parameters(ParametersBase):
    domains = ['square']
    meshes = ['005']

    def set_coefficients(self):
        self.T = 0.5;  self.K = K_STRIKE;  self.r = 0.03
        self.kappa = 7.0;  self.gamma = 0.3
        self.xi = 0.7;  self.rho = 0.5
        self.alpha_range = ALPHA_RANGE
        self.control_set_size = N_CONTROLS
        self.howmaxit = 3
        self.solver = PETScKrylovSolver('gmres', 'sor')
        self.solver.parameters['report'] = False
        self.solver.parameters['monitor_convergence'] = False
        self.solver.parameters['nonzero_initial_guess'] = True
        self.save_interval = 10

        rc = self.rho / np.sqrt(1.0 - self.rho**2)
        self.rc = rc
        self.ft = Expression('fmax(K - exp(x[0] + rc*x[1]), 0.0)',
                             degree=2, K=self.K, rc=rc)

        def diffusion(alpha):
            coeff = self.xi * np.sqrt(1.0 - self.rho**2) / 2.0
            return Expression('c*fmax(x[1], 0.0)', degree=1, c=coeff)
        self.__dict__['diffusion'] = diffusion

        def adv_y(alpha):
            return Expression(
                '-( -r + kappa*gamma*rho/xi'
                '   + (0.5*xi - kappa*rho)/sqrt(1.0-rho*rho)*x[1]'
                '   - a*rho*sqrt( fmax(xi*x[1],0.0)/sqrt(1.0-rho*rho) ) )',
                degree=1, r=self.r, kappa=self.kappa, gamma=self.gamma,
                xi=self.xi, rho=self.rho, a=alpha)
        self.__dict__['adv_x'] = adv_y

        def adv_z(alpha):
            return Expression(
                '-( -kappa*gamma*sqrt(1.0-rho*rho)/xi + kappa*x[1]'
                '   + a*sqrt( fmax(xi*x[1],0.0)*sqrt(1.0-rho*rho) ) )',
                degree=1, kappa=self.kappa, gamma=self.gamma,
                xi=self.xi, rho=self.rho, a=alpha)
        self.__dict__['adv_y'] = adv_z

        def lin(alpha):  return Constant(self.r)
        self.__dict__['lin'] = lin
        def RHSt(alpha): return Constant(0.0)
        self.__dict__['RHSt'] = RHSt

    def set_boundary_conditions(self, mesh):
        coords = mesh.coordinates()
        ymin, zmin = coords.min(0);  ymax, zmax = coords.max(0)
        tol = 1e-10
        rc = self.rho / np.sqrt(1.0 - self.rho**2)

        class LowS(SubDomain):
            def inside(self, x, on_boundary):
                return on_boundary and near(x[0], ymin, tol)
        class HighS(SubDomain):
            def inside(self, x, on_boundary):
                return on_boundary and near(x[0], ymax, tol)
        class HighV(SubDomain):
            def inside(self, x, on_boundary):
                return on_boundary and near(x[1], zmax, tol)
        class ZeroV(SubDomain):
            def inside(self, x, on_boundary):
                return on_boundary and near(x[1], 0.0, tol)

        self.omegas = {0: LowS(), 1: HighS(), 2: HighV(), 3: ZeroV()}

        self.regions = {"Dirichlet": [0, 1, 2, 3], "Robin": [], "RobinTime": []}

        payoff_edge = Expression('fmax(K - exp(x[0] + rc*x[1]), 0.0)',
                                 degree=2, K=self.K, rc=rc)
        self.RHS_bound = {
            0: Constant(self.K),   
            1: payoff_edge,        
            2: payoff_edge,
            3: payoff_edge,
        }
        self.time_dependent = {'rhs': False, 'pde': False, 'boundary': []}
