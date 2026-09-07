"""
parameters_power_general.py: Powered call with variable degree p:
    Lambda(S) = max(S - K, 0)^p / K^(p-1)

    p = 1    recovers the vanilla call (anchors the family)
    p = 1.5  mildly convex
    p = 2    the powered call used in the main comparison
    p = 3    strongly convex

The K^(p-1) normalisation keeps Lambda(S) comparable in magnitude to
(S - K) near the strike for every p, so bands are comparable across p
(Section 5.3). The degree p is set through the module-level P_DEGREE;
the convexity sweep (Appendix A.4) assigns it for each p in turn.
"""
import numpy as np
from dolfin import Expression, Constant, SubDomain, near, PETScKrylovSolver
from PDE_Solver.tools import ParametersBase

K_STRIKE = 1.60
ALPHA_RANGE = [-2.4, -1.6]
N_CONTROLS = 9
P_DEGREE = 2.0          # Payoff degree p: overridden by sweep scripts

class Parameters(ParametersBase):
    domains = ['square']
    meshes = ['005']

    def set_coefficients(self):
        self.T = 0.5;  self.K = K_STRIKE;  self.r = 0.03
        self.kappa = 7.0;  self.gamma = 0.3
        self.xi = 0.7;  self.rho = 0.5
        self.alpha_range = list(ALPHA_RANGE)
        self.control_set_size = N_CONTROLS
        self.howmaxit = 3
        self.solver = PETScKrylovSolver('gmres', 'sor')
        self.solver.parameters['report'] = False
        self.solver.parameters['monitor_convergence'] = False
        self.solver.parameters['nonzero_initial_guess'] = True
        self.save_interval = 10

        rc = self.rho / np.sqrt(1.0 - self.rho**2)
        self.rc = rc
        self.p = float(P_DEGREE)
        # Powered-call payoff: Lambda(S) = max(S-K,0)^p / K^(p-1),
        # with S = exp(y + rc*z), p=1 gives the vanilla call
        self.ft = Expression(
            'pow(fmax(exp(x[0] + rc*x[1]) - K, 0.0), p) / pow(K, p - 1.0)',
            degree=3, K=self.K, rc=rc, p=self.p)
        # Isotropic diffusion after the shear of Section 4.1:
        # a(z) = (xi*sqrt(1-rho^2)/2)*z, vanishing at the degenerate edge z=0
        def diffusion(alpha):
            coeff = self.xi * np.sqrt(1.0 - self.rho**2) / 2.0
            return Expression('c*fmax(x[1], 0.0)', degree=1, c=coeff)
        self.__dict__['diffusion'] = diffusion
        # adv_x / adv_y are the solver's coordinate slots x[0], x[1],
        # i.e. the thesis's transformed variables y and z. So the y-drift
        # is stored under 'adv_x' and the z-drift under 'adv_y'
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

        def lin(alpha):  return Constant(self.r) # Zeroth-order discount term -rV
        self.__dict__['lin'] = lin
        def RHSt(alpha): return Constant(0.0) # No source term
        self.__dict__['RHSt'] = RHSt

    def set_boundary_conditions(self, mesh):
        coords = mesh.coordinates()
        ymin, zmin = coords.min(0);  ymax, zmax = coords.max(0)
        tol = 1e-10
        rc = self.rho / np.sqrt(1.0 - self.rho**2)
        # Identify the four edges of the truncated rectangular domain
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
        # Boundary simplification (Section 5.5): all four edges imposed as
        # Dirichlet conditions carrying payoff data, in place of the mixed
        # Dirichlet-Robin set of the reference scheme
        self.regions = {"Dirichlet": [0, 1, 2, 3], "Robin": [], "RobinTime": []}
        # Each edge pinned to the powered payoff; the low-S edge is worthless (0)
        payoff_edge = Expression(
            'pow(fmax(exp(x[0] + rc*x[1]) - K, 0.0), p) / pow(K, p - 1.0)',
            degree=3, K=self.K, rc=rc, p=self.p)
        self.RHS_bound = {0: Constant(0.0), 1: payoff_edge,
                          2: payoff_edge, 3: payoff_edge}
        self.time_dependent = {'rhs': False, 'pde': False, 'boundary': []}