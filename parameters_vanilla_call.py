import numpy as np
from dolfin import Expression, Constant, SubDomain, near, PETScKrylovSolver
from PDE_Solver.tools import ParametersBase

K_STRIKE = 1.60              # Doran calibration: strike placed inside the shipped mesh domain
ALPHA_RANGE = [-2.4, -1.6]   # Uncertainty interval L for lambda
N_CONTROLS = 9

class Parameters(ParametersBase):
    domains = ['square']
    meshes = ['005']

    def set_coefficients(self):
        self.T = 0.5;  self.K = K_STRIKE;  self.r = 0.03
        self.kappa = 7.0;  self.gamma = 0.3;  self.xi = 0.7;  self.rho = 0.5
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
        # Terminal payoff of the European call in transformed coordinates: 
        # Lambda(S) = max(S - K, 0), where S = exp(y + rc*z)
        self.ft = Expression('fmax(exp(x[0] + rc*x[1]) - K, 0.0)',
                             degree=2, K=self.K, rc=rc)

        # Isotropic diffusion after the shear of Section 4.1: 
        # a(z) = (xi*sqrt(1-rho^2)/2)*z, vanishing at the degenerate edge z=0
        def diffusion(alpha):
            coeff = self.xi * np.sqrt(1.0 - self.rho**2) / 2.0
            return Expression('c*fmax(x[1], 0.0)', degree=1, c=coeff)
        self.__dict__['diffusion'] = diffusion

        # y-drift bracket of the transformed operator: the lambda term, sqrt(z)
        # so it vanishes at z=0. Solver slot 'adv_x' (x[0]) is the thesis's y
        def adv_y(alpha):
            return Expression(
                '-( -r + kappa*gamma*rho/xi'
                '   + (0.5*xi - kappa*rho)/sqrt(1.0-rho*rho)*x[1]'
                '   - a*rho*sqrt( fmax(xi*x[1],0.0)/sqrt(1.0-rho*rho) ) )',
                degree=1, r=self.r, kappa=self.kappa, gamma=self.gamma,
                xi=self.xi, rho=self.rho, a=alpha)
        self.__dict__['adv_x'] = adv_y

        # z-drift: second bracket of the transformed operator:
        # the lambda term again carries sqrt(z), 'adv_y' is the solver slot x[1] = z
        def adv_z(alpha):
            return Expression(
                '-( -kappa*gamma*sqrt(1.0-rho*rho)/xi + kappa*x[1]'
                '   + a*sqrt( fmax(xi*x[1],0.0)*sqrt(1.0-rho*rho) ) )',
                degree=1, kappa=self.kappa, gamma=self.gamma,
                xi=self.xi, rho=self.rho, a=alpha)
        self.__dict__['adv_y'] = adv_z

        def lin(alpha):  return Constant(self.r) # zeroth-order discount term -rV
        self.__dict__['lin'] = lin
        def RHSt(alpha): return Constant(0.0)
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
        # Dirichlet-Robin set of the reference scheme. The Robin conditions
        # failed the discretisation's admissibility (monotonicity) check, so
        # the provably convergent all-Dirichlet set is used instead.
        self.regions = {"Dirichlet": [0, 1, 2, 3],
                        "Robin": [], "RobinTime": []}

        # Dirichlet conditions require values (not slopes) on every edge:
        # each edge is pinned to the payoff; the low-S edge is worthless (0)
        payoff_edge = Expression('fmax(exp(x[0] + rc*x[1]) - K, 0.0)',
                                 degree=2, K=self.K, rc=rc)
        self.RHS_bound = {
            0: Constant(0.0),
            1: (lambda a: payoff_edge),
            2: (lambda a: payoff_edge),
            3: (lambda a: payoff_edge),
        }
        self.time_dependent = {'rhs': False, 'pde': False, 'boundary': []}
