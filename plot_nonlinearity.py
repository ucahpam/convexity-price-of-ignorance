"""
plot_nonlinearity.py -- the paper's Figure 5(b) analogue:
difference between the NONLINEAR worst-case solution and the LINEAR
evolution with a FIXED control lambda = -2.4 (the interval's endpoint).
If the problem were secretly linear the difference would vanish;
the structure of the difference IS the measured effect of nonlinearity.
Runs one extra solve: a control set collapsed to the single point -2.4
(control_set_size=1 => the HJB sup is over one option => linear PDE).
Usage (inside container, from /opt/FEISol):
  python3 /shared/plot_nonlinearity.py vanilla_call    # or vanilla_power
Reads the existing <exp>_sup run; runs fixed-lambda companion; plots
difference surface -> /shared/nonlinearity_<exp>.png
"""
import sys, glob
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from dolfin import Mesh, XDMFFile, FunctionSpace, Function

base = sys.argv[1] if len(sys.argv) > 1 else 'vanilla_call'
if base == 'vanilla_power':
    import parameters_power_general as mod
    mod.P_DEGREE = 2.0
else:
    import parameters_vanilla_call as mod
from PDE_Solver.hjb_mixed import FBVP, Solver

FIXED = -2.4   # the fixed control for the linear companion run

mesh = Mesh()
with XDMFFile(sorted(glob.glob('meshes/square/*005*.xdmf'))[0]) as f:
    f.read(mesh)
V = FunctionSpace(mesh, 'CG', 1)
xy = mesh.coordinates()

def read_final(exp):
    u = Function(V)
    for name in ['value_func', 'v', 'u', 'w']:
        try:
            with XDMFFile(f'out/{exp}/square/005/v.xdmf') as f:
                f.read_checkpoint(u, name, -1)
            return u.compute_vertex_values(mesh)
        except Exception:
            continue
    raise SystemExit(f'could not read {exp}; run the {exp} solve first')

# 1) the nonlinear worst case (already solved)
nl = read_final(f'{base}_sup')

# 2) the linear fixed-lambda companion (solve now, 1-point control set)
mod.ALPHA_RANGE = [FIXED, FIXED]
old_n = getattr(mod, 'N_CONTROLS', None)
mod.N_CONTROLS = 1
par = mod.Parameters('square', '005', f'{base}_fixed')
par.control_set_size = 1
fbvp = FBVP('005', par)
Solver(fbvp, howard_inf=True, get_error=False).time_iter()
if old_n is not None:
    mod.N_CONTROLS = old_n
lin = read_final(f'{base}_fixed')

diff = nl - lin
print(f'{base}: max |nonlinear - linear(lambda={FIXED})| = {np.abs(diff).max():.5f}')
print('should be >= 0 up to discretisation (sup over set >= single member):',
      f'min = {diff.min():.2e}')

fig, ax = plt.subplots(figsize=(7.5, 5.5))
t = ax.tricontourf(xy[:, 0], xy[:, 1], diff, 40, cmap='viridis')
fig.colorbar(t, ax=ax, label='V_nonlinear - V_fixed')
ax.set_xlabel('y'); ax.set_ylabel('z')
ax.set_title(f'{base}: effect of nonlinearity (sup over L vs fixed '
             f'lambda={FIXED})\n(paper Fig 5b analogue)', fontsize=10)
plt.tight_layout()
plt.savefig(f'/shared/nonlinearity_{base}.png', dpi=150)
print(f'saved /shared/nonlinearity_{base}.png')