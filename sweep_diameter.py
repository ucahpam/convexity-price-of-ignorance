"""
sweep_diameter.py -- the paper's Figure 6 analogue:
V_inf, V_sup AND Delta_inf, Delta_sup measured AT A FIXED POINT,
plotted against the DIAMETER of the control set (symmetric, centred at
lambda = -2; the paper centres at -1.25). Diameter 0 = a single fixed
lambda: the two curves start together and fan out as ignorance grows --
exactly the paper's two-panel figure.
Measurement point: at-the-money, mid variance: z*=0.5,
y* = ln(K) - rc*z* (so S = K there).
Usage: python3 /shared/sweep_diameter.py vanilla_call   # or vanilla_power
Output: /shared/diameter_gaps_<exp>.png + a printed table.
"""
import sys, glob
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from dolfin import (Mesh, XDMFFile, FunctionSpace, Function, project,
                    Expression)

base = sys.argv[1] if len(sys.argv) > 1 else 'vanilla_call'
if base == 'vanilla_power':
    import parameters_power_general as mod
    mod.P_DEGREE = 2.0
else:
    import parameters_vanilla_call as mod
from PDE_Solver.hjb_mixed import FBVP, Solver

DIAMETERS = [0.0, 0.2, 0.4, 0.8, 1.2, 1.6]
CENTRE = -2.0
K = 1.60
rho = 0.5
rc = rho / np.sqrt(1 - rho**2)

mesh = Mesh()
with XDMFFile(sorted(glob.glob('meshes/square/*005*.xdmf'))[0]) as f:
    f.read(mesh)
V = FunctionSpace(mesh, 'CG', 1)
xy = mesh.coordinates()

# measurement vertex: nearest to (y*, z*) with S=K at z*=0.5
zstar = 0.5
ystar = np.log(K) - rc * zstar
istar = int(np.argmin((xy[:, 0] - ystar)**2 + (xy[:, 1] - zstar)**2))
S_star = np.exp(xy[istar, 0] + rc * xy[istar, 1])
print(f'measurement vertex: (y,z)=({xy[istar,0]:.3f},{xy[istar,1]:.3f}), '
      f'S={S_star:.3f} (target K={K})')

S_inv = Expression('exp(-(x[0] + rc*x[1]))', degree=2, rc=rc)

def read_u(exp):
    u = Function(V)
    for name in ['value_func', 'v', 'u', 'w']:
        try:
            with XDMFFile(f'out/{exp}/square/005/v.xdmf') as f:
                f.read_checkpoint(u, name, -1)
            return u
        except Exception:
            continue
    raise SystemExit(f'could not read {exp}')

def run(name, sup, d):
    mod.ALPHA_RANGE = [CENTRE - d/2, CENTRE + d/2]
    if d == 0.0:
        mod.N_CONTROLS = 1
    else:
        mod.N_CONTROLS = 9
    par = mod.Parameters('square', '005', name)
    par.control_set_size = mod.N_CONTROLS
    fbvp = FBVP('005', par)
    Solver(fbvp, howard_inf=sup, get_error=False).time_iter()
    u = read_u(name)
    val = u.compute_vertex_values(mesh)[istar]
    delta = project(S_inv * u.dx(0), V).compute_vertex_values(mesh)[istar]
    return val, delta

rows = []
for d in DIAMETERS:
    vs, ds_ = run(f'swD_{base}_d{d}_sup', True, d)
    if d == 0.0:
        vi, di = vs, ds_          # single control: sup == inf
    else:
        vi, di = run(f'swD_{base}_d{d}_inf', False, d)
    rows.append((d, vi, vs, di, ds_))
    print(f'd={d:.1f}  V_inf={vi:.4f} V_sup={vs:.4f} '
          f'D_inf={di:.4f} D_sup={ds_:.4f}')

d_, vi_, vs_, di_, ds__ = map(np.array, zip(*rows))
fig, ax = plt.subplots(1, 2, figsize=(12.5, 5))
ax[0].plot(d_, vi_, 'o--', label='V_inf'); ax[0].plot(d_, vs_, 'o--', label='V_sup')
ax[0].set_xlabel('control set diameter'); ax[0].set_ylabel('option value at S=K')
ax[0].set_title('Gap in option value (paper Fig 6a analogue)')
ax[0].legend(); ax[0].grid(alpha=.3)
ax[1].plot(d_, di_, 'o--', label='dV_inf/dS'); ax[1].plot(d_, ds__, 'o--', label='dV_sup/dS')
ax[1].set_xlabel('control set diameter'); ax[1].set_ylabel('Delta at S=K')
ax[1].set_title('Gap in Delta (paper Fig 6b analogue)')
ax[1].legend(); ax[1].grid(alpha=.3)
plt.suptitle(f'{base}: worst/best gaps vs control-set diameter '
             f'(centred at {CENTRE})', fontsize=11)
plt.tight_layout()
plt.savefig(f'/shared/diameter_gaps_{base}.png', dpi=150)
print(f'saved /shared/diameter_gaps_{base}.png')
pv = (vs_[-1]-vi_[-1])/vi_[-1]*100 if vi_[-1] else 0
pd = (ds__[-1]-di_[-1])/di_[-1]*100 if di_[-1] else 0
print(f'at largest diameter: value gap {pv:.1f}%, Delta gap {pd:.1f}% '
      f'(paper reports ~16% / ~6% for its butterfly at K=50)')