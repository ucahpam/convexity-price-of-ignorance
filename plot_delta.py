import sys, glob
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from dolfin import (Mesh, XDMFFile, FunctionSpace, Function, project,
                    Expression)

exps = sys.argv[1:] if len(sys.argv) > 1 else ['vanilla_call_sup',
                                               'vanilla_call_inf']
rho = 0.5
rc = rho / np.sqrt(1 - rho**2)

mesh = Mesh()
with XDMFFile(sorted(glob.glob('meshes/square/*005*.xdmf'))[0]) as f:
    f.read(mesh)
V = FunctionSpace(mesh, 'CG', 1)
xy = mesh.coordinates()

def load(exp):
    u = Function(V)
    for name in ['value_func', 'v', 'u', 'w']:
        try:
            with XDMFFile(f'out/{exp}/square/005/v.xdmf') as f:
                f.read_checkpoint(u, name, -1)
            return u
        except Exception:
            continue
    raise SystemExit(f'could not read {exp}')

S_inv = Expression('exp(-(x[0] + rc*x[1]))', degree=2, rc=rc)

fig, ax = plt.subplots(1, 2, figsize=(13, 5))
zs, tolz = 0.5, 0.03
for exp in exps:
    w = load(exp)
    delta = project(S_inv * w.dx(0), V)     
    dvals = delta.compute_vertex_values(mesh)
    print(f'{exp}: Delta range [{dvals.min():.4f}, {dvals.max():.4f}]')
    if exp == exps[0]:
        t = ax[0].tricontourf(xy[:, 0], xy[:, 1], dvals, 40, cmap='coolwarm')
        fig.colorbar(t, ax=ax[0])
        ax[0].set_xlabel('y'); ax[0].set_ylabel('z')
        ax[0].set_title(f'{exp}: Delta = (1/S) dw/dy')
    sel = np.abs(xy[:, 1] - zs) < tolz
    order = np.argsort(xy[sel, 0])
    S = np.exp(xy[sel, 0][order] + rc * xy[sel, 1][order])
    ax[1].plot(S, dvals[sel][order], 'o-', ms=4, label=exp)

ax[1].set_xlabel('S'); ax[1].set_ylabel('Delta')
ax[1].legend(); ax[1].set_title(f'Delta vs S along z={zs} — band ends compared')
plt.tight_layout()
tag = exps[0].replace('_sup', '').replace('_inf', '')   # e.g. vanilla_power
outfile = f'/shared/delta_comparison_{tag}.png'
plt.savefig(outfile, dpi=150)
print('saved', outfile)
