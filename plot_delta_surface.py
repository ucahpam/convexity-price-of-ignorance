import glob
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa
import numpy as np
from dolfin import (Mesh, XDMFFile, FunctionSpace, Function, project,
                    Expression)

rho = 0.5
rc = rho / np.sqrt(1 - rho**2)

mesh = Mesh()
with XDMFFile(sorted(glob.glob('meshes/square/*005*.xdmf'))[0]) as f:
    f.read(mesh)
V = FunctionSpace(mesh, 'CG', 1)
xy = mesh.coordinates()
tri = mesh.cells()
S_inv = Expression('exp(-(x[0] + rc*x[1]))', degree=2, rc=rc)

def delta_vals(exp):
    u = Function(V)
    for name in ['value_func', 'v', 'u', 'w']:
        try:
            with XDMFFile(f'out/{exp}/square/005/v.xdmf') as f:
                f.read_checkpoint(u, name, -1)
            return project(S_inv * u.dx(0), V).compute_vertex_values(mesh)
        except Exception:
            continue
    raise SystemExit(f'could not read {exp}')

fig = plt.figure(figsize=(14, 6))
for i, (base, label) in enumerate([('vanilla_call', 'call'),
                                   ('vanilla_power', 'power call')]):
    dd = delta_vals(f'{base}_sup') - delta_vals(f'{base}_inf')
    print(f'{label}: Delta difference range [{dd.min():.5f}, {dd.max():.5f}]')
    ax = fig.add_subplot(1, 2, i+1, projection='3d')
    ax.plot_trisurf(xy[:, 0], xy[:, 1], tri, dd, cmap='coolwarm',
                    linewidth=0.1, antialiased=True)
    ax.set_xlabel('y'); ax.set_ylabel('z')
    ax.set_zlabel('Delta_sup - Delta_inf')
    ax.set_title(f'({chr(97+i)}) {label}', fontsize=11)
plt.suptitle('Delta difference d(V_sup - V_inf)/dS at t=0 '
             '(paper Fig 7 analogue)', fontsize=12)
plt.tight_layout()
plt.savefig('/shared/delta_diff_surfaces.png', dpi=150)
print('saved /shared/delta_diff_surfaces.png')
