import glob
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from dolfin import Mesh, XDMFFile, FunctionSpace, Function

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
            return u.compute_vertex_values(mesh)
        except Exception:
            continue
    raise SystemExit(f'could not read {exp}')

pairs = {'call':  ('vanilla_call_sup',  'vanilla_call_inf'),
         'power': ('vanilla_power_sup', 'vanilla_power_inf')}

fig, ax = plt.subplots(1, 2, figsize=(13, 5))
for i, (label, (sup, inf)) in enumerate(pairs.items()):
    band = load(sup) - load(inf)
    print(f'{label}: band range [{band.min():.5f}, {band.max():.5f}]')
    t = ax[i].tricontourf(xy[:, 0], xy[:, 1], band, 40, cmap='magma')
    fig.colorbar(t, ax=ax[i])
    ax[i].set_xlabel('y'); ax[i].set_ylabel('z')
    ax[i].set_title(f'{label}: V_sup - V_inf  (uncertainty premium)')

plt.tight_layout()
plt.savefig('/shared/uncertainty_bands.png', dpi=150)
print('saved /shared/uncertainty_bands.png')
