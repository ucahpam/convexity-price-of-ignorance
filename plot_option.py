"""
plot_option.py -- plot a FEISol option run from its checkpoint file.
Run INSIDE the container from /opt/FEISol:
  docker exec -it -w /opt/FEISol feisol-demos python3 /shared/plot_option.py vanilla_call_sup
"""
import sys, glob
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from dolfin import Mesh, XDMFFile, FunctionSpace, Function

EXP = sys.argv[1] if len(sys.argv) > 1 else 'vanilla_call_sup'
CHECKPOINT = f'out/{EXP}/square/005/v.xdmf'

# --- load the same mesh the solver used ---
mesh_candidates = sorted(glob.glob('meshes/square/*005*.xdmf'))
if not mesh_candidates:
    raise SystemExit('No square/005 mesh found - are you in /opt/FEISol?')
mesh = Mesh()
with XDMFFile(mesh_candidates[0]) as f:
    f.read(mesh)
V = FunctionSpace(mesh, 'CG', 1)
u = Function(V)

# --- read the LAST saved checkpoint (= smallest time, the price today) ---
loaded = False
for name in ['value_func', 'v', 'u', 'w']:
    try:
        with XDMFFile(CHECKPOINT) as f:
            f.read_checkpoint(u, name, -1)   # -1 = last written
        print(f'Loaded checkpoint field "{name}"')
        loaded = True
        break
    except Exception:
        continue
if not loaded:
    raise SystemExit('Could not read checkpoint - tell Claude the field name '
                     'from: h5ls out/' + EXP + '/square/005/v.h5')

vals = u.compute_vertex_values(mesh)
xy = mesh.coordinates()
print(f'value range: [{vals.min():.4f}, {vals.max():.4f}]')

# --- figure 1: the value surface over (y, z) ---
fig, ax = plt.subplots(1, 2, figsize=(13, 5))
t0 = ax[0].tricontourf(xy[:, 0], xy[:, 1], vals, 40, cmap='viridis')
fig.colorbar(t0, ax=ax[0])
ax[0].set_xlabel('y'); ax[0].set_ylabel('z')
ax[0].set_title(f'{EXP}: option value at t=0 over (y, z)')

# --- figure 2: slice along z = 0 (there S = e^y) vs the payoff ---
K, rho = 1.60, 0.5
rc = rho / np.sqrt(1 - rho**2)
zs = 0.5                                   # interior slice, away from pinned edges
edge = np.abs(xy[:, 1] - zs) < 0.03
y_edge = xy[edge, 0]; z_edge = xy[edge, 1]; v_edge = vals[edge]
order = np.argsort(y_edge)
S = np.exp(y_edge[order] + rc * z_edge[order])   # true S at each node
ax[1].plot(S, v_edge[order], 'o-', label='option value (t=0, v=0 edge)')
Sgrid = np.linspace(S.min(), S.max(), 200)
if 'put' in EXP:
    payoff = np.maximum(K - Sgrid, 0); plab = 'payoff max(K-S,0)'
elif 'power' in EXP:
    payoff = np.maximum(Sgrid - K, 0)**2 / K; plab = 'payoff max(S-K,0)^2/K'
else:
    payoff = np.maximum(Sgrid - K, 0); plab = 'payoff max(S-K,0)'
ax[1].plot(Sgrid, payoff, 'k--', label=plab)
ax[1].set_xlabel('S'); ax[1].set_ylabel('value')
ax[1].legend(); ax[1].set_title('value vs payoff along z=0')

plt.tight_layout()
out = f'/shared/{EXP}.png'
plt.savefig(out, dpi=150)
print('saved', out)