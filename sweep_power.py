"""
sweep_power.py: band width versus payoff convexity (Section 6.2).
For each p in [1.0, 1.5, 2.0, 3.0] (p=1 is the vanilla call, anchoring
the family), solves the worst and best case problems with the powered
payoff Lambda = max(S-K,0)^p / K^(p-1) and records the peak band
V_sup - V_inf. Produces Figure 6.3, the band-versus-convexity curve.
Run: docker exec -it -w /opt/FEISol feisol-demos python3 sweep_power.py
"""
import glob
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from dolfin import Mesh, XDMFFile, FunctionSpace, Function

import parameters_power_general as mpow
from PDE_Solver.hjb_mixed import FBVP, Solver

P_LIST = [1.0, 1.5, 2.0, 3.0]

mesh = Mesh()
with XDMFFile(sorted(glob.glob('meshes/square/*005*.xdmf'))[0]) as f:
    f.read(mesh)
V = FunctionSpace(mesh, 'CG', 1)

def read_final(exp):
    u = Function(V)
    for name in ['value_func', 'v', 'u', 'w']:
        try:
            with XDMFFile(f'out/{exp}/square/005/v.xdmf') as f:
                f.read_checkpoint(u, name, -1)
            return u.compute_vertex_values(mesh)
        except Exception:
            continue
    raise SystemExit(f'could not read {exp}')

def run(name, sup):
    par = mpow.Parameters('square', '005', name)
    fbvp = FBVP('005', par)
    Solver(fbvp, howard_inf=sup, get_error=False).time_iter()
    return read_final(name)

bands = []
for p in P_LIST:
    mpow.P_DEGREE = p
    sup = run(f'swP_p{p}_sup', True)
    inf = run(f'swP_p{p}_inf', False)
    band = float((sup - inf).max())
    bands.append(band)
    print(f'p={p:.1f}  peak band = {band:.5f}')

fig, ax = plt.subplots(figsize=(7.5, 5.2))
ax.plot(P_LIST, bands, 'o-')
ax.set_xlabel('payoff degree p   [ Lambda = max(S-K,0)^p / K^(p-1) ]')
ax.set_ylabel('peak of V_sup - V_inf')
ax.set_title('Band width vs payoff convexity — the pattern plot')
ax.grid(alpha=.3)
plt.tight_layout()
plt.savefig('/shared/sweep_power.png', dpi=150)
print('saved /shared/sweep_power.png')
print('table (p, peak band, ratio to p=1):')
for p, b in zip(P_LIST, bands):
    print(f'  {p:.1f}  {b:.5f}  x{b/bands[0]:.2f}')