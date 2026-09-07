"""
sweep_interval.py: band width versus the width of the uncertainty
interval L (Section 6.1). For each half-width h in [0.1, 0.2, 0.4, 0.8],
sets L = [-2-h, -2+h] (centred on Doran's lambda = -2) and solves the
worst and best case problems for both the vanilla call and the p=2
powered call, recording the peak band. Since the operator is affine in
lambda, the band is expected to grow approximately linearly through the
origin in the half-width; the sweep tests whether this holds. Produces
Figure 6.1.
Run: docker exec -it -w /opt/FEISol feisol-demos python3 sweep_interval.py
"""
import glob
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from dolfin import Mesh, XDMFFile, FunctionSpace, Function

import parameters_vanilla_call as mcall
import parameters_power_general as mpow
from PDE_Solver.hjb_mixed import FBVP, Solver

HALF_WIDTHS = [0.1, 0.2, 0.4, 0.8]
CENTRE = -2.0 # Interval centred on Doran's lambda estimate

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

def run(module, name, sup):
    par = module.Parameters('square', '005', name)
    fbvp = FBVP('005', par)
    Solver(fbvp, howard_inf=sup, get_error=False).time_iter()
    return read_final(name)

results = {'call': [], 'power': []}
for h in HALF_WIDTHS:
    L = [CENTRE - h, CENTRE + h]
    for tag, module in [('call', mcall), ('power', mpow)]:
        module.ALPHA_RANGE = L
        if module is mpow:
            module.P_DEGREE = 2.0
        sup = run(module, f'swL_{tag}_h{h}_sup', True)
        inf = run(module, f'swL_{tag}_h{h}_inf', False)
        band = float((sup - inf).max())
        results[tag].append(band)
        print(f'{tag:5s}  halfwidth={h:.1f}  L={L}  peak band = {band:.5f}')

fig, ax = plt.subplots(figsize=(7.5, 5.2))
for tag, marker in [('call', 'o-'), ('power', 's-')]:
    ax.plot(HALF_WIDTHS, results[tag], marker, label=f'{tag} (peak band)')
ax.set_xlabel('half-width of uncertainty interval L (centred at -2)')
ax.set_ylabel('peak of V_sup - V_inf')
ax.set_title('Band width vs interval width — linear-through-origin test')
ax.legend(); ax.grid(alpha=.3)
plt.tight_layout()
plt.savefig('/shared/sweep_interval.png', dpi=150)
print('saved /shared/sweep_interval.png')
print('table (halfwidth, call, power):')
for h, c, p in zip(HALF_WIDTHS, results['call'], results['power']):
    print(f'  {h:.1f}  {c:.5f}  {p:.5f}  ratio {p/c:.2f}')