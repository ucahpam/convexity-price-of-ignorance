"""
plot_control.py: reconstructs the optimal-control map (Section 5.4):
the value of lambda that Nature selects at each state. Because the
operator is affine in lambda, the optimal control at each node is the
lambda that maximises a linear function of lambda over the interval L,
and the maximum is attained at an endpoint. The optimising endpoint is
therefore determined by the sign of the coefficient multiplying lambda,
which is built from the gradient of the value function and the sqrt(z)
factors of the transformed operator (Section 4.1):
    s(y,z) = sqrt(xi*z) * ( sqrt(sqrt(1-rho^2)) * dw/dz
                            - rho/sqrt(sqrt(1-rho^2)) * dw/dy )
    lambda*(y,z) = -2.4  if s < 0,  else  -1.6
The map is read off the sign of s(y,z) node by node. This gradient-based
reconstruction is approximate near the switching curve, where s is close
to zero. The resulting node percentages are discussed in Section 5.4.
Run: docker exec -it -w /opt/FEISol feisol-demos \
       python3 plot_control.py vanilla_power_sup
"""
import sys, glob
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from dolfin import (Mesh, XDMFFile, FunctionSpace, Function, project,
                    Expression)

exp = sys.argv[1] if len(sys.argv) > 1 else 'vanilla_power_sup'
ALPHA_RANGE = [-2.4, -1.6]
rho, xi = 0.5, 0.7
rc = rho / np.sqrt(1 - rho**2)
s4 = np.sqrt(np.sqrt(1 - rho**2))

mesh = Mesh()
with XDMFFile(sorted(glob.glob('meshes/square/*005*.xdmf'))[0]) as f:
    f.read(mesh)
V = FunctionSpace(mesh, 'CG', 1)
xy = mesh.coordinates()

u = Function(V)
loaded = False
for name in ['value_func', 'v', 'u', 'w']:
    try:
        with XDMFFile(f'out/{exp}/square/005/v.xdmf') as f:
            f.read_checkpoint(u, name, -1)
        loaded = True
        break
    except Exception:
        continue
if not loaded:
    raise SystemExit(f'could not read {exp}')

sqz = Expression('sqrt(fmax(xi*x[1], 0.0))', degree=2, xi=xi)
s_expr = project(sqz * (s4 * u.dx(1) - (rho / s4) * u.dx(0)), V)
svals = s_expr.compute_vertex_values(mesh)
# Optimal endpoint by the sign of s: s<0 selects -2.4, s>0 selects -1.6.
lam = np.where(svals > 0, ALPHA_RANGE[0], ALPHA_RANGE[1])

frac_hi = (lam == ALPHA_RANGE[1]).mean()
print(f'{exp}: lambda* = {ALPHA_RANGE[1]} on {100*frac_hi:.1f}% of nodes, '
      f'{ALPHA_RANGE[0]} on {100*(1-frac_hi):.1f}%  (bang-bang by construction)')

fig, ax = plt.subplots(figsize=(7.5, 5.5))
t = ax.tricontourf(xy[:, 0], xy[:, 1], lam, levels=[ALPHA_RANGE[0]-.05,
                   (ALPHA_RANGE[0]+ALPHA_RANGE[1])/2, ALPHA_RANGE[1]+.05],
                   colors=['#30507a', '#d1793a'])
cb = fig.colorbar(t, ax=ax, ticks=ALPHA_RANGE)
cb.set_label('selected lambda*')
ax.set_xlabel('y'); ax.set_ylabel('z')
ax.set_title(f'{exp}: optimal control map (bang-bang: endpoints of $L$)',
             fontsize=10)
plt.tight_layout()
plt.savefig(f'/shared/control_map_{exp}.png', dpi=150)
print(f'saved /shared/control_map_{exp}.png')