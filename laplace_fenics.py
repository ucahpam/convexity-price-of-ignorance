from dolfin import (UnitSquareMesh, FunctionSpace, Expression, DirichletBC,
                    TrialFunction, TestFunction, Function,
                    dot, grad, dx, solve, errornorm)

n = 32
mesh = UnitSquareMesh(n, n)
V = FunctionSpace(mesh, 'CG', 1)

u_exact = Expression('sin(pi*x[0])*sin(pi*x[1])', degree=4)
f       = Expression('2*pi*pi*sin(pi*x[0])*sin(pi*x[1])', degree=4)

bc = DirichletBC(V, u_exact, lambda x, on_b: on_b)

u, v = TrialFunction(V), TestFunction(V)
a = dot(grad(u), grad(v)) * dx
L = f * v * dx

u_h = Function(V)
solve(a == L, u_h, bc)
print('n =', n, '  L2 error =', errornorm(u_exact, u_h, 'L2'))

import matplotlib
matplotlib.use('Agg')          
import matplotlib.pyplot as plt
from dolfin import plot

plt.figure(figsize=(7,5))
p = plot(u_h)
plt.colorbar(p)
plt.title(f'Laplace solution, n={n}')
plt.savefig('/shared/laplace_solution.png', dpi=150)
print('saved /shared/laplace_solution.png')