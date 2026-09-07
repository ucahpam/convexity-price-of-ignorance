"""band_peak.py -- read the call band peak for a GIVEN mesh resolution.
Reads the sup and inf checkpoints from the mesh-specific output directory
(not the hardcoded 005), so it works for the convergence study.

Run inside the container, passing the mesh name (01, 005, 0025, ...):
    docker exec -it -w /opt/FEISol feisol-demos python3 /shared/band_peak.py 01
"""
import sys, glob
import numpy as np
from dolfin import Mesh, XDMFFile, FunctionSpace, Function

mesh_name = sys.argv[1] if len(sys.argv) > 1 else '005'

# load the matching mesh
mesh = Mesh()
mesh_file = f'meshes/square/square_{mesh_name}.xdmf'
with XDMFFile(mesh_file) as f:
    f.read(mesh)
V = FunctionSpace(mesh, 'CG', 1)

def load(exp):
    u = Function(V)
    path = f'out/{exp}/square/{mesh_name}/v.xdmf'
    for name in ['value_func', 'v', 'u', 'w']:
        try:
            with XDMFFile(path) as f:
                f.read_checkpoint(u, name, -1)
            return u.compute_vertex_values(mesh)
        except Exception:
            continue
    raise SystemExit(f'could not read {path}')

sup = load('vanilla_call_sup')
inf = load('vanilla_call_inf')
band = sup - inf
print(f'mesh {mesh_name}: call band peak = {band.max():.5f}  '
      f'(min {band.min():.5f}, {len(band)} nodes)')
