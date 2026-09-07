"""
band_peak.py: reads the peak call band at a given mesh resolution, for
the mesh-refinement convergence study (Section 5.1). The mesh name is
passed as an argument, and the checkpoints are read from the matching
resolution's output directory, so the same script serves every mesh in
the refinement sequence.
Run inside the container, passing the mesh name (01, 005, 0025, ...):
    docker exec -it -w /opt/FEISol feisol-demos python3 band_peak.py 01
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
    # Read the final-time (t=0) value function from the run's checkpoint
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
    
# Peak band is the largest pointwise gap between worst and best case
sup = load('vanilla_call_sup')
inf = load('vanilla_call_inf')
band = sup - inf
print(f'mesh {mesh_name}: call band peak = {band.max():.5f}  '
      f'(min {band.min():.5f}, {len(band)} nodes)')
