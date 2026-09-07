import os, pathlib

OPTION = 'power'          # 'call' | 'put' | 'power'
WORST_CASE = False

if OPTION == 'call':
    from parameters_vanilla_call import Parameters
elif OPTION == 'put':
    from parameters_vanilla_put import Parameters
elif OPTION == 'power':
    from parameters_power_call import Parameters
else:
    raise ValueError(OPTION)

EXPERIMENT_NAME = f'vanilla_{OPTION}_{"sup" if WORST_CASE else "inf"}'
from PDE_Solver.hjb_mixed import FBVP, Solver
os.chdir(pathlib.Path(__file__).parent.absolute())

if __name__ == '__main__':
    for domain in Parameters.domains:
        for mesh_name in Parameters.meshes:
            print(f'Running {EXPERIMENT_NAME}: {domain}/{mesh_name}')
            par = Parameters(domain, mesh_name, EXPERIMENT_NAME)
            fbvp = FBVP(mesh_name, par)
            solver = Solver(fbvp, howard_inf=WORST_CASE, get_error=False)
            solver.time_iter()
    print(f'Done: {EXPERIMENT_NAME}')