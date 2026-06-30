from auto_test import *

### Stationary Single Monopole, R/lambda = 1 ### 
config = 'input_config.txt'
source = 'input_source.txt'
observer = 'input_observer.txt'
default_solver_output = 'FWH_observer_time_01_RT.dat' 
analytical_output = 'FWH_observer_time_01_validation.dat'

date = current_date()
X_targets = [20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0]
P_targets = [20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0]
M = 0.0
m_str = f"{int(M * 100):02d}" 

c = 340.0
f_source = 1
omega = 2 * np.pi * f_source
# omega = 100 # rad/s
# f_source = omega / (2 * np.pi) # Hz

ux = source_speed_omega(source, M, c, omega)

R = c / f_source # R = lambda
T_sim = 1 / f_source

output_dir = f"data_{date}/single_mono_stationary_noncompact_1hz"
os.makedirs(output_dir, exist_ok=True)

for x in X_targets:
    N = grid_sizing(config, M, x, (c / f_source), R)

    if N * N >= 1e6:
        processes = 8
    elif N * N >= 5e5:
        processes = 4
    elif N * N >= 1e5:
        processes = 2
    else:
        processes = 1
    
    for p in P_targets:
        dt = timestep_sizing(config, M, p, f_source, T_sim)
        print(f"Running Case: P = {p}, X = {x}, M = {M}, Grid Count = {N*N}, Time Step Size = {dt:.6f}, Timestep Count = {np.ceil(T_sim / dt)}")

        subprocess.run(["mpirun", "-np", str(processes), "../bin/FWH_Sphere"], check=True)

        new_filename = f"P{int(p)}_X{int(x)}_M{m_str}.dat"

        destination = os.path.join(output_dir, new_filename)
        shutil.move(default_solver_output, destination)

        if p == P_targets[-1] and x == X_targets[-1]:
            ana_filename = f"ana_M{m_str}.dat"
            ana_destination = os.path.join(output_dir, ana_filename)
            shutil.move(analytical_output, ana_destination)

print("Batch Processing Complete")