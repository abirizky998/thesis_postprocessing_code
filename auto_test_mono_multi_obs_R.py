from auto_test import *

### Moving Monopole, Multi-Observer, R/lambda = 1 ### 
config = 'input_config.txt'
source = 'input_source.txt'
observer = 'input_observer.txt'

date = current_date()
X_targets = [10.0, 12.5, 15.0, 17.5, 20.0, 22.5, 25.0]
P_targets = [20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0]
M = 0.9  # Moving Source
m_str = f"{int(M * 100):02d}" 

c = 340.0
f_source = 100.0 # Hz
omega = 2 * np.pi * f_source

lambda_s = c / f_source
R = lambda_s * (1 - M)
T_sim = (1 + M) / f_source

ux = source_speed_omega(source, M, c, omega)
observer_distance = ux * T_sim + 50.0

observers_list = ['01', '02', '03', '04', '05', '06', '07', '08']
observer_setup(observer, D_obs=observer_distance, n=len(observers_list))

output_dir = f"data_{date}/multi_mono_{m_str}M_{int(f_source)}hz"

for obs in observers_list:
    os.makedirs(os.path.join(output_dir, f"obs_{obs}"), exist_ok=True)

for x in X_targets[::-1]:
    N = grid_sizing(config, M, x, lambda_s, R)

    elements = N * N
    if elements >= 1e6: processes = 8
    elif elements >= 5e5: processes = 8
    elif elements >= 1e5: processes = 4
    else: processes = 1
    
    for p in P_targets[::-1]:
        dt = timestep_sizing(config, M, p, f_source, T_sim)

        print(f"Running Case: P = {p}, X = {x}, M = {M}, Grid Count = {N*N}, Time Step Size = {dt:.6f}, Timestep Count = {np.ceil(T_sim / dt)}")
        subprocess.run(["mpirun", "-np", str(processes), "../bin/FWH_Sphere"], check=True)

        new_filename = f"P{int(p)}_X{int(x)}_M{m_str}.dat"

        for obs in observers_list:
            solver_file = f"FWH_observer_time_{obs}_RT.dat"
            dest_dir = os.path.join(output_dir, f"obs_{obs}")
            shutil.move(solver_file, os.path.join(dest_dir, new_filename))

            if p == P_targets[-1] and x == X_targets[-1]:
                ana_file = f"FWH_observer_time_{obs}_validation.dat"
                if os.path.exists(ana_file):
                    shutil.move(ana_file, os.path.join(dest_dir, f"ana_M{m_str}.dat"))

print("Batch Processing Complete")