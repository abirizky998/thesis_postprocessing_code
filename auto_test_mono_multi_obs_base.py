from auto_test import *

### Multiple Observer Test Case ###
date = "260610"

M = 0.0
X_targets = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0]
P_targets = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0]

R = 20.0
speed_of_sound = 340.0
omega = 10.0
f_source = omega / (2 * np.pi)

config_path = 'input_config.txt'
source_path = 'input_source.txt'

observers = ['01', '02', '03', '04', '05', '06', '07', '08']
base_output_dir = f'data_{date}/dual_mono/'

output_dir = f'data_{date}/multi_obs/'
os.makedirs(output_dir, exist_ok=True)

for obs in observers:
    os.makedirs(os.path.join(base_output_dir, f"obs_{obs}"), exist_ok=True)

ux = M * speed_of_sound
source_speed = f"{ux} 0.0 0.0 0.0 0.0 0.0"
edit_line(source_path, 14, source_speed)
T_sim = 75.0 / ux if ux > 0.0 else 2.5
m_str = f"{int(M * 100):02d}"

for x in X_targets:
    dx = speed_of_sound * (1 - M) / (x * f_source)
    N = int(np.round(2 * np.pi * R / dx))

    config_val_spatial = f"{R}, {N}, {N}"
    edit_line(config_path, 6, config_val_spatial)

    elements = N * N
    if elements >= 1e6:
        processes = 8
    elif elements >= 5e5:
        processes = 4
    elif elements >= 1e5:
        processes = 2
    else:
        processes = 1

    for p in P_targets:
        dt = (1 - M) / (p * f_source)
        total_steps = int(np.ceil(T_sim / dt))

        config_val_time = f"{total_steps} {dt:.6f}"
        edit_line(config_path, 14, config_val_time)

        cfl = x / p
        print(f"Running Case: M={M:.2f} | X/P={cfl} | N={N} | dt={dt:.6f}")

        subprocess.run(["mpirun", "-np", str(processes), "../bin/FWH_Sphere"], check=True)

        new_filename = f"P{int(p)}_X{int(x)}_M{m_str}.dat"

        for obs in observers:
            solver_file = f"FWH_observer_time_{obs}_RT.dat"
            dest_dir = os.path.join(base_output_dir, f"obs_{obs}")
            destination = os.path.join(dest_dir, new_filename)

            if os.path.exists(solver_file):
                shutil.move(solver_file, destination)

        if p == P_targets[-1] and x==X_targets[-1]:
            for obs in observers:
                ana_file = f"FWH_observer_time_{obs}_validation.dat"
                dest_dir = os.path.join(base_output_dir, f"obs_{obs}")
                ana_destination = os.path.join(dest_dir, f"ana_M{m_str}.dat")

                if os.path.exists(ana_file):
                    shutil.copy(ana_file, ana_destination)

print("Batch Processing Complete")