from auto_test import *

### P ANALYSIS ###
mach_numbers = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
cfl_targets = [0.25, 0.5, 0.75, 1.0, 1.25, 1.5]

P_target = 15.0
R = 20.0
speed_of_sound = 340.0
omega = 10.0
f_source = omega / (2 * np.pi)
T_sim = 2.5

config_path = 'input_config.txt'
source_path = 'input_source.txt'
default_solver_output = 'FWH_observer_time_01_RT.dat' 
analytical_output = 'FWH_observer_time_01_validation.dat'
output_dir = 'data_260528/test_cfl_PPP10/'

os.makedirs(output_dir, exist_ok=True)

for M in mach_numbers:
    ux = M * speed_of_sound
    source_val = f"{ux}, 0.0, 0.0, 0.0, 0.0, 0.0"
    edit_line(source_path, 14, source_val)

    m_str = f"{int(M * 100):02d}"
    dt = (1 - M) / (P_target * f_source)
    N_step = int(np.ceil(T_sim / dt))

    config_val_time = f"{N_step}, {dt:.6f}"
    edit_line(config_path, 14, config_val_time)

    for cfl in cfl_targets:
        X_target = cfl * P_target
        dx = speed_of_sound * (1 - M) / (X_target * f_source)
        N = int(np.round(2 * np.pi * R / dx))
        config_val_spatial = f"{R}, {N}, {N}"
        edit_line(config_path, 6, config_val_spatial)

        print(f"Running Case= M={M:.2f} | CFL={cfl} | N={N} | dt={dt:.6f}")

        if M == 0.95:
            processes = 2
        # elif M >= 0.75:
            # processes = 2
        else:
            processes = 1

        subprocess.run(["mpirun", "-np", str(processes), "../bin/FWH_Sphere"], check=True)

        cfl_str = f"{int(cfl * 100):03d}"
        new_filename = f"cfl{cfl_str}_M{m_str}.dat"
        destination = os.path.join(output_dir, new_filename)
        shutil.move(default_solver_output, destination)

        if cfl == 0.25:
            ana_filename = f"ana_M{m_str}.dat"
            ana_destination = os.path.join(output_dir, ana_filename)
            shutil.move(analytical_output, ana_destination)

print("Batch processing complete.")