from auto_test import *

### P vs X Matrix Gen ###
### NOT DOPPLER SHIFTED ###
M = 0.7
X_targets = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0]
P_targets = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0]

speed_of_sound = 340.0
omega = 10.0
R = 20.0
f_source = omega / (2 * np.pi)
T_sim = 2.5            

config_path = 'input_config.txt'
source_path = 'input_source.txt'
default_solver_output = 'FWH_observer_time_01_RT.dat' 
analytical_output = 'FWH_observer_time_01_validation.dat'
output_dir = 'data_260610/single_mono/'
os.makedirs(output_dir, exist_ok=True)

ux = M * speed_of_sound
source_val = f"{ux}, 0.0, 0.0, 0.0, 0.0, 0.0"
edit_line(source_path, 14, source_val) 
m_str = f"{int(M * 100):02d}" 

for x in X_targets:
    dx = speed_of_sound / (x * f_source)
    N = int(np.round((2 * np.pi * R) / dx))
    
    config_val_spatial = f"{R}, {N}, {N}" 
    edit_line(config_path, 6, config_val_spatial) 
    
    for p in P_targets:
        dt = 1 / (p * f_source)
        total_steps = int(np.ceil(T_sim / dt))
        
        config_val_time = f"{total_steps} {dt:.6f}"
        edit_line(config_path, 14, config_val_time)
        cfl = x / p
        
        print(f"Running Case: M={M:.2f} | CFL={cfl} | N={N} | dt={dt:.6f}")
        
        elements = N * N
        if elements > 1e6:
            processes = 8
        elif elements > 2.5e5:
            processes = 4
        elif elements > 1e5:
            processes = 2
        else:
            processes = 1

        subprocess.run(["mpirun", "-np", str(processes), "../bin/FWH_Sphere"], check=True) 
        
        # cfl_str = f"{int(cfl * 100):03d}"
        new_filename = f"P{int(p)}_X{int(x)}_M{m_str}.dat"
        destination = os.path.join(output_dir, new_filename)
        shutil.move(default_solver_output, destination)

        if p == P_targets[-1] and x == X_targets[0]:
            ana_filename = f"ana_M{m_str}.dat"
            ana_destination = os.path.join(output_dir, ana_filename)
            shutil.copy(analytical_output, ana_destination)

print("Batch processing complete.")
