from auto_test import *
import numpy as np
import os
import subprocess
import shutil
import time

config = 'input_config.txt'
source = 'input_source.txt'
observer = 'input_observer.txt'
date = current_date()

c = 340.0
rho = 1.2
r_obs = 50.0
n_observer = 72
f_base = 107.25 # 1x BPF

# --- INSERT YOUR 12-HARMONIC DFT ARRAYS HERE ---
# Make sure these arrays only contain 12 elements (no DC offset)
amps = np.array([3.39017904, 4.41553483, 4.33133909, 4.05853316, 3.68401838,
       3.00466103, 2.60251833, 2.20761803, 1.81316128, 1.49356841,
       1.56660969, 1.11126176])
phases = np.array([-0.53425042,  2.19246152, -1.4319258 ,  1.29567134, -2.37311503,
        0.30186215,  2.99472323, -0.62744329,  2.08961611, -1.36787151,
        1.35893209, -2.542671]) 

output_dir = f"data_{date}/Compact_Point_Rotor_Test"
os.makedirs(output_dir, exist_ok=True)

# Observer Ring Setup
observer_setup(observer, D_obs=r_obs, n=n_observer, normal='z')
timestep_sizing(config, M=0.0, P=1000.0, f_s=len(amps) * f_base, T_sim=3/f_base)

lines = [
    "# Automatically generated 3-Blade Spatial Harmonic Cluster\n",
    "# Number of source sets\n",
    f"{len(amps)}\n"
]

for idx in range(len(amps)):
    n = idx + 1
    omega_n = 2 * np.pi * n * f_base
    strength_n = (4 * np.pi * r_obs * amps[idx]) / (omega_n)
    phase_n = phases[idx]

    lines.extend([
        "# Source type\n",
        "Monopole Harmonic\n",
        "# Source frequency [rad/s], source strength, and source phase [rad]\n",
        f"{omega_n:.6f} {strength_n:.8e} {phase_n:.6f}\n",
        "# Source definition\n",
        "Point\n",
        "# Source initial location [m] (x, y, z) - Stationary at origin\n",
        "0.000000 0.000000 0.000000\n",
        "# Source motion type and duration [s]\n",
        "Stationary 1000.0\n"
    ])

with open(source, 'w') as f:
    f.writelines(lines)

x_vals = [20.0, 40.0, 60.0, 80.0]
p_vals = [20.0, 40.0, 60.0, 80.0]

total_runs = len(x_vals) * len(p_vals)
current_run = 1

for x in x_vals:
    n = grid_sizing(config, M=0.0, X=x, lambda_s=c / f_base, R = (r_obs)/2)
    processes = 64

    for p in p_vals:
        print(f"--- Running Case {current_run}/{total_runs} | X = {x}, P = {p} ---")
        timestep_sizing(config, M=0.0, P=p, f_s=f_base, T_sim=3/f_base)

        subprocess.run(['mpirun', '-np', str(processes), '../bin/FWH_Sphere'], check=True)
        time.sleep(10)

        for i in range(1, n_observer + 1):
            obs_str = f"{i:02d}"
            
            ana_file = f"FWH_observer_time_{obs_str}_validation.dat" 
            ana_dest_file = os.path.join(output_dir, f"ana_spatial_{obs_str}.dat")
            
            if current_run == 1 and os.path.exists(ana_file):
                shutil.move(ana_file, ana_dest_file)
            elif os.path.exists(ana_file):
                os.remove(ana_file) # Delete duplicates from subsequent runs
            
            # Save the numerical file with X and P tags
            num_file = f'FWH_observer_time_{obs_str}_RT.dat'
            num_dest_file = os.path.join(output_dir, f"X{int(x)}_P{int(p)}_{obs_str}.dat") 
            
            if os.path.exists(num_file):
                shutil.move(num_file, num_dest_file)

        current_run += 1

print("Spatial Simulation Complete.")