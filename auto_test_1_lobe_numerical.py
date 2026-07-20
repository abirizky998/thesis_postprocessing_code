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
n_observer = 4
n_source = 1
r_eff = 1.5 # change this later for offset case
freq = 100 # hz
omega = 2 * np.pi * freq

output_dir = f"data_{date}/1lobe-offset"
os.makedirs(output_dir, exist_ok=True)

observer_setup(observer, D_obs=r_obs, n=n_observer, normal='z')
timestep_sizing(config, M=0.0, P=10, f_s=freq, T_sim=1/freq)

lines = [
    "# Automatically generated 3-Blade Spatial Harmonic Cluster\n",
    "# Number of source sets\n",
    f"{n_source}\n"
]

for n in range(n_source):
    x_b = r_eff * np.cos((2*n-1) * np.pi / n_source)
    y_b = r_eff * np.sin((2*n-1) * np.pi / n_source)

    strength_n = 1.0
    phase_shift = 0

    lines.extend([
        "# Source type\n",
        "Monopole Harmonic\n",
        "# Source frequency [rad/s], source strength, and source phase [rad]\n",
        f"{omega:.6f} {strength_n:.8e} {phase_shift:.6f}\n",
        "# Source definition\n",
        "Point\n",
        "# Source initial location [m] (x, y, z)\n",
        f"{x_b:.6f} {y_b:.6f} 0.0\n",
        "# Source motion type and duration [s]\n",
        "Stationary 1000.0\n"
    ])

with open(source, 'w') as f:
    f.writelines(lines)

x_values = [10.0]
p_values = [10.0]
total_runs = len(x_values) * len(p_values)
current_run = 1

for x in x_values:
    n = grid_sizing(config, M=0.0, X=x, lambda_s=c/freq, R=(r_eff + r_obs) / 2)
    processes = 8
    for p in p_values:
        # Update config file
        timestep_sizing(config, M=0.0, P=p, f_s=freq, T_sim=1/freq)
        
        # Execute solver
        subprocess.run(["mpirun", "-np", str(processes), "../bin/FWH_Sphere"], check=True)

        time.sleep(10)

        for i in range(1, n_observer + 1):
            obs_str = f"{i:02d}"
            
            # The analytical file doesn't care about X, but we save it once as a baseline
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