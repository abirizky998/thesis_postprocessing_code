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

# Physical Constants
c = 345.0
rho = 1.2
r_obs = 7.28
n_observer = 36

# Kinematics (Full 72-Source Disk)
n_source = 9
n_blades = 3
r_eff = 1.3         
f_fundamental = 107.25
Mr = 0.0

# Extract ONLY the fundamental harmonic (n=1)
amp_target = 3.39017904
phase_target = -0.53425042
omega = 2 * np.pi * f_fundamental

output_dir = f"data_{date}/{n_source}Sources_Fundamental_Sweep"
os.makedirs(output_dir, exist_ok=True)

# 1. Setup Observers (Full Ring)
observer_setup(observer, D_obs=r_obs, circle=1.0, n=n_observer, normal='z')

# 2. Build the Source File Once (Geometry and Kinematics are constant)
lines = [
    "# Automatically generated 72-Source Array (Fundamental Harmonic Only)\n",
    "# Number of source sets\n",
    f"{n_source}\n"
]

for s in range(n_source):
    # Full 360-degree sweep (5-degree spacing)
    theta_s = s * (2 * np.pi / n_source)
    x_b = r_eff * np.cos(theta_s)
    y_b = r_eff * np.sin(theta_s)

    # Source strength divided equally among 72 physical points
    amp_s = amp_target / n_source
    strength = (4 * np.pi * r_obs * amp_s * (1 - Mr)) / omega
    
    # Tyler-Sofrin stagger for fundamental (n=1)
    phase_shift = (phase_target - (1 * n_blades * theta_s)) % (2 * np.pi)

    lines.extend([
        "# Source type\n",
        "Monopole Harmonic\n",
        "# Source frequency [rad/s], source strength, and source phase [rad]\n"
        f"{omega:.6f} {strength:.8e} {phase_shift:.6f}\n",
        "# Source definition\n",
        "Point\n",
        "# Source initial location [m] (x, y, z)\n",
        f"{x_b:.6f} {y_b:.6f} 0.000000\n",
        "# Source motion type and duration [s]\n",
        "Stationary 1000.0\n"
    ])

with open(source, 'w') as f:
    f.writelines(lines)

# 3. Parameter Sweep Setup
x_values = [10.0, 20.0] # 40.0
p_values = [10.0, 20.0] # 10 30 40

# Calculate Simulation Time (Acoustic delay + 1 full mechanical rotation)
T_sim = 1 / f_fundamental

total_runs = len(x_values) * len(p_values)
current_run = 1

# 4. Execution Loop
for x in x_values:
    n = grid_sizing(config, M=0.0, X=x, lambda_s=c / f_fundamental, R=(r_eff + r_obs)/2)
    # if n * n > 1e5: processes = 4
    # elif n * n > 5e4: processes = 2
    # else: processes = 1
    processes = 8
    for p in p_values:
        print(f"--- Running Case {current_run}/{total_runs} | X = {x}, P = {p} ---")
        
        # Update config file
        timestep_sizing(config, M=0.0, P=p, f_s=f_fundamental, T_sim=T_sim)
        
        # Execute solver
        subprocess.run(["mpirun", "-np", str(processes), "../bin/FWH_Sphere"], check=True)

        time.sleep(10)
        
        # Post-Process and Rename
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

print("\n--- Full Parameter Sweep Complete ---")