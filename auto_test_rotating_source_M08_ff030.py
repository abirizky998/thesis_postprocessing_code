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
r_obs = 7.28
n_observer = 3
# processes = 8

n_source = 1
r_eff = 1.3 
mach_number = 0.8
Omega_rot = c * mach_number / r_eff
freq_rot = Omega_rot / (2 * np.pi)
freq_ratio = 3.0

omega_har = Omega_rot * freq_ratio
freq_har = omega_har / (2 * np.pi)

output_dir = f"data_{date}/M{mach_number}_fratio{freq_ratio}"
os.makedirs(output_dir, exist_ok=True)
observer_setup(observer, D_obs=r_obs, circle=1.0, n=n_observer, normal='z')

lines = [
    "# Single Rotating Harmonic Monopole\n",
    "# Number of source sets\n",
    f"{n_source}\n",
    "# Source type\n",
    "Monopole Harmonic\n",
    "# Source frequency [rad/s], source strength, and source phase [rad]\n"
    f"{omega_har} 1.0 0.0\n",
    "# Source definition\n",
    "Point\n",
    "# Source initial location [m] (x, y, z)\n",
    f"{r_eff:.6f} 0.00000 0.000000\n",
    "# Source motion type and duration [s]\n",
    "Rotation 1000.0\n",
    "Rotation angular velocity\n",
    f"0.0 0.0 {Omega_rot:.6f}"
]

with open(source, 'w') as f:
    f.writelines(lines)

x_values = [20.0, 40.0, 60.0, 80.0, 100.0]#[20.0, 40.0, 60.0, 80.0, 100.0] # 40.0
p_values = [20.0, 40.0, 60.0, 80.0, 100.0] # 10 30 40


T_sim = 1/(min(freq_har, freq_rot))

total_runs = len(x_values) * len(p_values)
current_run = 1

for x in x_values[::-1]:
    n = grid_sizing(config, M=0.0, X=x, lambda_s=c / freq_har, R=1.3 * r_eff)
    if n**2 >= 1e5: processes = 8
    elif n**2 >= 5e4: processes = 6
    elif n**2 >= 2e4: processes = 4
    elif n**2 >= 1e4: processes = 2
    else: processes = 1
    for p in p_values[::-1]:
        print(f"--- Running Case {current_run}/{total_runs} | X = {x}, P = {p} --- | Grid Count = {n**2} Time Step Count = {T_sim * p * freq_har}")
        
        # Update config file
        timestep_sizing(config, M=0.0, P=p, f_s=freq_har, T_sim=T_sim)
        
        # Execute solver
        subprocess.run(["mpirun", "-np", str(processes), "../../bin/FWH_Sphere"], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

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