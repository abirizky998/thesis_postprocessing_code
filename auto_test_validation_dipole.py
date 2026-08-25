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
r_obs = 10
n_observer = 1
processes = 1

n_source = 2
# r_eff = 0.0
mach_number = 0.0
ux = 5
Omega_rot = 0
# Omega_rot = c * mach_number / r_eff
# freq_rot = Omega_rot / (2 * np.pi)
# freq_ratio = 12.0

# omega_har = Omega_rot * freq_ratio
# freq_har = omega_har / (2 * np.pi)
omega_har = 10 # rad/s
freq_har = omega_har / (2 * np.pi)
T_sim = 3/freq_har
r_eff = 0.005 * ((c / freq_har) / (2 *np.pi))

output_dir = f"data_{date}/dual_monopole_directivity_translating"
os.makedirs(output_dir, exist_ok=True)
# observer_setup(observer, D_obs=r_obs, circle=1.0, n=n_observer, normal='z')

lines = [
    "# Single Harmonic Monopole\n",
    "# Number of source sets\n",
    f"{n_source}\n",
    "# Source type\n",
    "Monopole Harmonic\n",
    "# Source frequency [rad/s], source strength, and source phase [rad]\n"
    f"{omega_har} 1.0 0.0\n",
    "# Source definition\n",
    "Point\n",
    "# Source initial location [m] (x, y, z)\n",
    f"0.00000 {r_eff:.6f} 0.000000\n",
    "# Source motion type and duration [s]\n",
    "General 1000.0\n",
    "#Translational and Rotational Speed\n",
    f"{ux:6f} 0.0 0.0 0.0 0.0 {Omega_rot:.6f}\n",
    "# Source type\n",
    "Monopole Harmonic\n",
    "# Source frequency [rad/s], source strength, and source phase [rad]\n"
    f"{omega_har} -1.0 0.0\n",
    "# Source definition\n",
    "Point\n",
    "# Source initial location [m] (x, y, z)\n",
    f"0.00000 {- r_eff:.6f} 0.000000\n",
    "# Source motion type and duration [s]\n",
    "General 1000.0\n",
    "#Translational and Rotational Speed\n",
    f"{ux:6f} 0.0 0.0 0.0 0.0 {Omega_rot:.6f}"
]

with open(source, 'w') as f:
    f.writelines(lines)

# p_values = [100, 170, 275] # 10 30 40
p_values = [30]
x_values = [275] # 40.0 

total_runs = len(x_values) * len(p_values)
current_run = 1

for x in x_values:
    n = grid_sizing(config, M=0.0, X=x, lambda_s=c / freq_har, R=5.0)
    for p in p_values:
        print(f"--- Running Case {current_run}/{total_runs} | X = {x}, P = {p} --- | Grid Count = {n**2} Time Step Count = {T_sim * p * freq_har}")
        
        # Update config file
        timestep_sizing(config, M=0.0, P=p, f_s=freq_har, T_sim=T_sim)
        processes = 6
        # Execute solver
        subprocess.run(["mpirun", "-np", str(processes), "../bin/FWH_Sphere"], check=True)#, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        # time.sleep(10)
        
        # Post-Process and Rename
        for i in range(1, n_observer + 1):
            obs_str = f"{i:02d}"
            
            # The analytical file doesn't care about X, but we save it once as a baseline
            ana_file = f"FWH_observer_time_{obs_str}_validation.dat" 
            ana_dest_file = os.path.join(output_dir, f"ana_spatial_{obs_str}.dat")
            
            if current_run == total_runs and os.path.exists(ana_file):
                shutil.move(ana_file, ana_dest_file)
            elif os.path.exists(ana_file):
                os.remove(ana_file) # Delete duplicates from subsequent runs
            
            # Save the numerical file with X and P tags
            num_file = f'FWH_observer_time_{obs_str}_AT.dat'
            num_dest_file = os.path.join(output_dir, f"X{int(x)}_P{int(p)}_{obs_str}.dat") 
            
            if os.path.exists(num_file):
                shutil.move(num_file, num_dest_file)
                
        current_run += 1

print("\n--- Full Parameter Sweep Complete ---")