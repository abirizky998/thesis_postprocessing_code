from auto_test import *
import numpy as np
import os
import subprocess
import shutil
import time

config = 'input_config.txt'
source = 'input_source.txt'
observer = 'input_observer.txt'
date = "260902"#current_date()

c = 340.0
rho = 1.2
r_obs = 7.28
n_observer = 1
x = 15
p = 80
# processes=64

B = 12
V = 8
# r_eff = 2.1355 #m, fore rotor radius
r_eff = 3.96 / 2

# f_rot = 16 # Hz
rotor_speed_rpm = 983 
Omega_rot = 2 * np.pi * rotor_speed_rpm / 60
f_rot = rotor_speed_rpm / 60

# Interaction frequency in front rotor's reference frame is omega = B_aft * (Omega_fore + Omega_aft)
# the interaction frequency becomes the f_source in question
omega_source = V * Omega_rot
# f_source = V * 2 * Omega_rot
# omega_source = 2 * np.pi * f_source
f_ratio = omega_source / Omega_rot

planes = ['z']
observer_setup_scatter(observer, D_obs=r_obs, n_observer=n_observer, normals=planes)

# Account for advanced time algorithm
T_sim = 4 / f_rot

# Usually 2 x BPF1 + 1 x BPF2 is the inter-rotor frequency of interest
interaction_frequency = (2 * B * f_rot) 

n = grid_sizing(config, M=0.0, X=x, lambda_s=c/interaction_frequency, R=1.3 * r_eff)
timestep_sizing(config, M=0.0, P=p, f_s=interaction_frequency, T_sim=T_sim)

output_dir = f"data_{date}/OpenFan_XPSweep_NEW"
os.makedirs(output_dir, exist_ok=True)

lines = [
    f"# Automatically generated {B}-Source\n",
    "# Number of source sets\n",
    f"{B}\n"
]

for s in range(B):
    theta_s = s * (2 * np.pi / B)
    x_b = r_eff * np.cos(theta_s)
    y_b = r_eff * np.sin(theta_s)
  
    phase_shift = V * theta_s

    lines.extend([
        "# Source type\n",
        "Monopole Harmonic\n",
        "# Source frequency [rad/s], source strength, and source phase [rad]\n",
        f"{omega_source:.6f} 1.0 {phase_shift:.6f}\n",
        "# Source definition\n",
        "Point\n",
        "# Source initial location [m] (x, y, z)\n",
        f"{x_b:.6f} {y_b:.6f} 0.000000\n",
        "# Source motion type and duration [s]\n",
        "General 1000.0\n",
        "# Source Motion Definition\n",
        f"0.0 0.0 0.0 0.0 0.0 {Omega_rot}\n"
    ])

with open(source, 'w') as f:
    f.writelines(lines)

print(f"--- Running Open Rotor Case | Grid Count = {n ** 2} | Timestep Count = {T_sim * p * interaction_frequency} ---")
# subprocess.run(["mpirun", "-np", str(processes), "../bin/FWH_Sphere"], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
subprocess.run(['../bin/FWH_Sphere'], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


for i, plane in enumerate(planes, start=1):
    obs_str = f"{i:02d}"
    ana_file = f"FWH_observer_time_{obs_str}_validation.dat"
    # num_file = f'FWH_observer_time_{obs_str}_AT.dat'
    
    # Create subdirectories for each plane so the sliced microphone files don't overwrite each other
    plane_dir = os.path.join(output_dir, f"plane_{plane}")
    os.makedirs(plane_dir, exist_ok=True)
    
    # Slice the master file into the respective plane's folder
    slice_scatter_output(ana_file, plane_dir, prefix='analytical')
    # slice_scatter_output(num_file, plane_dir, prefix='numerical')
    
    # Move the master file into the main output directory
    ana_dest_file = os.path.join(output_dir, f"OpenRotor_ScatterMaster_{plane}_Analytical.dat")
    # num_dest_file = os.path.join(output_dir, f"OpenRotor_ScatterMaster_{plane}_Numerical.dat") 

    shutil.move(ana_file, ana_dest_file)
    # shutil.move(num_file, num_dest_file)

print("Sweep Complete.")