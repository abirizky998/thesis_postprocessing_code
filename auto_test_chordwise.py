from auto_test import *
import numpy as np
import os
import subprocess

config = 'input_config.txt'
source = 'input_source.txt'
observer = 'input_observer.txt'
date = current_date()

# Domain / Kinematics
M = 0.0 
c = 340.0 
# f_rot_x = 100.0 # Hz
Omega_x = 71.5 # rad/s # 2 * np.pi * f_rot_x
T_sim = 2 * np.pi / Omega_x # Run 2 rotations to clear retarded time transients

# Rotor Geometry Constants
R_p = 1.3          # Propeller radius [m]
eta = 0.8          # Dominant radial station
chord = 0.1        # Chord length at eta [m]
beta_c_deg = 45.0  # Twist angle [deg]
t_max = 0.01       # Maximum thickness [m]
n_blades = 3
n_chord_pts = 50   # Resolution of the chord line

# 1. Setup Source Array
source_setup_chordwise_array(
    config_path=source, 
    R_p=R_p, eta=eta, chord=chord, 
    beta_c_deg=beta_c_deg, t_max=t_max, 
    n_pts=n_chord_pts, n_blades=n_blades, 
    Omega_x=Omega_x
)

output_dir = f"data_{date}/chordwise_{n_blades}blades_{n_chord_pts}pts_{int(Omega_x)}Hz"
os.makedirs(output_dir, exist_ok=True)

# 2. Setup Observer Ring (Normal to X-axis)
n_observer = 12
observer_distance = 50.0
observer_setup(observer, D_obs=observer_distance, n=n_observer, normal='x')

# 3. Timestep Sizing
# Resolution is tied to the Blade Passing Frequency (BPF), not rotation frequency
f_bpf = n_blades * Omega_x
dt = timestep_sizing(config_path=config, M=M, P=40.0, f_s=f_bpf, T_sim=T_sim)

print(f"Chordwise array initialized: {n_blades} blades, {n_chord_pts} points/blade.")
print(f"Timestep: {dt:.8e} s, Total simulation time: {T_sim} s")

subprocess.run(["mpirun", "-np", "8", "../bin/FWH_Sphere"]) # Execute solver