import os
import shutil
import subprocess
import numpy as np
import datetime

def edit_line(filepath, line_idx, new_value):
    with open(filepath, 'r') as f:
        lines = f.readlines()
    lines[line_idx] = f"{new_value}\n"
    with open(filepath, 'w') as f:
        f.writelines(lines)

def current_date():
    return datetime.date.today().strftime('%y%m%d')

def timestep_sizing(config_path, M, P, f_s, T_sim):
    dt = (1 - M) / (P * f_s)
    total_step = int(np.ceil(T_sim / dt))
    config_val_time = f"{total_step} {dt:.8e}"
    edit_line(config_path, 14, config_val_time)
    return dt

def grid_sizing(config_path, M, X, lambda_s, R):
    dx = (1 - M) * lambda_s / X
    N = int(np.round(2 * np.pi * R / dx))
    config_val_spatial = f"{R} {N} {N}"
    edit_line(config_path, 6, config_val_spatial)
    return N

def source_speed_omega(config_path, M, speed_of_sound, omega, strength, Omega_rot=0.0, delta=0.0, n_sources=1, is_dipole=False):
    ux = M * speed_of_sound
    motion_vector = f"{ux:.6f} 0.0 0.0 {Omega_rot:.6f} 0.0 0.0"
    
    # -----------------------------------------------------------------
    # SCENARIO 1: The Dipole (Requires 2 separate source blocks because 
    # the amplitude changes from 1.0 to -1.0)
    # -----------------------------------------------------------------
    if is_dipole:
        lines = [
            "# Automatically generated source file\n",
            "# Number of source sets\n",
            "2\n",
            
            # Source 1 (Positive)
            "# Source 1 type\n",
            "Monopole Harmonic\n",
            "# Source frequency [rad/s], source strength, and source phase [rad]\n",
            f"{omega:.6f} 1.0 0.0\n",
            "# Source definition\n",
            "Point\n",
            "# Source initial location [m] (x, y, z)\n",
            f"0.0 {delta:.6f} 0.0\n",
            "# Source motion type and duration [s]\n",
            "General 1000.0\n",
            "# Source translational velocity vector and rotation rate vector\n",
            f"{motion_vector}\n",
            
            # Source 2 (Negative)
            "# Source 2 type\n",
            "Monopole Harmonic\n",
            "# Source frequency [rad/s], source strength, and source phase [rad]\n",
            f"{omega:.6f} -1.0 0.0\n",
            "# Source definition\n",
            "Point\n",
            "# Source initial location [m] (x, y, z)\n",
            f"0.0 {-delta:.6f} 0.0\n",
            "# Source motion type and duration [s]\n",
            "General 1000.0\n",
            "# Source translational velocity vector and rotation rate vector\n",
            f"{motion_vector}\n"
        ]
        
    # -----------------------------------------------------------------
    # SCENARIO 2: Multiple Monopoles (Uses 'Scatter' to define a single 
    # source block with multiple radiating points)
    # -----------------------------------------------------------------
    else:
        # Map sources in a circle on the Y-Z plane
        angles = np.linspace(0, 2 * np.pi, n_sources, endpoint=False)
        y_coords = delta * np.cos(angles)
        z_coords = delta * np.sin(angles)
        
        # Determine definition type based on count
        definition_type = "Point" if n_sources == 1 else "Scatter"
        
        lines = [
            "# Automatically generated source file\n",
            "# Number of source sets (Always 1 for grouped monopoles)\n",
            "1\n",
            
            "# Source type\n",
            "Monopole Harmonic\n",
            "# Source frequency [rad/s], source strength, and source phase [rad]\n",
            f"{omega:.6f} {strength:.6f} 0.0\n",
            "# Source definition\n",
            f"{definition_type}\n"
        ]
        
        # If Scatter, we must provide the integer count before the coordinates
        if n_sources > 1:
            lines.extend([
                "# Number of scattered points\n",
                f"{n_sources}\n"
            ])
            
        lines.append("# Source initial location(s) [m] (x, y, z)\n")
        
        # Append the coordinate block
        for i in range(n_sources):
            lines.append(f"0.0 {y_coords[i]:.6f} {z_coords[i]:.6f}\n")
            
        lines.extend([
            "# Source motion type and duration [s]\n",
            "General 1000.0\n",
            "# Source translational velocity vector and rotation rate vector\n",
            f"{motion_vector}\n"
        ])

    # Write to file
    with open(config_path, 'w') as f:
        f.writelines(lines)
        
    return ux

def observer_setup(config_path, D_obs, n, circle = 1.0, is_dipole=False, offset_deg=1.0, normal='z'):
    angles = np.linspace(0, circle * 2 * np.pi, n, endpoint=False)

    if is_dipole:
        offset_rad = np.radians(offset_deg)
        for i in range(len(angles)):
            if np.isclose(angles[i], 0.0) or np.isclose(angles[i], np.pi):
                angles[i] += offset_rad
    if normal == 'z':
        x = D_obs * np.cos(angles)
        y = D_obs * np.sin(angles)
    elif normal == 'x':
        y = D_obs * np.cos(angles)
        z = D_obs * np.sin(angles)
    elif normal == 'y':
        x = D_obs * np.cos(angles)
        z = D_obs * np.sin(angles)

    lines = [
        "# Automatically generated observer file\n",
        "# N observer sets\n",
        f"{n}\n"
    ]

    for i in range(n):
        if normal == 'z':
            lines.extend([
                f"# obs {i+1}\n",
                "Point\n",
                "# initial loc\n",
                f"{x[i]:.6f} {y[i]:.6f} 0.0\n",
                "# motion type and duration\n",
                "Stationary 1000.0\n"
            ])
        elif normal == 'x':
            lines.extend([
                f"# obs {i+1}\n",
                "Point\n",
                "# initial loc\n",
                f"0.0 {y[i]:.6f} {z[i]:.6f}\n",
                "# motion type and duration\n",
                "Stationary 1000.0\n"
            ])
        elif normal == 'y':
            lines.extend([
                f"# obs {i+1}\n",
                "Point\n",
                "# initial loc\n",
                f"{x[i]:.6f} 0.0 {z[i]:.6f}\n",
                "# motion type and duration\n",
                "Stationary 1000.0\n"
            ])

    with open(config_path, 'w') as f:
        f.writelines(lines)

def naca0012_thickness(x_norm, t_max):
    # Analytical thickness distribution for NACA 0012
    # x_norm is the normalized chord position from 0 (LE) to 1 (TE)
    return 5.0 * t_max * (0.2969 * np.sqrt(x_norm) - 0.1260 * x_norm - 
                          0.3516 * x_norm**2 + 0.2843 * x_norm**3 - 0.1015 * x_norm**4)

def source_setup_chordwise_array(config_path, R_p, eta, chord, beta_c_deg, t_max, n_pts, n_blades, Omega_x):
    r_eta = R_p * eta
    beta_c = np.radians(beta_c_deg)
    
    s_array = np.linspace(-0.5, 0.5, n_pts)
    ds = 1.0 / (n_pts - 1)
    
    lines = [
        "# Automatically generated source file\n",
        "# Number of source sets\n",
        f"{n_pts * n_blades}\n"
    ]
    
    for b in range(n_blades):
        phi_blade = b * (2 * np.pi / n_blades) # 0, 120, 240 degrees
        
        for s in s_array:
            # Map s [-0.5, 0.5] to x_norm [0, 1] for NACA polynomial
            x_norm = s + 0.5
            thickness = naca0012_thickness(x_norm, t_max)
            amplitude = thickness * ds  # Volume represented by this point
            
            # Local coordinates prior to rotation
            # Twist beta_c rotates the chord line around the spanwise Y-axis
            x_local = -s * chord * np.sin(beta_c)
            y_local = r_eta
            z_local = s * chord * np.cos(beta_c)
            
            # Global rotation around X-axis for blade positioning
            x_global = x_local
            y_global = y_local * np.cos(phi_blade) - z_local * np.sin(phi_blade)
            z_global = y_local * np.sin(phi_blade) + z_local * np.cos(phi_blade)
            
            lines.extend([
                "# Source type\n",
                "Monopole Harmonic\n",
                "# Source frequency [rad/s], source strength, and source phase [rad]\n",
                f"0.0 {amplitude:.8e} 1.570796327\n",  # omega=0, phase=pi/2
                "# Source definition\n",
                "Point\n",
                "# Source initial location [m] (x, y, z)\n",
                f"{x_global:.6f} {y_global:.6f} {z_global:.6f}\n",
                "# Source motion type and duration [s]\n",
                "General 1000.0\n",
                "# Source translational velocity vector and rotation rate vector\n",
                f"0.0 0.0 0.0 {Omega_x:.6f} 0.0 0.0\n"
            ])
            
    with open(config_path, 'w') as f:
        f.writelines(lines)