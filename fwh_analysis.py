import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import os
from pathlib import Path
import scipy

class FWHDataset:
    X_TARGET = 20.0
    P_TARGET = 15.0
    R = 20.0              
    C = 340.0
    OMEGA = 10.0
    F_SOURCE = OMEGA / (2 * np.pi)
    LAMBDA_SOURCE = C / F_SOURCE

    def __init__(self, filepath, name=None):
        self.filepath = Path(filepath)
        self.name = name or self.filepath.name
        self.data = self._load_data()
        self.metadata = self._parse_metadata()

    def _load_data(self):
        try:
            df = pd.read_csv(self.filepath, sep=r'\s+', header=None, comment='#')
            if df.iloc[0].apply(lambda x: isinstance(x, str)).any():
                df = df.iloc[1:].reset_index(drop=True)
                df = df.apply(pd.to_numeric)
        except Exception:
            df = pd.read_csv(self.filepath, sep=r'\s+', skiprows=1, header=None, comment='#')

        if df.shape[1] == 3:
            df.columns = ['Time','P_obs','P_analytical']
        elif df.shape[1] == 2:
            df.columns = ['Time', 'P_obs']

        return df
    
    def _parse_metadata(self):
        dx = self.LAMBDA_SOURCE / self.X_TARGET
        meta = {
            'mach': None, 'mach_str': None, 'radius': self.R, 
            'grid': None, 'time_lbl': None, 'cfl': None,
            'dx': dx, 'dt': None
        }
        
        # 1. NEW MATRIX FORMAT: Pxx_Xxx_Mxx.dat
        px_mach_match = re.search(r"P(\d+)_X(\d+)_M(\d+)", self.name)
        if px_mach_match:
            p_val = float(px_mach_match.group(1))
            x_val = float(px_mach_match.group(2))
            meta['mach_str'] = px_mach_match.group(3)
            meta['mach'] = float(meta['mach_str']) / 100.0
            
            meta['cfl'] = x_val / p_val
            meta['dx'] = self.C * (1 - meta['mach']) / (x_val * self.F_SOURCE)
            meta['dt'] = (1 - meta['mach']) / (p_val * self.F_SOURCE)
            meta['grid'] = int(np.round((2 * np.pi * self.R) / meta['dx']))
            return meta

        # 2. OLD CFL FORMAT: cflXXX_MXX.dat
        cfl_mach_match = re.search(r"cfl(\d+)_M(\d+)", self.name)
        if cfl_mach_match:
            meta['cfl'] = float(cfl_mach_match.group(1)) / 100.0
            meta['mach_str'] = cfl_mach_match.group(2)
            meta['mach'] = float(meta['mach_str']) / 100.0 
            meta['dt'] = (meta['cfl'] * dx) / self.C
            meta['grid'] = int(np.round((2 * np.pi * self.R) / dx))
            return meta
        
        # 3. OLD RADIUS FORMAT: rXXX_MXX.dat
        rad_mach_match = re.search(r"r(\d+)_M(\d+)\.dat", self.name)
        if rad_mach_match:
            meta['radius'] = float(rad_mach_match.group(1))
            meta['mach_str'] = rad_mach_match.group(2)
            meta['mach'] = float(meta['mach_str']) / 100.0
            meta['dx'] = self.C * (1 - meta['mach']) / (self.X_TARGET * self.F_SOURCE)
            meta['grid'] = int(np.round((2 * np.pi * meta['radius']) / meta['dx']))
            meta['cfl'] = 1.0 
            meta['dt'] = (meta['cfl'] * meta['dx']) / self.C
            return meta
            
        return meta

    def l2_error(self, ref_data, col='P_obs'):
        ref_df = ref_data.data if hasattr(ref_data, 'data') else ref_data

        if len(self.data) != len(ref_df):
            t_fine = ref_df['Time'].values
            t_coarse = self.data['Time'].values
            p_coarse = self.data[col].values
            p_interp = np.interp(t_fine, t_coarse, p_coarse)
            abs_error = np.linalg.norm(ref_df[col].to_numpy() - p_interp)
        else:
            abs_error = np.linalg.norm(ref_df[col].to_numpy() - self.data[col].to_numpy())

        return abs_error / (np.linalg.norm(ref_df[col]) + 1e-15)
    
    def rmse_error(self, ref_data, col='P_obs'):
        ref_df = ref_data.data if hasattr(ref_data, 'data') else ref_data

        if len(self.data) != len(ref_df):
            t_fine = ref_df['Time'].values
            t_coarse = self.data['Time'].values
            p_coarse = self.data[col].values
            p_interp = np.interp(t_fine, t_coarse, p_coarse)
            diff = ref_df[col].to_numpy() - p_interp
        else:
            diff = ref_df[col].to_numpy() - self.data[col].to_numpy()

        return np.sqrt(np.mean(diff**2))
    
    # def db_error(self, ref_data, col='P_obs'):
    #     ref_df = ref_data.data if hasattr(ref_data, 'data') else ref_data

    #     if len(self.data) != len(ref_df):
    #         t_fine = ref_df['Time'].values
    #         t_coarse = self.data['Time'].values
    #         p_coarse = self.data[col].values
    #         p_interp = np.interp(t_fine, t_coarse, p_coarse)
            
    #         # diff = ref_df[col].to_numpy() - p_interp
    #     else:
    #         # diff = ref_df[col].to_numpy() - self.data[col].to_numpy()
    
    # @staticmethod
    # def compile_fwh_batch(directory, error_method='l2', use_doppler=True, verbose=False):
    #     folder_path = Path(directory)
        
    #     # 1. Load local analytical references
    #     local_refs = {}
    #     for ana_file in folder_path.glob("ana_M*.dat"):
    #         m_match = re.search(r"ana_M(\d+)", ana_file.name)
    #         if m_match:
    #             local_refs[m_match.group(1)] = FWHDataset(filepath=ana_file)
                
    #     if verbose:
    #         print(f"DEBUG: Loaded Analytical References for Machs: {list(local_refs.keys())}")
        
    #     folder_data = []
    #     files_scanned = 0
        
    #     # 2. Process all data files
    #     for filepath in folder_path.glob("*.dat"):
    #         if filepath.name.startswith("ana_") or filepath.name.startswith('P'):
    #             continue
                
    #         files_scanned += 1
    #         case = FWHDataset(filepath=filepath)
    #         meta = case.metadata
            
    #         if meta.get('cfl') is None:
    #             if verbose: print(f"DEBUG: Skipped {filepath.name} -> Could not parse CFL/Metadata.")
    #             continue
                
    #         ref_key = meta['mach_str']
            
    #         if ref_key not in local_refs:
    #             if verbose: print(f"DEBUG: Skipped {filepath.name} -> Ref key '{ref_key}' not found.")
    #             continue 
                
    #         ref_df = local_refs[ref_key]
    #         if error_method == 'l2':
    #             error = case.l2_error(ref_df, col='P_obs')
    #         elif error_method == 'rmse':
    #             error = case.rmse_error(ref_df, col='P_obs')

    #         if len(case.data) > 1:
    #             actual_dt = case.data['Time'].iloc[1] - case.data['Time'].iloc[0]
    #         else:
    #             actual_dt = meta['dt'] 
                
    #         actual_dx = (case.C * actual_dt) / meta['cfl']
    #         true_N = int(np.round((2 * np.pi * case.R) / actual_dx))
    #         true_elements = true_N * true_N
            
    #         if use_doppler:
    #             p_val = (1.0 / actual_dt) / (case.F_SOURCE / (1 - meta['mach']))
    #             x_val = (case.LAMBDA_SOURCE * (1 - meta['mach'])) / actual_dx
    #         else:
    #             p_val = (1.0 / actual_dt) / case.F_SOURCE
    #             x_val = case.LAMBDA_SOURCE / actual_dx
                
    #         folder_data.append({
    #             'mach': meta['mach'],
    #             'cfl': meta['cfl'],
    #             'dt': actual_dt,
    #             'P': p_val,
    #             'X': x_val,
    #             'error': error,
    #             'elements': true_elements,
    #             'batch_name': folder_path.name
    #         })
            
    #     if verbose:
    #         print(f"\nDEBUG: Scanned {files_scanned} files. Successfully compiled {len(folder_data)} files.")
            
    #     return folder_data
    @staticmethod
    def compile_fwh_batch(directory, error_method='l2', use_doppler=True, verbose=False):
        folder_path = Path(directory)
        local_refs = {}
        
        # 1. Load Analytical References (Handles Old and New Formats)
        for ana_file in folder_path.glob("ana_*.dat"):
            # Match old format: ana_M00.dat
            m_match = re.search(r"ana_M(\d+)", ana_file.name)
            if m_match:
                local_refs[m_match.group(1)] = FWHDataset(filepath=ana_file)
                
            # Match new format: ana_spatial_01.dat
            obs_match = re.search(r"ana_spatial_(\d+)", ana_file.name)
            if obs_match:
                local_refs[f"obs_{obs_match.group(1)}"] = FWHDataset(filepath=ana_file)
                
        if verbose:
            print(f"DEBUG: Loaded References: {list(local_refs.keys())}")
            
        folder_data = []
        
        # 2. Process Data Files
        for filepath in folder_path.glob("*.dat"):
            if filepath.name.startswith("ana_") or filepath.name.startswith("FWH_"):
                continue
                
            case = FWHDataset(filepath=filepath)
            meta = case.metadata
            
            # Determine File Schema
            new_match = re.search(r"X(\d+(?:\.\d+)?)_P(\d+(?:\.\d+)?)_(\d+)\.dat", filepath.name)
            old_match = re.search(r"P(\d+(?:\.\d+)?)_X(\d+(?:\.\d+)?)_M(\d+)\.dat", filepath.name)
            
            p_val, x_val, mach_val, cfl_val, true_elements, actual_dt = None, None, 0.0, None, None, None
            ref_key, obs_id = None, None
            
            # Schema A: New Multipole Format (X20_P40_01.dat)
            if new_match:
                x_val = float(new_match.group(1))
                p_val = float(new_match.group(2))
                obs_id = new_match.group(3)
                ref_key = f"obs_{obs_id}"
                
                if len(case.data) > 1:
                    actual_dt = case.data['Time'].iloc[1] - case.data['Time'].iloc[0]
                else:
                    actual_dt = 1.0 / (p_val * case.F_SOURCE) 
                    
                true_elements = int((2 * np.pi * case.R * x_val / case.LAMBDA_SOURCE)**2)
                
            # Schema B: Old Fundamental Format (P40_X20_M00.dat)
            elif old_match or meta.get('cfl') is not None:
                cfl_val = meta.get('cfl')
                if cfl_val is None:
                    continue
                    
                ref_key = meta.get('mach_str')
                mach_val = meta.get('mach', 0.0)
                
                if len(case.data) > 1:
                    actual_dt = case.data['Time'].iloc[1] - case.data['Time'].iloc[0]
                else:
                    actual_dt = meta.get('dt', 1e-4)
                    
                actual_dx = (case.C * actual_dt) / cfl_val
                true_elements = int(np.round((2 * np.pi * case.R) / actual_dx))**2
                
                if use_doppler:
                    p_val = (1.0 / actual_dt) / (case.F_SOURCE / (1 - mach_val))
                    x_val = (case.LAMBDA_SOURCE * (1 - mach_val)) / actual_dx
                else:
                    p_val = (1.0 / actual_dt) / case.F_SOURCE
                    x_val = case.LAMBDA_SOURCE / actual_dx
            else:
                continue
                
            # 3. Calculate Error & Append
            if ref_key not in local_refs:
                continue
                
            ref_df = local_refs[ref_key]
            error = case.l2_error(ref_df, col='P_obs') if error_method == 'l2' else case.rmse_error(ref_df, col='P_obs')
            
            folder_data.append({
                'mach': mach_val,
                'cfl': cfl_val,
                'dt': actual_dt,
                'P': p_val,
                'X': x_val,
                'error': error,
                'elements': true_elements,
                'batch_name': folder_path.name,
                'observer': obs_id if obs_id else folder_path.name 
            })
            
        return folder_data

class FWHPlotter:
    @staticmethod
    def plot_pressure(reference, *comparisons, labels=None, title=None, scale=1.0, error_method='l2',
                      ref_col='P_analytical', comp_col='P_obs', legend=True, x_axis='absolute', time_method='RT'):
        
        fig, ax = plt.subplots()
        ref_df = reference.data if hasattr(reference, 'data') else reference

        if x_axis == 'absolute':
            ax.plot(ref_df['Time'], ref_df[ref_col] * scale, ls='-', c='black', label='Analytical Solution')
        elif x_axis == 'normalised':
            ax.plot(ref_df['Time'] / ref_df['Time'].max(), ref_df[ref_col] * scale, ls='-', c='black', label='Analytical Solution', alpha=0.7)

        if not labels or len(labels) != len(comparisons):
            labels = [f'Dataset {i+1}' for i in range(len(comparisons))]

        styles = [':', '-.', '--']
        colors = ['red', 'blue', 'green', 'orange', 'purple', 'cyan']

        for i, comp in enumerate(comparisons):
            comp_df = comp.data if hasattr(comp, 'data') else comp
            
            # Clip the zero-padded buffer for Advanced Time
            if time_method == 'AT':
                t_max = ref_df['Time'].max()
                comp_df = comp_df[comp_df['Time'] <= t_max]

            if error_method == 'l2':
                error = comp.l2_error(reference, col=comp_col)
            elif error_method == 'rmse':
                error = comp.rmse_error(reference, col=comp_col)
            
            ls = styles[i % len(styles)]
            c = colors[i % len(colors)]

            if error_method == 'l2':
                full_label = f'{labels[i]} (Error: {error * 100:.2f}%)' 
            elif error_method == 'rmse':
                full_label = f"{labels[i]} (RMSE: {error * 1e3:.2f}x$10^{{-3}}$ Pa)"

            if x_axis == 'absolute':
                x_data = comp_df['Time']
            elif x_axis == 'normalised':
                x_data = comp_df['Time'] / ref_df['Time'].max()

            ax.plot(x_data, comp_df[comp_col] * scale, ls=ls, c=c, label=full_label)

        if x_axis == 'absolute':
            ax.set_xlabel('Time (s)')
        elif x_axis == 'normalised':
            if ref_df['Time'].max() < 1.0:
                ax.set_xlabel(f't/T, T={(ref_df["Time"].max() * 1000):.3f} ms')
            else:
                ax.set_xlabel(f't/T, T={(ref_df["Time"].max()):.3f} s')

        if scale == 1.0:
            ax.set_ylabel("Pressure (Pa)")
        else:
            scale_str = f"x10^{{{int(np.log10(scale))}}}" if scale >= 10 else f"x{scale}"
            ax.set_ylabel(f'Pressure ({scale_str} Pa)')

        if title:
            ax.set_title(title)

        ax.grid(True, linestyle='--', alpha=0.7)
        if legend: ax.legend(loc='upper right')
        plt.tight_layout()
        plt.show()
    



# LATEST BEFORE ADVANCED TIME UPDATE  
#     @staticmethod
#     def plot_pressure(reference, *comparisons, labels=None, title=None, scale=1.0, error_method='l2',
#                       ref_col='P_analytical', comp_col='P_obs', legend=True, x_axis='absolute'):
        
#         fig, ax = plt.subplots()
#         ref_df = reference.data if hasattr(reference, 'data') else reference

#         if x_axis == 'absolute':
#             ax.plot(ref_df['Time'], ref_df[ref_col] * scale, ls='-', c='black', label=f'Analytical Solution')
#         elif x_axis == 'normalised':
#             ax.plot(ref_df['Time'] / ref_df['Time'].max(), ref_df[ref_col] * scale, ls = '-', c='black', label=f'Analytical Solution')

#         if not labels or len(labels) != len(comparisons):
#             labels = [f'Dataset {i+1}' for i in range(len(comparisons))]

#         styles = [':', '-.', '--']
#         colors = ['red', 'blue', 'green', 'orange', 'purple', 'cyan']

#         for i, comp in enumerate(comparisons):
#             comp_df = comp.data if hasattr(comp, 'data') else comp
#             if error_method == 'l2':
#                 error = comp.l2_error(reference, col=comp_col)
#             elif error_method == 'rmse':
#                 error = comp.rmse_error(reference, col=comp_col)
#             ls = styles[i % len(styles)]
#             c = colors[i % len(colors)]

#             full_label = f'{labels[i]} (Error: {error * 100:.2f}%)'

#             if x_axis == 'absolute':
#                 x_data = comp_df['Time']
#             elif x_axis == 'normalised':
#                 x_data = comp_df['Time'] / ref_df['Time'].max()

#             ax.plot(x_data, comp_df[comp_col] * scale, ls=ls, c=c, label=full_label)

#         if x_axis == 'absolute':
#             ax.set_xlabel('Time (s)')
#         elif x_axis == 'normalised':
#             if ref_df['Time'].max() < 1.0:
#                 ax.set_xlabel(f't/T, T={(ref_df["Time"].max() * 1000):.3f} ms')
#             else:
#                 ax.set_xlabel(f't/T, T={(ref_df["Time"].max()):.3f} s')

#         if scale == 1.0:
#             ax.set_ylabel("Pressure (Pa)")
#         else:
#             scale_str = f"x10^{{{int(np.log10(scale))}}}" if scale >= 10 else f"x{scale}"
#             ax.set_ylabel(f'Pressure ({scale_str} Pa)')

#         if title:
#             ax.set_title(title)

#         ax.grid(True, linestyle='--', alpha=0.7)
#         if legend: ax.legend(loc='upper right')#, bbox_to_anchor=(1.35, 1))
#         plt.tight_layout()
#         plt.show()

    # @staticmethod
    # def plot_directivity_snapshot(datasets, target_time, col='P_obs', is_polar=False, ylim=None, title=None):
    #     import numpy as np
    #     import matplotlib.pyplot as plt

    #     if not datasets:
    #         print("No datasets provided.")
    #         return

    #     n_obs = len(datasets)
    #     # Calculate angles in radians and degrees based on the number of observers
    #     angles_rad = np.linspace(0, 2 * np.pi, n_obs, endpoint=False)
    #     angles_deg = np.degrees(angles_rad)

    #     pressures = []
    #     actual_time = 0.0

    #     # Extract pressure at the closest time for each observer
    #     for ds in datasets:
    #         df = ds.data
    #         # Find the row index where the time is closest to the target_time
    #         idx = (np.abs(df['Time'] - target_time)).argmin()
    #         p_val = df[col].iloc[idx]
    #         pressures.append(p_val)
    #         actual_time = df['Time'].iloc[idx] 

    #     if is_polar:
    #         fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={'projection': 'polar'})
            
    #         # Append the first element to the end to close the circular loop visually
    #         plot_angles = np.append(angles_rad, angles_rad[0])
    #         plot_pressures = np.append(pressures, pressures[0])

    #         ax.plot(plot_angles, plot_pressures, marker='o', linestyle='-', color='b', 
    #                 linewidth=2, markersize=6, label=f'Snapshot at t ≈ {actual_time:.4f} s')
            
    #         # Standard math orientation: 0 degrees at East, counter-clockwise
    #         ax.set_theta_zero_location('E') 
    #         ax.set_title(title or f"Acoustic Directivity Polar Map", pad=20, weight='bold')
            
    #         # Customize radial grids for cleaner pressure reading
    #         ax.grid(True, linestyle='--', alpha=0.7)
    #         ax.legend(loc='upper right', bbox_to_anchor=(1.2, 1.1))

    #     else:
    #         fig, ax = plt.subplots(figsize=(12, 6))
    #         ax.plot(angles_deg, pressures, marker='o', linestyle='-', color='b', 
    #                 linewidth=2, markersize=8, label=f'Snapshot at t ≈ {actual_time:.4f} s')

    #         ax.set_title(title or f"Pressure vs Observer Angle ($\theta$)", weight='bold')
    #         ax.set_xlabel(r"Observer Angle, $\theta$ (Degrees)")
    #         ax.set_ylabel("Pressure (Pa)")

    #         ax.set_ylim(bottom=(-ylim if ylim is not None else None),
    #                     top = (ylim if ylim is not None else none))

    #         # Set x-ticks to match the exact observer angles and label them clearly
    #         ax.set_xticks(angles_deg)
    #         ax.set_xticklabels([f"{int(deg)}°\n(Obs {i+1:02d})" for i, deg in enumerate(angles_deg)])

    #         ax.grid(True, linestyle='--', alpha=0.6)
    #         ax.legend()

    #     plt.tight_layout()
    #     plt.show()

    ### LATEST PRIOR TO LATEST UPDATE
    # @staticmethod
    # def plot_directivity_snapshot(datasets, target_times, col='P_obs', ylim=None, is_polar=False, title=None, plot=True, legend=False):
    #     import numpy as np
    #     import matplotlib.pyplot as plt

    #     if not datasets:
    #         print("No datasets provided.")
    #         return None, None

    #     if np.isscalar(target_times):
    #         target_times = [target_times]

    #     n_obs = len(datasets)
    #     angles_rad = np.linspace(0, 2 * np.pi, n_obs, endpoint=False)
    #     angles_deg = np.degrees(angles_rad)
        
    #     extracted_data = {} # Dictionary to store pressures at each time step

    #     if plot:
    #         if is_polar:
    #             fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={'projection': 'polar'})
    #         else:
    #             fig, ax = plt.subplots(figsize=(8, 6))

    #         num_times = len(target_times)
    #         colors = plt.cm.viridis(np.linspace(0, 0.9, num_times)) if num_times > 1 else ['b']

    #     for i, target_time in enumerate(target_times):
    #         pressures = []
            
    #         idx_0 = (np.abs(datasets[0].data['Time'] - target_time)).argmin()
    #         actual_time = datasets[0].data['Time'].iloc[idx_0]

    #         for ds in datasets:
    #             df = ds.data
    #             idx = (np.abs(df['Time'] - target_time)).argmin()
    #             pressures.append(df[col].iloc[idx])
                
    #         extracted_data[actual_time] = np.array(pressures)

    #         if plot:
    #             label_str = f't ≈ {actual_time:.4f} s'

    #             if is_polar:
    #                 plot_angles = np.append(angles_rad, angles_rad[0])
    #                 plot_pressures = np.append(pressures, pressures[0])
    #                 ax.plot(plot_angles, plot_pressures, marker='o', linestyle='-', 
    #                         color=colors[i], linewidth=2, markersize=5, label=label_str)
    #             else:
    #                 ax.plot(angles_deg, pressures, linestyle='-', 
    #                         color=colors[i], linewidth=2, markersize=6, label=label_str)

    #     if plot:
    #         if is_polar:
    #             ax.set_theta_zero_location('E') 
    #             ax.set_title(title or "Acoustic Directivity Time Evolution", pad=20, weight='bold')
    #             ax.grid(True, linestyle='--', alpha=0.7)
    #             ax.legend(loc='upper right', bbox_to_anchor=(1.35, 1.1))
    #         else:
    #             ax.set_title(title or r"Pressure vs Observer Angle ($\theta$)", weight='bold')
    #             ax.set_xlabel(r"Observer Angle, $\theta$ (Degrees)")
    #             ax.set_ylabel("Pressure (Pa)")
    #             ax.set_ylim(bottom=-ylim if ylim is not None else None,
    #                         top=ylim if ylim is not None else None)
    #             if n_obs <= 12: step = 1
    #             elif n_obs <= 36: step = n_obs // 12
    #             else: step // 8
    #             ax.set_xticks(angles_deg[::step])
    #             ax.grid(True, linestyle='--', alpha=0.6)
    #             ax.legend(loc='center left', bbox_to_anchor=(1, 0.5)) if legend is True else None

    #         plt.tight_layout()
    #         plt.show()
            
    #     return angles_rad, extracted_data

    ### LATEST LATEST!!!
    @staticmethod
    def plot_directivity_snapshot(reference, *comparisons, target_time, col='P_obs', labels=None, 
                                  is_polar=False, title=None, legend=True, ylim=None):
        import numpy as np
        import matplotlib.pyplot as plt
        import matplotlib.lines as mlines

        # Handle scalar vs array inputs
        target_times = [target_time] if np.isscalar(target_time) else target_time

        n_obs = len(reference)
        angles_rad = np.linspace(0, 2 * np.pi, n_obs, endpoint=False)
        angles_deg = np.degrees(angles_rad)
        
        if is_polar:
            fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={'projection': 'polar'})
        else:
            fig, ax = plt.subplots(figsize=(8, 6))

        if not labels or len(labels) != len(comparisons):
            labels = [f'Dataset {i+1}' for i in range(len(comparisons))]

        styles = ['--', '-.', ':']
        
        num_times = len(target_times)
        colors = plt.cm.viridis(np.linspace(0, 0.9, num_times)) if num_times > 1 else ['black']

        time_labels = []
        
        for t_idx, t_val in enumerate(target_times):
            c_time = colors[t_idx]
            
            # 1. Reference Extraction & Plotting
            ref_pressures = []
            idx_0 = (np.abs(reference[0].data['Time'] - t_val)).argmin()
            actual_time = reference[0].data['Time'].iloc[idx_0]
            time_labels.append(f't = {actual_time:.4f} s')

            for ds in reference:
                df = ds.data if hasattr(ds, 'data') else ds
                idx = (np.abs(df['Time'] - t_val)).argmin()
                ref_pressures.append(df[col].iloc[idx])
                
            if is_polar:
                plot_angles = np.append(angles_rad, angles_rad[0])
                plot_ref = np.append(ref_pressures, ref_pressures[0])
                ax.plot(plot_angles, plot_ref, ls='-', c=c_time, lw=2)
            else:
                ax.plot(angles_deg, ref_pressures, ls='-', c=c_time, lw=2)

            # 2. Comparison Extraction & Plotting
            for i, comp in enumerate(comparisons):
                comp_pressures = []
                for ds in comp:
                    df = ds.data if hasattr(ds, 'data') else ds
                    idx = (np.abs(df['Time'] - t_val)).argmin()
                    comp_pressures.append(df[col].iloc[idx])
                
                ls = styles[i % len(styles)]
                
                if is_polar:
                    plot_comp = np.append(comp_pressures, comp_pressures[0])
                    ax.plot(plot_angles, plot_comp, ls=ls, c=c_time, marker='o', markersize=5)
                else:
                    ax.plot(angles_deg, comp_pressures, ls=ls, c=c_time, marker='o', markersize=5)

        # 3. Custom Legend Construction
        if legend:
            # Dataset mappings
            legend_elements = [mlines.Line2D([], [], color='k', ls='-', lw=2, label='Reference')]
            for i, label in enumerate(labels):
                legend_elements.append(mlines.Line2D([], [], color='k', ls=styles[i % len(styles)], marker='o', markersize=5, label=label))
            
            # Time mappings
            if num_times > 1:
                legend_elements.append(mlines.Line2D([], [], color='w', label=' ')) # Spacer
                for i, t_label in enumerate(time_labels):
                    legend_elements.append(mlines.Line2D([], [], color=colors[i], ls='-', lw=2, label=t_label))

            loc = 'upper right' if is_polar else 'center left'
            bbox = (1.35, 1.1) if is_polar else (1.05, 0.5)
            ax.legend(handles=legend_elements, loc=loc, bbox_to_anchor=bbox)

        # 4. Formatting
        title_str = title or "Pressure Lobes"
        if is_polar:
            ax.set_theta_zero_location('E') 
            ax.set_title(title_str, pad=20, weight='bold')
            ax.grid(True, linestyle='--', alpha=0.7)
        else:
            ax.set_title(title_str, weight='bold')
            ax.set_xlabel("Observer Angle (Degrees)")
            ax.set_ylabel("Pressure (Pa)")
            if ylim: ax.set_ylim(bottom=-ylim, top=ylim)
            step = 1 if n_obs <= 12 else (n_obs // 12 if n_obs <= 36 else n_obs // 8)
            ax.set_xticks(angles_deg[::step])
            ax.grid(True, linestyle='--', alpha=0.6)

        plt.tight_layout()
        plt.show()

    ### LATEST
    # @staticmethod
    # def plot_directivity_snapshot(reference, *comparisons, target_time, col='P_obs', labels=None, 
    #                               is_polar=False, title=None, legend=True, ylim=None):
    #     import numpy as np
    #     import matplotlib.pyplot as plt

    #     n_obs = len(reference)
    #     angles_rad = np.linspace(0, 2 * np.pi, n_obs, endpoint=False)
    #     angles_deg = np.degrees(angles_rad)
        
    #     if is_polar:
    #         fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={'projection': 'polar'})
    #     else:
    #         fig, ax = plt.subplots(figsize=(8, 6))

    #     # 1. Extract and Plot Reference
    #     ref_pressures = []
    #     idx_0 = (np.abs(reference[0].data['Time'] - target_time)).argmin()
    #     actual_time = reference[0].data['Time'].iloc[idx_0]

    #     for ds in reference:
    #         df = ds.data if hasattr(ds, 'data') else ds
    #         idx = (np.abs(df['Time'] - target_time)).argmin()
    #         ref_pressures.append(df[col].iloc[idx])
            
    #     if is_polar:
    #         plot_angles = np.append(angles_rad, angles_rad[0])
    #         plot_ref = np.append(ref_pressures, ref_pressures[0])
    #         ax.plot(plot_angles, plot_ref, ls='-', c='black', lw=2, label='Reference')
    #     else:
    #         ax.plot(angles_deg, ref_pressures, ls='-', c='black', lw=2, label='Reference')

    #     # 2. Extract and Plot Comparisons
    #     if not labels or len(labels) != len(comparisons):
    #         labels = [f'Dataset {i+1}' for i in range(len(comparisons))]

    #     styles = ['--', '-.', ':']
    #     colors = ['red', 'blue', 'green', 'orange', 'purple', 'cyan']

    #     extracted_comps = []
    #     for i, comp in enumerate(comparisons):
    #         comp_pressures = []
    #         for ds in comp:
    #             df = ds.data if hasattr(ds, 'data') else ds
    #             idx = (np.abs(df['Time'] - target_time)).argmin()
    #             comp_pressures.append(df[col].iloc[idx])
            
    #         extracted_comps.append(comp_pressures)
            
    #         ls = styles[i % len(styles)]
    #         c = colors[i % len(colors)]
            
    #         if is_polar:
    #             plot_comp = np.append(comp_pressures, comp_pressures[0])
    #             ax.plot(plot_angles, plot_comp, ls=ls, c=c, marker='o', markersize=5, label=labels[i])
    #         else:
    #             ax.plot(angles_deg, comp_pressures, ls=ls, c=c, marker='o', markersize=5, label=labels[i])

    #     # 3. Formatting
    #     if is_polar:
    #         ax.set_theta_zero_location('E') 
    #         ax.set_title(title or f"Acoustic Directivity at t = {actual_time:.4f} s", pad=20, weight='bold')
    #         ax.grid(True, linestyle='--', alpha=0.7)
    #         if legend: ax.legend(loc='upper right', bbox_to_anchor=(1.35, 1.1))
    #     else:
    #         ax.set_title(title or f"Pressure vs Observer Angle at t = {actual_time:.4f} s", weight='bold')
    #         ax.set_xlabel("Observer Angle (Degrees)")
    #         ax.set_ylabel("Pressure (Pa)")
    #         if ylim: ax.set_ylim(bottom=-ylim, top=ylim)
    #         step = 1 if n_obs <= 12 else (n_obs // 12 if n_obs <= 36 else n_obs // 8)
    #         ax.set_xticks(angles_deg[::step])
    #         ax.grid(True, linestyle='--', alpha=0.6)
    #         if legend: ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    #     plt.tight_layout()
    #     plt.show()
        
    #     return angles_rad, ref_pressures, extracted_comps
    # @staticmethod
    # def plot_directivity_snapshot(datasets, target_times, col='P_obs', ylim=None, is_polar=False, title=None):
    #     import numpy as np
    #     import matplotlib.pyplot as plt

    #     if not datasets:
    #         print("No datasets provided.")
    #         return

    #     # Ensure target_times is iterable so older scripts passing a single float don't break
    #     if np.isscalar(target_times):
    #         target_times = [target_times]

    #     n_obs = len(datasets)
    #     angles_rad = np.linspace(0, 2 * np.pi, n_obs, endpoint=False)
    #     angles_deg = np.degrees(angles_rad)

    #     # Setup Canvas
    #     if is_polar:
    #         fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={'projection': 'polar'})
    #     else:
    #         fig, ax = plt.subplots(figsize=(12, 6))

    #     # Generate sequential colors based on how many time steps you want to plot
    #     num_times = len(target_times)
    #     colors = plt.cm.viridis(np.linspace(0, 0.9, num_times)) if num_times > 1 else ['b']

    #     for i, target_time in enumerate(target_times):
    #         pressures = []
            
    #         # Extract actual solver time from the first observer (assumes uniform dt across array)
    #         idx_0 = (np.abs(datasets[0].data['Time'] - target_time)).argmin()
    #         actual_time = datasets[0].data['Time'].iloc[idx_0]

    #         for ds in datasets:
    #             df = ds.data
    #             idx = (np.abs(df['Time'] - target_time)).argmin()
    #             pressures.append(df[col].iloc[idx])

    #         label_str = f't ≈ {actual_time:.4f} s'

    #         if is_polar:
    #             plot_angles = np.append(angles_rad, angles_rad[0])
    #             plot_pressures = np.append(pressures, pressures[0])
    #             ax.plot(plot_angles, plot_pressures, marker='o', linestyle='-', 
    #                     color=colors[i], linewidth=2, markersize=5, label=label_str)
    #         else:
    #             ax.plot(angles_deg, pressures, linestyle='-', 
    #                     color=colors[i], linewidth=2, markersize=6, label=label_str)

    #     # Formatting
    #     if is_polar:
    #         ax.set_theta_zero_location('E') 
    #         ax.set_title(title or "Acoustic Directivity Time Evolution", pad=20, weight='bold')
    #         ax.grid(True, linestyle='--', alpha=0.7)
    #         # Push legend outside the polar plot so it doesn't overlap the lobes
    #         ax.legend(loc='upper right', bbox_to_anchor=(1.35, 1.1))
    #     else:
    #         ax.set_title(title or r"Pressure vs Observer Angle ($\theta$)", weight='bold')
    #         ax.set_xlabel(r"Observer Angle, $\theta$ (Degrees)")
    #         ax.set_ylabel("Pressure (Pa)")
    #         ax.set_ylim(bottom=-ylim if ylim is not None else None,
    #                     top=ylim if ylim is not None else None)
    #         ax.set_xticks(angles_deg)
    #         # ax.set_xticks(angles_deg, rotation=45)
    #         # ax.set_xticklabels([f"{int(deg)}°\n(Obs {j+1:02d})" for j, deg in enumerate(angles_deg)],
    #                         #    rotation=45, ha='right')
    #         ax.grid(True, linestyle='--', alpha=0.6)
    #         # Push legend to the right of the Cartesian plot
    #         ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    #     plt.tight_layout()
    #     plt.show()

    @staticmethod
    def plot_cfl_convergence_by_mach(directory, analysis, analytical_refs, ref_val, error_method='l2', ymin=3e-5, ymax=5e-2, use_doppler=True, legend=True):
        from matplotlib.lines import Line2D
        import matplotlib.cm as cm
        from pathlib import Path
        import re
        import numpy as np
        
        mach_data = {}
        marker_keys = set()

        search_pattern = "r*_M*.dat" if analysis == 'r' else "*.dat"
        fixed_val = ref_val[0]

        for filepath in Path(directory).glob(search_pattern):
            if filepath.name.startswith('ana_'):
                continue

            case = FWHDataset(filepath=filepath)
            meta = case.metadata

            if analysis in ['dt', 'dx'] and meta.get('cfl') is None:
                continue
            if analysis == 'r' and meta.get('radius') is None:
                continue

            # --- 1. FILTERING: Isolate the specific row/column of the parameter matrix ---
            px_match = re.search(r"P(\d+)_X(\d+)_M(\d+)", filepath.name)
            if px_match:
                p_val = float(px_match.group(1))
                x_val = float(px_match.group(2))
                
                # If plotting P (dt), lock X to the fixed_val
                if analysis == 'dt' and x_val != fixed_val:
                    continue
                # If plotting X (dx), lock P to the fixed_val
                if analysis == 'dx' and p_val != fixed_val:
                    continue

            ref_key = meta['mach_str']

            # --- 2. FLEXIBLE REFERENCES: Handle Dict vs Single Object ---
            if isinstance(analytical_refs, dict):
                if ref_key not in analytical_refs:
                    continue
                ref_df = analytical_refs[ref_key]
            else:
                # Assume a single FWHDataset was passed directly
                ref_df = analytical_refs
            
            # --- 3. BUG FIX: Actually call rmse_error ---
            if error_method == 'l2':
                error = case.l2_error(ref_df, col='P_obs')
            elif error_method == 'rmse':
                error = case.rmse_error(ref_df, col='P_obs')

            actual_dt = case.data['Time'].iloc[1] - case.data['Time'].iloc[0]
            actual_dx = (case.C * actual_dt) / meta['cfl']

            # Use plot_x_val to avoid clashing with the regex x_val variable above
            if analysis == 'dt':
                if use_doppler:
                    plot_x_val = (1.0 / actual_dt) / (case.F_SOURCE / (1 - meta['mach']))
                else:   
                    plot_x_val = (1.0 / actual_dt) / case.F_SOURCE
                marker_val = meta['cfl']

            elif analysis == 'dx':
                if use_doppler:
                    plot_x_val = (case.LAMBDA_SOURCE * (1 - meta['mach'])) / actual_dx
                else:
                    plot_x_val = case.LAMBDA_SOURCE / actual_dx
                marker_val = meta['cfl']

            elif analysis == 'r':
                if use_doppler:
                    plot_x_val = meta['radius'] / (case.LAMBDA_SOURCE * (1 - meta['mach']))
                else:
                    plot_x_val = meta['radius'] / case.LAMBDA_SOURCE
                marker_val = meta['radius']

            if meta['mach'] not in mach_data:
                mach_data[meta['mach']] = []
                
            mach_data[meta['mach']].append((plot_x_val, error, marker_val))
            marker_keys.add(marker_val)

        
        fig, ax = plt.subplots(figsize=(10, 7))
        
        sorted_machs = sorted(mach_data.keys())
        sorted_markers = sorted(list(marker_keys))
        
        # Color lines by Mach number (cool to warm)
        colors = cm.coolwarm(np.linspace(0, 1, len(sorted_machs)))
        
        # Assign distinct markers to CFL values so you can still track them
        markers = ['o', 's', '^', 'D', 'v', 'p', '*', 'X']
        marker_map = {val: markers[i % len(markers)] for i, val in enumerate(sorted_markers)}

        # 3. Plotting the Convergence Curves
        for idx, mach in enumerate(sorted_machs):
            data = mach_data[mach]
            data.sort(key=lambda x: x[0]) 
            
            x_vals = [p[0] for p in data]
            y_vals = [p[1] for p in data]
            mach_color = colors[idx]
            
            ax.plot(x_vals, y_vals, linestyle='-', color=mach_color, zorder=1)
            
            for p in data:
                x, y, m_val = p
                ax.scatter(x, y, marker=marker_map[m_val], 
                        color=mach_color, s=70, zorder=2, edgecolor='black', linewidth=0.5)

        # 4. Custom Split Legend
            legend_elements = [Line2D([0], [0], color='w', label=r'$\bf{Mach}$', lw=0)]
            for idx, mach in enumerate(sorted_machs):
                legend_elements.append(Line2D([0], [0], color=colors[idx], lw=2, label=f"{mach:.2f}"))
                
            legend_elements.append(Line2D([0], [0], color='w', label='', lw=0)) 
            
            marker_title = r'$\bf{Radius\ (m)}$' if analysis == 'r' else r'$\bf{X/P}$'
            legend_elements.append(Line2D([0], [0], color='w', label=marker_title, lw=0))
            for m_val in sorted_markers:
                lbl = f"{int(m_val)}" if analysis == 'r' else f'{m_val}'
                legend_elements.append(Line2D([0], [0], marker=marker_map[m_val], color='w', 
                                            markerfacecolor='gray', markeredgecolor='black', 
                                            markersize=8, label=lbl))

        title_shift = "Doppler-Shifted " if use_doppler else "Source "

        if analysis == "dt":
            ax.set_title(f"Error vs {title_shift}Points Per Period\n(Spatial Grid Fixed at {ref_val[0]} PPW and CS Radius at {ref_val[1]:.1f} m)")
            ax.set_xlabel(f"Temporal Resolution ($f_{{sample}} / f_{{eff}}$)")
        if analysis == "dx":
            ax.set_title(f"Error vs {title_shift}Points Per Wavelength\n(Temporal Resolution Fixed at {ref_val[0]} PPP and CS Radius at {ref_val[1]:.1f} m)")
            ax.set_xlabel(f"Spatial Resolution ($\lambda / d$)")
        if analysis == 'r':
            ax.set_title(f"Error vs Normalised CS Radius\n(Spatial Grid Fixed at {ref_val[0]} PPW and CFL={ref_val[1]})")
            ax.set_xlabel(f"Normalised Radius (R / $\lambda$)")
            
        if error_method == 'l2':
            ax.set_ylabel("Log L2 Error (%)")
            ax.set_yscale('log')
        elif error_method == 'rmse':
            ax.set_ylabel("RMSE Error (Pa)")
        
        ax.set_ylim(bottom=ymin, top=ymax)
        ax.grid(True, which="both", linestyle='--', alpha=0.6)
        if legend:
            ax.legend(handles=legend_elements, bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_px_contour_space(compiled_data, add_title=None, use_doppler=True, shape_by='elements', draw_mach_lines=False, vmax=1e-1, vmin=1e-3):
        import matplotlib.cm as cm
        import matplotlib.colors as mcolors
        from matplotlib.lines import Line2D
        import matplotlib.pyplot as plt
        import numpy as np
        from scipy.interpolate import griddata

        if not compiled_data:
            print("No data provided to plot.")
            return

        fig, ax = plt.subplots(figsize=(12, 8))
        
        X_coords = np.array([d['X'] for d in compiled_data])
        P_coords = np.array([d['P'] for d in compiled_data])
        errors = np.array([d['error'] for d in compiled_data])

        c_min = vmin if vmin is not None else min(errors)
        c_max = vmax if vmax is not None else max(errors)
        
        norm = mcolors.LogNorm(vmin=c_min, vmax=c_max)
        cmap = cm.coolwarm
        
        # 1. Generate the Contour Mesh (in Log Space for accurate scaling)
        xi = np.logspace(np.log10(X_coords.min()), np.log10(X_coords.max()), 100)
        pi = np.logspace(np.log10(P_coords.min()), np.log10(P_coords.max()), 100)
        XI, PI = np.meshgrid(xi, pi)
        
        # Interpolate the error data onto the grid using linear triangulation
        ZI = griddata((X_coords, P_coords), errors, (XI, PI), method='linear')
        
        # Plot the background contour (levels map to the log scale)
        contour = ax.contourf(XI, PI, ZI, levels=np.logspace(np.log10(errors.min()), np.log10(errors.max()), 20), 
                              norm=norm, cmap=cmap, alpha=0.6, zorder=0)

        # ---------------------------------------------------------
        # NEW: Draw explicit contour lines (isocurves) for major error thresholds
        # ---------------------------------------------------------
        # Dynamically determine clean logarithmic decades (e.g., 10^-3, 10^-2) 
        min_pow = int(np.floor(np.log10(errors.min())))
        max_pow = int(np.ceil(np.log10(errors.max())))
        
        # If the range is very tight, do half-decades to ensure we get enough lines. Otherwise, full decades.
        if (max_pow - min_pow) <= 2:
            major_levels = np.logspace(min_pow, max_pow, num=int(2*(max_pow-min_pow)+1))
        else:
            major_levels = np.logspace(min_pow, max_pow, num=int((max_pow-min_pow)+1))
            
        # Filter levels to strictly fall within the min/max of the data
        major_levels = [lvl for lvl in major_levels if errors.min() <= lvl <= errors.max()]
        
        # Fallback if filtering removes too many lines
        if len(major_levels) < 3:
            major_levels = np.logspace(np.log10(errors.min()), np.log10(errors.max()), 5)

        # Draw the lines in black with partial transparency
        contour_lines = ax.contour(XI, PI, ZI, levels=major_levels, colors='black', linewidths=1.2, alpha=0.5, zorder=1)
        
        # Stamp the labels directly onto the lines (inline=True breaks the line so the text is readable)
        ax.clabel(contour_lines, inline=True, fontsize=10, fmt='%.1e', colors='black')
        # ---------------------------------------------------------

        # Dynamic Marker Shapes based on the toggle
        def get_marker(d, mode):
            if mode == 'elements':
                val = d['elements']
                if val < 50000: return 'o'       
                elif val < 250000: return 's'    
                elif val < 1000000: return '^'   
                else: return 'D'
            elif mode == 'dt':
                val = d['dt']
                if val > 5e-3: return 'o'        
                elif val > 1e-3: return 's'      
                elif val > 1e-4: return '^'      
                else: return 'D'                 

        # 2. Draw Mach Lines (if toggled)
        if draw_mach_lines:
            mach_groups = {}
            for d in compiled_data:
                m = d['mach']
                if m not in mach_groups:
                    mach_groups[m] = []
                mach_groups[m].append(d)
                
            for m, points in mach_groups.items():
                points.sort(key=lambda item: item['X'])
                xs = [p['X'] for p in points]
                ps = [p['P'] for p in points]
                ax.plot(xs, ps, color='black', linestyle='-', linewidth=1.5, alpha=0.5, zorder=2)
                ax.text(xs[-1], ps[-1], f" M={m:.2f}", fontsize=8, color='black', verticalalignment='center', zorder=3)

        # 3. Scatter Plotting (Overlaying the actual anchor points)
        for d in compiled_data:
            color_val = cmap(norm(d['error']))
            marker_shape = get_marker(d, shape_by)
            ax.scatter(d['X'], d['P'], marker=marker_shape, color=color_val, 
                       edgecolor='black', s=50, linewidth=0.8, zorder=4)

        # 4. Formatting
        title_prefix = "Doppler-Shifted " if use_doppler else "Non-Doppler Shifted "
        ax.set_title(f"{add_title}\n {title_prefix}Temporal vs Spatial Resolution Matrix\n Shaped by {shape_by.upper()}")
        ax.set_xlabel(f"Spatial Resolution, X (Grid Points Per Wavelength)")
        ax.set_ylabel(f"Temporal Resolution, P (Time Steps Per Period)")
        
        # Colorbar

        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax)
        # cbar = fig.colorbar(contour, ax=ax)
        cbar.set_label('Log L2 Error (%)')
        
        # Dynamic Legend
        legend_title = r'$\bf{Element\ Count}$' if shape_by == 'elements' else r'$\bf{Time\ Step\ (\Delta t)}$'
        lbls = ['< 50k', '50k - 250k', '250k - 1M', '> 1M'] if shape_by == 'elements' else ['> 5e-3 s', '1e-3 to 5e-3 s', '1e-4 to 1e-3 s', '< 1e-4 s']
        
        legend_elements = [
            Line2D([0], [0], color='w', label=legend_title, lw=0),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', markeredgecolor='black', markersize=8, label=lbls[0]),
            Line2D([0], [0], marker='s', color='w', markerfacecolor='gray', markeredgecolor='black', markersize=8, label=lbls[1]),
            Line2D([0], [0], marker='^', color='w', markerfacecolor='gray', markeredgecolor='black', markersize=8, label=lbls[2]),
            Line2D([0], [0], marker='D', color='w', markerfacecolor='gray', markeredgecolor='black', markersize=8, label=lbls[3])
        ]
        ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.15, 1.0))

        plt.tight_layout()
        plt.show()

    # @staticmethod
    # def plot_cfl_convergence_by_mach(directory, analysis, analytical_refs, ref_val, error_method='l2', ymin=3e-5, ymax=5e-2, use_doppler=True):
    #     from matplotlib.lines import Line2D
    #     import matplotlib.cm as cm
        
    #     mach_data = {}
    #     marker_keys = set()

    #     search_pattern = "r*_M*.dat" if analysis == 'r' else "*.dat"

    #     for filepath in Path(directory).glob(search_pattern):
    #         if filepath.name.startswith('ana_'):
    #             continue

    #         case = FWHDataset(filepath=filepath)
    #         meta = case.metadata

    #         if analysis in ['dt', 'dx'] and meta.get('cfl') is None:
    #             continue
    #         if analysis == 'r' and meta.get('radius') is None:
    #             continue

    #         # ref_key = str(meta['mach']).replace('.','')
    #         ref_key = meta['mach_str']

    #         if ref_key not in analytical_refs:
    #             continue
    #         ref_df = analytical_refs[ref_key]
            
    #         if error_method == 'l2':
    #             error = case.l2_error(ref_df, col='P_obs')
    #         elif error_method == 'rmse':
    #             error = case.rmse_error(ref_df, col='P_obs')

    #         actual_dt = case.data['Time'].iloc[1] - case.data['Time'].iloc[0]
    #         actual_dx = (case.C * actual_dt) / meta['cfl']

    #         if analysis == 'dt':
    #             if use_doppler:
    #                 x_val = (1.0 / actual_dt) / (case.F_SOURCE / (1 - meta['mach']))
    #             else:   
    #                 x_val = (1.0 / actual_dt) / case.F_SOURCE
    #             marker_val = meta['cfl']

    #         elif analysis == 'dx':
    #             if use_doppler:
    #                 x_val = (case.LAMBDA_SOURCE * (1 - meta['mach'])) / actual_dx
    #             else:
    #                 x_val = case.LAMBDA_SOURCE / actual_dx
    #             marker_val = meta['cfl']

    #         elif analysis == 'r':
    #             if use_doppler:
    #                 x_val = meta['radius'] / (case.LAMBDA_SOURCE * (1 - meta['mach']))
    #             else:
    #                 x_val = meta['radius'] / case.LAMBDA_SOURCE
    #             # x_val = case.LAMBDA_SOURCE / meta['radius']
    #             marker_val = meta['radius']

    #         if meta['mach'] not in mach_data:
    #             mach_data[meta['mach']] = []
                
    #         mach_data[meta['mach']].append((x_val, error, marker_val))
    #         marker_keys.add(marker_val)

        
    #     fig, ax = plt.subplots(figsize=(10, 7))
        
    #     sorted_machs = sorted(mach_data.keys())
    #     sorted_markers = sorted(list(marker_keys))
        
    #     # Color lines by Mach number (cool to warm)
    #     colors = cm.coolwarm(np.linspace(0, 1, len(sorted_machs)))
        
    #     # Assign distinct markers to CFL values so you can still track them
    #     markers = ['o', 's', '^', 'D', 'v', 'p', '*', 'X']
    #     marker_map = {val: markers[i % len(markers)] for i, val in enumerate(sorted_markers)}

    #     # 3. Plotting the Convergence Curves
    #     for idx, mach in enumerate(sorted_machs):
    #         data = mach_data[mach]
    #         data.sort(key=lambda x: x[0]) 
            
    #         x_vals = [p[0] for p in data]
    #         y_vals = [p[1] for p in data]
    #         mach_color = colors[idx]
            
    #         # Draw the convergence line for this specific Mach number
    #         ax.plot(x_vals, y_vals, linestyle='-', color=mach_color, zorder=1)
            
    #         # Overlay the CFL markers
    #         for p in data:
    #             x, y, m_val = p
    #             ax.scatter(x, y, marker=marker_map[m_val], 
    #                     color=mach_color, s=70, zorder=2, edgecolor='black', linewidth=0.5)

    #     # 4. Custom Split Legend
    #     legend_elements = [Line2D([0], [0], color='w', label=r'$\bf{Mach}$', lw=0)]
    #     for idx, mach in enumerate(sorted_machs):
    #         legend_elements.append(Line2D([0], [0], color=colors[idx], lw=2, label=f"{mach:.2f}"))
            
    #     legend_elements.append(Line2D([0], [0], color='w', label='', lw=0)) 
        
    #     marker_title = r'$\bf{Radius\ (m)}$' if analysis == 'r' else r'$\bf{X/P}$'
    #     legend_elements.append(Line2D([0], [0], color='w', label=marker_title, lw=0))
    #     for m_val in sorted_markers:
    #         lbl = f"{int(m_val)}" if analysis == 'r' else f'{m_val}'
    #         legend_elements.append(Line2D([0], [0], marker=marker_map[m_val], color='w', 
    #                                     markerfacecolor='gray', markeredgecolor='black', 
    #                                     markersize=8, label=lbl))

    #     title_shift = "Doppler-Shifted " if use_doppler else "Source "

    #     if analysis == "dt":
    #         ax.set_title(f"Error vs {title_shift}Points Per Period\n(Spatial Grid Fixed at {ref_val[0]} PPW and CS Radius at {ref_val[1]} m)")
    #         ax.set_xlabel(f"Temporal Resolution ($f_{{sample}} / f_{{eff}}$)")
    #     if analysis == "dx":
    #         ax.set_title(f"Error vs {title_shift}Points Per Wavelength\n(Temporal Resolution Fixed at {ref_val[0]} PPP and CS Radius at {ref_val[1]} m)")
    #         ax.set_xlabel(f"Spatial Resolution ($\lambda% / d)")
    #     if analysis == 'r':
    #         ax.set_title(f"Error vs Normalised CS Radius\n(Spatial Grid Fixed at {ref_val[0]} PPW and CFL={ref_val[1]})")
    #         ax.set_xlabel(f"Normalised Radius (R / $\lambda$)")
    #     if error_method == 'l2':
    #         ax.set_ylabel("Log L2 Error (%)")
    #         ax.set_yscale('log')
    #     elif error_method == 'rmse':
    #         ax.set_ylabel("RMSE Error (Pa)")
        
    #     ax.set_ylim(bottom=ymin, top=ymax)
    #     ax.grid(True, which="both", linestyle='--', alpha=0.6)
        
    #     ax.legend(handles=legend_elements, bbox_to_anchor=(1.05, 1), loc='upper left')
    #     plt.tight_layout()
    #     plt.show()

    @staticmethod
    def plot_px_parameter_space(compiled_data, use_doppler=True, shape_by='elements', 
                                draw_mach_lines=False, mach_number=None, vmin=1e-3, vmax=1e-1,
                                legend=True, add_title=None):
        import matplotlib.cm as cm
        import matplotlib.colors as mcolors
        from matplotlib.lines import Line2D
        import matplotlib.pyplot as plt

        if not compiled_data:
            print("No data provided to plot.")
            return


        fig, ax = plt.subplots(figsize=(12, 8))
        
        errors = [d['error'] for d in compiled_data]
        
        # Color Mapping (Log Scale Error)
        c_min = vmin if vmin is not None else min(errors)
        c_max = vmax if vmax is not None else max(errors)

        norm = mcolors.LogNorm(vmin=c_min, vmax=c_max)
        cmap = cm.coolwarm
        
        # Dynamic Marker Shapes based on the toggle
        def get_marker(d, mode):
            if mode == 'elements':
                val = d['elements']
                if val < 50000: return 'o'       
                elif val < 250000: return 's'    
                elif val < 1000000: return '^'   
                else: return 'D'
            elif mode == 'dt':
                val = d['dt']
                if val > 5e-3: return 'o'        
                elif val > 1e-3: return 's'      
                elif val > 1e-4: return '^'      
                else: return 'D'                 

        # 1. Draw Mach Lines (if toggled)
        if draw_mach_lines:
            mach_groups = {}
            for d in compiled_data:
                m = d['mach']
                if m not in mach_groups:
                    mach_groups[m] = []
                mach_groups[m].append(d)
                
            for m, points in mach_groups.items():
                points.sort(key=lambda item: item['X'])
                xs = [p['X'] for p in points]
                ps = [p['P'] for p in points]
                ax.plot(xs, ps, color='gray', linestyle='-', linewidth=1.0, alpha=0.5, zorder=1)
                ax.text(xs[-1], ps[-1], f" M={m:.2f}", fontsize=8, color='dimgray', verticalalignment='center')

        # 2. Scatter Plotting
        for d in compiled_data:
            color_val = cmap(norm(d['error']))
            marker_shape = get_marker(d, shape_by)
            ax.scatter(d['X'], d['P'], marker=marker_shape, color=color_val, 
                       edgecolor='black', s=80, linewidth=0.5, alpha=0.8, zorder=2)

        # 3. Formatting
        title_prefix = f"Doppler-Shifted " if use_doppler else f"Source "
        shape_string = f"Shaped by {shape_by.upper()}" if legend else ''

        if mach_number is None:
            ax.set_title(f"{add_title}\nTemporal ({title_prefix}P) vs Spatial ({title_prefix}X)\nColoured by Error, {shape_string}")
        else:
            ax.set_title(f"{add_title}\nTemporal ({title_prefix}P) vs Spatial ({title_prefix}X)\nColoured by Error, {shape_string} Mach number = {mach_number}")

        ax.set_xlabel(f"Spatial Resolution, X ({title_prefix}Points Per Wavelength)")
        ax.set_ylabel(f"Temporal Resolution, P ({title_prefix}Points Per Period)")
        
        # ax.set_xscale('log')
        # ax.set_yscale('log')
        ax.grid(True, which='both', linestyle='--', alpha=0.4)
        
        # Colorbar
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax)
        cbar.set_label('Log L2 Error (%)')
        
        # Dynamic Legend
        if legend:
            legend_title = r'$\bf{Element\ Count}$' if shape_by == 'elements' else r'$\bf{Time\ Step\ (\Delta t)}$'
            lbls = ['< 50k', '50k - 250k', '250k - 1M', '> 1M'] if shape_by == 'elements' else ['> 5e-3 s', '1e-3 to 5e-3 s', '1e-4 to 1e-3 s', '< 1e-4 s']
            
            legend_elements = [
                Line2D([0], [0], color='w', label=legend_title, lw=0),
                Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', markeredgecolor='black', markersize=8, label=lbls[0]),
                Line2D([0], [0], marker='s', color='w', markerfacecolor='gray', markeredgecolor='black', markersize=8, label=lbls[1]),
                Line2D([0], [0], marker='^', color='w', markerfacecolor='gray', markeredgecolor='black', markersize=8, label=lbls[2]),
                Line2D([0], [0], marker='D', color='w', markerfacecolor='gray', markeredgecolor='black', markersize=8, label=lbls[3])
            ]
            ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.15, 1.0))

        # plt.tight_layout(right=0.75)
        plt.subplots_adjust(right=0.75)
        plt.show()

    # @staticmethod
    # def plot_px_contour_space(compiled_data, use_doppler=True, shape_by='elements', draw_mach_lines=False, vmax=1e-2, vmin=1e-3):
    #     import matplotlib.cm as cm
    #     import matplotlib.colors as mcolors
    #     from matplotlib.lines import Line2D
    #     import matplotlib.pyplot as plt
    #     import numpy as np
    #     from scipy.interpolate import griddata

    #     if not compiled_data:
    #         print("No data provided to plot.")
    #         return

    #     fig, ax = plt.subplots(figsize=(12, 8))
        
    #     X_coords = np.array([d['X'] for d in compiled_data])
    #     P_coords = np.array([d['P'] for d in compiled_data])
    #     errors = np.array([d['error'] for d in compiled_data])

    #     c_min = vmin if vmin is not None else min(errors)
    #     c_max = vmax if vmax is not None else max(errors)
        
    #     norm = mcolors.LogNorm(vmin=c_min, vmax=c_max)
    #     cmap = cm.coolwarm
        
    #     # 1. Generate the Contour Mesh (in Log Space for accurate scaling)
    #     # We create a fine 100x100 grid spanning the min and max of your data
    #     xi = np.logspace(np.log10(X_coords.min()), np.log10(X_coords.max()), 100)
    #     pi = np.logspace(np.log10(P_coords.min()), np.log10(P_coords.max()), 100)
    #     XI, PI = np.meshgrid(xi, pi)
        
    #     # Interpolate the error data onto the grid using linear triangulation
    #     ZI = griddata((X_coords, P_coords), errors, (XI, PI), method='linear')
        
    #     # Plot the background contour (levels map to the log scale)
    #     contour = ax.contourf(XI, PI, ZI, levels=np.logspace(np.log10(errors.min()), np.log10(errors.max()), 20), 
    #                           norm=norm, cmap=cmap, alpha=0.6, zorder=0)

    #     # Dynamic Marker Shapes based on the toggle
    #     def get_marker(d, mode):
    #         if mode == 'elements':
    #             val = d['elements']
    #             if val < 50000: return 'o'       
    #             elif val < 250000: return 's'    
    #             elif val < 1000000: return '^'   
    #             else: return 'D'
    #         elif mode == 'dt':
    #             val = d['dt']
    #             if val > 5e-3: return 'o'        
    #             elif val > 1e-3: return 's'      
    #             elif val > 1e-4: return '^'      
    #             else: return 'D'                 

    #     # 2. Draw Mach Lines (if toggled)
    #     if draw_mach_lines:
    #         mach_groups = {}
    #         for d in compiled_data:
    #             m = d['mach']
    #             if m not in mach_groups:
    #                 mach_groups[m] = []
    #             mach_groups[m].append(d)
                
    #         for m, points in mach_groups.items():
    #             points.sort(key=lambda item: item['X'])
    #             xs = [p['X'] for p in points]
    #             ps = [p['P'] for p in points]
    #             ax.plot(xs, ps, color='black', linestyle='-', linewidth=1.5, alpha=0.5, zorder=1)
    #             ax.text(xs[-1], ps[-1], f" M={m:.2f}", fontsize=8, color='black', verticalalignment='center', zorder=2)

    #     # 3. Scatter Plotting (Overlaying the actual anchor points)
    #     for d in compiled_data:
    #         # We color the edge black and fill with the exact error color so it blends but remains visible
    #         color_val = cmap(norm(d['error']))
    #         marker_shape = get_marker(d, shape_by)
    #         ax.scatter(d['X'], d['P'], marker=marker_shape, color=color_val, 
    #                    edgecolor='black', s=50, linewidth=0.8, zorder=3)

    #     # 4. Formatting
    #     title_prefix = "Doppler-Shifted " if use_doppler else "Source "
    #     ax.set_title(f"Acoustic Error Contour Map: Temporal ({title_prefix}PPP) vs Spatial ({title_prefix}PPW)\nShaped by {shape_by.upper()}")
    #     ax.set_xlabel(f"Spatial Resolution, X ({title_prefix}Points Per Wavelength)")
    #     ax.set_ylabel(f"Temporal Resolution, P ({title_prefix}Points Per Period)")
        
    #     # ax.set_xscale('log')
    #     # ax.set_yscale('log')
        
    #     # Colorbar
    #     cbar = fig.colorbar(contour, ax=ax)
    #     cbar.set_label('Log L2 Error (%)')
        
    #     # Dynamic Legend
    #     legend_title = r'$\bf{Element\ Count}$' if shape_by == 'elements' else r'$\bf{Time\ Step\ (\Delta t)}$'
    #     lbls = ['< 50k', '50k - 250k', '250k - 1M', '> 1M'] if shape_by == 'elements' else ['> 5e-3 s', '1e-3 to 5e-3 s', '1e-4 to 1e-3 s', '< 1e-4 s']
        
    #     legend_elements = [
    #         Line2D([0], [0], color='w', label=legend_title, lw=0),
    #         Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', markeredgecolor='black', markersize=8, label=lbls[0]),
    #         Line2D([0], [0], marker='s', color='w', markerfacecolor='gray', markeredgecolor='black', markersize=8, label=lbls[1]),
    #         Line2D([0], [0], marker='^', color='w', markerfacecolor='gray', markeredgecolor='black', markersize=8, label=lbls[2]),
    #         Line2D([0], [0], marker='D', color='w', markerfacecolor='gray', markeredgecolor='black', markersize=8, label=lbls[3])
    #     ]
    #     ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.0, 1.0))

    #     plt.tight_layout()
    #     plt.show()

    # @staticmethod
    # def plot_3d_parameter_surface(compiled_data, use_doppler=True, shape_by='elements'):
    #     import matplotlib.pyplot as plt
    #     import matplotlib.cm as cm
    #     import matplotlib.colors as mcolors
    #     import numpy as np
        
    #     if not compiled_data:
    #         print("No data provided to plot.")
    #         return

    #     # Create a wider figure to fit two colorbars
    #     fig = plt.figure(figsize=(14, 9))
    #     ax = fig.add_subplot(111, projection='3d')
        
    #     # Extract data
    #     X_coords = np.array([d['X'] for d in compiled_data])
    #     P_coords = np.array([d['P'] for d in compiled_data])
    #     errors = np.array([d['error'] for d in compiled_data])
        
    #     # Extract the cost metric for the dots
    #     costs = np.array([d[shape_by] for d in compiled_data])
        
    #     # We pass the raw error values to Z to keep the axis readable, 
    #     # but tell matplotlib to treat the Z-axis logarithmically
    #     Z_coords = errors
        
    #     # 1. Plot the Surface (Colored by Error using coolwarm)
    #     # norm_error = mcolors.LogNorm(vmin=errors.min(), vmax=errors.max())
    #     # surf = ax.plot_trisurf(X_coords, P_coords, Z_coords, 
    #     #                        cmap=cm.coolwarm, norm=norm_error, 
    #     #                        edgecolor='none', alpha=0.6, linewidth=0.2)
    #     # 1. Plot the Surface (Colored by Error using coolwarm)
    #     norm_error = mcolors.LogNorm(vmin=errors.min(), vmax=errors.max())
    #     surf = ax.plot_trisurf(X_coords, P_coords, Z_coords, 
    #                            cmap=cm.coolwarm, norm=norm_error, 
    #                            edgecolor='none',  # Ensures no chaotic mesh lines are drawn
    #                            antialiased=True,  # Smooths the surface rendering
    #                            alpha=0.6)
        
    #     # 2. Plot the Dots (Colored by Cost using viridis to contrast the surface)
    #     norm_cost = mcolors.LogNorm(vmin=costs.min(), vmax=costs.max())
    #     scatter = ax.scatter(X_coords, P_coords, Z_coords, 
    #                          c=costs, cmap=cm.viridis, norm=norm_cost, 
    #                          s=40, edgecolor='black', linewidth=0.5, zorder=3)

    #     # Formatting
    #     title_prefix = "Doppler-Shifted " if use_doppler else "Source "
    #     ax.set_title(f"3D Acoustic Error Topography\nSurface: L2 Error | Dots: {shape_by.upper()}")
    #     ax.set_xlabel(f"Spatial Resolution, X ({title_prefix}PPW)")
    #     ax.set_ylabel(f"Temporal Resolution, P ({title_prefix}PPP)")
    #     ax.set_zlabel("L2 Error (%)")
        
    #     # Force the Z-axis to behave logarithmically 
    #     ax.set_zscale('log')
        
    #     # Adjust viewing angle
    #     ax.view_init(elev=30, azim=225) 
        
    #     # Add two distinct colorbars so Fan can read both metrics
    #     cbar_error = fig.colorbar(surf, ax=ax, shrink=0.5, aspect=15, pad=0.05)
    #     cbar_error.set_label('Surface: L2 Error (%)')
        
    #     cbar_cost = fig.colorbar(scatter, ax=ax, shrink=0.5, aspect=15, pad=0.1)
    #     cbar_cost.set_label(f'Dots: {shape_by.upper()} Cost')
        
    #     plt.tight_layout()
    #     plt.show()

