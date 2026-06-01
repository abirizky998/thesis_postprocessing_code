import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import os
from pathlib import Path

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
        
        # cfl_mach_match = re.search(r"cfl(?P<cfl>\d+)_M(?P<mach>\d+)", self.name)
        # if cfl_mach_match:
        #     meta['cfl'] = float(cfl_mach_match.group('cfl')) / 100.0
        #     meta['mach_str'] = cfl_mach_match.group('mach')
        #     meta['mach'] = float(f"0.{meta['mach_str'][1:]}")
        #     meta['dt'] = (meta['cfl'] * dx) / self.C
        #     meta['grid'] = int(np.round((2 * np.pi * self.R) / dx))
            
        #     return meta
        cfl_mach_match = re.search(r"cfl(\d+)_M(\d+)", self.name)
        if cfl_mach_match:
            meta['cfl'] = float(cfl_mach_match.group(1)) / 100.0
            meta['mach_str'] = cfl_mach_match.group(2)
            
            # FIXED: Directly divide by 100 since the string is exactly int(M*100)
            meta['mach'] = float(meta['mach_str']) / 100.0 
            
            meta['dt'] = (meta['cfl'] * dx) / self.C
            meta['grid'] = int(np.round((2 * np.pi * self.R) / dx))
            
            return meta
        
        rad_mach_match = re.search(r"r(\d+)_M(\d+)\.dat", self.name)
        if rad_mach_match:
            meta['radius'] = float(rad_mach_match.group(1))
            meta['mach_str'] = rad_mach_match.group(2)
            meta['mach'] = float(meta['mach_str']) / 100.0
            
            # Calculate Doppler-shifted dx and corresponding N
            meta['dx'] = self.C * (1 - meta['mach']) / (self.X_TARGET * self.F_SOURCE)
            meta['grid'] = int(np.round((2 * np.pi * meta['radius']) / meta['dx']))
            
            # Time was locked at CFL = 1.0 for this sweep
            meta['cfl'] = 1.0 
            meta['dt'] = (meta['cfl'] * meta['dx']) / self.C
            return meta
        
        # # LEGACY STUFF
        # rad_mach_match = re.search(r"r(\d+)_M(\d+)([cmf])", self.name)
        # if rad_mach_match:
        #     meta['radius'] = int(rad_mach_match.group(1))
        #     meta['mach_str'] = rad_mach_match.group(2)
        #     meta['mach'] = float(f"0.{meta['mach_str'][1:]}")
        #     meta['grid'] = rad_mach_match.group(3)
        #     return meta
            
        # time_match = re.search(r"t_M(\d+)([cmf])", self.name)
        # if time_match:
        #     meta['mach_str'] = time_match.group(1)
        #     meta['mach'] = float(f"0.{meta['mach_str'][1:]}")
        #     meta['time_lbl'] = time_match.group(2)
            
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

class FWHPlotter:
    @staticmethod
    def plot_pressure(reference, *comparisons, labels=None, title=None, scale=1.0,
                      ref_col='P_analytical', comp_col='P_obs'):
        
        fig, ax = plt.subplots(figsize=(8,5))
        ref_df = reference.data if hasattr(reference, 'data') else reference
        ax.plot(ref_df['Time'], ref_df[ref_col] * scale, ls='-', c='black', label=f'Reference ({ref_col})')

        if not labels or len(labels) != len(comparisons):
            labels = [f'Dataset {i+1}' for i in range(len(comparisons))]

        styles = [':', '-.', '--']
        colors = ['red', 'blue', 'green', 'orange', 'purple', 'cyan']

        for i, comp in enumerate(comparisons):
            comp_df = comp.data if hasattr(comp, 'data') else comp

            error = comp.l2_error(reference, col=comp_col)
            ls = styles[i % len(styles)]
            c = colors[i % len(colors)]

            full_label = f'{labels[i]} (Error: {error * 100:.2f}%)'
            ax.plot(comp_df['Time'], comp_df[comp_col] * scale, ls=ls, c=c, label=full_label)

        ax.set_xlabel('Time (s)')

        if scale == 1.0:
            ax.set_ylabel("Pressure (Pa)")
        else:
            scale_str = f"x10^{{{int(np.log10(scale))}}}" if scale >= 10 else f"x{scale}"
            ax.set_ylabel(f'Pressure ({scale_str} Pa)')

        if title:
            ax.set_title(title)

        ax.grid(True, linestyle='--', alpha=0.7)
        ax.legend(loc='upper right', bbox_to_anchor=(1.35, 1))
        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_cfl_convergence(directory, analytical_refs, use_doppler=True):
        """
        Plots L2 Error vs Points Per Period (PPP) for the CFL batch dataset.
        Lines connect constant CFL values. Markers differentiate Mach numbers.
        """
        from matplotlib.lines import Line2D
        
        # Group data by CFL value
        cfl_data = {}
        
        for filepath in Path(directory).glob("cfl*.dat"):
            case = FWHDataset(filepath=filepath)
            meta = case.metadata
            
            if meta['cfl'] is None:
                continue
            
            ref_key = str(meta['mach']).replace('.', '')
            ref_df = analytical_refs[ref_key]

            # ref_df = analytical_refs[meta['mach_str']]
            error = case.l2_error(ref_df, col='P_obs')
            
            # PPP = X / CFL
            points_per_period = case.X_TARGET / meta['cfl']
            
            if meta['cfl'] not in cfl_data:
                cfl_data[meta['cfl']] = []
                
            cfl_data[meta['cfl']].append((points_per_period, error, meta['mach'], meta['mach_str']))

        # Setup Plot
        fig, ax = plt.subplots(figsize=(10, 7))
        
        mach_keys = sorted(list(set([m for c in cfl_data.values() for _, _, _, m in c])))
        markers = ['o', 's', '^', 'D', 'v', 'p', '*', 'X', '<', '>']
        mach_marker_map = {m: markers[i % len(markers)] for i, m in enumerate(mach_keys)}
        
        # Color mapping for lines based on CFL
        import matplotlib.cm as cm
        sorted_cfls = sorted(cfl_data.keys())
        colors = cm.viridis(np.linspace(0, 0.9, len(sorted_cfls)))

        # Plotting
        for idx, cfl in enumerate(sorted_cfls):
            data = cfl_data[cfl]
            data.sort(key=lambda x: x[2]) # Sort by Mach to draw logical lines
            
            cfl_color = colors[idx]
            
            x_vals = [p[0] for p in data]
            y_vals = [p[1] for p in data]
            
            # Connect the points of the same CFL
            ax.plot(x_vals, y_vals, linestyle='-', color=cfl_color, zorder=1)
            
            # Add Mach markers
            for p in data:
                x, y, mach_val, m_str = p
                ax.scatter(x, y, marker=mach_marker_map[m_str], 
                           color=cfl_color, s=70, zorder=2, edgecolor='black', linewidth=0.5)

        # Custom Legend
        legend_elements = [Line2D([0], [0], color='w', label=r'$\bf{CFL\ Targets}$', lw=0)]
        for idx, cfl in enumerate(sorted_cfls):
            legend_elements.append(Line2D([0], [0], color=colors[idx], lw=2, label=f"CFL = {cfl}"))
            
        legend_elements.append(Line2D([0], [0], color='w', label='', lw=0)) 
        
        legend_elements.append(Line2D([0], [0], color='w', label=r'$\bf{Mach\ Numbers}$', lw=0))
        for m_str in mach_keys:
            legend_elements.append(Line2D([0], [0], marker=mach_marker_map[m_str], color='w', 
                                          markerfacecolor='gray', markeredgecolor='black', 
                                          markersize=8, label=f"M = {float(m_str)/100:.2f}"))

        title_shift = "Doppler-Shifted " if use_doppler else "Source "
        ax.set_title(f"Time Convergence: L2 Error vs Points Per Period (Spatial PPW = {FWHDataset.X_TARGET})")
        ax.set_xlabel(f"Points Per Period ({title_shift}Frequency)")
        ax.set_ylabel("Log Error")
        ax.set_yscale('log')
        ax.grid(True, which="both", linestyle='--', alpha=0.6)
        
        ax.legend(handles=legend_elements, bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_cfl_convergence_by_mach(directory, analysis, analytical_refs, ref_val, use_doppler=True):
        from matplotlib.lines import Line2D
        import matplotlib.cm as cm
        
        mach_data = {}
        marker_keys = set()

        search_pattern = "r*_M*.dat" if analysis == 'r' else "cfl*.dat"

        for filepath in Path(directory).glob(search_pattern):
            case = FWHDataset(filepath=filepath)
            meta = case.metadata

            if analysis in ['dt', 'dx'] and meta.get('cfl') is None:
                continue
            if analysis == 'r' and meta.get('radius') is None:
                continue

            ref_key = str(meta['mach']).replace('.','')
            if ref_key not in analytical_refs:
                continue
            ref_df = analytical_refs[ref_key]
            error = case.l2_error(ref_df, col='P_obs')
            
            if analysis == 'dt':
                x_val = case.X_TARGET / meta['cfl']
                marker_val = meta['cfl']
            elif analysis == 'dx':
                x_val = case.P_TARGET * meta['cfl']
                marker_val = meta['cfl']
            elif analysis == 'r':
                x_val = meta['radius'] / case.LAMBDA_SOURCE
                marker_val = meta['radius']

            if meta['mach'] not in mach_data:
                mach_data[meta['mach']] = []
                
            mach_data[meta['mach']].append((x_val, error, marker_val))
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
            
            # Draw the convergence line for this specific Mach number
            ax.plot(x_vals, y_vals, linestyle='-', color=mach_color, zorder=1)
            
            # Overlay the CFL markers
            for p in data:
                x, y, m_val = p
                ax.scatter(x, y, marker=marker_map[m_val], 
                        color=mach_color, s=70, zorder=2, edgecolor='black', linewidth=0.5)

        # 4. Custom Split Legend
        legend_elements = [Line2D([0], [0], color='w', label=r'$\bf{Mach}$', lw=0)]
        for idx, mach in enumerate(sorted_machs):
            legend_elements.append(Line2D([0], [0], color=colors[idx], lw=2, label=f"{mach:.2f}"))
            
        legend_elements.append(Line2D([0], [0], color='w', label='', lw=0)) 
        
        marker_title = r'$\bf{Radius\ (m)}$' if analysis == 'r' else r'$\bf{CFL}$'
        legend_elements.append(Line2D([0], [0], color='w', label=marker_title, lw=0))
        for m_val in sorted_markers:
            lbl = f"{int(m_val)}" if analysis == 'r' else f'{m_val}'
            legend_elements.append(Line2D([0], [0], marker=marker_map[m_val], color='w', 
                                        markerfacecolor='gray', markeredgecolor='black', 
                                        markersize=8, label=lbl))

        title_shift = "Doppler-Shifted " if use_doppler else "Source "

        if analysis == "dt":
            ax.set_title(f"Error vs {title_shift}Points Per Period\n(Spatial Grid Fixed at {ref_val[0]} PPW and CS Radius at {ref_val[1]} m)")
            ax.set_xlabel(f"Temporal Resolution ($f_{{sample}} / f_{{eff}}$)")
        if analysis == "dx":
            ax.set_title(f"Error vs {title_shift}Points Per Wavelength\n(Temporal Resolution Fixed at {ref_val[0]} PPP and CS Radius at {ref_val[1]} m)")
            ax.set_xlabel(f"Spatial Resolution ($\lambda / d$)")
        if analysis == 'r':
            ax.set_title(f"Error vs {title_shift} meter\n(Spatial Grid Fixed at {ref_val[0]} PPW and CFL={ref_val[1]})")
            ax.set_xlabel(f"Normalised Radius ($R / \lambda$)")
        ax.set_ylabel("Log L2 Error (%)")
        ax.set_yscale('log')
        ax.grid(True, which="both", linestyle='--', alpha=0.6)
        
        ax.legend(handles=legend_elements, bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.show()