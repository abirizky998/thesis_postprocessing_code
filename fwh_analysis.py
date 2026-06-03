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
    
    @staticmethod
    def compile_fwh_batch(directory, use_doppler=True, verbose=False):
        folder_path = Path(directory)
        
        # 1. Load local analytical references
        local_refs = {}
        for ana_file in folder_path.glob("ana_M*.dat"):
            m_match = re.search(r"ana_M(\d+)", ana_file.name)
            if m_match:
                local_refs[m_match.group(1)] = FWHDataset(filepath=ana_file)
                
        if verbose:
            print(f"DEBUG: Loaded Analytical References for Machs: {list(local_refs.keys())}")
        
        folder_data = []
        files_scanned = 0
        
        # 2. Process all data files
        for filepath in folder_path.glob("*.dat"):
            if filepath.name.startswith("ana_"):
                continue
                
            files_scanned += 1
            case = FWHDataset(filepath=filepath)
            meta = case.metadata
            
            if meta.get('cfl') is None:
                if verbose: print(f"DEBUG: Skipped {filepath.name} -> Could not parse CFL/Metadata.")
                continue
                
            ref_key = meta['mach_str']
            
            if ref_key not in local_refs:
                if verbose: print(f"DEBUG: Skipped {filepath.name} -> Ref key '{ref_key}' not found.")
                continue 
                
            ref_df = local_refs[ref_key]
            error = case.l2_error(ref_df, col='P_obs')
            
            if len(case.data) > 1:
                actual_dt = case.data['Time'].iloc[1] - case.data['Time'].iloc[0]
            else:
                actual_dt = meta['dt'] 
                
            actual_dx = (case.C * actual_dt) / meta['cfl']
            true_N = int(np.round((2 * np.pi * case.R) / actual_dx))
            true_elements = true_N * true_N
            
            if use_doppler:
                p_val = (1.0 / actual_dt) / (case.F_SOURCE / (1 - meta['mach']))
                x_val = (case.LAMBDA_SOURCE * (1 - meta['mach'])) / actual_dx
            else:
                p_val = (1.0 / actual_dt) / case.F_SOURCE
                x_val = case.LAMBDA_SOURCE / actual_dx
                
            folder_data.append({
                'mach': meta['mach'],
                'cfl': meta['cfl'],
                'dt': actual_dt,
                'P': p_val,
                'X': x_val,
                'error': error,
                'elements': true_elements,
                'batch_name': folder_path.name
            })
            
        if verbose:
            print(f"\nDEBUG: Scanned {files_scanned} files. Successfully compiled {len(folder_data)} files.")
            
        return folder_data
    
class FWHPlotter:
    # @staticmethod
    # def plot_pressure(reference, *comparisons, labels=None, title=None, scale=1.0,
    #                   ref_col='P_analytical', comp_col='P_obs'):
        
    #     fig, ax = plt.subplots(figsize=(8,5))
    #     ref_df = reference.data if hasattr(reference, 'data') else reference
    #     ax.plot(ref_df['Time'], ref_df[ref_col] * scale, ls='-', c='black', label=f'Reference ({ref_col})')

    #     if not labels or len(labels) != len(comparisons):
    #         labels = [f'Dataset {i+1}' for i in range(len(comparisons))]

    #     styles = [':', '-.', '--']
    #     colors = ['red', 'blue', 'green', 'orange', 'purple', 'cyan']

    #     for i, comp in enumerate(comparisons):
    #         comp_df = comp.data if hasattr(comp, 'data') else comp

    #         error = comp.l2_error(reference, col=comp_col)
    #         ls = styles[i % len(styles)]
    #         c = colors[i % len(colors)]

    #         full_label = f'{labels[i]} (Error: {error * 100:.2f}%)'
    #         ax.plot(comp_df['Time'], comp_df[comp_col] * scale, ls=ls, c=c, label=full_label)

    #     ax.set_xlabel('Time (s)')

    #     if scale == 1.0:
    #         ax.set_ylabel("Pressure (Pa)")
    #     else:
    #         scale_str = f"x10^{{{int(np.log10(scale))}}}" if scale >= 10 else f"x{scale}"
    #         ax.set_ylabel(f'Pressure ({scale_str} Pa)')

    #     if title:
    #         ax.set_title(title)

    #     ax.grid(True, linestyle='--', alpha=0.7)
    #     ax.legend(loc='upper right', bbox_to_anchor=(1.35, 1))
    #     plt.tight_layout()
    #     plt.show()

    # @staticmethod
    # def plot_cfl_convergence(directory, analytical_refs, use_doppler=True):
    #     """
    #     Plots L2 Error vs Points Per Period (PPP) for the CFL batch dataset.
    #     Lines connect constant CFL values. Markers differentiate Mach numbers.
    #     """
    #     from matplotlib.lines import Line2D
        
    #     # Group data by CFL value
    #     cfl_data = {}
        
    #     for filepath in Path(directory).glob("cfl*.dat"):
    #         case = FWHDataset(filepath=filepath)
    #         meta = case.metadata
            
    #         if meta['cfl'] is None:
    #             continue
            
    #         ref_key = str(meta['mach']).replace('.', '')
    #         ref_df = analytical_refs[ref_key]

    #         # ref_df = analytical_refs[meta['mach_str']]
    #         error = case.l2_error(ref_df, col='P_obs')
            
    #         # PPP = X / CFL
    #         points_per_period = case.X_TARGET / meta['cfl']
            
    #         if meta['cfl'] not in cfl_data:
    #             cfl_data[meta['cfl']] = []
                
    #         cfl_data[meta['cfl']].append((points_per_period, error, meta['mach'], meta['mach_str']))

    #     # Setup Plot
    #     fig, ax = plt.subplots(figsize=(10, 7))
        
    #     mach_keys = sorted(list(set([m for c in cfl_data.values() for _, _, _, m in c])))
    #     markers = ['o', 's', '^', 'D', 'v', 'p', '*', 'X', '<', '>']
    #     mach_marker_map = {m: markers[i % len(markers)] for i, m in enumerate(mach_keys)}
        
    #     # Color mapping for lines based on CFL
    #     import matplotlib.cm as cm
    #     sorted_cfls = sorted(cfl_data.keys())
    #     colors = cm.viridis(np.linspace(0, 0.9, len(sorted_cfls)))

    #     # Plotting
    #     for idx, cfl in enumerate(sorted_cfls):
    #         data = cfl_data[cfl]
    #         data.sort(key=lambda x: x[2]) # Sort by Mach to draw logical lines
            
    #         cfl_color = colors[idx]
            
    #         x_vals = [p[0] for p in data]
    #         y_vals = [p[1] for p in data]
            
    #         # Connect the points of the same CFL
    #         ax.plot(x_vals, y_vals, linestyle='-', color=cfl_color, zorder=1)
            
    #         # Add Mach markers
    #         for p in data:
    #             x, y, mach_val, m_str = p
    #             ax.scatter(x, y, marker=mach_marker_map[m_str], 
    #                        color=cfl_color, s=70, zorder=2, edgecolor='black', linewidth=0.5)

    #     # Custom Legend
    #     legend_elements = [Line2D([0], [0], color='w', label=r'$\bf{CFL\ Targets}$', lw=0)]
    #     for idx, cfl in enumerate(sorted_cfls):
    #         legend_elements.append(Line2D([0], [0], color=colors[idx], lw=2, label=f"CFL = {cfl}"))
            
    #     legend_elements.append(Line2D([0], [0], color='w', label='', lw=0)) 
        
    #     legend_elements.append(Line2D([0], [0], color='w', label=r'$\bf{Mach\ Numbers}$', lw=0))
    #     for m_str in mach_keys:
    #         legend_elements.append(Line2D([0], [0], marker=mach_marker_map[m_str], color='w', 
    #                                       markerfacecolor='gray', markeredgecolor='black', 
    #                                       markersize=8, label=f"M = {float(m_str)/100:.2f}"))

    #     title_shift = "Doppler-Shifted " if use_doppler else "Source "
    #     ax.set_title(f"Time Convergence: L2 Error vs Points Per Period (Spatial PPW = {FWHDataset.X_TARGET})")
    #     ax.set_xlabel(f"Points Per Period ({title_shift}Frequency)")
    #     ax.set_ylabel("Log Error")
    #     ax.set_yscale('log')
    #     ax.grid(True, which="both", linestyle='--', alpha=0.6)
        
    #     ax.legend(handles=legend_elements, bbox_to_anchor=(1.05, 1), loc='upper left')
    #     plt.tight_layout()
    #     plt.show()

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

            actual_dt = case.data['Time'].iloc[1] - case.data['Time'].iloc[0]
            actual_dx = (case.C * actual_dt) / meta['cfl']

            if analysis == 'dt':
                if use_doppler:
                    x_val = (1.0 / actual_dt) / (case.F_SOURCE / (1 - meta['mach']))
                else:   
                    x_val = (1.0 / actual_dt) / case.F_SOURCE
                marker_val = meta['cfl']

            elif analysis == 'dx':
                if use_doppler:
                    x_val = (case.LAMBDA_SOURCE * (1 - meta['mach'])) / actual_dx
                else:
                    x_val = case.LAMBDA_SOURCE / actual_dx
                marker_val = meta['cfl']

            elif analysis == 'r':
                if use_doppler:
                    x_val = meta['radius'] / (case.LAMBDA_SOURCE * (1 - meta['mach']))
                else:
                    x_val = meta['radius'] / case.LAMBDA_SOURCE
                # x_val = case.LAMBDA_SOURCE / meta['radius']
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
            ax.set_title(f"Error vs Normalised CS Radius\n(Spatial Grid Fixed at {ref_val[0]} PPW and CFL={ref_val[1]})")
            ax.set_xlabel(f"Normalised Radius ($R / \lambda$)")
        ax.set_ylabel("Log L2 Error (%)")
        ax.set_yscale('log')
        ax.grid(True, which="both", linestyle='--', alpha=0.6)
        
        ax.legend(handles=legend_elements, bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_px_parameter_space(compiled_data, use_doppler=True, shape_by='elements', draw_mach_lines=False, mach_number=None):
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
        norm = mcolors.LogNorm(vmin=min(errors), vmax=max(errors))
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
        title_prefix = "Doppler-Shifted " if use_doppler else "Source "
        if mach_number is None:
            ax.set_title(f"Temporal ({title_prefix}PPP) vs Spatial ({title_prefix}PPW)\nColored by L2 Error, Shaped by {shape_by.upper()}")
        else:
            ax.set_title(f"Temporal ({title_prefix}PPP) vs Spatial ({title_prefix}PPW)\nColoured by L2 Error, Shaped by {shape_by.upper()}, Mach number = {mach_number}")
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
        legend_title = r'$\bf{Element\ Count}$' if shape_by == 'elements' else r'$\bf{Time\ Step\ (\Delta t)}$'
        lbls = ['< 50k', '50k - 250k', '250k - 1M', '> 1M'] if shape_by == 'elements' else ['> 5e-3 s', '1e-3 to 5e-3 s', '1e-4 to 1e-3 s', '< 1e-4 s']
        
        legend_elements = [
            Line2D([0], [0], color='w', label=legend_title, lw=0),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', markeredgecolor='black', markersize=8, label=lbls[0]),
            Line2D([0], [0], marker='s', color='w', markerfacecolor='gray', markeredgecolor='black', markersize=8, label=lbls[1]),
            Line2D([0], [0], marker='^', color='w', markerfacecolor='gray', markeredgecolor='black', markersize=8, label=lbls[2]),
            Line2D([0], [0], marker='D', color='w', markerfacecolor='gray', markeredgecolor='black', markersize=8, label=lbls[3])
        ]
        ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.0, 1.0))

        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_px_contour_space(compiled_data, use_doppler=True, shape_by='elements', draw_mach_lines=False):
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
        
        norm = mcolors.LogNorm(vmin=min(errors), vmax=max(errors))
        cmap = cm.coolwarm
        
        # 1. Generate the Contour Mesh (in Log Space for accurate scaling)
        # We create a fine 100x100 grid spanning the min and max of your data
        xi = np.logspace(np.log10(X_coords.min()), np.log10(X_coords.max()), 100)
        pi = np.logspace(np.log10(P_coords.min()), np.log10(P_coords.max()), 100)
        XI, PI = np.meshgrid(xi, pi)
        
        # Interpolate the error data onto the grid using linear triangulation
        ZI = griddata((X_coords, P_coords), errors, (XI, PI), method='linear')
        
        # Plot the background contour (levels map to the log scale)
        contour = ax.contourf(XI, PI, ZI, levels=np.logspace(np.log10(errors.min()), np.log10(errors.max()), 20), 
                              norm=norm, cmap=cmap, alpha=0.6, zorder=0)

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
                ax.plot(xs, ps, color='black', linestyle='-', linewidth=1.5, alpha=0.5, zorder=1)
                ax.text(xs[-1], ps[-1], f" M={m:.2f}", fontsize=8, color='black', verticalalignment='center', zorder=2)

        # 3. Scatter Plotting (Overlaying the actual anchor points)
        for d in compiled_data:
            # We color the edge black and fill with the exact error color so it blends but remains visible
            color_val = cmap(norm(d['error']))
            marker_shape = get_marker(d, shape_by)
            ax.scatter(d['X'], d['P'], marker=marker_shape, color=color_val, 
                       edgecolor='black', s=50, linewidth=0.8, zorder=3)

        # 4. Formatting
        title_prefix = "Doppler-Shifted " if use_doppler else "Source "
        ax.set_title(f"Acoustic Error Contour Map: Temporal ({title_prefix}PPP) vs Spatial ({title_prefix}PPW)\nShaped by {shape_by.upper()}")
        ax.set_xlabel(f"Spatial Resolution, X ({title_prefix}Points Per Wavelength)")
        ax.set_ylabel(f"Temporal Resolution, P ({title_prefix}Points Per Period)")
        
        # ax.set_xscale('log')
        # ax.set_yscale('log')
        
        # Colorbar
        cbar = fig.colorbar(contour, ax=ax)
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
        ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.0, 1.0))

        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_3d_parameter_surface(compiled_data, use_doppler=True, shape_by='elements'):
        import matplotlib.pyplot as plt
        import matplotlib.cm as cm
        import matplotlib.colors as mcolors
        import numpy as np
        
        if not compiled_data:
            print("No data provided to plot.")
            return

        # Create a wider figure to fit two colorbars
        fig = plt.figure(figsize=(14, 9))
        ax = fig.add_subplot(111, projection='3d')
        
        # Extract data
        X_coords = np.array([d['X'] for d in compiled_data])
        P_coords = np.array([d['P'] for d in compiled_data])
        errors = np.array([d['error'] for d in compiled_data])
        
        # Extract the cost metric for the dots
        costs = np.array([d[shape_by] for d in compiled_data])
        
        # We pass the raw error values to Z to keep the axis readable, 
        # but tell matplotlib to treat the Z-axis logarithmically
        Z_coords = errors
        
        # 1. Plot the Surface (Colored by Error using coolwarm)
        # norm_error = mcolors.LogNorm(vmin=errors.min(), vmax=errors.max())
        # surf = ax.plot_trisurf(X_coords, P_coords, Z_coords, 
        #                        cmap=cm.coolwarm, norm=norm_error, 
        #                        edgecolor='none', alpha=0.6, linewidth=0.2)
        # 1. Plot the Surface (Colored by Error using coolwarm)
        norm_error = mcolors.LogNorm(vmin=errors.min(), vmax=errors.max())
        surf = ax.plot_trisurf(X_coords, P_coords, Z_coords, 
                               cmap=cm.coolwarm, norm=norm_error, 
                               edgecolor='none',  # Ensures no chaotic mesh lines are drawn
                               antialiased=True,  # Smooths the surface rendering
                               alpha=0.6)
        
        # 2. Plot the Dots (Colored by Cost using viridis to contrast the surface)
        norm_cost = mcolors.LogNorm(vmin=costs.min(), vmax=costs.max())
        scatter = ax.scatter(X_coords, P_coords, Z_coords, 
                             c=costs, cmap=cm.viridis, norm=norm_cost, 
                             s=40, edgecolor='black', linewidth=0.5, zorder=3)

        # Formatting
        title_prefix = "Doppler-Shifted " if use_doppler else "Source "
        ax.set_title(f"3D Acoustic Error Topography\nSurface: L2 Error | Dots: {shape_by.upper()}")
        ax.set_xlabel(f"Spatial Resolution, X ({title_prefix}PPW)")
        ax.set_ylabel(f"Temporal Resolution, P ({title_prefix}PPP)")
        ax.set_zlabel("L2 Error (%)")
        
        # Force the Z-axis to behave logarithmically 
        ax.set_zscale('log')
        
        # Adjust viewing angle
        ax.view_init(elev=30, azim=225) 
        
        # Add two distinct colorbars so Fan can read both metrics
        cbar_error = fig.colorbar(surf, ax=ax, shrink=0.5, aspect=15, pad=0.05)
        cbar_error.set_label('Surface: L2 Error (%)')
        
        cbar_cost = fig.colorbar(scatter, ax=ax, shrink=0.5, aspect=15, pad=0.1)
        cbar_cost.set_label(f'Dots: {shape_by.upper()} Cost')
        
        plt.tight_layout()
        plt.show()