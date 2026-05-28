import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import os
from pathlib import Path

class FWHDataset:
    X_TARGET = 20.0        
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
        
        cfl_mach_match = re.search(r"cfl(?P<cfl>\d+)_M(?P<mach>\d+)", self.name)
        if cfl_mach_match:
            meta['cfl'] = float(cfl_mach_match.group('cfl')) / 100.0
            meta['mach_str'] = cfl_mach_match.group('mach')
            meta['mach'] = float(f"0.{meta['mach_str'][1:]}")
            meta['dt'] = (meta['cfl'] * dx) / self.C
            meta['grid'] = int(np.round((2 * np.pi * self.R) / dx))
            
            return meta

        rad_mach_match = re.search(r"r(\d+)_M(\d+)([cmf])", self.name)
        if rad_mach_match:
            meta['radius'] = int(rad_mach_match.group(1))
            meta['mach_str'] = rad_mach_match.group(2)
            meta['mach'] = float(f"0.{meta['mach_str'][1:]}")
            meta['grid'] = rad_mach_match.group(3)
            return meta
            
        time_match = re.search(r"t_M(\d+)([cmf])", self.name)
        if time_match:
            meta['mach_str'] = time_match.group(1)
            meta['mach'] = float(f"0.{meta['mach_str'][1:]}")
            meta['time_lbl'] = time_match.group(2)
            
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