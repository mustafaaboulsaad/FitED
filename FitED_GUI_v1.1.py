"""
Author: Mustafa Mahmoud Aboulsaad
Email: mustafa.aboulsaad@outlook.com
"""

from __future__ import annotations

import json
import time
import zipfile
from pathlib import Path
from tempfile import TemporaryDirectory

import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import copy

from pl_fitting_backend_step16_aic import (
    BUILTIN_PEAK_KINDS,
    FIT_SELECTION_CRITERIA,
    load_spectrum,
    crop_roi,
    smooth_if_requested,
    build_composite_model,
    normalize_custom_profile_definition,
    fit_selection_score,
    fit_metric_summary,
)



APP_TITLE = "FitED"
AUTHOR_NAME = "Mustafa Mahmoud Ibrahim Aboulsaad"
APP_VERSION = "v1.1"
SOFTWARE_DOI = "10.5281/zenodo.20044656"
Citation = "Aboulsaad, M. M. FitED. Zenodo. https://doi.org/10.5281/zenodo.20044656"
LICENSE_NAME = "FitED Non-Commercial Software License"
DISCLAIMER_TEXT = (
    "This software is provided as-is, without warranty of any kind, express or implied. "
    "The author shall not be liable for any claim, damages, or other liability arising from, "
    "out of, or in connection with the software or the use or other dealings in the software."
)

STARTUP_NOTICE = f"""{APP_TITLE} {APP_VERSION}

Author: {AUTHOR_NAME}

If you use this software in academic work, please cite:
{Citation}

This software is for non-commercial use only.

{DISCLAIMER_TEXT}

By clicking Accept, you acknowledge these terms."""

class ScrollableFrame(ttk.Frame):
    def __init__(self, parent, *args, **kwargs):
        super().__init__(parent, *args, **kwargs)
        self.canvas = tk.Canvas(self, highlightthickness=0)
        self.v_scroll = ttk.Scrollbar(self, orient='vertical', command=self.canvas.yview)
        self.h_scroll = ttk.Scrollbar(self, orient='horizontal', command=self.canvas.xview)
        self.inner = ttk.Frame(self.canvas)

        self.inner.bind(
            '<Configure>',
            lambda e: self.canvas.configure(scrollregion=self.canvas.bbox('all')),
        )
        self.window_id = self.canvas.create_window((0, 0), window=self.inner, anchor='nw')
        self.canvas.configure(yscrollcommand=self.v_scroll.set, xscrollcommand=self.h_scroll.set)

        self.canvas.grid(row=0, column=0, sticky='nsew')
        self.v_scroll.grid(row=0, column=1, sticky='ns')
        self.h_scroll.grid(row=1, column=0, sticky='ew')
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)

        self.canvas.bind('<Configure>', self._on_canvas_configure)
        self.inner.bind('<Enter>', lambda e: self._bind_mousewheel())
        self.inner.bind('<Leave>', lambda e: self._unbind_mousewheel())

    def _on_canvas_configure(self, event):
        self.canvas.itemconfigure(self.window_id, width=event.width)

    def _on_mousewheel(self, event):
        if event.delta:
            self.canvas.yview_scroll(int(-1 * (event.delta / 120)), 'units')
        elif getattr(event, 'num', None) == 4:
            self.canvas.yview_scroll(-1, 'units')
        elif getattr(event, 'num', None) == 5:
            self.canvas.yview_scroll(1, 'units')

    def _bind_mousewheel(self):
        self.canvas.bind_all('<MouseWheel>', self._on_mousewheel)
        self.canvas.bind_all('<Button-4>', self._on_mousewheel)
        self.canvas.bind_all('<Button-5>', self._on_mousewheel)

    def _unbind_mousewheel(self):
        self.canvas.unbind_all('<MouseWheel>')
        self.canvas.unbind_all('<Button-4>')
        self.canvas.unbind_all('<Button-5>')


class DesktopPLFitterApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title(APP_TITLE)
        self.root.geometry('1380x900')
        self.root.minsize(1180, 760)

        self.current_file: Path | None = None
        self.x_full: np.ndarray | None = None
        self.y_full: np.ndarray | None = None

        self.fit_result = None
        self.last_components = None
        self.last_best_fit = None
        self.last_roi = None
        self.last_session_path: Path | None = None
        self.custom_profiles: dict[str, dict] = {}
        self.custom_background_profiles: dict[str, dict] = {}

        self._build_state()
        self._build_ui()
        self._build_plot()
        self._setup_hover()
        self.pick_centers_mode = False
        self.next_center_pick_index = 0
        self.center_pick_lines = []
        self.last_fit_peak_count = 0
        self.picked_peak_indices = set()

        # Peak-drag interaction state. Dragging updates only the model preview;
        # click Run fit afterward if you want the optimizer/export to use it.
        self.peak_drag_state = None
        self.peak_drag_pick_tolerance_px = 18
        self._last_peak_drag_draw = 0.0
    
    def _custom_profile_has_center(self, row):
        if row['kind'].get() != 'Custom':
            return False
    
        profile_name = row.get('custom_profile')
        if profile_name is None:
            return False
    
        profile = self.custom_profiles.get(profile_name.get())
        if not profile:
            return False
    
        param_names = {p['name'].strip().lower() for p in profile.get('parameters', [])}
        return 'center' in param_names
    
    def _get_row_custom_param_defs(self, row):
        defs = []
        custom_params = row.get('custom_params', {})
        for name, cfg in custom_params.items():
            defs.append({
                'name': name,
                'value': float(cfg['value_var'].get()),
                'min': float(cfg['min_var'].get()),
                'max': float(cfg['max_var'].get()),
            })
        return defs
    
    def _current_fit_criterion(self):
        """Return the selected model-selection criterion for auto-fit trials."""
        if hasattr(self, 'fit_criterion_var'):
            return self.fit_criterion_var.get()
        return 'AIC'

    def _fit_selection_score(self, result, criterion=None):
        """Score a fit result using the backend criterion helper. Lower is better."""
        return fit_selection_score(result, criterion or self._current_fit_criterion())

    def _fit_metric_summary(self, result):
        """Compact text for the status bar and messages."""
        return fit_metric_summary(result)

    def _run_autoprefit_search_custom(self, x, y_raw, peak_defs, n_trials=None):
        if n_trials is None:
            n_trials = max(1, int(self.autofit_trials_var.get()))
    
        best_result = None
        best_defs = None
        best_score = np.inf
        last_error = None
    
        for trial in range(n_trials):
            self.status_var.set(f'Custom auto pre-fit: trial {trial + 1}/{n_trials} ...')
            self.root.update_idletasks()
    
            trial_defs = copy.deepcopy(peak_defs)
    
            for p in trial_defs:
                if p.get('kind') != 'Custom':
                    continue
    
                custom_params = p.get('custom_params', {})
                for pname, cfg in custom_params.items():
                    val = float(cfg['value'])
                    pmin = float(cfg['min'])
                    pmax = float(cfg['max'])
    
                    if not np.isfinite(pmin):
                        pmin = val * 0.2 if val != 0 else -10.0
                    if not np.isfinite(pmax):
                        pmax = val * 5.0 if val != 0 else 10.0
    
                    if pmin > pmax:
                        pmin, pmax = pmax, pmin
    
                    # log-like randomization for positive-only params
                    if pmin >= 0 and pmax > 0:
                        lo = max(pmin, 1e-12)
                        hi = max(pmax, lo * 1.0001)
                        trial_val = np.exp(np.random.uniform(np.log(lo), np.log(hi)))
                    else:
                        # linear randomization otherwise
                        trial_val = np.random.uniform(pmin, pmax)
    
                    cfg['value'] = float(np.clip(trial_val, pmin, pmax))
    
            try:
                model, params = build_composite_model(
                    trial_defs,
                    background_kind=self.background_var.get(),
                    poly_order=int(self.poly_order_var.get()),
                    x=x,
                    y=y_raw,
                    custom_profiles=self.custom_profiles,   # only if your backend expects this
                )
    
                result = model.fit(
                    y_raw,
                    params=params,
                    x=x,
                    weights=self._weights(y_raw),
                    nan_policy='raise'
                )
    
                if not np.all(np.isfinite(result.best_fit)):
                    continue
    
                score = self._fit_selection_score(result)
                if np.isfinite(score) and score < best_score:
                    best_score = score
                    best_result = result
                    best_defs = copy.deepcopy(trial_defs)
    
            except Exception as exc:
                last_error = exc
                continue
    
        if best_result is None and last_error is not None:
            raise RuntimeError(f'Custom auto pre-fit failed. Last error: {last_error}')
    
        return best_result, best_defs
    
    def _clear_fit_and_pick_state(self):
        self.fit_result = None
        self.last_components = None
        self.last_best_fit = None
        self.last_roi = None
        self.last_session_path = None
        self.last_fit_peak_count = 0
    
        self.pick_centers_mode = False
        self.next_center_pick_index = 0
        self.picked_peak_indices.clear()
        self._clear_center_pick_lines()
            
    def rebuild_peaks_fresh(self):
        count = max(1, int(self.peak_count_var.get()))

        x_min = float(np.min(self.x_full)) if self.x_full is not None else 0.0
        x_max = float(np.max(self.x_full)) if self.x_full is not None else 100.0
        x_span = max(x_max - x_min, 1.0)

        row_states = []
        for i in range(count):
            center_guess = x_min + (i + 1) * x_span / (count + 1)
            amp_guess = 0.01
            fwhm_guess = max(x_span / 80.0, 1e-4)

            row_states.append({
                'active': True,
                'kind': 'Pseudo-Voigt',
                'custom_profile': '',
                'center': center_guess,
                'amplitude': amp_guess,
                'fwhm': fwhm_guess,
                'center_min': center_guess - 2 * fwhm_guess,
                'center_max': center_guess + 2 * fwhm_guess,
                'amplitude_min': 0.0,
                'amplitude_max': max(amp_guess * 5, 1.0),
                'fwhm_min': max(x_span / 1000.0, 1e-3),
                'fwhm_max': max(fwhm_guess * 5, x_span / 50.0),
                'fraction': 0.5,
                'sigma': max(fwhm_guess / 2.354820045, 1e-8),
                'gamma': max(fwhm_guess / 2.0, 1e-8),
                'custom_params': {},
            })

        self._draw_peak_rows(row_states)

    def _peak_kind_values(self):
        return BUILTIN_PEAK_KINDS

    def _default_peak_state(self, idx, count):
        x_min = float(np.min(self.x_full)) if self.x_full is not None else 0.0
        x_max = float(np.max(self.x_full)) if self.x_full is not None else 100.0
        x_span = max(x_max - x_min, 1.0)
        center_guess = x_min + (idx + 1) * x_span / (count + 1)
        amp_guess = 0.01
        fwhm_guess = max(x_span / 80.0, 1e-4)
        return {
            'active': True,
            'kind': 'Pseudo-Voigt',
            'custom_profile': '',
            'center': center_guess,
            'amplitude': amp_guess,
            'fwhm': fwhm_guess,
            'center_min': center_guess - 2 * fwhm_guess,
            'center_max': center_guess + 2 * fwhm_guess,
            'amplitude_min': 0.0,
            'amplitude_max': max(amp_guess * 5, 1.0),
            'fwhm_min': max(x_span / 1000.0, 1e-3),
            'fwhm_max': max(fwhm_guess * 5, x_span / 50.0),
            'fraction': 0.5,
            'sigma': max(fwhm_guess / 2.354820045, 1e-8),
            'gamma': max(fwhm_guess / 2.0, 1e-8),
            'custom_params': {},
        }

    def _row_state_from_row(self, row):
        state = {
            'active': bool(row['active'].get()),
            'kind': row['kind'].get(),
            'custom_profile': row.get('custom_profile').get() if 'custom_profile' in row else '',
            'center': float(row['center'].get()),
            'amplitude': float(row['amplitude'].get()),
            'fwhm': float(row['fwhm'].get()),
            'center_min': float(row['center_min'].get()),
            'center_max': float(row['center_max'].get()),
            'amplitude_min': float(row['amplitude_min'].get()),
            'amplitude_max': float(row['amplitude_max'].get()),
            'fwhm_min': float(row['fwhm_min'].get()),
            'fwhm_max': float(row['fwhm_max'].get()),
            'fraction': float(row['fraction'].get()),
            'sigma': float(row['sigma'].get()),
            'gamma': float(row['gamma'].get()),
            'custom_params': {},
        }
        for name, vars_dict in row.get('custom_params', {}).items():
            state['custom_params'][name] = {
                'value': float(vars_dict['value'].get()),
                'min': float(vars_dict['min'].get()),
                'max': float(vars_dict['max'].get()),
            }
        return state

    def _draw_peak_rows(self, row_states):
        for child in self.peaks_container.winfo_children():
            child.destroy()

        self.peak_rows = []

        for i, state in enumerate(row_states):
            frm = ttk.LabelFrame(self.peaks_container, text=f'Peak {i+1}', padding=6)
            frm.pack(fill=tk.X, pady=4)

            kind_var = tk.StringVar(value=state.get('kind', 'Pseudo-Voigt'))
            custom_profile_var = tk.StringVar(value=state.get('custom_profile', ''))
            center_var = tk.DoubleVar(value=state.get('center', 0.0))
            amp_var = tk.DoubleVar(value=state.get('amplitude', 1.0))
            fwhm_var = tk.DoubleVar(value=state.get('fwhm', 1.0))
            cmin_var = tk.DoubleVar(value=state.get('center_min', 0.0))
            cmax_var = tk.DoubleVar(value=state.get('center_max', 1.0))
            amin_var = tk.DoubleVar(value=state.get('amplitude_min', 0.0))
            amax_var = tk.DoubleVar(value=state.get('amplitude_max', 1.0))
            wmin_var = tk.DoubleVar(value=state.get('fwhm_min', 1e-6))
            wmax_var = tk.DoubleVar(value=state.get('fwhm_max', 1.0))
            frac_var = tk.DoubleVar(value=state.get('fraction', 0.5))
            sigma_var = tk.DoubleVar(value=state.get('sigma', max(state.get('fwhm', 1.0) / 2.354820045, 1e-8)))
            gamma_var = tk.DoubleVar(value=state.get('gamma', max(state.get('fwhm', 1.0) / 2.0, 1e-8)))
            active_var = tk.BooleanVar(value=state.get('active', True))
            custom_param_vars = {}

            row0 = ttk.Frame(frm)
            row0.pack(fill=tk.X, anchor='w', pady=(2, 0))
            ttk.Checkbutton(row0, text='Use', variable=active_var).pack(side='left', padx=(0, 14))
            ttk.Label(row0, text='Kind').pack(side='left', padx=(0, 4))
            ttk.Combobox(row0, textvariable=kind_var, values=self._peak_kind_values(), state='readonly', width=16).pack(side='left', padx=(0, 14))
            ttk.Label(row0, text='Profile').pack(side='left', padx=(0, 4))
            custom_combo = ttk.Combobox(row0, textvariable=custom_profile_var, values=list(self.custom_profiles.keys()), state='readonly', width=18)
            custom_combo.pack(side='left', padx=(0, 30))

            ttk.Label(row0, text='G/L').pack(side='left', padx=(0, 4))
            ttk.Entry(row0, textvariable=frac_var, width=10).pack(side='left', padx=(0, 0))
            #ttk.Label(row0, text='Sigma').pack(side='left', padx=(0, 4))
            #ttk.Entry(row0, textvariable=sigma_var, width=10).pack(side='left', padx=(0, 14))
            #ttk.Label(row0, text='Gamma').pack(side='left', padx=(0, 4))
            #ttk.Entry(row0, textvariable=gamma_var, width=10).pack(side='left', padx=(0, 0))

            row1 = ttk.Frame(frm)
            row1.pack(fill=tk.X, anchor='w', pady=(4, 0))
            ttk.Label(row1, text='Center').pack(side='left', padx=(0, 4))
            ttk.Entry(row1, textvariable=center_var, width=10).pack(side='left', padx=(0, 14))
            ttk.Label(row1, text='c min').pack(side='left', padx=(0, 4))
            ttk.Entry(row1, textvariable=cmin_var, width=10).pack(side='left', padx=(0, 14))
            ttk.Label(row1, text='c max').pack(side='left', padx=(0, 4))
            ttk.Entry(row1, textvariable=cmax_var, width=10).pack(side='left', padx=(0, 57))
            ttk.Label(row1, text='Sigma').pack(side='left', padx=(0, 4))
            ttk.Entry(row1, textvariable=sigma_var, width=10).pack(side='left', padx=(0, 0))
            

            row2 = ttk.Frame(frm)
            row2.pack(fill=tk.X, anchor='w', pady=(4, 0))
            ttk.Label(row2, text='Area').pack(side='left', padx=(0, 4))
            ttk.Entry(row2, textvariable=amp_var, width=10).pack(side='left', padx=(0, 14))
            ttk.Label(row2, text='a min').pack(side='left', padx=(0, 4))
            ttk.Entry(row2, textvariable=amin_var, width=10).pack(side='left', padx=(0, 14))
            ttk.Label(row2, text='a max').pack(side='left', padx=(0, 4))
            ttk.Entry(row2, textvariable=amax_var, width=10).pack(side='left', padx=(0, 59.5))
            ttk.Label(row2, text='Gamma').pack(side='left', padx=(0, 4))
            ttk.Entry(row2, textvariable=gamma_var, width=10).pack(side='left', padx=(0, 0))

            row3 = ttk.Frame(frm)
            row3.pack(fill=tk.X, anchor='w', pady=(4, 0))
            ttk.Label(row3, text='FWHM').pack(side='left', padx=(0, 4))
            ttk.Entry(row3, textvariable=fwhm_var, width=10).pack(side='left', padx=(0, 14))
            ttk.Label(row3, text='w min').pack(side='left', padx=(0, 4))
            ttk.Entry(row3, textvariable=wmin_var, width=10).pack(side='left', padx=(0, 14))
            ttk.Label(row3, text='w max').pack(side='left', padx=(0, 4))
            ttk.Entry(row3, textvariable=wmax_var, width=10).pack(side='left', padx=(0, 0))

            custom_frame = ttk.LabelFrame(frm, text='Custom profile parameters', padding=6)

            def rebuild_custom_fields(*_):
                for child in custom_frame.winfo_children():
                    child.destroy()
                custom_param_vars.clear()
                profile = self.custom_profiles.get(custom_profile_var.get())
                is_custom = kind_var.get().strip().lower() == 'custom'

                custom_combo.configure(values=list(self.custom_profiles.keys()), state='readonly' if is_custom and self.custom_profiles else 'disabled')
                if not is_custom:
                    custom_frame.pack_forget()
                    return

                custom_frame.pack(fill=tk.X, pady=(6, 0))
                if not profile:
                    ttk.Label(custom_frame, text='No custom profile selected.').pack(anchor='w')
                    return

                saved = state.get('custom_params', {})
                row_idx = 0
                for param in profile.get('parameters', []):
                    name = param['name']
                    if name in {'center', 'amplitude', 'fwhm'}:
                        continue
                    cfg = saved.get(name, {})
                    value_var = tk.DoubleVar(value=cfg.get('value', param.get('default', 1.0)))
                    min_var = tk.DoubleVar(value=cfg.get('min', param.get('min', float('-inf'))))
                    max_var = tk.DoubleVar(value=cfg.get('max', param.get('max', float('inf'))))
                    custom_param_vars[name] = {'value': value_var, 'min': min_var, 'max': max_var}

                    ttk.Label(custom_frame, text=name).grid(row=row_idx, column=0, sticky='w', padx=(0, 4), pady=2)
                    ttk.Entry(custom_frame, textvariable=value_var, width=12).grid(row=row_idx, column=1, sticky='w', padx=(0, 8), pady=2)
                    ttk.Label(custom_frame, text='min').grid(row=row_idx, column=2, sticky='w', padx=(0, 4), pady=2)
                    ttk.Entry(custom_frame, textvariable=min_var, width=12).grid(row=row_idx, column=3, sticky='w', padx=(0, 8), pady=2)
                    ttk.Label(custom_frame, text='max').grid(row=row_idx, column=4, sticky='w', padx=(0, 4), pady=2)
                    ttk.Entry(custom_frame, textvariable=max_var, width=12).grid(row=row_idx, column=5, sticky='w', pady=2)
                    row_idx += 1

                if row_idx == 0:
                    ttk.Label(custom_frame, text='No extra parameters beyond center/amplitude/fwhm.').pack(anchor='w')

            kind_var.trace_add('write', rebuild_custom_fields)
            custom_profile_var.trace_add('write', rebuild_custom_fields)
            rebuild_custom_fields()

            self.peak_rows.append({
                'active': active_var,
                'kind': kind_var,
                'custom_profile': custom_profile_var,
                'center': center_var,
                'amplitude': amp_var,
                'fwhm': fwhm_var,
                'center_min': cmin_var,
                'center_max': cmax_var,
                'amplitude_min': amin_var,
                'amplitude_max': amax_var,
                'fwhm_min': wmin_var,
                'fwhm_max': wmax_var,
                'fraction': frac_var,
                'sigma': sigma_var,
                'gamma': gamma_var,
                'custom_params': custom_param_vars,
            })

        self.root.update_idletasks()
        self.peaks_scroll.canvas.configure(scrollregion=self.peaks_scroll.canvas.bbox('all'))

    
        self._clear_fit_and_pick_state()
        self.rebuild_peaks_fresh()
        self._plot_loaded_data()
    
        self.status_var.set('Peaks, picked centers, and fit state reset for current spectrum. ROI kept unchanged.')
    
    def refine_with_added_peaks(self):
        if self.x_full is None or self.y_full is None:
            messagebox.showinfo('No data', 'Load a file first.')
            return
    
        if self.fit_result is None:
            messagebox.showinfo('No previous fit', 'Run Auto pre-fit or Run fit on the main peaks first.')
            return
    
        try:
            x, y_raw, _ = self._get_roi_data()
            peak_defs = self._collect_peak_defs()
            active_count = len(peak_defs)
    
            if active_count <= self.last_fit_peak_count:
                raise ValueError('No newly added active peaks detected.')
    
            old_count = self.last_fit_peak_count
            params_prev = self.fit_result.params
    
            # Start from current UI values
            staged_defs = copy.deepcopy(peak_defs)
    
            # Keep previously fitted peaks close to the old solution
            for i in range(min(old_count, len(staged_defs))):
                p = staged_defs[i]
                prefix = f'p{i+1}_'
    
                if f'{prefix}center' in params_prev:
                    c = float(params_prev[f'{prefix}center'].value)
                    p['center'] = c
                    halfw = max(0.15 * (float(p['center_max']) - float(p['center_min'])), 1e-6)
                    p['center_min'] = c - halfw
                    p['center_max'] = c + halfw
    
                if f'{prefix}amplitude' in params_prev:
                    a = max(float(params_prev[f'{prefix}amplitude'].value), 1e-12)
                    p['amplitude'] = a
                    p['amplitude_min'] = 0.0
                    p['amplitude_max'] = max(a * 2.0, float(p['amplitude_max']))
    
                if f'{prefix}sigma' in params_prev:
                    sigma = max(float(params_prev[f'{prefix}sigma'].value), 1e-8)
                    p['sigma'] = sigma
                    fwhm = 2.354820045 * sigma
                    p['fwhm'] = fwhm
                    p['fwhm_min'] = max(fwhm * 0.7, 1e-8)
                    p['fwhm_max'] = max(fwhm * 1.5, p['fwhm_min'] * 1.2)
    
                if f'{prefix}gamma' in params_prev:
                    p['gamma'] = max(float(params_prev[f'{prefix}gamma'].value), 1e-8)
    
            # Seed only the newly added peaks from centers
            if old_count < len(staged_defs):
                new_defs = self._seed_peak_defs_from_centers(x, y_raw, staged_defs[old_count:])
                staged_defs[old_count:] = new_defs
    
            # Stage 1: fit with old peaks constrained, new peaks freer
            best_result = None
            best_score = np.inf
            n_trials = max(1, int(self.autofit_trials_var.get()))
    
            for trial in range(n_trials):
                self.status_var.set(f'Refine with added peaks: trial {trial + 1}/{n_trials} ...')
                self.root.update_idletasks()
    
                trial_defs = copy.deepcopy(staged_defs)
    
                # Only vary the newly added peaks strongly
                for i in range(old_count, len(trial_defs)):
                    p = trial_defs[i]
    
                    c = float(p['center'])
                    cmin = float(p['center_min'])
                    cmax = float(p['center_max'])
                    fwhm = max(float(p['fwhm']), 1e-12)
                    amp = max(float(p['amplitude']), 1e-12)
    
                    width_scale = [0.8, 1.0, 1.25][trial % 3]
                    amp_scale = [0.7, 1.0, 1.4][(trial // 3) % 3]
    
                    jitter = 0.12 * (cmax - cmin)
                    p['center'] = np.clip(c + np.random.uniform(-jitter, jitter), cmin, cmax)
                    p['fwhm'] = np.clip(fwhm * width_scale, float(p['fwhm_min']), float(p['fwhm_max']))
                    p['amplitude'] = np.clip(amp * amp_scale, 1e-12, float(p['amplitude_max']))
                    p['sigma'] = max(float(p['fwhm']) / 2.354820045, 1e-8)
                    p['gamma'] = max(float(p['fwhm']) / 2.0, 1e-8)
    
                try:
                    model, params = build_composite_model(
                        trial_defs,
                        background_kind=self.background_var.get(),
                        poly_order=int(self.poly_order_var.get()),
                        x=x,
                        y=y_raw,
                    )
    
                    result = model.fit(
                        y_raw,
                        params=params,
                        x=x,
                        weights=self._weights(y_raw),
                        nan_policy='raise'
                    )
    
                    score = self._fit_selection_score(result)
                    if np.isfinite(score) and score < best_score:
                        best_score = score
                        best_result = result
    
                except Exception:
                    continue
    
            if best_result is None:
                raise RuntimeError('Refinement failed for all trials.')
    
            # Stage 2: one final full refinement starting from stage-1 best result
            model2, params2 = build_composite_model(
                peak_defs,
                background_kind=self.background_var.get(),
                poly_order=int(self.poly_order_var.get()),
                x=x,
                y=y_raw,
                custom_profiles=self.custom_profiles,
            )
    
            for name, par in params2.items():
                if name in best_result.params:
                    val = best_result.params[name].value
                    try:
                        params2[name].set(value=val)
                    except Exception:
                        pass
    
            final_result = model2.fit(
                y_raw,
                params=params2,
                x=x,
                weights=self._weights(y_raw),
                nan_policy='raise'
            )
    
            self.fit_result = final_result
            self.last_components = final_result.eval_components(x=x)
            self.last_best_fit = final_result.best_fit
            self.last_roi = (x.copy(), y_raw.copy())
            self.last_fit_peak_count = active_count
    
            # write fit values back into UI
            self._apply_peak_defs_to_ui(peak_defs, fit_result=final_result)
    
            self.ax_main.clear()
            self.ax_resid.clear()
            self._refresh_hover_axis()
            self.ax_main.plot(x, y_raw, 'k.', ms=3, alpha=0.6, label='Data')
            self.ax_main.plot(x, final_result.best_fit, linewidth=2.1, label='Refined fit')
            for name, comp in self.last_components.items():
                self.ax_main.plot(x, comp, '--', linewidth=1.0, alpha=0.85, label=name)
            self.ax_main.set_title('Refined fit with added peaks')
            self.ax_main.set_ylabel('Y')
            self.ax_main.legend(fontsize=8, ncol=2)
    
            resid = y_raw - final_result.best_fit
            self.ax_resid.axhline(0.0, linestyle='--', linewidth=1.0)
            self.ax_resid.plot(x, resid, linewidth=1.0)
            self.ax_resid.set_xlabel('X')
            self.ax_resid.set_ylabel('Residual')
            self.fig.tight_layout()
            self.canvas.draw_idle()
    
            criterion = self._current_fit_criterion()
            final_metric = self._fit_selection_score(final_result, criterion)
            self.status_var.set(
                f'Refine with added peaks complete. Best {criterion}: {final_metric:.6g}. '
                f'{self._fit_metric_summary(final_result)}'
            )
    
        except Exception as exc:
            messagebox.showerror('Refine with added peaks error', str(exc))
    def start_pick_centers(self):
        active_rows = [row for row in self.peak_rows if row['active'].get()]
        if not active_rows:
            messagebox.showinfo('No active peaks', 'Activate at least one peak first.')
            return
    
        self.pick_centers_mode = True
    
        # Start from the first active peak that has not yet been explicitly picked
        start_idx = 0
        for i in range(len(active_rows)):
            if i not in self.picked_peak_indices:
                start_idx = i
                break
        else:
            # if all were picked already, start again from the first one
            start_idx = 0
            self.picked_peak_indices.clear()
    
        self.next_center_pick_index = start_idx
        self._clear_center_pick_lines()
    
        self.status_var.set(
            f'Pick mode ON: double-click peaks on the main plot to set centers '
            f'(starting from peak {self.next_center_pick_index + 1}).'
        )
    
    def _on_pick_center(self, event):
        if not self.pick_centers_mode:
            return
    
        if event.inaxes != self.ax_main:
            return
    
        if event.xdata is None:
            return
    
        # Matplotlib double click
        if not getattr(event, 'dblclick', False):
            return
    
        active_rows = [row for row in self.peak_rows if row['active'].get()]
        if self.next_center_pick_index >= len(active_rows):
            self.pick_centers_mode = False
            self.status_var.set('All active peak centers have been picked.')
            return
    
        x_pick = float(event.xdata)
        row = active_rows[self.next_center_pick_index]
        row['center'].set(x_pick)
        self.picked_peak_indices.add(self.next_center_pick_index)
    
        line = self.ax_main.axvline(x_pick, linestyle='--', linewidth=1.0, alpha=0.8)
        self.center_pick_lines.append(line)
        self.canvas.draw_idle()
    
        self.next_center_pick_index += 1
    
        if self.next_center_pick_index >= len(active_rows):
            self.pick_centers_mode = False
            self.status_var.set('Finished picking centers. Now click Auto pre-fit.')
        else:
            self.status_var.set(
                f'Center set for peak {self.next_center_pick_index}. '
                f'Double-click next peak.'
            )
    
    def _clear_center_pick_lines(self):
        for line in self.center_pick_lines:
            try:
                line.remove()
            except Exception:
                pass
        self.center_pick_lines = []
        
        
    def autofill_from_centers(self):
        if self.x_full is None or self.y_full is None:
            messagebox.showinfo('No data', 'Load a file first.')
            return
    
        try:
            x, y_raw, _ = self._get_roi_data()
            peak_defs = self._collect_peak_defs()
    
            active_rows = [row for row in self.peak_rows if row['active'].get()]
    
            # decide whether this is a center-based peak fit or a general custom fit
            all_custom_no_center = (
                len(active_rows) > 0 and
                all(row['kind'].get() == 'Custom' and not self._custom_profile_has_center(row)
                    for row in active_rows)
            )
    
            if all_custom_no_center:
                best_result, best_peak_defs = self._run_autoprefit_search_custom(x, y_raw, peak_defs)
            else:
                seeded_peak_defs = self._seed_peak_defs_from_centers(x, y_raw, peak_defs)
                best_result, best_peak_defs = self._run_autoprefit_search(x, y_raw, seeded_peak_defs)
    
            if best_result is None:
                raise RuntimeError('Automatic pre-fit failed for all attempts.')
    
            self._apply_peak_defs_to_ui(best_peak_defs, fit_result=best_result)
    
            self.fit_result = best_result
            self.last_components = best_result.eval_components(x=x)
            self.last_best_fit = best_result.best_fit
            self.last_roi = (x.copy(), y_raw.copy())
    
            self.ax_main.clear()
            self.ax_resid.clear()
            self._refresh_hover_axis()
            self.ax_main.plot(x, y_raw, 'k.', ms=3, alpha=0.6, label='Data')
            self.ax_main.plot(x, best_result.best_fit, linewidth=2.1, label='Auto pre-fit')
    
            for name, comp in self.last_components.items():
                self.ax_main.plot(x, comp, '--', linewidth=1.0, alpha=0.85, label=name)
    
            self.ax_main.set_title('Automatic pre-fit')
            self.ax_main.set_ylabel('Y')
            self.ax_main.legend(fontsize=8, ncol=2)
    
            resid = y_raw - best_result.best_fit
            self.ax_resid.axhline(0.0, linestyle='--', linewidth=1.0)
            self.ax_resid.plot(x, resid, linewidth=1.0)
            self.ax_resid.set_xlabel('X')
            self.ax_resid.set_ylabel('Residual')
            self.fig.tight_layout()
            self.canvas.draw_idle()
    
            criterion = self._current_fit_criterion()
            best_metric = self._fit_selection_score(best_result, criterion)
            self.status_var.set(
                f'Automatic pre-fit complete. Best {criterion}: {best_metric:.6g}. '
                f'{self._fit_metric_summary(best_result)}'
            )
    
        except Exception as exc:
            messagebox.showerror('Auto pre-fit error', str(exc))
    
    
    def _seed_peak_defs_from_centers(self, x, y_raw, peak_defs):
        x = np.asarray(x, dtype=float)
        y_raw = np.asarray(y_raw, dtype=float)
    
        x_span = max(float(np.max(x) - np.min(x)), 1e-12)
        y_min = float(np.min(y_raw))
        y_max = float(np.max(y_raw))
        y_span = max(y_max - y_min, 1.0)
    
        centers = [float(p['center']) for p in peak_defs]
        centers_sorted = sorted(centers)
    
        new_defs = copy.deepcopy(peak_defs)
    
        for p in new_defs:
            c = float(p['center'])
            idx = int(np.argmin(np.abs(x - c)))
            local_y = float(y_raw[idx])
    
            if len(centers_sorted) == 1:
                spacing = x_span / 5.0
            else:
                pos = centers_sorted.index(c)
                left_gap = np.inf if pos == 0 else abs(c - centers_sorted[pos - 1])
                right_gap = np.inf if pos == len(centers_sorted) - 1 else abs(centers_sorted[pos + 1] - c)
                spacing = min(left_gap, right_gap)
                if not np.isfinite(spacing) or spacing <= 0:
                    spacing = x_span / 5.0
    
            fwhm_guess = max(spacing * 0.5, x_span / 200.0)
            fwhm_min = max(fwhm_guess * 0.15, x_span / 3000.0)
            fwhm_max = max(fwhm_guess * 4.0, x_span / 20.0)
    
            center_half_window = max(0.75 * spacing, 1.2 * fwhm_guess)
            center_min = c - center_half_window
            center_max = c + center_half_window
    
            height_guess = max(local_y - y_min, 0.03 * y_span)
            area_guess = max(height_guess * fwhm_guess, 1e-9)
    
            p['amplitude'] = area_guess
            p['fwhm'] = fwhm_guess
            p['center_min'] = center_min
            p['center_max'] = center_max
            p['amplitude_min'] = 0.0
            p['amplitude_max'] = max(area_guess * 12.0, y_span * x_span)
            p['fwhm_min'] = fwhm_min
            p['fwhm_max'] = fwhm_max
            p['sigma'] = max(fwhm_guess / 2.354820045, 1e-8)
            p['gamma'] = max(fwhm_guess / 2.0, 1e-8)
            if p.get('kind', '').strip().lower() == 'custom' and p.get('custom_profile'):
                profile = self.custom_profiles.get(p['custom_profile'])
                if profile:
                    for param in profile.get('parameters', []):
                        name = param['name']
                        if name in {'center', 'amplitude', 'fwhm'}:
                            continue
                        p.setdefault(name, float(param.get('default', 1.0)))
                        p.setdefault(f'{name}_min', float(param.get('min', float('-inf'))))
                        p.setdefault(f'{name}_max', float(param.get('max', float('inf'))))
    
        return new_defs

    #def _run_autoprefit_search(self, x, y_raw, base_peak_defs, n_trials=35):
    def _run_autoprefit_search(self, x, y_raw, base_peak_defs, n_trials=None):
        if n_trials is None:
            n_trials = max(1, int(self.autofit_trials_var.get()))
        best_result = None
        best_defs = None
        best_score = np.inf
        last_error = None
    
        x = np.asarray(x, dtype=float)
        y_raw = np.asarray(y_raw, dtype=float)
        x_span = max(float(np.max(x) - np.min(x)), 1e-12)
    
        for trial in range(n_trials):
            self.status_var.set(f'Auto pre-fit: trial {trial + 1}/{n_trials} ...')
            self.root.update_idletasks()
            
            trial_defs = copy.deepcopy(base_peak_defs)
    
            # progressively widen exploration a bit
            """
            center_scale = 1.0 + 0.03 * (trial % 5)
            width_scale = [0.5, 0.75, 1.0, 1.35, 1.8][trial % 5]
            amp_scale = [0.4, 0.7, 1.0, 1.5, 2.2][(trial // 5) % 5]
            """
            center_scale = 1.0 + 0.02 * (trial % 3)
            width_scale = [0.8, 1.0, 1.25][trial % 3]
            amp_scale = [0.7, 1.0, 1.4][(trial // 3) % 3]
    
            for p in trial_defs:
                c = float(p['center'])
                cmin = float(p['center_min'])
                cmax = float(p['center_max'])
                fwhm = max(float(p['fwhm']), 1e-12)
                amp = max(float(p['amplitude']), 1e-12)
    
                # randomize initial value within allowed window
                jitter = 0.15 * (cmax - cmin)
                c_trial = np.clip(
                    c + np.random.uniform(-jitter, jitter),
                    cmin,
                    cmax
                )
    
                fwhm_trial = max(fwhm * width_scale * np.random.uniform(0.85, 1.15), 1e-12)
                amp_trial = max(amp * amp_scale * np.random.uniform(0.85, 1.15), 1e-12)
    
                # optionally widen bounds a little for exploration
                #half_window = max((cmax - cmin) * 0.5 * center_scale, x_span / 5000.0)
                half_window = max((cmax - cmin) * 0.30 * center_scale, x_span / 8000.0)
                new_cmin = c - half_window
                new_cmax = c + half_window
    
                #new_fmin = max(float(p['fwhm_min']) * 0.8, x_span / 5000.0)
                #new_fmax = max(float(p['fwhm_max']) * 1.2, new_fmin * 1.5)
                new_fmin = max(float(p['fwhm_min']) * 0.9, x_span / 6000.0)
                new_fmax = max(float(p['fwhm_max']) * 1.05, new_fmin * 1.4)
                new_amin = 0.0
                new_amax = max(float(p['amplitude_max']) * 1.2, amp_trial * 5.0)
    
                p['center'] = c_trial
                p['center_min'] = new_cmin
                p['center_max'] = new_cmax
                p['fwhm'] = np.clip(fwhm_trial, new_fmin, new_fmax)
                p['fwhm_min'] = new_fmin
                p['fwhm_max'] = new_fmax
                p['amplitude'] = np.clip(amp_trial, 1e-12, new_amax)
                p['amplitude_min'] = new_amin
                p['amplitude_max'] = new_amax
                p['sigma'] = max(float(p['fwhm']) / 2.354820045, 1e-8)
                p['gamma'] = max(float(p['fwhm']) / 2.0, 1e-8)
    
            try:
                model, params = build_composite_model(
                    trial_defs,
                    background_kind=self.background_var.get(),
                    poly_order=int(self.poly_order_var.get()),
                    x=x,
                    y=y_raw,
                    custom_profiles=self.custom_profiles,
                custom_background_profiles=self.custom_background_profiles,
                custom_background_profile_name=self.background_profile_var.get(),
                background_params=self._collect_background_params(),
                )
    
                result = model.fit(
                    y_raw,
                    params=params,
                    x=x,
                    weights=self._weights(y_raw),
                    nan_policy='raise'
                )
    
                if not np.all(np.isfinite(result.best_fit)):
                    continue
    
                score = self._fit_selection_score(result)
    
                if score < best_score:
                    best_score = score
                    best_result = result
                    best_defs = copy.deepcopy(trial_defs)
    
            except Exception as exc:
                last_error = exc
                continue
    
        if best_result is None and last_error is not None:
            raise RuntimeError(f"Automatic pre-fit failed in all trials. Last error: {last_error}")
    
        return best_result, best_defs

    def _apply_peak_defs_to_ui(self, peak_defs, fit_result=None):
        active_rows = [row for row in self.peak_rows if row['active'].get()]

        for row, p in zip(active_rows, peak_defs):
            row['center'].set(float(p.get('center', row['center'].get())))
            row['amplitude'].set(float(p.get('amplitude', row['amplitude'].get())))
            row['fwhm'].set(float(p.get('fwhm', row['fwhm'].get())))
            row['center_min'].set(float(p.get('center_min', row['center_min'].get())))
            row['center_max'].set(float(p.get('center_max', row['center_max'].get())))
            row['amplitude_min'].set(float(p.get('amplitude_min', row['amplitude_min'].get())))
            row['amplitude_max'].set(float(p.get('amplitude_max', row['amplitude_max'].get())))
            row['fwhm_min'].set(float(p.get('fwhm_min', row['fwhm_min'].get())))
            row['fwhm_max'].set(float(p.get('fwhm_max', row['fwhm_max'].get())))
            row['sigma'].set(float(p.get('sigma', row['sigma'].get())))
            row['gamma'].set(float(p.get('gamma', row['gamma'].get())))
            if 'custom_profile' in row and 'custom_profile' in p:
                row['custom_profile'].set(p.get('custom_profile', row['custom_profile'].get()))

            if row['kind'].get().strip().lower() == 'custom':
                profile = self.custom_profiles.get(row['custom_profile'].get())
                if profile:
                    for param in profile['parameters']:
                        name = param['name']
                        if name in {'center', 'amplitude', 'fwhm'}:
                            continue
                        vars_dict = row['custom_params'].get(name)
                        if vars_dict:
                            vars_dict['value'].set(float(p.get(name, vars_dict['value'].get())))
                            vars_dict['min'].set(float(p.get(f'{name}_min', vars_dict['min'].get())))
                            vars_dict['max'].set(float(p.get(f'{name}_max', vars_dict['max'].get())))

        if fit_result is not None:
            for i, row in enumerate(active_rows, start=1):
                prefix = f'p{i}_'
                params = fit_result.params

                if f'{prefix}center' in params:
                    row['center'].set(float(params[f'{prefix}center'].value))
                if f'{prefix}amplitude' in params:
                    row['amplitude'].set(float(params[f'{prefix}amplitude'].value))
                if f'{prefix}sigma' in params:
                    sigma = float(params[f'{prefix}sigma'].value)
                    kind = row['kind'].get().lower()
                    if kind == 'lorentzian':
                        row['fwhm'].set(2.0 * sigma)
                    else:
                        row['fwhm'].set(2.354820045 * sigma)
                    row['sigma'].set(sigma)
                if f'{prefix}gamma' in params:
                    row['gamma'].set(float(params[f'{prefix}gamma'].value))

                if row['kind'].get().strip().lower() == 'custom':
                    profile = self.custom_profiles.get(row['custom_profile'].get())
                    if profile:
                        for param in profile['parameters']:
                            name = param['name']
                            if name in {'center', 'amplitude', 'fwhm'}:
                                continue
                            full_name = f'{prefix}{name}'
                            vars_dict = row['custom_params'].get(name)
                            if vars_dict and full_name in params:
                                vars_dict['value'].set(float(params[full_name].value))


    def _is_peak_param_name(self, name):
        head = str(name).split('_', 1)[0]
        return len(head) > 1 and head.startswith('p') and head[1:].isdigit()

    def _active_peak_rows(self):
        return [row for row in self.peak_rows if row['active'].get()]

    def _build_model_from_current_rows(self, x, y_raw):
        peak_defs = self._collect_peak_defs()
        model, params = build_composite_model(
            peak_defs,
            background_kind=self.background_var.get(),
            poly_order=int(self.poly_order_var.get()),
            x=x,
            y=y_raw,
            custom_profiles=self.custom_profiles,
            custom_background_profiles=self.custom_background_profiles,
            custom_background_profile_name=self.background_profile_var.get(),
            background_params=self._collect_background_params(),
        )

        # Keep fitted background values during dragging. Peak values come from the UI rows,
        # which _apply_peak_defs_to_ui already updates after Auto pre-fit / Run fit.
        if self.fit_result is not None:
            for name, old_par in self.fit_result.params.items():
                if name in params and not self._is_peak_param_name(name):
                    try:
                        params[name].set(value=float(old_par.value))
                    except Exception:
                        pass

        return model, params

    def _evaluate_current_drag_model(self, x, y_raw):
        model, params = self._build_model_from_current_rows(x, y_raw)
        best = model.eval(params=params, x=x)
        comps = model.eval_components(params=params, x=x)
        return best, comps

    def _component_peak_number(self, component_name):
        name = str(component_name)
        if not name.startswith('p'):
            return None
        head = name.split('_', 1)[0]
        if len(head) <= 1 or not head[1:].isdigit():
            return None
        return int(head[1:])

    def _nearest_peak_component(self, event, x, comps):
        active_rows = self._active_peak_rows()
        if not active_rows or event.xdata is None or event.ydata is None:
            return None

        best = None
        best_dist_px = float('inf')
        for comp_name, comp in comps.items():
            peak_number = self._component_peak_number(comp_name)
            if peak_number is None:
                continue

            row_index = peak_number - 1
            if row_index < 0 or row_index >= len(active_rows):
                continue

            comp = np.asarray(comp, dtype=float)
            if comp.size == 0:
                continue

            idx = int(np.argmin(np.abs(x - event.xdata)))
            px, py = self.ax_main.transData.transform((float(x[idx]), float(comp[idx])))
            dist_px = ((px - event.x) ** 2 + (py - event.y) ** 2) ** 0.5

            if dist_px < best_dist_px:
                best_dist_px = dist_px
                best = {
                    'row': active_rows[row_index],
                    'row_index': row_index,
                    'peak_number': peak_number,
                    'component_name': comp_name,
                    'component_y': float(comp[idx]),
                }

        if best is not None and best_dist_px <= float(self.peak_drag_pick_tolerance_px):
            return best
        return None

    def _set_dragged_peak_center(self, row, new_center):
        x_min = float(self.roi_min_var.get())
        x_max = float(self.roi_max_var.get())
        if x_min > x_max:
            x_min, x_max = x_max, x_min

        new_center = float(np.clip(new_center, x_min, x_max))
        row['center'].set(new_center)

        # Make sure the parameter bounds still contain the manually dragged center.
        margin = max(abs(float(row['fwhm'].get())) * 2.0, (x_max - x_min) / 1000.0, 1e-9)
        if new_center <= float(row['center_min'].get()):
            row['center_min'].set(new_center - margin)
        if new_center >= float(row['center_max'].get()):
            row['center_max'].set(new_center + margin)

    def _redraw_drag_preview(self, force=False):
        now = time.perf_counter()
        if not force and (now - self._last_peak_drag_draw) < (1.0 / 30.0):
            return
        self._last_peak_drag_draw = now

        x, y_raw, _ = self._get_roi_data()
        best, comps = self._evaluate_current_drag_model(x, y_raw)
        resid = y_raw - best
        chisq = float(np.nansum(resid ** 2))

        self.last_roi = (x.copy(), y_raw.copy())
        self.last_best_fit = np.asarray(best, dtype=float).copy()
        self.last_components = {name: np.asarray(val, dtype=float).copy() for name, val in comps.items()}

        xlim = self.ax_main.get_xlim()
        ylim = self.ax_main.get_ylim()
        rylim = self.ax_resid.get_ylim()

        self.ax_main.clear()
        self.ax_resid.clear()
        self._refresh_hover_axis()

        self.ax_main.plot(x, y_raw, 'k.', ms=3, alpha=0.6, label='Data')
        self.ax_main.plot(x, best, linewidth=2.1, label='Dragged model')
        for name, comp in comps.items():
            self.ax_main.plot(x, comp, '--', linewidth=1.0, alpha=0.85, label=name)

        self.ax_main.set_title('Manual peak drag preview')
        self.ax_main.set_ylabel('Y')
        self.ax_main.set_xlim(xlim)
        self.ax_main.set_ylim(ylim)
        self.ax_main.legend(fontsize=8, ncol=2)

        self.ax_resid.axhline(0.0, linestyle='--', linewidth=1.0)
        self.ax_resid.plot(x, resid, linewidth=1.0)
        self.ax_resid.set_xlabel('X')
        self.ax_resid.set_ylabel('Residual')
        self.ax_resid.set_ylim(rylim)

        state = self.peak_drag_state
        if state is not None:
            row = state['row']
            self.status_var.set(
                f"Dragging peak {state['peak_number']}: center={float(row['center'].get()):.6g}; "
                f"preview chi-square={chisq:.6g}. Click Run fit to finalize."
            )
        else:
            self.status_var.set(f'Peak drag preview updated. Preview chi-square={chisq:.6g}.')

        self.canvas.draw_idle()

    def _on_peak_drag_press(self, event):
        if self.pick_centers_mode:
            return
        if event.inaxes != self.ax_main or event.button != 1:
            return
        if self.x_full is None or self.y_full is None or event.xdata is None:
            return

        try:
            x, y_raw, _ = self._get_roi_data()
            _, comps = self._evaluate_current_drag_model(x, y_raw)
            hit = self._nearest_peak_component(event, x, comps)
        except Exception:
            hit = None

        if hit is None:
            return

        row = hit['row']
        self.peak_drag_state = {
            'row': row,
            'row_index': hit['row_index'],
            'peak_number': hit['peak_number'],
            'start_x': float(event.xdata),
            'start_center': float(row['center'].get()),
        }
        if hasattr(self, 'hover_annot'):
            self.hover_annot.set_visible(False)
        self.status_var.set(f"Dragging peak {hit['peak_number']}. Move mouse to change center; release to stop.")

    def _on_peak_drag_motion(self, event):
        state = getattr(self, 'peak_drag_state', None)
        if state is None:
            return
        if event.inaxes != self.ax_main or event.xdata is None:
            return

        dx = float(event.xdata) - state['start_x']
        self._set_dragged_peak_center(state['row'], state['start_center'] + dx)

        try:
            self._redraw_drag_preview(force=False)
        except Exception as exc:
            self.status_var.set(f'Peak drag preview error: {exc}')

    def _on_peak_drag_release(self, event):
        if getattr(self, 'peak_drag_state', None) is None:
            return
        try:
            self._redraw_drag_preview(force=True)
        except Exception as exc:
            self.status_var.set(f'Peak drag release error: {exc}')
        finally:
            self.peak_drag_state = None
            self.status_var.set('Peak moved. Inspect residuals, then click Run fit to finalize/export.')

    def _setup_hover(self):
        self.hover_annot = self.ax_main.annotate(
            "",
            xy=(0, 0),
            xytext=(12, 12),
            textcoords="offset points",
            bbox=dict(boxstyle="round", fc="white", ec="black", alpha=0.9),
            arrowprops=dict(arrowstyle="->"),
        )
        self.hover_annot.set_visible(False)

        self.canvas.mpl_connect("motion_notify_event", self._on_hover)

    def _refresh_hover_axis(self):
        # Reattach annotation to the current main axis after clear()
        if hasattr(self, "hover_annot"):
            try:
                self.hover_annot.remove()
            except Exception:
                pass

        self.hover_annot = self.ax_main.annotate(
            "",
            xy=(0, 0),
            xytext=(12, 12),
            textcoords="offset points",
            bbox=dict(boxstyle="round", fc="white", ec="black", alpha=0.9),
            arrowprops=dict(arrowstyle="->"),
        )
        self.hover_annot.set_visible(False)

    def _on_hover(self, event):
        if getattr(self, 'peak_drag_state', None) is not None:
            return

        if event.inaxes != self.ax_main:
            if hasattr(self, "hover_annot") and self.hover_annot.get_visible():
                self.hover_annot.set_visible(False)
                self.canvas.draw_idle()
            return

        best_line = None
        best_index = None
        best_dist = float("inf")

        for line in self.ax_main.lines:
            label = line.get_label()
            if label.startswith("_"):
                continue

            xdata = line.get_xdata()
            ydata = line.get_ydata()

            if len(xdata) == 0:
                continue

            contains, info = line.contains(event)
            if contains and "ind" in info and len(info["ind"]) > 0:
                idx = info["ind"][0]
                dx = xdata[idx] - event.xdata
                dy = ydata[idx] - event.ydata
                dist = dx * dx + dy * dy
                if dist < best_dist:
                    best_dist = dist
                    best_line = line
                    best_index = idx

        if best_line is not None:
            x = best_line.get_xdata()[best_index]
            y = best_line.get_ydata()[best_index]

            self.hover_annot.xy = (x, y)
            self.hover_annot.set_text(
                f"{best_line.get_label()}\nX = {x:.4f}\nY = {y:.4f}"
            )
            self.hover_annot.set_visible(True)
            self.canvas.draw_idle()
        else:
            if self.hover_annot.get_visible():
                self.hover_annot.set_visible(False)
                self.canvas.draw_idle()

    def _build_state(self):
        self.delimiter_var = tk.StringVar(value='tab')
        self.skiprows_var = tk.IntVar(value=0)
        self.xcol_var = tk.IntVar(value=0)
        self.ycol_var = tk.IntVar(value=1)

        self.roi_min_var = tk.DoubleVar(value=0.0)
        self.roi_max_var = tk.DoubleVar(value=1.0)

        self.smooth_enabled_var = tk.BooleanVar(value=False)
        self.smooth_window_var = tk.IntVar(value=9)
        self.smooth_poly_var = tk.IntVar(value=2)

        self.background_var = tk.StringVar(value='linear')
        self.background_profile_var = tk.StringVar(value='')
        self.background_custom_param_vars: dict[str, dict] = {}
        self._background_custom_state_cache: dict[str, dict] = {}
        self.poly_order_var = tk.IntVar(value=2)
        self.weighting_var = tk.StringVar(value='none')

        self.peak_count_var = tk.IntVar(value=3)
        self.peak_rows: list[dict] = []
        self.autofit_trials_var = tk.IntVar(value=8)
        # Criterion used to choose the best trial during auto pre-fit/refinement.
        # Lower is better for AIC, BIC, chi-square, and reduced chi-square.
        self.fit_criterion_var = tk.StringVar(value='AIC')

    def _build_ui(self):
        outer = ttk.Panedwindow(self.root, orient=tk.HORIZONTAL)
        outer.pack(fill=tk.BOTH, expand=True)

        controls_wrap = ScrollableFrame(outer)
        plot_frame = ttk.Frame(outer, padding=10)
        outer.add(controls_wrap, weight=0)
        outer.add(plot_frame, weight=1)
        controls = controls_wrap.inner
        controls.configure(padding=10)
        self.controls_scroll = controls_wrap
        self.plot_frame = plot_frame

        file_box = ttk.LabelFrame(controls, text='1. Load data', padding=8)
        file_box.pack(fill=tk.X, pady=(0, 8))

        ttk.Button(file_box, text='Open spectrum file', command=self.open_file).grid(row=0, column=0, sticky='ew', padx=(0, 6), pady=4)
        self.file_label = ttk.Label(file_box, text='No file selected', width=42)
        self.file_label.grid(row=0, column=1, columnspan=3, sticky='w', pady=4)

        ttk.Label(file_box, text='Delimiter').grid(row=1, column=0, sticky='w')
        ttk.Combobox(file_box, textvariable=self.delimiter_var, state='readonly', width=10,
                     values=['tab', 'comma', 'semicolon', 'space', 'auto']).grid(row=1, column=1, sticky='w', padx=(0, 6))
        ttk.Label(file_box, text='Skip rows').grid(row=1, column=2, sticky='w')
        ttk.Spinbox(file_box, from_=0, to=1000, textvariable=self.skiprows_var, width=8).grid(row=1, column=3, sticky='w')

        ttk.Label(file_box, text='X col').grid(row=2, column=0, sticky='w')
        ttk.Spinbox(file_box, from_=0, to=20, textvariable=self.xcol_var, width=8).grid(row=2, column=1, sticky='w', padx=(0, 6))
        ttk.Label(file_box, text='Y col').grid(row=2, column=2, sticky='w')
        ttk.Spinbox(file_box, from_=0, to=20, textvariable=self.ycol_var, width=8).grid(row=2, column=3, sticky='w')

        ttk.Button(file_box, text='Reload file', command=self.reload_current_file).grid(row=1, column=4, sticky='ew', pady=(6, 0), padx=(0, 6))
        ttk.Button(file_box, text='Reset peaks / fit state', command=self.reset_peaks_and_fit_state).grid(row=2, column=4, columnspan=2, sticky='ew', pady=(6, 0))

        fit_box = ttk.LabelFrame(controls, text='2. Fit settings', padding=8)
        fit_box.pack(fill=tk.X, pady=(0, 8))

        ttk.Label(fit_box, text='ROI min').grid(row=0, column=0, sticky='w')
        ttk.Entry(fit_box, textvariable=self.roi_min_var, width=12).grid(row=0, column=1, sticky='w', padx=(0, 6))
        ttk.Label(fit_box, text='ROI max').grid(row=0, column=2, sticky='w')
        ttk.Entry(fit_box, textvariable=self.roi_max_var, width=12).grid(row=0, column=3, sticky='w')

        ttk.Checkbutton(fit_box, text='Preview smoothing', variable=self.smooth_enabled_var).grid(row=1, column=0, sticky='w', pady=(6, 0))
        ttk.Label(fit_box, text='SG window').grid(row=1, column=1, sticky='e', pady=(6, 0))
        ttk.Spinbox(fit_box, from_=5, to=101, increment=2, textvariable=self.smooth_window_var, width=8).grid(row=1, column=2, sticky='w', pady=(6, 0))
        ttk.Label(fit_box, text='SG poly').grid(row=1, column=3, sticky='e', pady=(6, 0))
        ttk.Spinbox(fit_box, from_=1, to=7, textvariable=self.smooth_poly_var, width=8).grid(row=1, column=4, sticky='w', pady=(6, 0))

        ttk.Label(fit_box, text='Background').grid(row=2, column=0, sticky='w', pady=(6, 0))
        self.background_combo = ttk.Combobox(fit_box, textvariable=self.background_var, state='readonly', width=12,
                     values=['none', 'constant', 'linear', 'polynomial', 'custom'])
        self.background_combo.grid(row=2, column=1, sticky='w', pady=(6, 0))
        ttk.Label(fit_box, text='Poly order').grid(row=2, column=2, sticky='w', pady=(6, 0))
        self.background_poly_spin = ttk.Spinbox(fit_box, from_=1, to=7, textvariable=self.poly_order_var, width=8)
        self.background_poly_spin.grid(row=2, column=3, sticky='w', pady=(6, 0))
        ttk.Label(fit_box, text='Background profile').grid(row=3, column=0, sticky='w', pady=(6, 0))
        self.background_profile_combo = ttk.Combobox(fit_box, textvariable=self.background_profile_var, state='readonly', width=18, values=list(self.custom_background_profiles.keys()))
        self.background_profile_combo.grid(row=3, column=1, sticky='w', pady=(6, 0))
        ttk.Button(fit_box, text='Manage custom backgrounds', command=self.open_custom_background_profile_manager).grid(row=3, column=2, columnspan=2, sticky='ew', pady=(6, 0), padx=(6, 0))
        self.background_custom_frame = ttk.LabelFrame(fit_box, text='Custom background parameters', padding=6)
        self.background_custom_frame.grid(row=4, column=0, columnspan=5, sticky='ew', pady=(8, 0))
        ttk.Label(fit_box, text='Weights').grid(row=5, column=0, sticky='w', pady=(6, 0))
        ttk.Combobox(
            fit_box,
            textvariable=self.weighting_var,
            state='readonly',
            width=12,
            values=['none', 'poisson-like', 'sqrt(y)', '1/y']
        ).grid(row=5, column=1, sticky='w', pady=(6, 0), padx=(0, 12))
        
        ttk.Label(fit_box, text='Auto-fit trials').grid(row=5, column=2, sticky='w', pady=(6, 0))
        ttk.Spinbox(
            fit_box,
            from_=1,
            to=100,
            textvariable=self.autofit_trials_var,
            width=8
        ).grid(row=5, column=3, sticky='w', pady=(6, 0))

        ttk.Label(fit_box, text='Auto-fit criterion').grid(row=6, column=0, sticky='w', pady=(6, 0))
        ttk.Combobox(
            fit_box,
            textvariable=self.fit_criterion_var,
            state='readonly',
            width=20,
            values=FIT_SELECTION_CRITERIA
        ).grid(row=6, column=1, columnspan=3, sticky='w', pady=(6, 0))
        
        """
        ttk.Label(fit_box, text='Weights').grid(row=3, column=0, sticky='w', pady=(6, 0))
        ttk.Combobox(fit_box, textvariable=self.weighting_var, state='readonly', width=12,
                     values=['none', 'poisson-like', 'sqrt(y)', '1/y']).grid(row=3, column=1, sticky='w', pady=(6, 0))
        ttk.Label(fit_box, text='Auto-fit trials').grid(row=4, column=0, sticky='w', pady=(6, 0))
        ttk.Spinbox(fit_box, from_=1, to=100, textvariable=self.autofit_trials_var, width=8).grid(row=4, column=1, sticky='w', pady=(6, 0))
        """
        peaks_box = ttk.LabelFrame(controls, text='3. Peaks', padding=8)
        peaks_box.pack(fill=tk.BOTH, expand=True, pady=(0, 8))

        top_peaks = ttk.Frame(peaks_box)
        top_peaks.pack(fill=tk.X)
        ttk.Label(top_peaks, text='Number of peaks').pack(side=tk.LEFT)
        ttk.Spinbox(top_peaks, from_=1, to=15, textvariable=self.peak_count_var, width=6,
                    command=self.rebuild_peaks).pack(side=tk.LEFT, padx=(8, 8))
        ttk.Button(top_peaks, text='Apply peak count', command=self.rebuild_peaks).pack(side=tk.LEFT)

        self.peaks_scroll = ScrollableFrame(peaks_box)
        self.peaks_scroll.pack(fill=tk.BOTH, expand=True, pady=(8, 0))
        self.peaks_container = self.peaks_scroll.inner

        action_box = ttk.LabelFrame(controls, text='4. Actions', padding=8)
        action_box.pack(fill=tk.X)
        ttk.Button(action_box, text='Preview', command=self.preview).pack(fill=tk.X, pady=2)
        ttk.Button(action_box, text='Manage custom profiles', command=self.open_custom_profile_manager).pack(fill=tk.X, pady=2)
        ttk.Button(action_box, text='Pick centers from plot', command=self.start_pick_centers).pack(fill=tk.X, pady=2)
        ttk.Button(action_box, text='Auto pre-fit', command=self.autofill_from_centers).pack(fill=tk.X, pady=2)
        ttk.Button(action_box, text='Refine with added peaks', command=self.refine_with_added_peaks).pack(fill=tk.X, pady=2)
        ttk.Button(action_box, text='Run fit', command=self.run_fit).pack(fill=tk.X, pady=2)
        ttk.Separator(action_box).pack(fill=tk.X, pady=6)
        ttk.Button(action_box, text='Save session (.json)', command=self.save_session).pack(fill=tk.X, pady=2)
        ttk.Button(action_box, text='Load session (.json)', command=self.load_session).pack(fill=tk.X, pady=2)
        ttk.Separator(action_box).pack(fill=tk.X, pady=6)
        ttk.Button(action_box, text='Save ZIP results', command=self.save_zip_results).pack(fill=tk.X, pady=2)

        self.status_var = tk.StringVar(value='Ready.')
        ttk.Label(controls, textvariable=self.status_var, wraplength=360, foreground='#444').pack(fill=tk.X, pady=(8, 0))

        self.background_var.trace_add('write', self._refresh_background_controls)
        self.background_profile_var.trace_add('write', self._refresh_background_controls)
        self._refresh_background_controls()
        self.rebuild_peaks()

    def _background_kind_uses_custom_profile(self):
        return self.background_var.get().strip().lower() == 'custom'

    def _collect_background_params(self):
        params = {}
        for name, vars_dict in self.background_custom_param_vars.items():
            params[name] = {
                'value': float(vars_dict['value'].get()),
                'min': float(vars_dict['min'].get()),
                'max': float(vars_dict['max'].get()),
            }
        return params

    def _refresh_background_controls(self, *_):
        if not hasattr(self, 'background_profile_combo'):
            return

        kind = self.background_var.get().strip().lower()
        self.background_profile_combo.configure(values=list(self.custom_background_profiles.keys()))

        if kind == 'polynomial':
            try:
                self.background_poly_spin.configure(state='normal')
            except Exception:
                pass
        else:
            try:
                self.background_poly_spin.configure(state='disabled')
            except Exception:
                pass

        for child in self.background_custom_frame.winfo_children():
            child.destroy()
        self.background_custom_param_vars.clear()

        if kind != 'custom':
            self.background_profile_combo.configure(state='disabled')
            self.background_custom_frame.grid_remove()
            return

        self.background_profile_combo.configure(state='readonly' if self.custom_background_profiles else 'disabled')
        self.background_custom_frame.grid()

        profile = self.custom_background_profiles.get(self.background_profile_var.get())
        if not profile:
            ttk.Label(self.background_custom_frame, text='No custom background profile selected.').grid(row=0, column=0, sticky='w')
            return

        row_idx = 0
        current_values = getattr(self, '_background_custom_state_cache', {}) if isinstance(getattr(self, '_background_custom_state_cache', {}), dict) else {}
        for param in profile.get('parameters', []):
            name = param['name']
            cfg = current_values.get(name, {})
            value_var = tk.DoubleVar(value=float(cfg.get('value', param.get('default', 1.0))))
            min_var = tk.DoubleVar(value=float(cfg.get('min', param.get('min', float('-inf')))))
            max_var = tk.DoubleVar(value=float(cfg.get('max', param.get('max', float('inf')))))
            self.background_custom_param_vars[name] = {'value': value_var, 'min': min_var, 'max': max_var}

            ttk.Label(self.background_custom_frame, text=name).grid(row=row_idx, column=0, sticky='w', padx=(0, 4), pady=2)
            ttk.Entry(self.background_custom_frame, textvariable=value_var, width=12).grid(row=row_idx, column=1, sticky='w', padx=(0, 8), pady=2)
            ttk.Label(self.background_custom_frame, text='min').grid(row=row_idx, column=2, sticky='w', padx=(0, 4), pady=2)
            ttk.Entry(self.background_custom_frame, textvariable=min_var, width=12).grid(row=row_idx, column=3, sticky='w', padx=(0, 8), pady=2)
            ttk.Label(self.background_custom_frame, text='max').grid(row=row_idx, column=4, sticky='w', padx=(0, 4), pady=2)
            ttk.Entry(self.background_custom_frame, textvariable=max_var, width=12).grid(row=row_idx, column=5, sticky='w', pady=2)
            row_idx += 1

    def _profile_collection_to_lines(self, profile: dict) -> str:
        return self._profile_parameters_to_lines(profile)

    def open_custom_background_profile_manager(self):
        win = tk.Toplevel(self.root)
        win.title('Custom background profile manager')
        win.geometry('920x560')

        left = ttk.Frame(win, padding=8)
        right = ttk.Frame(win, padding=8)
        left.pack(side=tk.LEFT, fill=tk.Y)
        right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        ttk.Label(left, text='Background profiles').pack(anchor='w')
        listbox = tk.Listbox(left, width=28, height=20)
        listbox.pack(fill=tk.Y, expand=True, pady=(6, 6))
        for name in self.custom_background_profiles:
            listbox.insert(tk.END, name)

        name_var = tk.StringVar()
        ttk.Label(right, text='Background profile name').pack(anchor='w')
        ttk.Entry(right, textvariable=name_var).pack(fill=tk.X)

        ttk.Label(right, text='Parameters: one per line -> name, default, min, max').pack(anchor='w', pady=(8, 2))
        params_text = tk.Text(right, height=10)
        params_text.pack(fill=tk.X)

        ttk.Label(right, text='Expression').pack(anchor='w', pady=(8, 2))
        expr_text = tk.Text(right, height=10)
        expr_text.pack(fill=tk.BOTH, expand=True)

        ttk.Label(
            right,
            text='Use x and safe functions such as exp, log, sqrt, abs, where, minimum, maximum, clip.',
        ).pack(anchor='w', pady=(6, 0))

        def load_selected(event=None):
            if not listbox.curselection():
                return
            name = listbox.get(listbox.curselection()[0])
            profile = self.custom_background_profiles[name]
            name_var.set(name)
            params_text.delete('1.0', tk.END)
            params_text.insert('1.0', self._profile_parameters_to_lines(profile))
            expr_text.delete('1.0', tk.END)
            expr_text.insert('1.0', profile.get('expression', ''))

        listbox.bind('<<ListboxSelect>>', load_selected)

        def save_profile():
            try:
                profile = normalize_custom_profile_definition({
                    'name': name_var.get().strip(),
                    'parameters': self._profile_lines_to_parameters(params_text.get('1.0', tk.END)),
                    'expression': expr_text.get('1.0', tk.END).strip(),
                })
            except Exception as exc:
                messagebox.showerror('Invalid background profile', str(exc), parent=win)
                return

            old_name = None
            if listbox.curselection():
                old_name = listbox.get(listbox.curselection()[0])

            self.custom_background_profiles[profile['name']] = profile
            if old_name and old_name != profile['name'] and old_name in self.custom_background_profiles:
                self.custom_background_profiles.pop(old_name, None)

            current_profile = self.background_profile_var.get()
            if current_profile == old_name or current_profile == profile['name']:
                self.background_profile_var.set(profile['name'])
                self._background_custom_state_cache = {}

            listbox.delete(0, tk.END)
            for nm in self.custom_background_profiles:
                listbox.insert(tk.END, nm)

            self._refresh_background_controls()
            self.status_var.set(f"Saved custom background profile '{profile['name']}'.")

        def delete_profile():
            if not listbox.curselection():
                return
            name = listbox.get(listbox.curselection()[0])
            self.custom_background_profiles.pop(name, None)
            if self.background_profile_var.get() == name:
                self.background_profile_var.set('')
                self._background_custom_state_cache = {}
            listbox.delete(listbox.curselection()[0])
            self._refresh_background_controls()

        btns = ttk.Frame(right)
        btns.pack(fill=tk.X, pady=(8, 0))
        ttk.Button(btns, text='Save / update', command=save_profile).pack(side=tk.LEFT)
        ttk.Button(btns, text='Delete', command=delete_profile).pack(side=tk.LEFT, padx=(6, 0))
        ttk.Button(btns, text='Close', command=win.destroy).pack(side=tk.RIGHT)

    def _build_plot(self):
        self.fig = Figure(figsize=(7.7, 6.8), dpi=100)
        self.ax_main = self.fig.add_subplot(211)
        self.ax_resid = self.fig.add_subplot(212, sharex=self.ax_main)
        self.fig.tight_layout()

        self.canvas = FigureCanvasTkAgg(self.fig, master=self.plot_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.canvas.mpl_connect("button_press_event", self._on_pick_center)
        self.canvas.mpl_connect("button_press_event", self._on_peak_drag_press)
        self.canvas.mpl_connect("motion_notify_event", self._on_peak_drag_motion)
        self.canvas.mpl_connect("button_release_event", self._on_peak_drag_release)

        self.ax_main.set_title('Load a file to begin')
        self.ax_main.set_ylabel('Y')
        self.ax_resid.set_xlabel('X')
        self.ax_resid.set_ylabel('Residual')
        self._refresh_hover_axis()
        self.canvas.draw_idle()

    def rebuild_peaks(self):
        count = max(1, int(self.peak_count_var.get()))
        old = self.peak_rows

        row_states = []
        for i in range(count):
            if i < len(old):
                row_states.append(self._row_state_from_row(old[i]))
            else:
                row_states.append(self._default_peak_state(i, count))

        self._draw_peak_rows(row_states)

    def _resolve_delimiter(self):
        mapping = {
            'tab': '\t',
            'comma': ',',
            'semicolon': ';',
            'space': r'\s+',
            'auto': None,
        }
        return mapping[self.delimiter_var.get()]

    def open_file(self):
        path = filedialog.askopenfilename(
            title='Open spectrum file',
            filetypes=[('Text/CSV files', '*.txt *.csv *.dat *.asc'), ('All files', '*.*')],
        )
        if not path:
            return
        self.current_file = Path(path)
        self.reload_current_file()

    def reload_current_file(self):
        if self.current_file is None:
            messagebox.showinfo('No file', 'Please select a file first.')
            return
        try:
            x, y = load_spectrum(
                self.current_file,
                x_col=int(self.xcol_var.get()),
                y_col=int(self.ycol_var.get()),
                delimiter=self._resolve_delimiter(),
                skiprows=int(self.skiprows_var.get()),
            )
        except Exception as exc:
            messagebox.showerror('Load error', f'Could not load file:\n{exc}')
            return

        self.x_full = np.asarray(x, dtype=float)
        self.y_full = np.asarray(y, dtype=float)
        self.file_label.config(text=self.current_file.name)
        self.roi_min_var.set(float(np.min(self.x_full)))
        self.roi_max_var.set(float(np.max(self.x_full)))
        self._clear_fit_and_pick_state()
        self.rebuild_peaks_fresh()
        self.status_var.set(f'Loaded {self.current_file.name} with {len(self.x_full)} points.')
        self._plot_loaded_data()
    
    
    def reset_peaks_and_fit_state(self):
        if self.x_full is None or self.y_full is None:
            messagebox.showinfo('No data', 'Load a file first.')
            return
    
        self._clear_fit_and_pick_state()
        self.rebuild_peaks_fresh()
        self._plot_loaded_data()
    
        self.status_var.set('Peaks, picked centers, and fit state reset for current spectrum. ROI kept unchanged.')
    
    def _plot_loaded_data(self):
        self.ax_main.clear()
        self.ax_resid.clear()
        self._refresh_hover_axis()
        data_line, = self.ax_main.plot(
            self.x_full,
            self.y_full,
            'k.-',
            linewidth=1.1,
            markersize=3,
            alpha=0.75,
            label='Loaded data',
        )
        data_line.set_picker(5)
        self.ax_main.set_title(self.current_file.name if self.current_file else 'Loaded spectrum')
        self.ax_main.set_ylabel('Y')
        self.ax_resid.set_xlabel('X')
        self.ax_resid.set_ylabel('Residual')
        self.ax_resid.axhline(0.0, linestyle='--', linewidth=1.0)
        self.fig.tight_layout()
        self.ax_main.relim()
        self.ax_main.autoscale_view()
        self.ax_resid.relim()
        self.ax_resid.autoscale_view()
        self.canvas.draw_idle()

    def _collect_peak_defs(self):
        peak_defs = []
        for row in self.peak_rows:
            if not row['active'].get():
                continue

            peak = {
                'kind': row['kind'].get(),
                'custom_profile': row.get('custom_profile').get() if 'custom_profile' in row else '',
                'center': float(row['center'].get()),
                'amplitude': float(row['amplitude'].get()),
                'fwhm': float(row['fwhm'].get()),
                'center_min': float(row['center_min'].get()),
                'center_max': float(row['center_max'].get()),
                'amplitude_min': float(row['amplitude_min'].get()),
                'amplitude_max': float(row['amplitude_max'].get()),
                'fwhm_min': float(row['fwhm_min'].get()),
                'fwhm_max': float(row['fwhm_max'].get()),
                'fraction': float(row['fraction'].get()),
                'sigma': float(row['sigma'].get()),
                'gamma': float(row['gamma'].get()),
            }

            if peak['kind'].strip().lower() == 'custom':
                profile = self.custom_profiles.get(peak['custom_profile'])
                if profile is None:
                    raise ValueError('A custom peak was selected but no custom profile was assigned.')
            
                peak['custom_params'] = {}
            
                for param in profile['parameters']:
                    name = param['name']
            
                    if name == 'center':
                        value = peak['center']
                        pmin = peak['center_min']
                        pmax = peak['center_max']
            
                    elif name == 'amplitude':
                        value = peak['amplitude']
                        pmin = peak['amplitude_min']
                        pmax = peak['amplitude_max']
            
                    elif name == 'fwhm':
                        value = peak['fwhm']
                        pmin = peak['fwhm_min']
                        pmax = peak['fwhm_max']
            
                    else:
                        vars_dict = row['custom_params'].get(name)
                        if vars_dict is None:
                            value = float(param.get('default', 1.0))
                            pmin = float(param.get('min', float('-inf')))
                            pmax = float(param.get('max', float('inf')))
                        else:
                            value = float(vars_dict['value'].get())
                            pmin = float(vars_dict['min'].get())
                            pmax = float(vars_dict['max'].get())
            
                    # keep old flat style for backend compatibility
                    peak[name] = value
                    peak[f'{name}_min'] = pmin
                    peak[f'{name}_max'] = pmax
            
                    # add nested style for custom auto-prefit
                    peak['custom_params'][name] = {
                        'value': value,
                        'min': pmin,
                        'max': pmax,
                    }

            peak_defs.append(peak)

        if not peak_defs:
            raise ValueError('At least one active peak is required.')
        return peak_defs

    def _get_roi_data(self):
        if self.x_full is None or self.y_full is None:
            raise ValueError('Load a file first.')
        x, y = crop_roi(self.x_full, self.y_full, float(self.roi_min_var.get()), float(self.roi_max_var.get()))
        if len(x) == 0:
            raise ValueError('No data points inside ROI.')
        y_plot = smooth_if_requested(
            y,
            window=int(self.smooth_window_var.get()),
            polyorder=int(self.smooth_poly_var.get()),
            enabled=bool(self.smooth_enabled_var.get()),
        )
        return x, y, y_plot

    def _weights(self, y):
        eps = 1e-12
        mode = self.weighting_var.get()
        if mode == 'none':
            return None
        if mode == 'poisson-like':
            return 1.0 / np.sqrt(np.clip(np.abs(y), 1.0, None))
        if mode == 'sqrt(y)':
            return np.sqrt(np.clip(np.abs(y), eps, None))
        if mode == '1/y':
            return 1.0 / np.clip(np.abs(y), eps, None)
        return None

    def get_session_state(self):
        return {
            'app_version': 'step16-custom-merged',
            'current_file': str(self.current_file) if self.current_file else None,
            'loader': {
                'delimiter': self.delimiter_var.get(),
                'skiprows': int(self.skiprows_var.get()),
                'x_col': int(self.xcol_var.get()),
                'y_col': int(self.ycol_var.get()),
            },
            'fit_settings': {
                'roi_min': float(self.roi_min_var.get()),
                'roi_max': float(self.roi_max_var.get()),
                'smooth_enabled': bool(self.smooth_enabled_var.get()),
                'smooth_window': int(self.smooth_window_var.get()),
                'smooth_poly': int(self.smooth_poly_var.get()),
                'background': self.background_var.get(),
                'background_profile': self.background_profile_var.get(),
                'background_params': self._collect_background_params(),
                'poly_order': int(self.poly_order_var.get()),
                'weighting': self.weighting_var.get(),
                'fit_criterion': self._current_fit_criterion(),
                'peak_count': int(self.peak_count_var.get()),
            },
            'custom_profiles': list(self.custom_profiles.values()),
            'custom_background_profiles': list(self.custom_background_profiles.values()),
            'peaks': [self._row_state_from_row(row) for row in self.peak_rows],
        }

    def apply_session_state(self, state: dict):
        self.custom_profiles = {}
        self.custom_background_profiles = {}
        for prof in state.get('custom_profiles', []):
            try:
                normalized = normalize_custom_profile_definition(prof)
                self.custom_profiles[normalized['name']] = normalized
            except Exception:
                pass

        for prof in state.get('custom_background_profiles', []):
            try:
                normalized = normalize_custom_profile_definition(prof)
                self.custom_background_profiles[normalized['name']] = normalized
            except Exception:
                pass

        loader = state.get('loader', {})
        self.delimiter_var.set(loader.get('delimiter', self.delimiter_var.get()))
        self.skiprows_var.set(loader.get('skiprows', self.skiprows_var.get()))
        self.xcol_var.set(loader.get('x_col', self.xcol_var.get()))
        self.ycol_var.set(loader.get('y_col', self.ycol_var.get()))

        settings = state.get('fit_settings', {})
        self.roi_min_var.set(settings.get('roi_min', self.roi_min_var.get()))
        self.roi_max_var.set(settings.get('roi_max', self.roi_max_var.get()))
        self.smooth_enabled_var.set(settings.get('smooth_enabled', self.smooth_enabled_var.get()))
        self.smooth_window_var.set(settings.get('smooth_window', self.smooth_window_var.get()))
        self.smooth_poly_var.set(settings.get('smooth_poly', self.smooth_poly_var.get()))
        self.background_var.set(settings.get('background', self.background_var.get()))
        self.background_profile_var.set(settings.get('background_profile', self.background_profile_var.get()))
        self._background_custom_state_cache = settings.get('background_params', {}) if isinstance(settings.get('background_params', {}), dict) else {}
        self.poly_order_var.set(settings.get('poly_order', self.poly_order_var.get()))
        self.weighting_var.set(settings.get('weighting', self.weighting_var.get()))
        if 'fit_criterion' in settings:
            self.fit_criterion_var.set(settings.get('fit_criterion', self.fit_criterion_var.get()))

        self._refresh_background_controls()

        peaks = state.get('peaks', [])
        self.peak_count_var.set(settings.get('peak_count', len(peaks) or self.peak_count_var.get()))
        self._draw_peak_rows(peaks if peaks else [self._default_peak_state(i, int(self.peak_count_var.get())) for i in range(int(self.peak_count_var.get()))])

        state_file = state.get('current_file')
        if state_file:
            maybe_file = Path(state_file)
            if maybe_file.exists():
                self.current_file = maybe_file
                self.reload_current_file()
                self._draw_peak_rows(peaks if peaks else [self._default_peak_state(i, int(self.peak_count_var.get())) for i in range(int(self.peak_count_var.get()))])
            else:
                self.status_var.set('Session loaded, but the saved spectrum file path was not found. Load the data file manually.')


    def _profile_lines_to_parameters(self, text: str):
        params = []
        for raw in text.splitlines():
            line = raw.strip()
            if not line:
                continue
            parts = [p.strip() for p in line.split(',')]
            if len(parts) != 4:
                raise ValueError('Each parameter line must be: name, default, min, max')

            def parse_bound(v: str):
                lv = v.lower()
                if lv in {'inf', '+inf'}:
                    return float('inf')
                if lv == '-inf':
                    return float('-inf')
                return float(v)

            params.append({
                'name': parts[0],
                'default': float(parts[1]),
                'min': parse_bound(parts[2]),
                'max': parse_bound(parts[3]),
            })
        return params

    def _profile_parameters_to_lines(self, profile: dict) -> str:
        return '\n'.join(
            f"{p['name']}, {p.get('default', 1.0)}, {p.get('min', '-inf')}, {p.get('max', 'inf')}"
            for p in profile.get('parameters', [])
        )

    def open_custom_profile_manager(self):
        win = tk.Toplevel(self.root)
        win.title('Custom profile manager')
        win.geometry('920x560')

        left = ttk.Frame(win, padding=8)
        right = ttk.Frame(win, padding=8)
        left.pack(side=tk.LEFT, fill=tk.Y)
        right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        ttk.Label(left, text='Profiles').pack(anchor='w')
        listbox = tk.Listbox(left, width=28, height=20)
        listbox.pack(fill=tk.Y, expand=True, pady=(6, 6))
        for name in self.custom_profiles:
            listbox.insert(tk.END, name)

        name_var = tk.StringVar()
        ttk.Label(right, text='Profile name').pack(anchor='w')
        ttk.Entry(right, textvariable=name_var).pack(fill=tk.X)

        ttk.Label(right, text='Parameters: one per line -> name, default, min, max').pack(anchor='w', pady=(8, 2))
        params_text = tk.Text(right, height=10)
        params_text.pack(fill=tk.X)

        ttk.Label(right, text='Expression').pack(anchor='w', pady=(8, 2))
        expr_text = tk.Text(right, height=10)
        expr_text.pack(fill=tk.BOTH, expand=True)

        ttk.Label(
            right,
            text='Use x and safe functions such as exp, log, sqrt, abs, where, minimum, maximum, clip.',
        ).pack(anchor='w', pady=(6, 0))

        def load_selected(event=None):
            if not listbox.curselection():
                return
            name = listbox.get(listbox.curselection()[0])
            profile = self.custom_profiles[name]
            name_var.set(name)
            params_text.delete('1.0', tk.END)
            params_text.insert('1.0', self._profile_parameters_to_lines(profile))
            expr_text.delete('1.0', tk.END)
            expr_text.insert('1.0', profile.get('expression', ''))

        listbox.bind('<<ListboxSelect>>', load_selected)

        def save_profile():
            try:
                profile = normalize_custom_profile_definition({
                    'name': name_var.get().strip(),
                    'parameters': self._profile_lines_to_parameters(params_text.get('1.0', tk.END)),
                    'expression': expr_text.get('1.0', tk.END).strip(),
                })
            except Exception as exc:
                messagebox.showerror('Custom profile error', str(exc), parent=win)
                return
            
            """
            self.custom_profiles[profile['name']] = profile
            listbox.delete(0, tk.END)
            for name in self.custom_profiles:
                listbox.insert(tk.END, name)
            self.rebuild_peaks()
            self.status_var.set(f"Saved custom profile '{profile['name']}'.")
            """
            
            old_name = None
            if listbox.curselection():
                old_name = listbox.get(listbox.curselection()[0])
            
            self.custom_profiles[profile['name']] = profile
            
            # If the profile was renamed, remove the old entry
            if old_name and old_name != profile['name'] and old_name in self.custom_profiles:
                self.custom_profiles.pop(old_name, None)
            
            # Force rows using this profile to forget cached custom parameter values
            for row in self.peak_rows:
                row_profile = row.get('custom_profile')
                if row_profile is None:
                    continue
            
                current_name = row_profile.get()
                if current_name == old_name or current_name == profile['name']:
                    row_profile.set(profile['name'])
                    if 'custom_params' in row:
                        row['custom_params'].clear()
            
            # Refresh listbox
            listbox.delete(0, tk.END)
            for nm in self.custom_profiles:
                listbox.insert(tk.END, nm)
            
            self.rebuild_peaks()
            self.status_var.set(f"Saved custom profile '{profile['name']}'.")

        def delete_profile():
            if not listbox.curselection():
                return
            name = listbox.get(listbox.curselection()[0])
            self.custom_profiles.pop(name, None)
            listbox.delete(listbox.curselection()[0])
            self.rebuild_peaks()

        btns = ttk.Frame(right)
        btns.pack(fill=tk.X, pady=(8, 0))
        ttk.Button(btns, text='Save / update', command=save_profile).pack(side=tk.LEFT)
        ttk.Button(btns, text='Delete', command=delete_profile).pack(side=tk.LEFT, padx=(6, 0))
        ttk.Button(btns, text='Close', command=win.destroy).pack(side=tk.RIGHT)

    def save_session(self):
        default_name = (self.current_file.stem if self.current_file else 'pl_fit_session') + '_session.json'
        target = filedialog.asksaveasfilename(
            title='Save session',
            defaultextension='.json',
            initialfile=default_name,
            filetypes=[('JSON files', '*.json')],
        )
        if not target:
            return
        try:
            Path(target).write_text(json.dumps(self.get_session_state(), indent=2), encoding='utf-8')
            self.last_session_path = Path(target)
        except Exception as exc:
            messagebox.showerror('Save session error', f'Could not save session:\n{exc}')
            return
        self.status_var.set(f'Session saved to {target}')
        messagebox.showinfo('Saved', f'Session saved to:\n{target}')

    def load_session(self):
        path = filedialog.askopenfilename(
            title='Load session',
            filetypes=[('JSON files', '*.json'), ('All files', '*.*')],
        )
        if not path:
            return
        try:
            state = json.loads(Path(path).read_text(encoding='utf-8'))
            self.apply_session_state(state)
            self.last_session_path = Path(path)
        except Exception as exc:
            messagebox.showerror('Load session error', f'Could not load session:\n{exc}')
            return
        self.status_var.set(f'Session loaded from {path}')
        messagebox.showinfo('Loaded', f'Session loaded from:\n{path}')

    def preview(self):
        try:
            x, y_raw, y_plot = self._get_roi_data()
            peak_defs = self._collect_peak_defs()
            model, params = build_composite_model(
                peak_defs,
                background_kind=self.background_var.get(),
                poly_order=int(self.poly_order_var.get()),
                x=x,
                y=y_raw,
                custom_profiles=self.custom_profiles,
                custom_background_profiles=self.custom_background_profiles,
                custom_background_profile_name=self.background_profile_var.get(),
                background_params=self._collect_background_params(),
            )
            preview = model.eval(params=params, x=x)
            comps = model.eval_components(params=params, x=x)
        except Exception as exc:
            messagebox.showerror('Preview error', str(exc))
            return

        self.ax_main.clear()
        self.ax_resid.clear()
        self._refresh_hover_axis()
        self.ax_main.plot(x, y_raw, 'k.', ms=3, alpha=0.6, label='Raw')
        if bool(self.smooth_enabled_var.get()):
            self.ax_main.plot(x, y_plot, linewidth=1.2, alpha=0.9, label='Smoothed preview')
        self.ax_main.plot(x, preview, linewidth=2.0, label='Current model preview')
        for name, comp in comps.items():
            self.ax_main.plot(x, comp, '--', linewidth=1.0, alpha=0.85, label=name)
        self.ax_main.set_title('Preview')
        self.ax_main.set_ylabel('Y')
        self.ax_main.legend(fontsize=8, ncol=2)

        resid = y_raw - preview
        self.ax_resid.axhline(0.0, linestyle='--', linewidth=1.0)
        self.ax_resid.plot(x, resid, linewidth=1.0)
        self.ax_resid.set_xlabel('X')
        self.ax_resid.set_ylabel('Residual')
        self.ax_main.relim()
        self.ax_main.autoscale_view()
        self.ax_resid.relim()
        self.ax_resid.autoscale_view()
        self.fig.tight_layout()
        self.canvas.draw_idle()
        self.status_var.set('Preview updated.')

    def run_fit(self):
        try:
            x, y_raw, _ = self._get_roi_data()
            peak_defs = self._collect_peak_defs()
            model, params = build_composite_model(
                peak_defs,
                background_kind=self.background_var.get(),
                poly_order=int(self.poly_order_var.get()),
                x=x,
                y=y_raw,
                custom_profiles=self.custom_profiles,
                custom_background_profiles=self.custom_background_profiles,
                custom_background_profile_name=self.background_profile_var.get(),
                background_params=self._collect_background_params(),
            )
            result = model.fit(y_raw, params=params, x=x, weights=self._weights(y_raw), nan_policy='raise')
            comps = result.eval_components(x=x)
            best = result.best_fit
        except Exception as exc:
            messagebox.showerror('Fit error', str(exc))
            return

        self.fit_result = result
        self.last_components = comps
        self.last_best_fit = best
        self.last_roi = (x.copy(), y_raw.copy())

        self.ax_main.clear()
        self.ax_resid.clear()
        self._refresh_hover_axis()
        self.ax_main.plot(x, y_raw, 'k.', ms=3, alpha=0.6, label='Data')
        self.ax_main.plot(x, best, linewidth=2.1, label='Best fit')
        for name, comp in comps.items():
            self.ax_main.plot(x, comp, '--', linewidth=1.0, alpha=0.85, label=name)
        self.ax_main.set_title('Best fit and components')
        self.ax_main.set_ylabel('Y')
        self.ax_main.legend(fontsize=8, ncol=2)

        resid = y_raw - best
        self.ax_resid.axhline(0.0, linestyle='--', linewidth=1.0)
        self.ax_resid.plot(x, resid, linewidth=1.0)
        self.ax_resid.set_xlabel('X')
        self.ax_resid.set_ylabel('Residual')
        self.fig.tight_layout()
        self.canvas.draw_idle()
        self.status_var.set(f'Fit complete. {self._fit_metric_summary(result)}')
        self.last_fit_peak_count = len([row for row in self.peak_rows if row['active'].get()])

    def save_zip_results(self):
        if self.fit_result is None or self.last_roi is None:
            messagebox.showinfo('Nothing to save', 'Run a fit first.')
            return

        default_name = (self.current_file.stem if self.current_file else 'pl_fit_export') + '_results.zip'
        target = filedialog.asksaveasfilename(
            title='Save ZIP results',
            defaultextension='.zip',
            initialfile=default_name,
            filetypes=[('ZIP archive', '*.zip')],
        )
        if not target:
            return

        x, y = self.last_roi
        try:
            with TemporaryDirectory() as tmpdir:
                tmpdir = Path(tmpdir)
                base = self.current_file.stem if self.current_file else 'pl_fit_export'

                curves_df = pd.DataFrame({
                    'x': x,
                    'y_data': y,
                    'y_fit': self.last_best_fit,
                    'residual': y - self.last_best_fit,
                })
                for name, comp in self.last_components.items():
                    curves_df[name] = comp
                curves_csv_path = tmpdir / f'{base}_curves.csv'
                curves_df.to_csv(curves_csv_path, index=False)

                rows = []
                for name, par in self.fit_result.params.items():
                    rows.append({
                        'parameter': name,
                        'value': par.value,
                        'stderr': par.stderr,
                        'min': par.min,
                        'max': par.max,
                        'vary': par.vary,
                        'expr': par.expr,
                    })
                params_df = pd.DataFrame(rows)
                params_csv_path = tmpdir / f'{base}_parameters.csv'
                params_df.to_csv(params_csv_path, index=False)

                report_path = tmpdir / f'{base}_fit_report.txt'
                report_path.write_text(self.fit_result.fit_report(min_correl=0.5), encoding='utf-8')

                session_path = tmpdir / f'{base}_session.json'
                session_path.write_text(json.dumps(self.get_session_state(), indent=2), encoding='utf-8')

                meta_df = pd.DataFrame([
                    {'field': 'source_file', 'value': str(self.current_file) if self.current_file else ''},
                    {'field': 'background', 'value': self.background_var.get()},
                    {'field': 'poly_order', 'value': int(self.poly_order_var.get())},
                    {'field': 'weighting', 'value': self.weighting_var.get()},
                    {'field': 'fit_criterion', 'value': self._current_fit_criterion()},
                    {'field': 'roi_min', 'value': float(self.roi_min_var.get())},
                    {'field': 'roi_max', 'value': float(self.roi_max_var.get())},
                    {'field': 'chisqr', 'value': float(self.fit_result.chisqr)},
                    {'field': 'redchi', 'value': float(self.fit_result.redchi)},
                    {'field': 'aic', 'value': float(getattr(self.fit_result, 'aic', np.nan))},
                    {'field': 'bic', 'value': float(getattr(self.fit_result, 'bic', np.nan))},
                    {'field': 'nfev', 'value': int(self.fit_result.nfev)},
                ])
                summary_csv_path = tmpdir / f'{base}_summary.csv'
                meta_df.to_csv(summary_csv_path, index=False)

                excel_path = tmpdir / f'{base}_results.xlsx'
                excel_created = False
                try:
                    import openpyxl  # noqa: F401
                    with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
                        curves_df.to_excel(writer, sheet_name='curves', index=False)
                        params_df.to_excel(writer, sheet_name='parameters', index=False)
                        meta_df.to_excel(writer, sheet_name='summary', index=False)
                    excel_created = True
                except Exception:
                    excel_created = False

                original_copy = None
                if self.current_file is not None and self.current_file.exists():
                    original_copy = tmpdir / self.current_file.name
                    original_copy.write_bytes(self.current_file.read_bytes())

                with zipfile.ZipFile(target, 'w', compression=zipfile.ZIP_DEFLATED) as zf:
                    for path in [curves_csv_path, params_csv_path, summary_csv_path, report_path, session_path]:
                        zf.write(path, path.name)
                    if excel_created and excel_path.exists():
                        zf.write(excel_path, excel_path.name)
                    if original_copy is not None:
                        zf.write(original_copy, original_copy.name)
        except Exception as exc:
            messagebox.showerror('Save error', f'Could not save ZIP:\n{exc}')
            return

        self.status_var.set(f'Saved ZIP results to {target}')
        messagebox.showinfo('Saved', f'Results saved to:\n{target}')


    def show_about_dialog(self):
        messagebox.showinfo(
            'About / License',
            STARTUP_NOTICE,
            parent=self.root,
        )


def show_startup_notice(root: tk.Tk) -> bool:
    accepted = {'value': False}
    dialog = tk.Toplevel(root)
    dialog.title('License / Citation Notice')
    dialog.resizable(True, True)
    dialog.geometry('760x420')
    dialog.minsize(680, 360)

    # Keep the startup notice independent of the hidden root so it cannot
    # end up invisible behind other windows on Windows.
    dialog.attributes('-topmost', True)

    frame = ttk.Frame(dialog, padding=16)
    frame.pack(fill='both', expand=True)

    ttk.Label(
        frame,
        text=f'{APP_TITLE} {APP_VERSION}',
        font=('Segoe UI', 13, 'bold')
    ).pack(anchor='w', pady=(0, 10))

    text = tk.Text(frame, wrap='word', height=14)
    text.pack(fill='both', expand=True)
    text.insert('1.0', STARTUP_NOTICE)
    text.configure(state='disabled')

    btns = ttk.Frame(frame)
    btns.pack(fill='x', pady=(12, 0))

    def accept():
        accepted['value'] = True
        dialog.destroy()

    def exit_app():
        accepted['value'] = False
        dialog.destroy()

    ttk.Button(btns, text='Accept', command=accept).pack(side='right', padx=(8, 0))
    ttk.Button(btns, text='Exit', command=exit_app).pack(side='right')

    dialog.protocol('WM_DELETE_WINDOW', exit_app)
    dialog.update_idletasks()

    # Center on screen and force it to the foreground.
    sw = dialog.winfo_screenwidth()
    sh = dialog.winfo_screenheight()
    ww = dialog.winfo_width()
    wh = dialog.winfo_height()
    x = max((sw - ww) // 2, 0)
    y = max((sh - wh) // 2, 0)
    dialog.geometry(f'+{x}+{y}')
    dialog.lift()
    dialog.focus_force()
    dialog.grab_set()
    root.wait_window(dialog)
    return accepted['value']


def main():
    root = tk.Tk()
    root.withdraw()
    style = ttk.Style(root)
    try:
        style.theme_use('clam')
    except Exception:
        pass

    if not show_startup_notice(root):
        root.destroy()
        return

    app = DesktopPLFitterApp(root)

    menubar = tk.Menu(root)
    help_menu = tk.Menu(menubar, tearoff=0)
    help_menu.add_command(label='About / License', command=app.show_about_dialog)
    menubar.add_cascade(label='Help', menu=help_menu)
    root.config(menu=menubar)

    root.deiconify()
    root.lift()
    root.focus_force()
    root.mainloop()


if __name__ == '__main__':
    main()
