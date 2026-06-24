from __future__ import annotations

import json
import time
import threading
import queue
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

from FitED_Backend_v4 import (
    BUILTIN_PEAK_KINDS,
    FIT_SELECTION_CRITERIA,
    FIT_OPTIMIZER_MODES,
    AUTO_PREFIT_SAMPLING_MODES,
    DEFAULT_AUTO_PREFIT_SAMPLING_MODE,
    PEAK_DETECTION_DIRECTIONS,
    BATCH_FIT_MODES,
    STABILITY_TEST_PROTOCOLS,
    MAX_RANDOM_SEED,
    RESIDUAL_SUGGESTION_DIRECTIONS,
    RESIDUAL_SUGGESTION_SENSITIVITIES,
    load_spectrum,
    crop_roi,
    smooth_if_requested,
    build_composite_model,
    normalize_custom_profile_definition,
    fit_selection_score,
    fit_metric_summary,
    full_fited_fit_report,
    fit_model_with_optimizer,
    compute_weights,
    seed_with_offset,
    rng_from_seed,
    raise_if_cancelled,
    build_model_from_context,
    worker_fit_once,
    seed_peak_defs_from_centers,
    latin_hypercube_unit_samples,
    finite_trial_bounds,
    sample_linear_from_unit,
    sample_log_from_unit,
    approx_voigt_fwhm,
    lhs_dimension_count_for_peak_def,
    lhs_dimension_count_for_peak_defs,
    use_lhs_for_trial,
    apply_fast_jitter_sampling_to_peak_def,
    apply_lhs_sampling_to_peak_def,
    randomize_custom_param_fast,
    randomize_custom_param_lhs,
    run_autoprefit_search_worker,
    run_autoprefit_search_custom_worker,
    refine_with_added_peaks_worker,
    run_stability_test_worker,
    summarize_stability_test,
    format_stability_test_report,
    split_batch_patterns,
    collect_batch_files,
    batch_context_for_file,
    copy_fit_result_values_into_params,
    batch_fit_one_file,
    batch_result_to_row,
    batch_failed_row,
    write_batch_outputs,
    default_peak_state_for_range,
    x_units_to_samples,
    negative_peak_seed_from_centers,
    detect_peaks_auto,
    residual_noise_sigma,
    residual_suggestion_threshold,
    detect_residual_peak_candidates,
    residual_candidate_peak_defs,
    safe_derived_expression_names,
    validate_derived_expression,
    params_to_value_dict,
    eval_derived_expression,
    eval_derived_expression_for_params,
    finite_difference_derivative_for_var,
    compute_one_derived_quantity,
    compute_derived_quantities,
    compute_derived_uncertainty_contribution_map,
    compute_parameter_correlation_matrix,
    compute_residual_diagnostics,
    confidence_ellipse_pair_summary,
    compute_confidence_ellipse_data,
    parse_auxiliary_parameter_lines,
    parse_parameter_constraint_lines,
    parse_derived_quantity_lines,
    format_derived_quantities_report,
    default_derived_quantity_text,
    build_session_state,
    normalize_session_payload,
)

APP_TITLE = "FitED"
AUTHOR_NAME = "Mustafa Mahmoud Ibrahim Aboulsaad"
APP_VERSION = "v1.4"
SOFTWARE_DOI = "10.5281/zenodo.19411620"
Citation = "Aboulsaad, M. M. I. (2026). FitED. Zenodo. https://doi.org/10.5281/zenodo.19411620"
LICENSE_NAME = "FitED Non-Commercial Software License"
DISCLAIMER_TEXT = (
    "This software is provided as-is, without warranty of any kind, express or implied. "
    "The author shall not be liable for any claim, damages, or other liability arising from, "
    "out of, or in connection with the software or the use or other dealings in the software."
)

STARTUP_NOTICE = f"""{APP_TITLE} {APP_VERSION}

Author: {AUTHOR_NAME}

If you use this software in academic work, please cite:
{Citation}, which is the DOI for all versions.

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
        
        self.derived_quantity_definitions = []
        self.last_derived_quantities = None
        self.last_derived_uncertainty_map = None
        self.last_confidence_ellipse_payloads = []
        self.auxiliary_parameter_definitions = []
        self.parameter_constraint_definitions = []
        self.last_stability_test_payload = None

        # Latest generated text reports kept inside the Reports tab so closing
        # a popup never makes the current report inaccessible. This is not a
        # report history yet: each slot stores only the latest report of its type.
        self.report_documents = {
            'fit': {
                'tab_label': 'Fit report',
                'title': 'Latest fit report',
                'text': '',
            },
            'stability': {
                'tab_label': 'Stability test',
                'title': 'Latest stability test report',
                'text': '',
            },
            'stability_best_fit': {
                'tab_label': 'Stability best fit',
                'title': 'Best fit report chosen by stability test',
                'text': '',
            },
            'derived': {
                'tab_label': 'Derived quantities',
                'title': 'Latest derived quantities report',
                'text': '',
            },
            'correlation': {
                'tab_label': 'Correlation matrix',
                'title': 'Latest parameter correlation matrix report',
                'text': '',
            },
            'residual_diagnostics': {
                'tab_label': 'Residual diagnostics',
                'title': 'Latest residual diagnostics report',
                'text': '',
            },
            'confidence_ellipse': {
                'tab_label': 'Confidence ellipse',
                'title': 'Latest 2D confidence ellipse report',
                'text': '',
            },
        }
        self.report_text_widgets = {}
        self.report_title_labels = {}
        self.report_histories = {key: [] for key in self.report_documents}
        self.report_history_listboxes = {}
        self.result_package_history = []
        self.result_package_listbox = None
        self.result_package_details_text = None
        self.session_history_counter = 0
        self.active_result_package_id = None

        self._build_state()
        
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
        
        self._build_ui()
        self._build_plot()
        self._setup_hover()

        # Background fit worker state. Long lmfit calls run outside the Tk event loop
        # so the GUI stays responsive. All Tk/Matplotlib updates still happen in
        # the main thread through _poll_fit_worker_queue().
        self.fit_worker_thread = None
        self.fit_worker_queue = None
        self.fit_cancel_event = None
        self.fit_worker_busy = False
    
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

    def _current_optimizer_mode(self):
        """Return the selected optimizer mode for lmfit calls."""
        if hasattr(self, 'optimizer_mode_var'):
            return self.optimizer_mode_var.get()
        return FIT_OPTIMIZER_MODES[0]
    
    def _current_autoprefit_sampling_mode(self):
        """Return the selected Auto pre-fit trial-generation method."""
        if hasattr(self, 'autoprefit_sampling_var'):
            mode = self.autoprefit_sampling_var.get()
        else:
            mode = DEFAULT_AUTO_PREFIT_SAMPLING_MODE
    
        if mode not in AUTO_PREFIT_SAMPLING_MODES:
            return DEFAULT_AUTO_PREFIT_SAMPLING_MODE
        return mode

    def _fit_selection_score(self, result, criterion=None):
        """Score a fit result using the backend criterion helper. Lower is better."""
        return fit_selection_score(result, criterion or self._current_fit_criterion())

    def _fit_metric_summary(self, result):
        """Compact text for the status bar and messages."""
        return fit_metric_summary(result)

    def _max_nfev(self):
        try:
            return max(1, int(self.max_nfev_var.get()))
        except Exception:
            return 10000

    def _current_random_seed(self):
        """Return an optional integer seed; blank means stochastic behavior."""
        if not hasattr(self, 'random_seed_var'):
            return None

        raw = str(self.random_seed_var.get()).strip()
        if raw == '':
            return None

        try:
            seed = int(raw)
        except Exception as exc:
            raise ValueError('Random seed must be an integer or left blank.') from exc

        if seed < 0 or seed > MAX_RANDOM_SEED:
            raise ValueError(f'Random seed must be between 0 and {MAX_RANDOM_SEED}, or left blank.')

        return int(seed)

    def _seed_with_offset(self, seed, offset=0):
        """Create a deterministic repeat/trial seed while staying in 32-bit range."""
        return seed_with_offset(seed, offset)

    def _rng_from_seed(self, seed):
        """Create a local NumPy Generator for backend stochastic sampling."""
        return rng_from_seed(seed)

    def _fit_kwargs(self):
        return {
            'max_nfev': self._max_nfev(),
            'optimizer_mode': self._current_optimizer_mode(),
            'random_seed': self._current_random_seed(),
        }

    def _build_model_from_context(self, context, peak_defs):
        """Delegate GUI-independent model construction to the backend."""
        return build_model_from_context(context, peak_defs)

    def _prepare_fit_context(self, require_active_peak=True):
        x, y_raw, y_plot = self._get_roi_data()
        peak_defs = self._collect_peak_defs(require_active_peak=require_active_peak)
        background_kind = self.background_var.get().strip().lower()

        if not peak_defs and background_kind == 'none':
            raise ValueError('At least one active peak is required when Background is none.')

        return {
            'x': x.copy(),
            'y_raw': y_raw.copy(),
            'y_plot': y_plot.copy(),
            'peak_defs': copy.deepcopy(peak_defs),
            'background_kind': self.background_var.get(),
            'poly_order': int(self.poly_order_var.get()),
            'custom_profiles': copy.deepcopy(self.custom_profiles),
            'custom_background_profiles': copy.deepcopy(self.custom_background_profiles),
            'custom_background_profile_name': self.background_profile_var.get(),
            'background_params': copy.deepcopy(self._collect_background_params()),
            'auxiliary_parameters': copy.deepcopy(self.auxiliary_parameter_definitions),
            'parameter_constraints': copy.deepcopy(self.parameter_constraint_definitions),
            'weights': None if self._weights(y_raw) is None else np.asarray(self._weights(y_raw), dtype=float).copy(),
            'criterion': self._current_fit_criterion(),
            'n_trials': max(1, int(self.autofit_trials_var.get())),
            'autoprefit_sampling_mode': self._current_autoprefit_sampling_mode(),
            'max_nfev': self._max_nfev(),
            'optimizer_mode': self._current_optimizer_mode(),
            'random_seed': self._current_random_seed(),
            'active_count': len([row for row in self.peak_rows if row['active'].get()]),
            'all_custom_no_center': (
                len([row for row in self.peak_rows if row['active'].get()]) > 0 and
                all(row['kind'].get() == 'Custom' and not self._custom_profile_has_center(row)
                    for row in self.peak_rows if row['active'].get())
            ),
        }

    def _set_fit_controls_busy(self, busy):
        self.fit_worker_busy = bool(busy)
        normal_state = 'disabled' if busy else 'normal'
        cancel_state = 'normal' if busy else 'disabled'
        for attr in [
            'preview_button',
            'custom_profiles_button',
            'pick_centers_button',
            'find_peaks_button',
            'autofit_button',
            'refine_button',
            'run_fit_button',
            'batch_fit_button',
            'residual_suggest_button',
            'derived_quantities_button',
            'stability_test_button',
            'parameter_constraints_button',
            'confidence_ellipse_button',
            'residual_diagnostics_button',
            'correlation_matrix_button',
        ]:
            if hasattr(self, attr):
                try:
                    getattr(self, attr).configure(state=normal_state)
                except Exception:
                    pass
        if hasattr(self, 'cancel_fit_button'):
            try:
                self.cancel_fit_button.configure(state=cancel_state)
            except Exception:
                pass

    def _start_fit_worker(self, label, worker_func, on_success):
        if getattr(self, 'fit_worker_busy', False):
            messagebox.showinfo('Fit already running', 'A fit is already running. Click Cancel running fit or wait for it to finish.')
            return

        self.fit_worker_queue = queue.Queue()
        self.fit_cancel_event = threading.Event()
        self._set_fit_controls_busy(True)
        if hasattr(self, 'progress_var'):
            self.progress_var.set(0.0)
        self.status_var.set(f'{label} started...')

        def _runner():
            try:
                payload = worker_func(self.fit_cancel_event, self.fit_worker_queue)
                self.fit_worker_queue.put(('done', payload, on_success))
            except RuntimeError as exc:
                if str(exc) == 'Fit cancelled.':
                    self.fit_worker_queue.put(('cancelled', str(exc), None))
                else:
                    self.fit_worker_queue.put(('error', str(exc), None))
            except Exception as exc:
                self.fit_worker_queue.put(('error', str(exc), None))

        self.fit_worker_thread = threading.Thread(target=_runner, daemon=True)
        self.fit_worker_thread.start()
        self.root.after(100, self._poll_fit_worker_queue)

    def _poll_fit_worker_queue(self):
        q = getattr(self, 'fit_worker_queue', None)
        if q is None:
            return

        keep_polling = True
        while True:
            try:
                kind, payload, callback = q.get_nowait()
            except queue.Empty:
                break

            if kind == 'progress':
                current, total, message = payload
                if hasattr(self, 'progress_bar'):
                    try:
                        self.progress_bar.configure(maximum=max(float(total), 1.0))
                        self.progress_var.set(float(current))
                    except Exception:
                        pass
                self.status_var.set(message)

            elif kind == 'done':
                keep_polling = False
                self._set_fit_controls_busy(False)
                if hasattr(self, 'progress_var'):
                    self.progress_var.set(0.0)
                callback(payload)

            elif kind == 'cancelled':
                keep_polling = False
                self._set_fit_controls_busy(False)
                if hasattr(self, 'progress_var'):
                    self.progress_var.set(0.0)
                self.status_var.set('Fit cancelled.')

            elif kind == 'error':
                keep_polling = False
                self._set_fit_controls_busy(False)
                if hasattr(self, 'progress_var'):
                    self.progress_var.set(0.0)
                messagebox.showerror('Fit worker error', str(payload))
                self.status_var.set('Fit failed.')

        if keep_polling and getattr(self, 'fit_worker_busy', False):
            self.root.after(100, self._poll_fit_worker_queue)

    def cancel_running_fit(self):
        if self.fit_cancel_event is not None:
            self.fit_cancel_event.set()
            self.status_var.set('Cancel requested. Waiting for the current fit call to finish...')

    def _raise_if_cancelled(self, cancel_event):
        """Delegate worker cancellation checks to the backend."""
        return raise_if_cancelled(cancel_event)

    def _worker_fit_once(self, context, peak_defs, cancel_event, progress_queue=None, message='Fitting...'):
        """Delegate one GUI-independent fit call to the backend."""
        return worker_fit_once(
            context,
            peak_defs,
            cancel_event,
            progress_queue=progress_queue,
            message=message,
        )

    def run_fit_background(self):
        try:
            context = self._prepare_fit_context(require_active_peak=False)
        except Exception as exc:
            messagebox.showerror('Fit error', str(exc))
            return

        def _worker(cancel_event, progress_queue):
            progress_queue.put(('progress', (0, 1, 'Run fit: fitting...'), None))
            result, comps, best = self._worker_fit_once(context, context['peak_defs'], cancel_event)
            progress_queue.put(('progress', (1, 1, 'Run fit: finished.'), None))
            return {
                'context': context,
                'result': result,
                'components': comps,
                'best_fit': best,
            }

        self._start_fit_worker('Run fit', _worker, self._display_run_fit_payload)

    def _display_run_fit_payload(self, payload):
        context = payload['context']
        x = context['x']
        y_raw = context['y_raw']
        result = payload['result']
        comps = payload['components']
        best = payload['best_fit']

        self.fit_result = result
        self.last_derived_quantities = None
        self.last_derived_uncertainty_map = None
        self.last_confidence_ellipse_payloads = []
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
        self.last_fit_peak_count = context['active_count']
        
        history_id = self._new_session_history_id()
        self._capture_result_package_history(
            result,
            'Run fit result',
            history_id=history_id,
        )
        self._show_fit_report_dialog(
            result,
            title='Run fit report',
            history_id=history_id,
        )

    def _seed_peak_defs_from_centers_worker(self, x, y_raw, peak_defs, custom_profiles):
        """Delegate center-based parameter seeding to the backend."""
        return seed_peak_defs_from_centers(x, y_raw, peak_defs, custom_profiles)

    def _latin_hypercube_unit_samples(self, n_samples, n_dimensions, rng=None):
        """Delegate Latin Hypercube unit sampling to the backend."""
        return latin_hypercube_unit_samples(n_samples, n_dimensions, rng=rng)
    
    
    def _finite_trial_bounds(self, lower, upper, center, fallback_span=1.0, positive=False):
        """Delegate safe finite trial-bound preparation to the backend."""
        return finite_trial_bounds(
            lower,
            upper,
            center,
            fallback_span=fallback_span,
            positive=positive,
        )
    
    
    def _sample_linear_from_unit(self, lower, upper, unit_value):
        """Delegate linear unit-interval sampling to the backend."""
        return sample_linear_from_unit(lower, upper, unit_value)
    
    
    def _sample_log_from_unit(self, lower, upper, unit_value):
        """Delegate log-scaled unit-interval sampling to the backend."""
        return sample_log_from_unit(lower, upper, unit_value)
    
    def _approx_voigt_fwhm_from_sigma_gamma(self, sigma, gamma):
        """Delegate Exact-Voigt FWHM back-calculation to the backend."""
        return approx_voigt_fwhm(sigma, gamma)
    
    
    def _lhs_dimension_count_for_peak_def(self, peak_def):
        """Delegate LHS dimensionality counting to the backend."""
        return lhs_dimension_count_for_peak_def(peak_def)
    
    
    def _lhs_dimension_count_for_peak_defs(self, peak_defs):
        """Delegate LHS dimensionality counting to the backend."""
        return lhs_dimension_count_for_peak_defs(peak_defs)
    
    def _use_lhs_for_trial(self, sampling_mode, trial_index, n_trials):
        """Delegate sampling-mode trial selection to the backend."""
        return use_lhs_for_trial(sampling_mode, trial_index, n_trials)
    
    
    def _apply_fast_jitter_sampling_to_peak_def(self, p, trial, x_span, rng=None):
        """Delegate Fast-Jitter Auto pre-fit sampling to the backend."""
        return apply_fast_jitter_sampling_to_peak_def(p, trial, x_span, rng=rng)
    
    
    def _apply_lhs_sampling_to_peak_def(self, p, lhs_row, dim_offset, x_span, y_span):
        """Delegate Latin-Hypercube Auto pre-fit sampling to the backend."""
        return apply_lhs_sampling_to_peak_def(
            p,
            lhs_row,
            dim_offset,
            x_span=x_span,
            y_span=y_span,
        )
    
    
    def _randomize_custom_param_fast(self, cfg, rng=None):
        """Delegate custom Fast-Jitter parameter randomization to the backend."""
        return randomize_custom_param_fast(cfg, rng=rng)
    
    
    def _randomize_custom_param_lhs(self, cfg, unit_value):
        """Delegate custom LHS parameter randomization to the backend."""
        return randomize_custom_param_lhs(cfg, unit_value)

    def _run_autoprefit_search_worker(self, context, base_peak_defs, cancel_event, progress_queue):
        """Delegate standard Auto pre-fit repeated search to the backend."""
        return run_autoprefit_search_worker(
            context,
            base_peak_defs,
            cancel_event,
            progress_queue,
        )

    def _run_autoprefit_search_custom_worker(self, context, peak_defs, cancel_event, progress_queue):
        """Delegate custom-profile Auto pre-fit repeated search to the backend."""
        return run_autoprefit_search_custom_worker(
            context,
            peak_defs,
            cancel_event,
            progress_queue,
        )

    def autofill_from_centers_background(self):
        try:
            context = self._prepare_fit_context(require_active_peak=True)
        except Exception as exc:
            messagebox.showerror('Auto pre-fit error', str(exc))
            return

        def _worker(cancel_event, progress_queue):
            if context['all_custom_no_center']:
                best_result, best_peak_defs = self._run_autoprefit_search_custom_worker(context, context['peak_defs'], cancel_event, progress_queue)
            else:
                seeded_peak_defs = self._seed_peak_defs_from_centers_worker(
                    context['x'],
                    context['y_raw'],
                    context['peak_defs'],
                    context['custom_profiles'],
                )
                best_result, best_peak_defs = self._run_autoprefit_search_worker(context, seeded_peak_defs, cancel_event, progress_queue)

            comps = best_result.eval_components(x=context['x'])
            return {
                'context': context,
                'result': best_result,
                'peak_defs': best_peak_defs,
                'components': comps,
                'best_fit': best_result.best_fit,
            }

        self._start_fit_worker('Auto pre-fit', _worker, self._display_autofit_payload)

    def _display_autofit_payload(self, payload):
        context = payload['context']
        x = context['x']
        y_raw = context['y_raw']
        best_result = payload['result']
        best_peak_defs = payload['peak_defs']

        self._apply_peak_defs_to_ui(best_peak_defs, fit_result=best_result)
        self.fit_result = best_result
        self.last_derived_quantities = None
        self.last_derived_uncertainty_map = None
        self.last_confidence_ellipse_payloads = []
        self.last_components = payload['components']
        self.last_best_fit = payload['best_fit']
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

        criterion = context['criterion']
        best_metric = fit_selection_score(best_result, criterion)
        self.status_var.set(
            f'Automatic pre-fit complete. Best {criterion}: {best_metric:.6g}. '
            f'{self._fit_metric_summary(best_result)}'
        )
        
        history_id = self._new_session_history_id()
        self._capture_result_package_history(
            best_result,
            'Automatic pre-fit result',
            history_id=history_id,
        )
        self._show_fit_report_dialog(
            best_result,
            title='Automatic pre-fit report',
            history_id=history_id,
        )

    def refine_with_added_peaks_background(self):
        if self.fit_result is None:
            messagebox.showinfo('No previous fit', 'Run Auto pre-fit or Run fit on the main peaks first.')
            return
        try:
            context = self._prepare_fit_context(require_active_peak=True)
        except Exception as exc:
            messagebox.showerror('Refine with added peaks error', str(exc))
            return

        active_count = len(context['peak_defs'])
        if active_count <= self.last_fit_peak_count:
            messagebox.showerror('Refine with added peaks error', 'No newly added active peaks detected.')
            return

        params_prev = self.fit_result.params.copy()
        old_count = self.last_fit_peak_count

        def _worker(cancel_event, progress_queue):
            return refine_with_added_peaks_worker(
                context,
                params_prev,
                old_count,
                active_count,
                cancel_event,
                progress_queue,
            )

        self._start_fit_worker('Refine with added peaks', _worker, self._display_refine_payload)

    def _display_refine_payload(self, payload):
        context = payload['context']
        x = context['x']
        y_raw = context['y_raw']
        final_result = payload['result']

        self.fit_result = final_result
        self.last_derived_quantities = None
        self.last_derived_uncertainty_map = None
        self.last_confidence_ellipse_payloads = []
        self.last_components = payload['components']
        self.last_best_fit = payload['best_fit']
        self.last_roi = (x.copy(), y_raw.copy())
        self.last_fit_peak_count = payload['active_count']
        self._apply_peak_defs_to_ui(payload['peak_defs'], fit_result=final_result)

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

        criterion = context['criterion']
        final_metric = fit_selection_score(final_result, criterion)
        self.status_var.set(
            f'Refine with added peaks complete. Best {criterion}: {final_metric:.6g}. '
            f'{self._fit_metric_summary(final_result)}'
        )
        
        history_id = self._new_session_history_id()
        self._capture_result_package_history(
            final_result,
            'Refine with added peaks result',
            history_id=history_id,
        )
        self._show_fit_report_dialog(
            final_result,
            title='Refine with added peaks report',
            history_id=history_id,
        )

    
    
    def open_fit_stability_dialog(self):
        """Open a dialog for repeated fit/search stability testing."""
        if self.x_full is None or self.y_full is None:
            messagebox.showinfo('No data', 'Load and configure one spectrum before running a stability test.')
            return

        win = tk.Toplevel(self.root)
        win.title('Fit stability test')
        win.geometry('760x470')
        win.minsize(680, 420)

        frame = ttk.Frame(win, padding=12)
        frame.pack(fill=tk.BOTH, expand=True)

        ttk.Label(
            frame,
            text='Fit stability test',
            font=('Segoe UI', 12, 'bold')
        ).pack(anchor='w', pady=(0, 8))

        ttk.Label(
            frame,
            text=(
                'This repeats the chosen FitED fit/search protocol with different stochastic '
                'explorations to test whether the decomposition is stable. The best repeated '
                'solution is displayed when the test finishes.'
            ),
            wraplength=700,
        ).pack(anchor='w', pady=(0, 10))

        warning = (
            'Interpretation warning: a good stability score does not prove that a decomposition '
            'is physically unique. Use the score spread, near-best parameter spread, correlations, '
            'residuals, and scientific constraints together.'
        )
        ttk.Label(
            frame,
            text=warning,
            wraplength=700,
            foreground='#8a4b00'
        ).pack(anchor='w', pady=(0, 12))

        form = ttk.Frame(frame)
        form.pack(fill=tk.X)

        ttk.Label(form, text='Protocol').grid(row=0, column=0, sticky='w', pady=4)
        ttk.Combobox(
            form,
            textvariable=self.stability_protocol_var,
            values=STABILITY_TEST_PROTOCOLS,
            state='readonly',
            width=28
        ).grid(row=0, column=1, sticky='w', padx=(8, 8), pady=4)

        ttk.Label(form, text='Repeated searches').grid(row=1, column=0, sticky='w', pady=4)
        ttk.Spinbox(
            form,
            from_=1,
            to=1000,
            textvariable=self.stability_repeats_var,
            width=10
        ).grid(row=1, column=1, sticky='w', padx=(8, 8), pady=4)

        ttk.Label(form, text='Near-best Δ criterion').grid(row=2, column=0, sticky='w', pady=4)
        ttk.Entry(
            form,
            textvariable=self.stability_delta_var,
            width=12
        ).grid(row=2, column=1, sticky='w', padx=(8, 8), pady=4)

        ttk.Label(form, text='Base random seed').grid(row=3, column=0, sticky='w', pady=4)
        ttk.Entry(
            form,
            textvariable=self.random_seed_var,
            width=14
        ).grid(row=3, column=1, sticky='w', padx=(8, 8), pady=4)

        ttk.Label(
            form,
            text='Blank = stochastic. Integer = reproducible repeated experiment.',
            wraplength=420
        ).grid(row=3, column=2, sticky='w', pady=4)

        ttk.Label(
            frame,
            text=(
                'Seed rule: with base seed S, repeat 1 uses S, repeat 2 uses S+1, and so on. '
                'For Auto pre-fit, each repeat still contains its own trial search.'
            ),
            wraplength=700
        ).pack(anchor='w', pady=(12, 8))

        btns = ttk.Frame(frame)
        btns.pack(fill=tk.X, pady=(12, 0))

        def start_test():
            protocol = self.stability_protocol_var.get()
            if protocol not in STABILITY_TEST_PROTOCOLS:
                messagebox.showerror('Stability test error', 'Choose a valid stability protocol.', parent=win)
                return

            try:
                repeats = max(1, int(self.stability_repeats_var.get()))
                delta_score = float(self.stability_delta_var.get())
                if not np.isfinite(delta_score) or delta_score < 0:
                    raise ValueError('Near-best Δ criterion must be a finite non-negative number.')

                require_active_peak = protocol == 'Repeat Auto pre-fit'
                context = self._prepare_fit_context(require_active_peak=require_active_peak)
                base_seed = context.get('random_seed')
            except Exception as exc:
                messagebox.showerror('Stability test error', str(exc), parent=win)
                return

            setup = {
                'protocol': protocol,
                'repeats': repeats,
                'delta_score': delta_score,
                'base_seed': base_seed,
                'context': context,
            }

            win.destroy()
            self._start_stability_test_background(setup)

        ttk.Button(btns, text='Start stability test', command=start_test).pack(side=tk.LEFT)
        ttk.Button(btns, text='Close', command=win.destroy).pack(side=tk.RIGHT)

        win.transient(self.root)
        win.lift()
        win.focus_force()


    def _start_stability_test_background(self, setup):
        """Run repeated stability testing in the worker thread through the backend."""
        def _worker(cancel_event, progress_queue):
            return run_stability_test_worker(setup, cancel_event, progress_queue)

        self._start_fit_worker(
            'Fit stability test',
            _worker,
            self._display_stability_test_payload
        )


    def _summarize_stability_test(self, records, failures, protocol, repeats, delta_score, base_seed, base_context):
        """Delegate stability-summary computation to the backend."""
        return summarize_stability_test(
            records,
            failures,
            protocol,
            repeats,
            delta_score,
            base_seed,
            base_context,
        )


    def _format_stability_test_report(self, payload):
        """Delegate stability-report formatting to the backend."""
        return format_stability_test_report(payload)


    def _report_placeholder_text(self, report_key):
        """Readable empty-state text for one Reports-tab document."""
        placeholders = {
            'fit': (
                'No fit report history is available yet.\n\n'
                'Run fit, Auto pre-fit, or Refine with added peaks. Each generated report '
                'will be kept here until you clear it or close FitED.'
            ),
            'stability': (
                'No stability-test report history is available yet.\n\n'
                'Run Fit stability test. Each stability summary report will be kept here '
                'until you clear it or close FitED.'
            ),
            'stability_best_fit': (
                'No best repeated stability-test fit report history is available yet.\n\n'
                'Run Fit stability test. FitED will still open the popup report and will '
                'also keep the selected best repeated solution report here.'
            ),
            'derived': (
                'No derived-quantities report history is available yet.\n\n'
                'Open Derived quantities, enter definitions, and click Compute. Each computed '
                'report will be kept here until you clear it or close FitED.'
            ),
            'correlation': (
                'No parameter-correlation report history is available yet.\n\n'
                'Run a fit, then open Parameter correlation matrix. The matrix report '
                'will be kept here until you clear it or close FitED.'
            ),
            'residual_diagnostics': (
                'No residual-diagnostics report history is available yet.\n\n'
                'Run a fit, then open Residual diagnostics. The residual statistics, '
                'autocorrelation, and Q-Q summary will be kept here until you clear it or close FitED.'
            ),
            'confidence_ellipse': (
                'No confidence-ellipse report history is available yet.\n\n'
                'Run a fit, then open 2D confidence ellipse and choose a parameter pair. '
                'The selected-pair summary will be kept here until you clear it or close FitED.'
            ),
        }
        return placeholders.get(str(report_key), 'No report is available yet.')


    def _new_session_history_id(self):
        """Return the next in-memory history identifier for this FitED session."""
        self.session_history_counter = int(getattr(self, 'session_history_counter', 0)) + 1
        return int(self.session_history_counter)


    def _history_source_stem(self):
        """Return a readable current source-file stem for report/package history labels."""
        try:
            if self.current_file is not None:
                stem = str(Path(self.current_file).stem).strip()
                if stem:
                    return stem
        except Exception:
            pass
        return 'fit_session'


    def _history_label(self, descriptor, history_id):
        """Build a readable session-history list label."""
        return f"{self._history_source_stem()} — {str(descriptor).strip()} — ID {int(history_id):03d}"


    def _default_report_title(self, report_key):
        """Default title used when a report-history tab has no selected entry."""
        defaults = {
            'fit': 'Fit report history',
            'stability': 'Stability test report history',
            'stability_best_fit': 'Stability best-fit report history',
            'derived': 'Derived quantities report history',
            'correlation': 'Parameter correlation matrix history',
            'residual_diagnostics': 'Residual diagnostics history',
            'confidence_ellipse': '2D confidence ellipse history',
        }
        return defaults.get(str(report_key), 'Report history')


    def _report_display_text(self, report_key):
        """Return the selected report text or an informative empty-state placeholder."""
        doc = getattr(self, 'report_documents', {}).get(report_key, {})
        text = str(doc.get('text', '') or '')
        return text if text.strip() else self._report_placeholder_text(report_key)


    def _refresh_report_document_view(self, report_key):
        """Refresh one Reports-tab document viewer after its selection/text changes."""
        if not hasattr(self, 'report_documents'):
            return

        doc = self.report_documents.get(report_key, {})
        title_label = getattr(self, 'report_title_labels', {}).get(report_key)
        if title_label is not None:
            try:
                title_label.configure(text=str(doc.get('title', self._default_report_title(report_key))))
            except Exception:
                pass

        text_widget = getattr(self, 'report_text_widgets', {}).get(report_key)
        if text_widget is not None:
            try:
                text_widget.configure(state='normal')
                text_widget.delete('1.0', tk.END)
                text_widget.insert('1.0', self._report_display_text(report_key))
                text_widget.configure(state='disabled')
            except Exception:
                pass


    def _refresh_reports_tab(self):
        """Refresh all Reports-tab text viewers and history listboxes."""
        for key in getattr(self, 'report_documents', {}):
            self._refresh_report_history_list(key)
            self._refresh_report_document_view(key)
        self._refresh_result_package_history_list()


    def _report_history_entries(self, report_key):
        """Return the in-memory history list for one report type."""
        histories = getattr(self, 'report_histories', {})
        return histories.get(report_key, [])


    def _refresh_report_history_list(self, report_key, select_history_id=None):
        """Refresh one report-history listbox and optionally select a specific entry."""
        entries = self._report_history_entries(report_key)
        listbox = getattr(self, 'report_history_listboxes', {}).get(report_key)

        doc = getattr(self, 'report_documents', {}).get(report_key)
        if doc is None:
            return

        selected_index = None
        if select_history_id is None:
            current_id = doc.get('history_id')
            for idx, entry in enumerate(entries):
                if entry.get('id') == current_id:
                    selected_index = idx
                    break
        else:
            for idx, entry in enumerate(entries):
                if entry.get('id') == select_history_id:
                    selected_index = idx
                    break

        if selected_index is None and entries:
            selected_index = len(entries) - 1

        if listbox is not None:
            try:
                listbox.delete(0, tk.END)
                for entry in entries:
                    listbox.insert(tk.END, entry.get('label', 'Report'))
                if selected_index is not None:
                    listbox.selection_clear(0, tk.END)
                    listbox.selection_set(selected_index)
                    listbox.activate(selected_index)
                    listbox.see(selected_index)
            except Exception:
                pass

        if selected_index is None:
            doc['history_id'] = None
            doc['title'] = self._default_report_title(report_key)
            doc['text'] = ''
        else:
            entry = entries[selected_index]
            doc['history_id'] = entry.get('id')
            doc['title'] = entry.get('title', self._default_report_title(report_key))
            doc['text'] = entry.get('text', '')


    def _select_report_history_entry(self, report_key, event=None):
        """Load the clicked report-history item into the right-side viewer."""
        listbox = getattr(self, 'report_history_listboxes', {}).get(report_key)
        entries = self._report_history_entries(report_key)
        if listbox is None or not entries:
            return

        selection = listbox.curselection()
        if not selection:
            return

        idx = int(selection[0])
        if idx < 0 or idx >= len(entries):
            return

        entry = entries[idx]
        doc = self.report_documents.get(report_key, {})
        doc['history_id'] = entry.get('id')
        doc['title'] = entry.get('title', self._default_report_title(report_key))
        doc['text'] = entry.get('text', '')
        self._refresh_report_document_view(report_key)


    def _set_report_document(self, report_key, report_text, title=None, history_id=None):
        """
        Add a report to the in-memory session history and select it in Reports.

        Popups are still shown by the calling code. This method only preserves the
        report text inside the Reports tab until the user clears it or closes FitED.
        """
        if not hasattr(self, 'report_documents'):
            return None
        if report_key not in self.report_documents:
            return None

        if history_id is None:
            history_id = self._new_session_history_id()

        display_title = str(title or self._default_report_title(report_key))
        entry = {
            'id': int(history_id),
            'label': self._history_label(display_title, history_id),
            'title': display_title,
            'text': str(report_text or ''),
            'source_file': str(self.current_file) if self.current_file else '',
            'report_key': str(report_key),
        }

        histories = getattr(self, 'report_histories', {})
        histories.setdefault(report_key, []).append(entry)

        doc = self.report_documents[report_key]
        doc['history_id'] = int(history_id)
        doc['title'] = display_title
        doc['text'] = str(report_text or '')

        self._refresh_report_history_list(report_key, select_history_id=history_id)
        self._refresh_report_document_view(report_key)
        return int(history_id)


    def _reset_report_documents(self):
        """
        Reset only the current viewer selection placeholders.

        Report histories intentionally remain in memory across data reloads/resets
        during this FitED session so the user can return to older trials.
        """
        for key, doc in getattr(self, 'report_documents', {}).items():
            entries = self._report_history_entries(key)
            if entries:
                latest = entries[-1]
                doc['history_id'] = latest.get('id')
                doc['title'] = latest.get('title', self._default_report_title(key))
                doc['text'] = latest.get('text', '')
            else:
                doc['history_id'] = None
                doc['title'] = self._default_report_title(key)
                doc['text'] = ''
        self._refresh_reports_tab()


    def _copy_report_document(self, report_key):
        """Copy the selected report text from the Reports tab to the clipboard."""
        doc = getattr(self, 'report_documents', {}).get(report_key, {})
        text = str(doc.get('text', '') or '').strip()
        if not text:
            messagebox.showinfo(
                'No report available',
                'Select or generate a report first.'
            )
            return

        self.root.clipboard_clear()
        self.root.clipboard_append(text)
        label = str(doc.get('tab_label', 'Report'))
        self.status_var.set(f'{label} copied to clipboard.')


    def _delete_selected_report_history(self, report_key):
        """Delete the selected report-history entry from one Reports sub-tab."""
        entries = self._report_history_entries(report_key)
        listbox = getattr(self, 'report_history_listboxes', {}).get(report_key)
        if listbox is None or not entries:
            messagebox.showinfo('No report selected', 'There is no report history entry to delete.')
            return

        selection = listbox.curselection()
        if not selection:
            messagebox.showinfo('No report selected', 'Select one report history entry first.')
            return

        idx = int(selection[0])
        if idx < 0 or idx >= len(entries):
            return

        removed = entries.pop(idx)
        next_id = None
        if entries:
            next_entry = entries[min(idx, len(entries) - 1)]
            next_id = next_entry.get('id')

        self._refresh_report_history_list(report_key, select_history_id=next_id)
        self._refresh_report_document_view(report_key)
        self.status_var.set(f"Deleted report history entry: {removed.get('label', 'report')}.")


    def _clear_report_history(self, report_key):
        """Clear all report-history entries from one Reports sub-tab."""
        entries = self._report_history_entries(report_key)
        if not entries:
            return

        label = self.report_documents.get(report_key, {}).get('tab_label', 'report')
        proceed = messagebox.askyesno(
            'Clear report history',
            f'Clear all {label} history entries for this FitED session?'
        )
        if not proceed:
            return

        entries.clear()
        self._refresh_report_history_list(report_key)
        self._refresh_report_document_view(report_key)
        self.status_var.set(f'Cleared {label} history.')


    def _capture_result_package_history(self, result, descriptor, history_id=None):
        """
        Snapshot a fitted result as a temporary in-memory export package.

        The package can later be written to a ZIP from Reports → Result packages.
        Nothing is saved permanently until the user chooses Save selected as ZIP.
        """
        if result is None or self.last_roi is None or self.last_best_fit is None:
            return None

        if history_id is None:
            history_id = self._new_session_history_id()

        x, y = self.last_roi
        x = np.asarray(x, dtype=float).copy()
        y = np.asarray(y, dtype=float).copy()
        best_fit = np.asarray(self.last_best_fit, dtype=float).copy()
        components = {
            str(name): np.asarray(comp, dtype=float).copy()
            for name, comp in (self.last_components or {}).items()
        }

        curves_df = pd.DataFrame({
            'x': x,
            'y_data': y,
            'y_fit': best_fit,
            'residual': y - best_fit,
        })
        for name, comp in components.items():
            curves_df[name] = comp

        param_rows = []
        for name, par in result.params.items():
            param_rows.append({
                'parameter': name,
                'value': par.value,
                'stderr': par.stderr,
                'min': par.min,
                'max': par.max,
                'vary': par.vary,
                'expr': par.expr,
            })
        params_df = pd.DataFrame(param_rows)

        meta_df = pd.DataFrame([
            {'field': 'source_file', 'value': str(self.current_file) if self.current_file else ''},
            {'field': 'history_descriptor', 'value': str(descriptor)},
            {'field': 'history_id', 'value': int(history_id)},
            {'field': 'background', 'value': self.background_var.get()},
            {'field': 'poly_order', 'value': int(self.poly_order_var.get())},
            {'field': 'weighting', 'value': self.weighting_var.get()},
            {'field': 'fit_criterion', 'value': self._current_fit_criterion()},
            {'field': 'optimizer_mode', 'value': self._current_optimizer_mode()},
            {'field': 'random_seed', 'value': getattr(result, 'fited_random_seed', '')},
            {'field': 'selected_optimizer_candidate', 'value': getattr(result, 'fited_selected_candidate', '')},
            {'field': 'optimizer_candidate_scores', 'value': json.dumps(getattr(result, 'fited_candidate_scores', {}))},
            {'field': 'roi_min', 'value': float(self.roi_min_var.get())},
            {'field': 'roi_max', 'value': float(self.roi_max_var.get())},
            {'field': 'chisqr', 'value': float(getattr(result, 'chisqr', np.nan))},
            {'field': 'redchi', 'value': float(getattr(result, 'redchi', np.nan))},
            {'field': 'aic', 'value': float(getattr(result, 'aic', np.nan))},
            {'field': 'bic', 'value': float(getattr(result, 'bic', np.nan))},
            {'field': 'nfev', 'value': int(getattr(result, 'nfev', -1))},
        ])

        derived_rows = None
        derived_df = None
        derived_uncertainty_map_payload = None
        derived_uncertainty_map_df = None

        correlation_matrix_payload = None
        correlation_matrix_df = None
        residual_diagnostics_payload = None
        residual_diagnostics_dfs = None
        confidence_ellipse_summary_df = None

        try:
            correlation_matrix_payload = compute_parameter_correlation_matrix(result)
            correlation_matrix_df = self._parameter_correlation_dataframe(correlation_matrix_payload)
            if correlation_matrix_df.empty:
                correlation_matrix_df = None
        except Exception:
            correlation_matrix_payload = None
            correlation_matrix_df = None

        try:
            residual_diagnostics_payload = compute_residual_diagnostics(
                x,
                y,
                best_fit,
                weights=self._weights(y),
            )
            residual_diagnostics_dfs = self._residual_diagnostics_to_dataframes(residual_diagnostics_payload)
        except Exception:
            residual_diagnostics_payload = None
            residual_diagnostics_dfs = None

        try:
            ellipse_rows = confidence_ellipse_pair_summary(result)
            confidence_ellipse_summary_df = pd.DataFrame(ellipse_rows)
            if confidence_ellipse_summary_df.empty:
                confidence_ellipse_summary_df = None
        except Exception:
            confidence_ellipse_summary_df = None

        try:
            session_state = copy.deepcopy(self.get_session_state())
        except Exception:
            session_state = {}

        fit_report = self._full_fited_fit_report(result)
        base = self._history_source_stem()
        label = self._history_label(str(descriptor), history_id)

        entry = {
            'id': int(history_id),
            'label': label,
            'descriptor': str(descriptor),
            'base': base,
            'source_file': str(self.current_file) if self.current_file else '',
            'original_file_path': str(self.current_file) if self.current_file else '',
            'curves_df': curves_df,
            'params_df': params_df,
            'meta_df': meta_df,
            'fit_report': fit_report,
            'session_state': session_state,
            'derived_rows': derived_rows,
            'derived_df': derived_df,
            'derived_uncertainty_map_payload': derived_uncertainty_map_payload,
            'derived_uncertainty_map_df': derived_uncertainty_map_df,
            'correlation_matrix_payload': correlation_matrix_payload,
            'correlation_matrix_df': correlation_matrix_df,
            'residual_diagnostics_payload': residual_diagnostics_payload,
            'residual_diagnostics_dfs': residual_diagnostics_dfs,
            'confidence_ellipse_summary_df': confidence_ellipse_summary_df,
            'confidence_ellipse_payloads': copy.deepcopy(getattr(self, 'last_confidence_ellipse_payloads', [])),
        }

        self.result_package_history.append(entry)
        self.active_result_package_id = int(history_id)
        self._refresh_result_package_history_list(select_history_id=history_id)
        return int(history_id)


    def _attach_derived_quantities_to_active_result_package(
        self,
        rows,
        report_text=None,
        uncertainty_map_payload=None,
    ):
        """
        Attach newly computed derived quantities and their uncertainty map
        to the currently active result package.
        """
        active_id = getattr(self, 'active_result_package_id', None)
        if active_id is None:
            return

        for entry in reversed(getattr(self, 'result_package_history', [])):
            if entry.get('id') != active_id:
                continue

            copied_rows = copy.deepcopy(rows) if rows else None
            entry['derived_rows'] = copied_rows
            entry['derived_df'] = pd.DataFrame(copied_rows) if copied_rows else None

            if report_text is not None:
                entry['derived_report'] = str(report_text)

            copied_map_payload = (
                copy.deepcopy(uncertainty_map_payload)
                if uncertainty_map_payload
                else None
            )
            entry['derived_uncertainty_map_payload'] = copied_map_payload

            map_df = self._derived_uncertainty_heatmap_dataframe(copied_map_payload)
            entry['derived_uncertainty_map_df'] = (
                map_df.copy() if not map_df.empty else None
            )

            self._refresh_result_package_details_view()
            return


    def _selected_result_package_entry(self):
        """Return the currently selected temporary result-package entry."""
        entries = getattr(self, 'result_package_history', [])
        listbox = getattr(self, 'result_package_listbox', None)
        if listbox is None or not entries:
            return None

        selection = listbox.curselection()
        if not selection:
            return None

        idx = int(selection[0])
        if idx < 0 or idx >= len(entries):
            return None
        return entries[idx]


    def _refresh_result_package_history_list(self, select_history_id=None):
        """Refresh the temporary result-package listbox."""
        entries = getattr(self, 'result_package_history', [])
        listbox = getattr(self, 'result_package_listbox', None)
        if listbox is None:
            self._refresh_result_package_details_view()
            return

        selected_index = None
        if select_history_id is not None:
            for idx, entry in enumerate(entries):
                if entry.get('id') == select_history_id:
                    selected_index = idx
                    break

        if selected_index is None and entries:
            current = self._selected_result_package_entry()
            current_id = current.get('id') if current else None
            if current_id is not None:
                for idx, entry in enumerate(entries):
                    if entry.get('id') == current_id:
                        selected_index = idx
                        break

        if selected_index is None and entries:
            selected_index = len(entries) - 1

        try:
            listbox.delete(0, tk.END)
            for entry in entries:
                listbox.insert(tk.END, entry.get('label', 'Result package'))
            if selected_index is not None:
                listbox.selection_clear(0, tk.END)
                listbox.selection_set(selected_index)
                listbox.activate(selected_index)
                listbox.see(selected_index)
        except Exception:
            pass

        self._refresh_result_package_details_view()


    def _result_package_details_text(self, entry):
        """Human-readable summary for one temporary result package."""
        if not entry:
            return (
                'No temporary result package has been created yet.\n\n'
                'After Run fit, Auto pre-fit, Refine with added peaks, or Fit stability test, '
                'FitED stores an in-memory export package here. Select one package and use '
                '"Save selected as ZIP" only when you decide it is worth keeping.'
            )

        meta_values = {}
        try:
            for _, row in entry.get('meta_df', pd.DataFrame()).iterrows():
                meta_values[str(row.get('field', ''))] = row.get('value', '')
        except Exception:
            meta_values = {}

        lines = [
            '[[FitED temporary result package]]',
            f"Name              = {entry.get('label', '')}",
            f"Source file       = {entry.get('source_file', '')}",
            f"Result type       = {entry.get('descriptor', '')}",
            f"History ID        = {entry.get('id', '')}",
            '',
            'This package is stored only in memory for the current FitED session.',
            'It is erased automatically when FitED closes unless you save it as a ZIP.',
            '',
            'Package content prepared for ZIP export:',
            '- fitted curves and residuals',
            '- fitted parameter table',
            '- fit summary table',
            '- fit report text',
            '- session JSON snapshot',
            '- Excel workbook when openpyxl is available',
        ]

        if entry.get('derived_df') is not None:
            lines.append('- derived quantities table')
        if entry.get('derived_uncertainty_map_df') is not None:
            lines.append('- derived uncertainty contribution map: CSV matrix, PNG heatmap, and Excel sheet')

        lines.extend([
            '- original spectrum file copy when the saved source path still exists',
            '',
            'Fit summary:',
            f"  chi-square = {meta_values.get('chisqr', '')}",
            f"  redchi     = {meta_values.get('redchi', '')}",
            f"  AIC        = {meta_values.get('aic', '')}",
            f"  BIC        = {meta_values.get('bic', '')}",
            f"  nfev       = {meta_values.get('nfev', '')}",
            f"  optimizer  = {meta_values.get('optimizer_mode', '')}",
            f"  candidate  = {meta_values.get('selected_optimizer_candidate', '')}",
        ])
        return '\n'.join(lines)


    def _refresh_result_package_details_view(self, event=None):
        """Refresh the right-side temporary result-package summary viewer."""
        widget = getattr(self, 'result_package_details_text', None)
        if widget is None:
            return

        entry = self._selected_result_package_entry()
        text = self._result_package_details_text(entry)
        try:
            widget.configure(state='normal')
            widget.delete('1.0', tk.END)
            widget.insert('1.0', text)
            widget.configure(state='disabled')
        except Exception:
            pass


    def _delete_selected_result_package(self):
        """Delete one selected temporary result package."""
        entries = getattr(self, 'result_package_history', [])
        listbox = getattr(self, 'result_package_listbox', None)
        if listbox is None or not entries:
            messagebox.showinfo('No package selected', 'There is no result package to delete.')
            return

        selection = listbox.curselection()
        if not selection:
            messagebox.showinfo('No package selected', 'Select one result package first.')
            return

        idx = int(selection[0])
        if idx < 0 or idx >= len(entries):
            return

        removed = entries.pop(idx)
        if removed.get('id') == getattr(self, 'active_result_package_id', None):
            self.active_result_package_id = None

        next_id = None
        if entries:
            next_id = entries[min(idx, len(entries) - 1)].get('id')

        self._refresh_result_package_history_list(select_history_id=next_id)
        self.status_var.set(f"Deleted temporary result package: {removed.get('label', 'package')}.")


    def _clear_result_package_history(self):
        """Clear all temporary result packages from the current FitED session."""
        entries = getattr(self, 'result_package_history', [])
        if not entries:
            return

        proceed = messagebox.askyesno(
            'Clear result package history',
            'Clear all temporary result packages for this FitED session?'
        )
        if not proceed:
            return

        entries.clear()
        self.active_result_package_id = None
        self._refresh_result_package_history_list()
        self.status_var.set('Cleared temporary result package history.')


    def _safe_history_zip_default_name(self, entry):
        """Return a safe default ZIP filename for one temporary result package."""
        base = str(entry.get('base', 'fit_session') or 'fit_session')
        descriptor = str(entry.get('descriptor', 'result') or 'result')
        safe_desc = ''.join(ch if ch.isalnum() else '_' for ch in descriptor).strip('_')
        safe_desc = safe_desc or 'result'
        history_id = int(entry.get('id', 0) or 0)
        return f'{base}_{safe_desc}_ID{history_id:03d}_results.zip'
    
    def _write_confidence_ellipse_exports(self, tmpdir, base, ellipse_payloads):
        """
        Write selected/generated covariance ellipse exports:
        - pair summary CSV
        - coordinate CSV
        - one PNG per selected parameter pair
        """
        created_paths = []
        ellipse_payloads = ellipse_payloads or []
    
        if not ellipse_payloads:
            return created_paths
    
        # -----------------------------
        # 1) Pair summary CSV
        # -----------------------------
        summary_rows = []
    
        for payload in ellipse_payloads:
            try:
                cov_mat = payload.get(
                    "covariance_matrix",
                    [[np.nan, np.nan], [np.nan, np.nan]]
                )
                eigenvalues = payload.get("eigenvalues", [np.nan, np.nan])
                vals = np.asarray(eigenvalues, dtype=float)
    
                if vals.size < 2:
                    vals = np.array([np.nan, np.nan], dtype=float)
    
                summary_rows.append({
                    "parameter_x": payload.get("parameter_x", ""),
                    "parameter_y": payload.get("parameter_y", ""),
                    "center_x": payload.get("center_x", payload.get("x_value", np.nan)),
                    "center_y": payload.get("center_y", payload.get("y_value", np.nan)),
                    "covariance": float(cov_mat[0][1]),
                    "correlation": payload.get("correlation", np.nan),
                    "ellipse_angle_deg": payload.get("ellipse_angle_deg", np.nan),
                    "width_1sigma": float(2.0 * np.sqrt(vals[0])) if np.isfinite(vals[0]) else np.nan,
                    "height_1sigma": float(2.0 * np.sqrt(vals[1])) if np.isfinite(vals[1]) else np.nan,
                    "width_2sigma": float(4.0 * np.sqrt(vals[0])) if np.isfinite(vals[0]) else np.nan,
                    "height_2sigma": float(4.0 * np.sqrt(vals[1])) if np.isfinite(vals[1]) else np.nan,
                })
            except Exception:
                pass
    
        if summary_rows:
            summary_path = tmpdir / f"{base}_confidence_ellipse_pair_summary.csv"
            pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
            created_paths.append(summary_path)
    
        # -----------------------------
        # 2) Coordinate CSV
        # -----------------------------
        coord_rows = []
    
        for payload in ellipse_payloads:
            px = payload.get("parameter_x", "")
            py = payload.get("parameter_y", "")
    
            for ellipse in payload.get("ellipses", []):
                sigma = ellipse.get("sigma", np.nan)
                xs = ellipse.get("x", [])
                ys = ellipse.get("y", [])
    
                for xv, yv in zip(xs, ys):
                    try:
                        coord_rows.append({
                            "parameter_x": px,
                            "parameter_y": py,
                            "sigma": sigma,
                            "x": float(xv),
                            "y": float(yv),
                        })
                    except Exception:
                        pass
    
        if coord_rows:
            coord_path = tmpdir / f"{base}_confidence_ellipse_coordinates.csv"
            pd.DataFrame(coord_rows).to_csv(coord_path, index=False)
            created_paths.append(coord_path)
    
        # -----------------------------
        # 3) PNG for each selected ellipse
        # -----------------------------
        used_names = set()
    
        for idx, payload in enumerate(ellipse_payloads, start=1):
            try:
                px = str(payload.get("parameter_x", "x"))
                py = str(payload.get("parameter_y", "y"))
    
                safe_px = "".join(ch if ch.isalnum() else "_" for ch in px).strip("_") or "x"
                safe_py = "".join(ch if ch.isalnum() else "_" for ch in py).strip("_") or "y"
    
                filename = f"{base}_confidence_ellipse_{safe_px}_vs_{safe_py}.png"
    
                if filename in used_names:
                    filename = f"{base}_confidence_ellipse_{safe_px}_vs_{safe_py}_{idx}.png"
    
                used_names.add(filename)
    
                png_path = tmpdir / filename
                fig = self._build_confidence_ellipse_figure(payload)
                fig.savefig(png_path, dpi=220, bbox_inches="tight")
                created_paths.append(png_path)
    
            except Exception:
                pass
    
        return created_paths



    def _write_result_package_zip(self, entry, target):
        """Write one in-memory result package to a user-selected ZIP archive."""
        target = Path(target)
        base = str(entry.get('base', 'fit_session') or 'fit_session')

        curves_df = entry.get('curves_df')
        params_df = entry.get('params_df')
        meta_df = entry.get('meta_df')
        derived_df = entry.get('derived_df')
        derived_uncertainty_map_df = entry.get('derived_uncertainty_map_df')
        correlation_matrix_df = entry.get('correlation_matrix_df')
        residual_diagnostics_payload = entry.get('residual_diagnostics_payload')
        residual_diagnostics_dfs = entry.get('residual_diagnostics_dfs')
        confidence_ellipse_summary_df = entry.get('confidence_ellipse_summary_df')

        if curves_df is None or params_df is None or meta_df is None:
            raise ValueError('The selected result package is incomplete and cannot be exported.')

        with TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            curves_csv_path = tmpdir / f'{base}_curves.csv'
            params_csv_path = tmpdir / f'{base}_parameters.csv'
            summary_csv_path = tmpdir / f'{base}_summary.csv'
            report_path = tmpdir / f'{base}_fit_report.txt'
            session_path = tmpdir / f'{base}_session.json'

            curves_df.to_csv(curves_csv_path, index=False)
            params_df.to_csv(params_csv_path, index=False)
            meta_df.to_csv(summary_csv_path, index=False)
            report_path.write_text(str(entry.get('fit_report', '')), encoding='utf-8')
            session_path.write_text(json.dumps(entry.get('session_state', {}), indent=2), encoding='utf-8')

            derived_csv_path = None
            if derived_df is not None:
                derived_csv_path = tmpdir / f'{base}_derived_quantities.csv'
                derived_df.to_csv(derived_csv_path, index=False)
            confidence_ellipse_paths = self._write_confidence_ellipse_exports(
                tmpdir,
                base,
                entry.get(
                    "confidence_ellipse_payloads",
                    getattr(self, "last_confidence_ellipse_payloads", [])
                ),
            )

            derived_map_csv_path = None
            derived_map_png_path = None

            if (
                derived_uncertainty_map_df is not None
                and not derived_uncertainty_map_df.empty
            ):
                map_export_df = derived_uncertainty_map_df.copy()
                map_export_df.index.name = 'derived_quantity'

                derived_map_csv_path = tmpdir / f'{base}_derived_uncertainty_map.csv'
                map_export_df.to_csv(derived_map_csv_path)

                derived_map_png_path = tmpdir / f'{base}_derived_uncertainty_map.png'
                map_fig = self._build_derived_uncertainty_heatmap_figure(map_export_df)
                map_fig.savefig(
                    derived_map_png_path,
                    dpi=220,
                    bbox_inches='tight',
                )

            correlation_csv_path = None
            correlation_png_path = None
            if correlation_matrix_df is not None and not correlation_matrix_df.empty:
                correlation_export_df = correlation_matrix_df.copy()
                correlation_export_df.index.name = 'parameter'
                correlation_csv_path = tmpdir / f'{base}_parameter_correlation_matrix.csv'
                correlation_export_df.to_csv(correlation_csv_path)

                correlation_png_path = tmpdir / f'{base}_parameter_correlation_heatmap.png'
                corr_fig = self._build_parameter_correlation_heatmap_figure(correlation_export_df)
                corr_fig.savefig(correlation_png_path, dpi=220, bbox_inches='tight')

            residual_csv_path = None
            residual_autocorr_csv_path = None
            residual_qq_csv_path = None
            residual_summary_path = None
            residual_png_path = None
            residual_df = autocorr_df = qq_df = residual_summary_df = None

            if residual_diagnostics_payload is not None:
                if residual_diagnostics_dfs is None:
                    residual_diagnostics_dfs = self._residual_diagnostics_to_dataframes(residual_diagnostics_payload)

                residual_df, autocorr_df, qq_df, residual_summary_df = residual_diagnostics_dfs

                if residual_df is not None and not residual_df.empty:
                    residual_csv_path = tmpdir / f'{base}_residual_diagnostics.csv'
                    residual_df.to_csv(residual_csv_path, index=False)

                if autocorr_df is not None and not autocorr_df.empty:
                    residual_autocorr_csv_path = tmpdir / f'{base}_residual_autocorrelation.csv'
                    autocorr_df.to_csv(residual_autocorr_csv_path, index=False)

                if qq_df is not None and not qq_df.empty:
                    residual_qq_csv_path = tmpdir / f'{base}_residual_qq_plot_data.csv'
                    qq_df.to_csv(residual_qq_csv_path, index=False)

                residual_summary_path = tmpdir / f'{base}_residual_diagnostics_summary.txt'
                residual_summary_path.write_text(
                    self._format_residual_diagnostics_report(residual_diagnostics_payload),
                    encoding='utf-8',
                )

                residual_png_path = tmpdir / f'{base}_residual_diagnostics.png'
                residual_fig = self._build_residual_diagnostics_figure(residual_diagnostics_payload)
                residual_fig.savefig(residual_png_path, dpi=220, bbox_inches='tight')

            #ellipse_summary_csv_path = None
            #if confidence_ellipse_summary_df is not None and not confidence_ellipse_summary_df.empty:
                #ellipse_summary_csv_path = tmpdir / f'{base}_confidence_ellipse_pair_summary.csv'
                #confidence_ellipse_summary_df.to_csv(ellipse_summary_csv_path, index=False)

            excel_path = tmpdir / f'{base}_results.xlsx'
            excel_created = False
            try:
                import openpyxl  # noqa: F401
                with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
                    curves_df.to_excel(writer, sheet_name='curves', index=False)
                    params_df.to_excel(writer, sheet_name='parameters', index=False)
                    meta_df.to_excel(writer, sheet_name='summary', index=False)
                    if derived_df is not None:
                        derived_df.to_excel(writer, sheet_name='derived_quantities', index=False)
                    if (
                        derived_uncertainty_map_df is not None
                        and not derived_uncertainty_map_df.empty
                    ):
                        map_excel_df = derived_uncertainty_map_df.copy()
                        map_excel_df.index.name = 'derived_quantity'
                        map_excel_df.to_excel(
                            writer,
                            sheet_name='derived_uncertainty_map',
                            index=True,
                        )
                    if correlation_matrix_df is not None and not correlation_matrix_df.empty:
                        corr_excel_df = correlation_matrix_df.copy()
                        corr_excel_df.index.name = 'parameter'
                        corr_excel_df.to_excel(
                            writer,
                            sheet_name='parameter_correlations',
                            index=True,
                        )
                    if residual_df is not None and not residual_df.empty:
                        residual_df.to_excel(writer, sheet_name='residuals', index=False)
                    if autocorr_df is not None and not autocorr_df.empty:
                        autocorr_df.to_excel(writer, sheet_name='residual_autocorr', index=False)
                    if qq_df is not None and not qq_df.empty:
                        qq_df.to_excel(writer, sheet_name='residual_qq', index=False)
                    if residual_summary_df is not None and not residual_summary_df.empty:
                        residual_summary_df.to_excel(writer, sheet_name='residual_summary', index=False)
                    if confidence_ellipse_summary_df is not None and not confidence_ellipse_summary_df.empty:
                        confidence_ellipse_summary_df.to_excel(writer, sheet_name='confidence_ellipses', index=False)
                excel_created = True
            except Exception:
                excel_created = False

            original_copy = None
            original_path = str(entry.get('original_file_path', '') or '')
            if original_path:
                source_path = Path(original_path)
                if source_path.exists() and source_path.is_file():
                    original_copy = tmpdir / source_path.name
                    original_copy.write_bytes(source_path.read_bytes())

            with zipfile.ZipFile(target, 'w', compression=zipfile.ZIP_DEFLATED) as zf:
                for path in [curves_csv_path, params_csv_path, summary_csv_path, report_path, session_path]:
                    zf.write(path, path.name)
                if derived_csv_path is not None and derived_csv_path.exists():
                    zf.write(derived_csv_path, derived_csv_path.name)
                for path in confidence_ellipse_paths:
                    if path is not None and path.exists():
                        zf.write(path, path.name)
                if derived_map_csv_path is not None and derived_map_csv_path.exists():
                    zf.write(derived_map_csv_path, derived_map_csv_path.name)
                if derived_map_png_path is not None and derived_map_png_path.exists():
                    zf.write(derived_map_png_path, derived_map_png_path.name)
                for path in [
                    correlation_csv_path,
                    correlation_png_path,
                    residual_csv_path,
                    residual_autocorr_csv_path,
                    residual_qq_csv_path,
                    residual_summary_path,
                    residual_png_path,
                ]:
                    if path is not None and path.exists():
                        zf.write(path, path.name)
                if excel_created and excel_path.exists():
                    zf.write(excel_path, excel_path.name)
                if original_copy is not None:
                    zf.write(original_copy, original_copy.name)


    def _save_selected_result_package_as_zip(self):
        """Ask for a ZIP path and permanently save the selected temporary result package."""
        entry = self._selected_result_package_entry()
        if entry is None:
            messagebox.showinfo('No package selected', 'Select one result package first.')
            return

        target = filedialog.asksaveasfilename(
            title='Save selected result package as ZIP',
            defaultextension='.zip',
            initialfile=self._safe_history_zip_default_name(entry),
            filetypes=[('ZIP archive', '*.zip')],
        )
        if not target:
            return

        try:
            self._write_result_package_zip(entry, target)
        except Exception as exc:
            messagebox.showerror('Save package error', f'Could not save selected package:\n{exc}')
            return

        self.status_var.set(f'Saved selected result package to {target}')
        messagebox.showinfo('Saved', f'Selected result package saved to:\n{target}')


    def _build_reports_tab(self, parent):
        """Create the Reports tab with session-only histories for reports and result packages."""
        reports_box = ttk.LabelFrame(parent, text='Current session report and result history', padding=8)
        reports_box.pack(fill=tk.BOTH, expand=True)

        ttk.Label(
            reports_box,
            text=(
                'Closing a popup does not remove it from this tab. FitED keeps report and result-package '
                'histories only while this software window is open. Load/reset actions do not erase these '
                'histories; use the delete/clear buttons when desired.'
            ),
            wraplength=370,
        ).pack(anchor='w', pady=(0, 8))

        self.reports_notebook = ttk.Notebook(reports_box)
        self.reports_notebook.pack(fill=tk.BOTH, expand=True)

        report_order = ['fit', 'stability', 'stability_best_fit', 'derived', 'correlation', 'residual_diagnostics', 'confidence_ellipse']
        for report_key in report_order:
            doc = self.report_documents[report_key]
            tab = ttk.Frame(self.reports_notebook, padding=6)
            self.reports_notebook.add(tab, text=doc['tab_label'])

            split = ttk.Panedwindow(tab, orient=tk.HORIZONTAL)
            split.pack(fill=tk.BOTH, expand=True)

            left = ttk.Frame(split, padding=(0, 0, 6, 0))
            right = ttk.Frame(split)
            split.add(left, weight=1)
            split.add(right, weight=2)

            ttk.Label(
                left,
                text='History list',
                font=('Segoe UI', 9, 'bold'),
            ).pack(anchor='w', pady=(0, 4))

            list_frame = ttk.Frame(left)
            list_frame.pack(fill=tk.BOTH, expand=True)

            listbox = tk.Listbox(
                list_frame,
                width=34,
                height=28,
                exportselection=False,
            )
            listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
            list_scroll = ttk.Scrollbar(list_frame, orient='vertical', command=listbox.yview)
            list_scroll.pack(side=tk.RIGHT, fill=tk.Y)
            listbox.configure(yscrollcommand=list_scroll.set)
            listbox.bind('<<ListboxSelect>>', lambda event, key=report_key: self._select_report_history_entry(key, event))
            self.report_history_listboxes[report_key] = listbox

            left_btns = ttk.Frame(left)
            left_btns.pack(fill=tk.X, pady=(8, 0))
            ttk.Button(
                left_btns,
                text='Delete selected',
                command=lambda key=report_key: self._delete_selected_report_history(key),
            ).pack(fill=tk.X, pady=(0, 4))
            ttk.Button(
                left_btns,
                text='Clear history',
                command=lambda key=report_key: self._clear_report_history(key),
            ).pack(fill=tk.X)

            title_label = ttk.Label(
                right,
                text=doc.get('title', self._default_report_title(report_key)),
                font=('Segoe UI', 9, 'bold'),
                wraplength=350,
            )
            title_label.pack(anchor='w', pady=(0, 6))
            self.report_title_labels[report_key] = title_label

            text_frame = ttk.Frame(right)
            text_frame.pack(fill=tk.BOTH, expand=True)

            report_text = tk.Text(
                text_frame,
                wrap='none',
                width=52,
                height=30,
            )
            report_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

            y_scroll = ttk.Scrollbar(text_frame, orient='vertical', command=report_text.yview)
            y_scroll.pack(side=tk.RIGHT, fill=tk.Y)

            x_scroll = ttk.Scrollbar(right, orient='horizontal', command=report_text.xview)
            x_scroll.pack(fill=tk.X, pady=(4, 0))

            report_text.configure(
                yscrollcommand=y_scroll.set,
                xscrollcommand=x_scroll.set,
            )
            self.report_text_widgets[report_key] = report_text

            btns = ttk.Frame(right)
            btns.pack(fill=tk.X, pady=(8, 0))
            ttk.Button(
                btns,
                text='Copy selected report',
                command=lambda key=report_key: self._copy_report_document(key),
            ).pack(side=tk.LEFT)

        package_tab = ttk.Frame(self.reports_notebook, padding=6)
        self.reports_notebook.add(package_tab, text='Result packages')

        package_split = ttk.Panedwindow(package_tab, orient=tk.HORIZONTAL)
        package_split.pack(fill=tk.BOTH, expand=True)

        package_left = ttk.Frame(package_split, padding=(0, 0, 6, 0))
        package_right = ttk.Frame(package_split)
        package_split.add(package_left, weight=1)
        package_split.add(package_right, weight=2)

        ttk.Label(
            package_left,
            text='Temporary package history',
            font=('Segoe UI', 9, 'bold'),
        ).pack(anchor='w', pady=(0, 4))

        package_list_frame = ttk.Frame(package_left)
        package_list_frame.pack(fill=tk.BOTH, expand=True)
        self.result_package_listbox = tk.Listbox(
            package_list_frame,
            width=34,
            height=28,
            exportselection=False,
        )
        self.result_package_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        package_list_scroll = ttk.Scrollbar(
            package_list_frame,
            orient='vertical',
            command=self.result_package_listbox.yview,
        )
        package_list_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.result_package_listbox.configure(yscrollcommand=package_list_scroll.set)
        self.result_package_listbox.bind('<<ListboxSelect>>', self._refresh_result_package_details_view)

        package_left_btns = ttk.Frame(package_left)
        package_left_btns.pack(fill=tk.X, pady=(8, 0))
        ttk.Button(
            package_left_btns,
            text='Save selected as ZIP',
            command=self._save_selected_result_package_as_zip,
        ).pack(fill=tk.X, pady=(0, 4))
        ttk.Button(
            package_left_btns,
            text='Delete selected',
            command=self._delete_selected_result_package,
        ).pack(fill=tk.X, pady=(0, 4))
        ttk.Button(
            package_left_btns,
            text='Clear package history',
            command=self._clear_result_package_history,
        ).pack(fill=tk.X)

        ttk.Label(
            package_right,
            text='Selected result-package summary',
            font=('Segoe UI', 9, 'bold'),
        ).pack(anchor='w', pady=(0, 6))

        package_text_frame = ttk.Frame(package_right)
        package_text_frame.pack(fill=tk.BOTH, expand=True)

        self.result_package_details_text = tk.Text(
            package_text_frame,
            wrap='none',
            width=52,
            height=30,
        )
        self.result_package_details_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        package_y_scroll = ttk.Scrollbar(
            package_text_frame,
            orient='vertical',
            command=self.result_package_details_text.yview,
        )
        package_y_scroll.pack(side=tk.RIGHT, fill=tk.Y)

        package_x_scroll = ttk.Scrollbar(
            package_right,
            orient='horizontal',
            command=self.result_package_details_text.xview,
        )
        package_x_scroll.pack(fill=tk.X, pady=(4, 0))

        self.result_package_details_text.configure(
            yscrollcommand=package_y_scroll.set,
            xscrollcommand=package_x_scroll.set,
        )

        self._refresh_reports_tab()

    def _show_stability_test_report_dialog(self, payload, history_id=None):
        """Display the text stability report in a scrollable popup."""
        report = self._format_stability_test_report(payload)
        self._set_report_document(
            'stability',
            report,
            title='Fit stability test report',
            history_id=history_id,
        )

        win = tk.Toplevel(self.root)
        win.title('Fit stability test report')
        win.geometry('1120x760')
        win.minsize(860, 560)

        frame = ttk.Frame(win, padding=10)
        frame.pack(fill=tk.BOTH, expand=True)

        ttk.Label(
            frame,
            text='Fit stability test report',
            font=('Segoe UI', 11, 'bold')
        ).pack(anchor='w', pady=(0, 8))

        text_frame = ttk.Frame(frame)
        text_frame.pack(fill=tk.BOTH, expand=True)

        report_text = tk.Text(text_frame, wrap='none', height=34, width=130)
        report_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        y_scroll = ttk.Scrollbar(text_frame, orient='vertical', command=report_text.yview)
        y_scroll.pack(side=tk.RIGHT, fill=tk.Y)

        x_scroll = ttk.Scrollbar(frame, orient='horizontal', command=report_text.xview)
        x_scroll.pack(fill=tk.X)

        report_text.configure(
            yscrollcommand=y_scroll.set,
            xscrollcommand=x_scroll.set
        )
        report_text.insert('1.0', report)
        report_text.configure(state='disabled')

        btns = ttk.Frame(frame)
        btns.pack(fill=tk.X, pady=(10, 0))

        def copy_report():
            self.root.clipboard_clear()
            self.root.clipboard_append(report)
            self.status_var.set('Fit stability test report copied to clipboard.')

        ttk.Button(btns, text='Copy report', command=copy_report).pack(side=tk.LEFT)
        ttk.Button(btns, text='Close', command=win.destroy).pack(side=tk.RIGHT)

        win.transient(self.root)
        win.lift()
        win.focus_force()


    def _display_stability_test_payload(self, payload):
        """Display the best repeated stability-test result and open its report."""
        best_record = payload['best_record']
        context = best_record['context']
        result = best_record['result']
        comps = best_record['components']
        best_fit = best_record['best_fit']
        x = context['x']
        y_raw = context['y_raw']

        if payload.get('protocol') == 'Repeat Auto pre-fit':
            self._apply_peak_defs_to_ui(best_record['peak_defs'], fit_result=result)

        self.fit_result = result
        self.last_derived_quantities = None
        self.last_derived_uncertainty_map = None
        self.last_components = comps
        self.last_best_fit = best_fit
        self.last_roi = (x.copy(), y_raw.copy())
        self.last_stability_test_payload = payload

        if payload.get('protocol') == 'Repeat Run fit':
            self.last_fit_peak_count = context.get('active_count', self.last_fit_peak_count)

        self.ax_main.clear()
        self.ax_resid.clear()
        self._refresh_hover_axis()
        self.ax_main.plot(x, y_raw, 'k.', ms=3, alpha=0.6, label='Data')
        self.ax_main.plot(x, best_fit, linewidth=2.1, label='Best stability-test solution')
        for name, comp in comps.items():
            self.ax_main.plot(x, comp, '--', linewidth=1.0, alpha=0.85, label=name)
        self.ax_main.set_title('Fit stability test: best repeated solution')
        self.ax_main.set_ylabel('Y')
        self.ax_main.legend(fontsize=8, ncol=2)

        resid = y_raw - best_fit
        self.ax_resid.axhline(0.0, linestyle='--', linewidth=1.0)
        self.ax_resid.plot(x, resid, linewidth=1.0)
        self.ax_resid.set_xlabel('X')
        self.ax_resid.set_ylabel('Residual')
        self.fig.tight_layout()
        self.canvas.draw_idle()

        criterion = payload.get('criterion', self._current_fit_criterion())
        self.status_var.set(
            f"Fit stability test complete. Best {criterion}: {payload.get('best_score', np.nan):.6g}; "
            f"near-best repeated solutions: {len(payload.get('near_best_records', []))}/"
            f"{payload.get('successful_repeats', 0)}."
        )

        history_id = self._new_session_history_id()
        self._capture_result_package_history(
            result,
            'Stability best-fit result',
            history_id=history_id,
        )
        self._show_stability_test_report_dialog(payload, history_id=history_id)
        self._show_fit_report_dialog(
            result,
            title='Fit stability test: best repeated solution report',
            report_key='stability_best_fit',
            history_id=history_id,
        )


    def _clear_fit_and_pick_state(self):
        self.fit_result = None
        self.last_components = None
        self.last_best_fit = None
        self.last_roi = None
        self.last_session_path = None
        self.last_fit_peak_count = 0
        self.last_derived_quantities = None
        self.last_derived_uncertainty_map = None
        self.last_stability_test_payload = None
        self.active_result_package_id = None
    
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
        return default_peak_state_for_range(idx, count, x_min=x_min, x_max=x_max)

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

            ttk.Label(row0, text='L-fraction (0=G, 1=L)').pack(side='left', padx=(0, 4))
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
            ttk.Entry(row1, textvariable=cmax_var, width=10).pack(side='left', padx=(0, 150))
            ttk.Label(row1, text='Sigma').pack(side='left', padx=(0, 4))
            ttk.Entry(row1, textvariable=sigma_var, width=10).pack(side='left', padx=(0, 0))
            

            row2 = ttk.Frame(frm)
            row2.pack(fill=tk.X, anchor='w', pady=(4, 0))
            ttk.Label(row2, text='Area').pack(side='left', padx=(0, 4))
            ttk.Entry(row2, textvariable=amp_var, width=10).pack(side='left', padx=(0, 14))
            ttk.Label(row2, text='a min').pack(side='left', padx=(0, 4))
            ttk.Entry(row2, textvariable=amin_var, width=10).pack(side='left', padx=(0, 14))
            ttk.Label(row2, text='a max').pack(side='left', padx=(0, 4))
            ttk.Entry(row2, textvariable=amax_var, width=10).pack(side='left', padx=(0, 152))
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

            #def rebuild_custom_fields(*_):
            def rebuild_custom_fields(
                *_,
                custom_frame=custom_frame,
                custom_param_vars=custom_param_vars,
                custom_profile_var=custom_profile_var,
                kind_var=kind_var,
                custom_combo=custom_combo,
                state=state,
            ):
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
    
        # Keep manually entered ROI values unchanged during reset.
        # self.roi_min_var.set(float(np.min(self.x_full)))
        # self.roi_max_var.set(float(np.max(self.x_full)))
    
        #self._clear_fit_and_pick_state()
        #self.rebuild_peaks_fresh()
        #self._plot_loaded_data()
    
        #self.status_var.set('Peaks, picked centers, and fit state reset for current spectrum. ROI kept unchanged.')
            
    def _x_units_to_samples(self, x, value):
        """Delegate x-unit to sample-count conversion to the backend."""
        return x_units_to_samples(x, value)
    
    
    def _negative_peak_seed_from_centers(self, x, y_raw, peak_defs):
        """Delegate negative-peak amplitude seeding to the backend."""
        return negative_peak_seed_from_centers(x, y_raw, peak_defs)
    
    
    def _display_found_peak_centers(self, x, y_raw, y_detection_display, centers, direction, used_smoothing):
        """Show detected peak centers on the main plot without running any fit."""
        self.ax_main.clear()
        self.ax_resid.clear()
        self._refresh_hover_axis()
    
        self.ax_main.plot(x, y_raw, 'k.', ms=3, alpha=0.6, label='Raw data')
    
        if used_smoothing:
            self.ax_main.plot(
                x,
                y_detection_display,
                linewidth=1.2,
                alpha=0.9,
                label='Detection signal'
            )
    
        for i, center in enumerate(centers, start=1):
            self.ax_main.axvline(center, linestyle='--', linewidth=1.0, alpha=0.85)
            try:
                ymax = float(np.nanmax(y_raw))
                ymin = float(np.nanmin(y_raw))
                ypos = ymin + 0.92 * (ymax - ymin)
                self.ax_main.text(center, ypos, str(i), rotation=90, va='top', ha='right', fontsize=8)
            except Exception:
                pass
    
        self.ax_main.set_title(f'Found {len(centers)} candidate peaks ({direction})')
        self.ax_main.set_ylabel('Y')
        self.ax_main.legend(fontsize=8)
    
        self.ax_resid.axis('off')
        self.ax_resid.text(
            0.01,
            0.5,
            'Find peaks only proposes initial centers. Inspect/edit the peak table, then click Auto pre-fit or Run fit.',
            transform=self.ax_resid.transAxes,
            va='center',
            ha='left',
        )
    
        self.fig.tight_layout()
        self.canvas.draw_idle()
    
    
    def find_peaks_auto_populate(self):
        """
        Automatically detect candidate peak centers and populate the peak table.

        This GUI handler preserves the existing interaction flow while the
        detection/seeding logic now lives in the backend.
        """
        if self.x_full is None or self.y_full is None:
            messagebox.showinfo('No data', 'Load a file first.')
            return

        if self.peak_rows:
            proceed = messagebox.askyesno(
                'Find peaks',
                'Find peaks will replace the current peak table with detected candidate peaks.\n\n'
                'This does not delete your data and does not run a fit.\n\n'
                'Continue?'
            )
            if not proceed:
                return

        try:
            x, y_raw, _ = self._get_roi_data()

            use_smooth = bool(self.find_peaks_use_smooth_var.get())
            y_detection_display = smooth_if_requested(
                y_raw,
                window=int(self.smooth_window_var.get()),
                polyorder=int(self.smooth_poly_var.get()),
                enabled=use_smooth,
            )

            detection = detect_peaks_auto(
                x,
                y_raw,
                y_detection_display,
                direction=self.find_peaks_direction_var.get(),
                prominence_pct=self.find_peaks_prominence_pct_var.get(),
                min_distance_x=self.find_peaks_min_distance_var.get(),
                min_width_x=self.find_peaks_min_width_var.get(),
                max_peaks=self.find_peaks_max_var.get(),
                custom_profiles=self.custom_profiles,
                default_x_min=(float(np.min(self.x_full)) if self.x_full is not None else None),
                default_x_max=(float(np.max(self.x_full)) if self.x_full is not None else None),
            )

            centers = detection['centers']
            direction = detection['direction']
            seeded_defs = detection['seeded_defs']

            if not centers:
                messagebox.showinfo(
                    'No peaks found',
                    'No peaks were detected with the current settings.\n\n'
                    'Try lowering Prominence %, lowering Min distance/width, changing direction, or adjusting ROI.'
                )
                self.status_var.set('Find peaks: no peaks detected.')
                return

            n_found = len(centers)

            self._clear_fit_and_pick_state()
            self.peak_count_var.set(n_found)
            self._draw_peak_rows(seeded_defs)

            self._display_found_peak_centers(
                x=x,
                y_raw=y_raw,
                y_detection_display=y_detection_display,
                centers=centers,
                direction=direction,
                used_smoothing=use_smooth,
            )

            self.status_var.set(
                f'Find peaks detected {n_found} candidate peak(s). '
                f'Inspect/edit the table, then click Auto pre-fit or Run fit.'
            )

        except Exception as exc:
            messagebox.showerror('Find peaks error', str(exc))
            self.status_var.set('Find peaks failed.')
            
    def _residual_noise_sigma(self, residual):
        """Delegate robust residual-noise estimation to the backend."""
        return residual_noise_sigma(residual)
    
    
    def _residual_suggestion_threshold(self, residual, y_raw):
        """Delegate residual suggestion threshold selection to the backend."""
        return residual_suggestion_threshold(
            residual,
            y_raw,
            sensitivity=self.residual_suggest_sensitivity_var.get(),
        )
    
    
    def _residual_default_distance_samples(self, x):
        """
        Candidate separation for residual peak detection.
    
        This reuses the Find-peaks min-distance control if available.
        """
        x = np.asarray(x, dtype=float)
    
        try:
            distance_samples = self._x_units_to_samples(x, self.find_peaks_min_distance_var.get())
        except Exception:
            distance_samples = None
    
        if distance_samples is None:
            distance_samples = max(1, int(len(x) / 200))
    
        return max(1, int(distance_samples))
    
    
    def _residual_default_width_samples(self, x):
        """
        Candidate width filter for residual peak detection.
    
        This reuses the Find-peaks min-width control if available.
        """
        try:
            return self._x_units_to_samples(x, self.find_peaks_min_width_var.get())
        except Exception:
            return None
    
    
    def _current_active_peak_centers(self):
        """Return current active peak centers from the GUI table."""
        centers = []
        for row in self.peak_rows:
            try:
                if row['active'].get():
                    centers.append(float(row['center'].get()))
            except Exception:
                continue
        return centers
    
    
    def _detect_missing_peak_candidates_from_residual(self):
        """
        Detect candidate missing components from the current fit residual.

        The data extraction remains in the GUI layer; the candidate-search logic
        is executed by the backend.
        """
        if self.fit_result is None or self.last_roi is None or self.last_best_fit is None:
            raise ValueError('Run fit, Auto pre-fit, or Refine before suggesting missing peaks from residual.')

        x, y_raw = self.last_roi
        existing_centers = self._current_active_peak_centers()

        return detect_residual_peak_candidates(
            x,
            y_raw,
            self.last_best_fit,
            smooth_enabled=bool(self.residual_suggest_use_smooth_var.get()),
            smooth_window=int(self.smooth_window_var.get()),
            smooth_poly=int(self.smooth_poly_var.get()),
            sensitivity=self.residual_suggest_sensitivity_var.get(),
            direction=self.residual_suggest_direction_var.get(),
            max_suggestions=self.residual_suggest_max_var.get(),
            min_distance_x=self.find_peaks_min_distance_var.get(),
            min_width_x=self.find_peaks_min_width_var.get(),
            existing_centers=existing_centers,
        )
    
    
    def _preview_residual_candidates(self, candidates, residual, residual_for_detection):
        """Draw candidate missing components on the current fit/residual plot."""
        if self.last_roi is None or self.last_best_fit is None:
            return
    
        x, y_raw = self.last_roi
        x = np.asarray(x, dtype=float)
        y_raw = np.asarray(y_raw, dtype=float)
        best = np.asarray(self.last_best_fit, dtype=float)
    
        self.ax_main.clear()
        self.ax_resid.clear()
        self._refresh_hover_axis()
    
        self.ax_main.plot(x, y_raw, 'k.', ms=3, alpha=0.6, label='Data')
        self.ax_main.plot(x, best, linewidth=2.1, label='Current fit')
    
        if self.last_components:
            for name, comp in self.last_components.items():
                self.ax_main.plot(x, comp, '--', linewidth=1.0, alpha=0.85, label=name)
    
        for i, cand in enumerate(candidates, start=1):
            center = cand['center']
            self.ax_main.axvline(center, linestyle=':', linewidth=1.3, alpha=0.95)
            self.ax_resid.axvline(center, linestyle=':', linewidth=1.3, alpha=0.95)
    
            try:
                y_top = float(np.nanmin(y_raw) + 0.92 * (np.nanmax(y_raw) - np.nanmin(y_raw)))
                self.ax_main.text(
                    center,
                    y_top,
                    f'R{i}',
                    rotation=90,
                    va='top',
                    ha='right',
                    fontsize=8
                )
            except Exception:
                pass
    
        self.ax_main.set_title('Suggested missing components from residual')
        self.ax_main.set_ylabel('Y')
        self.ax_main.legend(fontsize=8, ncol=2)
    
        self.ax_resid.axhline(0.0, linestyle='--', linewidth=1.0)
        self.ax_resid.plot(x, residual, linewidth=1.0, label='Residual')
    
        if bool(self.residual_suggest_use_smooth_var.get()):
            self.ax_resid.plot(
                x,
                residual_for_detection,
                linewidth=1.1,
                alpha=0.85,
                label='Detection residual'
            )
    
        self.ax_resid.set_xlabel('X')
        self.ax_resid.set_ylabel('Residual')
        self.ax_resid.legend(fontsize=8)
    
        self.fig.tight_layout()
        self.canvas.draw_idle()
    
    
    def _make_residual_candidate_peak_defs(self, candidates):
        """
        Convert selected residual candidates into new peak-row states.

        Existing rows stay owned by the GUI table; the construction/seeding of
        the new states is delegated to the backend.
        """
        if self.last_roi is None:
            raise ValueError('No stored ROI is available.')

        x, y_raw = self.last_roi
        current_states = [self._row_state_from_row(row) for row in self.peak_rows]
        return residual_candidate_peak_defs(
            x,
            y_raw,
            current_states,
            candidates,
            custom_profiles=self.custom_profiles,
            default_x_min=(float(np.min(self.x_full)) if self.x_full is not None else None),
            default_x_max=(float(np.max(self.x_full)) if self.x_full is not None else None),
        )
    
    
    def _add_residual_candidates_to_peak_table(self, candidates):
        """
        Append selected candidate peaks to the table.
    
        Important:
        keep self.fit_result and self.last_fit_peak_count because Refine with added
        peaks needs the previous fit as the Stage-1 reference.
        """
        if not candidates:
            return
    
        current_states, new_states = self._make_residual_candidate_peak_defs(candidates)
    
        active_before = sum(1 for state in current_states if bool(state.get('active', True)))
    
        # Refine with added peaks uses last_fit_peak_count to know which peaks are old.
        # Set it to the number of active peaks before adding the residual suggestions.
        self.last_fit_peak_count = active_before
    
        combined_states = current_states + new_states
        self.peak_count_var.set(len(combined_states))
        self._draw_peak_rows(combined_states)
    
        self.status_var.set(
            f'Added {len(new_states)} suggested component(s). '
            f'Click Refine with added peaks, then inspect residuals again.'
        )
    
    
    def _show_residual_suggestion_dialog(self, candidates):
        """Show candidate residual peaks and let the user choose which ones to add."""
        win = tk.Toplevel(self.root)
        win.title('Residual missing-peak suggestions')
        win.geometry('720x420')
        win.minsize(600, 320)
    
        frame = ttk.Frame(win, padding=10)
        frame.pack(fill=tk.BOTH, expand=True)
    
        ttk.Label(
            frame,
            text='Select candidate missing components to append to the peak table.',
            font=('Segoe UI', 10, 'bold')
        ).pack(anchor='w', pady=(0, 6))
    
        ttk.Label(
            frame,
            text=(
                'These are suggestions from structured residuals, not confirmed physical peaks. '
                'After adding, use Refine with added peaks.'
            ),
            wraplength=660
        ).pack(anchor='w', pady=(0, 10))
    
        table = ttk.Frame(frame)
        table.pack(fill=tk.BOTH, expand=True)
    
        headers = ['Use', '#', 'Center', 'Direction', 'Prominence', 'Residual value']
        for col, header in enumerate(headers):
            ttk.Label(table, text=header, font=('Segoe UI', 9, 'bold')).grid(
                row=0,
                column=col,
                sticky='w',
                padx=4,
                pady=(0, 4)
            )
    
        selected_vars = []
    
        for row_idx, cand in enumerate(candidates, start=1):
            var = tk.BooleanVar(value=True)
            selected_vars.append(var)
    
            ttk.Checkbutton(table, variable=var).grid(row=row_idx, column=0, sticky='w', padx=4)
            ttk.Label(table, text=str(row_idx)).grid(row=row_idx, column=1, sticky='w', padx=4)
            ttk.Label(table, text=f"{cand['center']:.6g}").grid(row=row_idx, column=2, sticky='w', padx=4)
            ttk.Label(table, text=cand['direction']).grid(row=row_idx, column=3, sticky='w', padx=4)
            ttk.Label(table, text=f"{cand['prominence']:.6g}").grid(row=row_idx, column=4, sticky='w', padx=4)
            ttk.Label(table, text=f"{cand['residual_value']:.6g}").grid(row=row_idx, column=5, sticky='w', padx=4)
    
        btns = ttk.Frame(frame)
        btns.pack(fill=tk.X, pady=(12, 0))
    
        def add_selected():
            selected = [
                cand for cand, var in zip(candidates, selected_vars)
                if bool(var.get())
            ]
            if not selected:
                messagebox.showinfo('No selection', 'Select at least one candidate to add.', parent=win)
                return
    
            self._add_residual_candidates_to_peak_table(selected)
            win.destroy()
    
        def select_all():
            for var in selected_vars:
                var.set(True)
    
        def select_none():
            for var in selected_vars:
                var.set(False)
    
        ttk.Button(btns, text='Select all', command=select_all).pack(side=tk.LEFT)
        ttk.Button(btns, text='Select none', command=select_none).pack(side=tk.LEFT, padx=(6, 0))
        ttk.Button(btns, text='Cancel', command=win.destroy).pack(side=tk.RIGHT)
        ttk.Button(btns, text='Add selected', command=add_selected).pack(side=tk.RIGHT, padx=(0, 8))
    
        win.transient(self.root)
        win.lift()
        win.focus_force()
    
    
    def suggest_missing_peaks_from_residual(self):
        """
        Suggest missing components from structured residuals.
    
        This is a post-fit diagnostic helper:
        residual = data - current fit
        Then scipy.find_peaks is applied to the residual signal.
        """
        try:
            candidates, residual, residual_for_detection = self._detect_missing_peak_candidates_from_residual()
        except Exception as exc:
            messagebox.showerror('Residual suggestion error', str(exc))
            return
    
        if not candidates:
            messagebox.showinfo(
                'No residual suggestions',
                'No clear missing components were found with the current residual settings.\n\n'
                'Try Aggressive sensitivity, changing direction, or running a better first fit.'
            )
            self.status_var.set('Residual suggestion: no candidates found.')
            return
    
        self._preview_residual_candidates(candidates, residual, residual_for_detection)
        self._show_residual_suggestion_dialog(candidates)
    
        self.status_var.set(
            f'Residual suggestion found {len(candidates)} candidate component(s). '
            f'Review the popup and choose which to add.'
        )
    
    def open_parameter_constraints_dialog(self):
        """
        Open a dialog for general lmfit-style parameter constraints.
        """
        win = tk.Toplevel(self.root)
        win.title("Parameter constraints")
        win.geometry("1050x720")
        win.minsize(850, 560)
    
        frame = ttk.Frame(win, padding=10)
        frame.pack(fill=tk.BOTH, expand=True)
    
        ttk.Label(
            frame,
            text="Parameter constraints",
            font=("Segoe UI", 11, "bold"),
        ).pack(anchor="w", pady=(0, 6))
    
        ttk.Label(
            frame,
            text=(
                "Use this to impose algebraic relationships between fit parameters during fitting. "
                "Examples: p2_center = p1_center + delta, p2_sigma = p1_sigma, "
                "p2_amplitude = p1_amplitude * ratio."
            ),
            wraplength=980,
        ).pack(anchor="w", pady=(0, 8))
    
        main = ttk.Panedwindow(frame, orient=tk.HORIZONTAL)
        main.pack(fill=tk.BOTH, expand=True)
    
        left = ttk.Frame(main, padding=(0, 0, 8, 0))
        right = ttk.Frame(main)
        main.add(left, weight=1)
        main.add(right, weight=2)
    
        ttk.Label(left, text="Available current model parameters").pack(anchor="w")
    
        param_text = tk.Text(left, wrap="none", width=38, height=30)
        param_text.pack(fill=tk.BOTH, expand=True, pady=(4, 0))
    
        try:
            if self.x_full is not None and self.y_full is not None:
                x, y_raw, _ = self._get_roi_data()
            else:
                x = np.asarray([0.0, 1.0], dtype=float)
                y_raw = np.asarray([0.0, 0.0], dtype=float)
    
            peak_defs = self._collect_peak_defs(require_active_peak=False)
    
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
    
            for name in params.keys():
                param_text.insert(tk.END, f"{name}\n")
    
        except Exception as exc:
            param_text.insert(
                tk.END,
                f"Could not build current parameter list:\n{exc}\n"
            )
    
        param_text.configure(state="disabled")
    
        ttk.Label(right, text="Auxiliary free/fixed parameters").pack(anchor="w")
        ttk.Label(
            right,
            text="Format: name, value, min, max, vary/fixed",
            foreground="#555",
        ).pack(anchor="w")
    
        aux_text = tk.Text(right, wrap="none", height=8)
        aux_text.pack(fill=tk.X, pady=(4, 8))
    
        if self.auxiliary_parameter_definitions:
            aux_lines = []
            for item in self.auxiliary_parameter_definitions:
                flag = "vary" if bool(item.get("vary", True)) else "fixed"
                aux_lines.append(
                    f"{item.get('name', '')}, {item.get('value', 0.0)}, "
                    f"{item.get('min', '-inf')}, {item.get('max', 'inf')}, {flag}"
                )
            aux_text.insert("1.0", "\n".join(aux_lines))
        else:
            aux_text.insert(
                "1.0",
                "# Example:\n# delta, 0.026, 0, 0.1, vary\n# ratio, 1.0, 0, inf, vary\n"
            )
    
        ttk.Label(right, text="Constraint expressions").pack(anchor="w")
        ttk.Label(
            right,
            text="Format: target_parameter = expression",
            foreground="#555",
        ).pack(anchor="w")
    
        con_text = tk.Text(right, wrap="none", height=12)
        con_text.pack(fill=tk.BOTH, expand=True, pady=(4, 0))
    
        if self.parameter_constraint_definitions:
            con_lines = [
                f"{item.get('target', '')} = {item.get('expression', '')}"
                for item in self.parameter_constraint_definitions
            ]
            con_text.insert("1.0", "\n".join(con_lines))
        else:
            con_text.insert(
                "1.0",
                "# Examples:\n# p2_center = p1_center + delta\n# p2_sigma = p1_sigma\n"
            )
    
        def save_constraints():
            try:
                aux_defs = self._parse_auxiliary_parameter_lines(
                    aux_text.get("1.0", tk.END)
                )
                con_defs = self._parse_parameter_constraint_lines(
                    con_text.get("1.0", tk.END)
                )
    
                # Validate by building the current model and applying constraints once.
                if self.x_full is not None and self.y_full is not None:
                    x, y_raw, _ = self._get_roi_data()
                else:
                    x = np.asarray([0.0, 1.0], dtype=float)
                    y_raw = np.asarray([0.0, 0.0], dtype=float)
    
                peak_defs = self._collect_peak_defs(require_active_peak=False)
    
                build_composite_model(
                    peak_defs,
                    background_kind=self.background_var.get(),
                    poly_order=int(self.poly_order_var.get()),
                    x=x,
                    y=y_raw,
                    custom_profiles=self.custom_profiles,
                    custom_background_profiles=self.custom_background_profiles,
                    custom_background_profile_name=self.background_profile_var.get(),
                    background_params=self._collect_background_params(),
                    auxiliary_parameters=aux_defs,
                    parameter_constraints=con_defs,
                )
    
            except Exception as exc:
                messagebox.showerror(
                    "Parameter constraint error",
                    str(exc),
                    parent=win,
                )
                return
    
            self.auxiliary_parameter_definitions = aux_defs
            self.parameter_constraint_definitions = con_defs
    
            self.status_var.set(
                f"Saved {len(aux_defs)} auxiliary parameter(s) and "
                f"{len(con_defs)} parameter constraint(s)."
            )
            messagebox.showinfo(
                "Saved",
                "Parameter constraints saved for the current FitED session.",
                parent=win,
            )
    
        def clear_constraints():
            aux_text.delete("1.0", tk.END)
            con_text.delete("1.0", tk.END)
    
        btns = ttk.Frame(frame)
        btns.pack(fill=tk.X, pady=(10, 0))
    
        ttk.Button(btns, text="Save / validate", command=save_constraints).pack(side=tk.LEFT)
        ttk.Button(btns, text="Clear text", command=clear_constraints).pack(side=tk.LEFT, padx=(6, 0))
        ttk.Button(btns, text="Close", command=win.destroy).pack(side=tk.RIGHT)
    
        win.transient(self.root)
        win.lift()
        win.focus_force()
    
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
            auxiliary_parameters=self.auxiliary_parameter_definitions,
            parameter_constraints=self.parameter_constraint_definitions,
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
        weights = self._weights(y_raw)
        
        if weights is None:
            preview_metric = float(np.nansum(resid ** 2))
            preview_metric_label = "RSS"
        else:
            preview_metric = float(np.nansum((resid * weights) ** 2))
            preview_metric_label = f"weighted chi-square ({self.weighting_var.get()})"

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
                f"preview {preview_metric_label}={preview_metric:.6g}. Click Run fit to finalize."
            )
        else:
            self.status_var.set(
                f'Peak drag preview updated. Preview {preview_metric_label}={preview_metric:.6g}.'
            )

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
        # Trial-generation strategy used before fitting in Auto pre-fit.
        # The default preserves the original fast jitter behavior.
        self.autoprefit_sampling_var = tk.StringVar(value=DEFAULT_AUTO_PREFIT_SAMPLING_MODE)
        # Maximum number of objective-function evaluations for each lmfit call.
        # This prevents a bad/messy fit from running for too long.
        self.max_nfev_var = tk.IntVar(value=10000)
        # Criterion used to choose the best trial during auto pre-fit/refinement.
        # Lower is better for AIC, BIC, chi-square, and reduced chi-square.
        self.fit_criterion_var = tk.StringVar(value='AIC')
        # Optimizer mode used by Run fit, Auto pre-fit, and refinement.
        # Default stays LM-only so easy spectra keep the original fast behavior.
        # Robust mode runs LM and DE+LM as separate candidates and keeps the
        # result with the lower selected criterion.
        self.optimizer_mode_var = tk.StringVar(value=FIT_OPTIMIZER_MODES[0])
        # Optional seed for reproducible stochastic fitting/search behavior.
        # Blank keeps stochastic exploration behavior.
        self.random_seed_var = tk.StringVar(value='')
        # Defaults for the repeated numerical stability diagnostic.
        self.stability_protocol_var = tk.StringVar(value=STABILITY_TEST_PROTOCOLS[0])
        self.stability_repeats_var = tk.IntVar(value=20)
        self.stability_delta_var = tk.DoubleVar(value=10.0)
        # Batch fitting uses the current session/model as a template.
        self.batch_fit_mode_var = tk.StringVar(value=BATCH_FIT_MODES[0])
        # Optional automatic peak detection settings.
        # These do not replace manual peak picking; they only auto-populate candidate centers.
        self.find_peaks_max_var = tk.IntVar(value=10)
        self.find_peaks_prominence_pct_var = tk.DoubleVar(value=5.0)
        self.find_peaks_min_distance_var = tk.DoubleVar(value=0.0)
        self.find_peaks_min_width_var = tk.DoubleVar(value=0.0)
        self.find_peaks_direction_var = tk.StringVar(value="positive")
        self.find_peaks_use_smooth_var = tk.BooleanVar(value=True)
        # Optional residual-based missing-component suggestion.
        # This does not replace Find peaks or Pick centers from plot.
        # It is used after a fit to suggest extra components from structured residuals.
        self.residual_suggest_max_var = tk.IntVar(value=1)
        self.residual_suggest_sensitivity_var = tk.StringVar(value="Normal")
        self.residual_suggest_direction_var = tk.StringVar(value="positive")
        self.residual_suggest_use_smooth_var = tk.BooleanVar(value=True)

    def _build_ui(self):
        outer = ttk.Panedwindow(self.root, orient=tk.HORIZONTAL)
        outer.pack(fill=tk.BOTH, expand=True)

        # The left workflow area is now organized as notebook tabs instead of
        # one very tall control stack. This only rearranges layout; the existing
        # fitting logic, variables, and button commands are unchanged.
        controls_panel = ttk.Frame(outer)
        plot_frame = ttk.Frame(outer, padding=10)
        outer.add(controls_panel, weight=0)
        outer.add(plot_frame, weight=1)
        self.plot_frame = plot_frame

        self.workflow_notebook = ttk.Notebook(controls_panel)
        self.workflow_notebook.pack(fill=tk.BOTH, expand=True)

        load_tab_wrap = ScrollableFrame(self.workflow_notebook)
        fit_tab_wrap = ScrollableFrame(self.workflow_notebook)
        peaks_tab = ttk.Frame(self.workflow_notebook, padding=10)
        actions_tab_wrap = ScrollableFrame(self.workflow_notebook)
        reports_tab = ttk.Frame(self.workflow_notebook, padding=10)

        self.workflow_notebook.add(load_tab_wrap, text='Load data')
        self.workflow_notebook.add(fit_tab_wrap, text='Fit settings')
        self.workflow_notebook.add(peaks_tab, text='Peaks')
        self.workflow_notebook.add(actions_tab_wrap, text='Actions')
        self.workflow_notebook.add(reports_tab, text='Reports')

        load_controls = load_tab_wrap.inner
        fit_controls = fit_tab_wrap.inner
        actions_controls = actions_tab_wrap.inner
        load_controls.configure(padding=10)
        fit_controls.configure(padding=10)
        actions_controls.configure(padding=10)

        # Retain named references for future UI work and compatibility with the
        # earlier scroll-based left control layout.
        self.load_controls_scroll = load_tab_wrap
        self.fit_controls_scroll = fit_tab_wrap
        self.actions_controls_scroll = actions_tab_wrap
        self.controls_scroll = fit_tab_wrap

        self._build_reports_tab(reports_tab)

        file_box = ttk.LabelFrame(load_controls, text='1. Load data', padding=8)
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

        # Preview was intentionally moved from Actions to Load data so it sits
        # with the first-stage data/setup controls.
      

        fit_box = ttk.LabelFrame(fit_controls, text='2. Fit settings', padding=8)
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
            values=['none', 'poisson-like', 'sqrt(y) emphasis', '1/y']
        ).grid(row=5, column=1, sticky='w', pady=(6, 0), padx=(0, 12))
        
        ttk.Label(fit_box, text='Auto-fit trials').grid(row=5, column=2, sticky='w', pady=(6, 0))
        ttk.Spinbox(
            fit_box,
            from_=1,
            to=100,
            textvariable=self.autofit_trials_var,
            width=8
        ).grid(row=5, column=3, sticky='w', pady=(6, 0))

        ttk.Label(fit_box, text='Optimizer').grid(row=6, column=0, sticky='w', pady=(6, 0))
        ttk.Combobox(
            fit_box,
            textvariable=self.optimizer_mode_var,
            state='readonly',
            width=34,
            values=FIT_OPTIMIZER_MODES
        ).grid(row=6, column=1, sticky='w', pady=(6, 0), padx=(0, 12))

        ttk.Label(fit_box, text='Max fit evals').grid(row=6, column=2, sticky='w', pady=(6, 0))
        ttk.Spinbox(
            fit_box,
            from_=50,
            to=1000000,
            increment=500,
            textvariable=self.max_nfev_var,
            width=10
        ).grid(row=6, column=3, sticky='w', pady=(6, 0))

        ttk.Label(fit_box, text='Auto-fit criterion').grid(row=7, column=0, sticky='w', pady=(6, 0))
        ttk.Combobox(
            fit_box,
            textvariable=self.fit_criterion_var,
            state='readonly',
            width=20,
            values=FIT_SELECTION_CRITERIA
        ).grid(row=7, column=1, columnspan=3, sticky='w', pady=(6, 0))
        
        ttk.Label(fit_box, text='Auto-fit sampling').grid(row=8, column=0, sticky='w', pady=(6, 0))
        ttk.Combobox(
            fit_box,
            textvariable=self.autoprefit_sampling_var,
            state='readonly',
            width=34,
            values=AUTO_PREFIT_SAMPLING_MODES
        ).grid(row=8, column=1, columnspan=3, sticky='w', pady=(6, 0))

        ttk.Label(fit_box, text='Random seed').grid(row=9, column=0, sticky='w', pady=(6, 0))
        ttk.Entry(
            fit_box,
            textvariable=self.random_seed_var,
            width=14
        ).grid(row=9, column=1, sticky='w', pady=(6, 0), padx=(0, 12))
        ttk.Label(
            fit_box,
            text='blank = stochastic; integer = reproducible',
        ).grid(row=9, column=2, columnspan=3, sticky='w', pady=(6, 0))
        
        """
        ttk.Label(fit_box, text='Weights').grid(row=3, column=0, sticky='w', pady=(6, 0))
        ttk.Combobox(fit_box, textvariable=self.weighting_var, state='readonly', width=12,
                     values=['none', 'poisson-like', 'sqrt(y)', '1/y']).grid(row=3, column=1, sticky='w', pady=(6, 0))
        ttk.Label(fit_box, text='Auto-fit trials').grid(row=4, column=0, sticky='w', pady=(6, 0))
        ttk.Spinbox(fit_box, from_=1, to=100, textvariable=self.autofit_trials_var, width=8).grid(row=4, column=1, sticky='w', pady=(6, 0))
        """
        detect_box = ttk.LabelFrame(fit_controls, text='Optional: Find peaks', padding=8)
        detect_box.pack(fill=tk.X, pady=(0, 8))
        
        ttk.Label(detect_box, text='Max peaks').grid(row=0, column=0, sticky='w')
        ttk.Spinbox(
            detect_box,
            from_=1,
            to=15,
            textvariable=self.find_peaks_max_var,
            width=8
        ).grid(row=0, column=1, sticky='w', padx=(6, 12))
        
        ttk.Label(detect_box, text='Prominence %').grid(row=0, column=2, sticky='w')
        ttk.Entry(
            detect_box,
            textvariable=self.find_peaks_prominence_pct_var,
            width=8
        ).grid(row=0, column=3, sticky='w', padx=(6, 0))
        
        ttk.Label(detect_box, text='Min distance').grid(row=1, column=0, sticky='w', pady=(6, 0))
        ttk.Entry(
            detect_box,
            textvariable=self.find_peaks_min_distance_var,
            width=8
        ).grid(row=1, column=1, sticky='w', padx=(6, 12), pady=(6, 0))
        
        ttk.Label(detect_box, text='Min width').grid(row=1, column=2, sticky='w', pady=(6, 0))
        ttk.Entry(
            detect_box,
            textvariable=self.find_peaks_min_width_var,
            width=8
        ).grid(row=1, column=3, sticky='w', padx=(6, 0), pady=(6, 0))
        
        ttk.Label(detect_box, text='Direction').grid(row=2, column=0, sticky='w', pady=(6, 0))
        ttk.Combobox(
            detect_box,
            textvariable=self.find_peaks_direction_var,
            values=PEAK_DETECTION_DIRECTIONS,
            state='readonly',
            width=10
        ).grid(row=2, column=1, sticky='w', padx=(6, 12), pady=(6, 0))
        
        ttk.Checkbutton(
            detect_box,
            text='Use smoothed preview for detection',
            variable=self.find_peaks_use_smooth_var
        ).grid(row=2, column=2, columnspan=2, sticky='w', pady=(6, 0))
        
        resid_box = ttk.LabelFrame(fit_controls, text='Optional: Missing peaks from residual', padding=8)
        resid_box.pack(fill=tk.X, pady=(0, 8))
        
        ttk.Label(resid_box, text='Max suggestions').grid(row=0, column=0, sticky='w')
        ttk.Spinbox(
            resid_box,
            from_=1,
            to=5,
            textvariable=self.residual_suggest_max_var,
            width=8
        ).grid(row=0, column=1, sticky='w', padx=(6, 12))
        
        ttk.Label(resid_box, text='Sensitivity').grid(row=0, column=2, sticky='w')
        ttk.Combobox(
            resid_box,
            textvariable=self.residual_suggest_sensitivity_var,
            values=RESIDUAL_SUGGESTION_SENSITIVITIES,
            state='readonly',
            width=14
        ).grid(row=0, column=3, sticky='w', padx=(6, 0))
        
        ttk.Label(resid_box, text='Direction').grid(row=1, column=0, sticky='w', pady=(6, 0))
        ttk.Combobox(
            resid_box,
            textvariable=self.residual_suggest_direction_var,
            values=RESIDUAL_SUGGESTION_DIRECTIONS,
            state='readonly',
            width=10
        ).grid(row=1, column=1, sticky='w', padx=(6, 12), pady=(6, 0))
        
        ttk.Checkbutton(
            resid_box,
            text='Use smoothed residual for suggestion',
            variable=self.residual_suggest_use_smooth_var
        ).grid(row=1, column=2, columnspan=2, sticky='w', pady=(6, 0))
        
        self.preview_button = ttk.Button(
            fit_controls,
            text='Preview',
            command=self.preview
        )
        self.preview_button.pack(fill=tk.X, pady=(0, 8))
        
        peaks_box = ttk.LabelFrame(peaks_tab, text='3. Peaks', padding=8)
        peaks_box.pack(fill=tk.BOTH, expand=True, pady=(0, 8))
        
        self.custom_profiles_button = ttk.Button(
            peaks_box,
            text='Manage custom profiles',
            command=self.open_custom_profile_manager
        )
        self.custom_profiles_button.pack(fill=tk.X, pady=(0, 8))
        
        top_peaks = ttk.Frame(peaks_box)
        top_peaks.pack(fill=tk.X)
        ttk.Label(top_peaks, text='Number of peaks').pack(side=tk.LEFT)
        ttk.Spinbox(top_peaks, from_=1, to=15, textvariable=self.peak_count_var, width=6,
                    command=self.rebuild_peaks).pack(side=tk.LEFT, padx=(8, 8))
        ttk.Button(top_peaks, text='Apply peak count', command=self.rebuild_peaks).pack(side=tk.LEFT)

        self.peaks_scroll = ScrollableFrame(peaks_box)
        self.peaks_scroll.pack(fill=tk.BOTH, expand=True, pady=(8, 0))
        self.peaks_container = self.peaks_scroll.inner

        action_box = ttk.LabelFrame(actions_controls, text='4. Actions', padding=8)
        action_box.pack(fill=tk.X)
      
        
        self.pick_centers_button = ttk.Button(action_box, text='Pick centers from plot', command=self.start_pick_centers)
        self.pick_centers_button.pack(fill=tk.X, pady=2)
        self.find_peaks_button = ttk.Button(action_box, text='Find peaks', command=self.find_peaks_auto_populate)
        self.find_peaks_button.pack(fill=tk.X, pady=2)
        self.parameter_constraints_button = ttk.Button(
            action_box,
            text='Parameter constraints',
            command=self.open_parameter_constraints_dialog
        )
        self.parameter_constraints_button.pack(fill=tk.X, pady=2)
        self.autofit_button = ttk.Button(action_box, text='Auto pre-fit', command=self.autofill_from_centers_background)
        self.autofit_button.pack(fill=tk.X, pady=2)
        
        self.residual_suggest_button = ttk.Button(
            action_box,
            text='Suggest missing peaks from residual',
            command=self.suggest_missing_peaks_from_residual
        )
        self.residual_suggest_button.pack(fill=tk.X, pady=2)
        
        self.refine_button = ttk.Button(action_box, text='Refine with added peaks', command=self.refine_with_added_peaks_background)
        self.refine_button.pack(fill=tk.X, pady=2)
        
        self.run_fit_button = ttk.Button(action_box, text='Run fit', command=self.run_fit_background)
        self.run_fit_button.pack(fill=tk.X, pady=2)

        self.stability_test_button = ttk.Button(
            action_box,
            text='Fit stability test',
            command=self.open_fit_stability_dialog
        )
        self.stability_test_button.pack(fill=tk.X, pady=2)
        
        self.derived_quantities_button = ttk.Button(
            action_box,
            text='Derived quantities',
            command=self.open_derived_quantities_dialog
        )
        self.derived_quantities_button.pack(fill=tk.X, pady=2)
        
        self.correlation_matrix_button = ttk.Button(
            action_box,
            text='Correlation matrix',
            command=self.open_parameter_correlation_matrix_dialog
        )
        self.correlation_matrix_button.pack(fill=tk.X, pady=2)

        self.residual_diagnostics_button = ttk.Button(
            action_box,
            text='Residual diagnostics',
            command=self.open_residual_diagnostics_dialog
        )
        self.residual_diagnostics_button.pack(fill=tk.X, pady=2)

        self.confidence_ellipse_button = ttk.Button(
            action_box,
            text='2D confidence ellipse',
            command=self.open_confidence_ellipse_dialog
        )
        self.confidence_ellipse_button.pack(fill=tk.X, pady=2)
        
        self.cancel_fit_button = ttk.Button(action_box, text='Cancel running fit', command=self.cancel_running_fit, state='disabled')
        self.cancel_fit_button.pack(fill=tk.X, pady=2)
        ttk.Separator(action_box).pack(fill=tk.X, pady=6)
        
        ttk.Button(action_box, text='Save session (.json)', command=self.save_session).pack(fill=tk.X, pady=2)
        ttk.Button(action_box, text='Load session (.json)', command=self.load_session).pack(fill=tk.X, pady=2)
        ttk.Separator(action_box).pack(fill=tk.X, pady=6)
        
        self.batch_fit_button = ttk.Button(
            action_box,
            text='Batch fit folder',
            command=self.open_batch_fit_dialog
        )
        self.batch_fit_button.pack(fill=tk.X, pady=2)
        
        ttk.Button(action_box, text='Save ZIP results', command=self.save_zip_results).pack(fill=tk.X, pady=2)

        # Keep status/progress always visible below the tabs.
        self.status_var = tk.StringVar(value='Ready.')
        ttk.Label(controls_panel, textvariable=self.status_var, wraplength=360, foreground='#444').pack(fill=tk.X, padx=10, pady=(8, 0))
        self.progress_var = tk.DoubleVar(value=0.0)
        self.progress_bar = ttk.Progressbar(controls_panel, variable=self.progress_var, maximum=100.0, mode='determinate')
        self.progress_bar.pack(fill=tk.X, padx=10, pady=(4, 10))

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
    
        # Keep manually entered ROI values unchanged during reset.
        # self.roi_min_var.set(float(np.min(self.x_full)))
        # self.roi_max_var.set(float(np.max(self.x_full)))
    
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

    def _collect_peak_defs(self, require_active_peak=True):
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

        if not peak_defs and require_active_peak:
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
        """Return weights from the current GUI-selected weighting mode."""
        return compute_weights(y, self.weighting_var.get())

    def get_session_state(self):
        """Collect GUI values, then let the backend compose the session payload."""
        loader = {
            'delimiter': self.delimiter_var.get(),
            'skiprows': int(self.skiprows_var.get()),
            'x_col': int(self.xcol_var.get()),
            'y_col': int(self.ycol_var.get()),
        }

        fit_settings = {
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
            'max_nfev': int(self.max_nfev_var.get()),
            'fit_criterion': self._current_fit_criterion(),
            'optimizer_mode': self._current_optimizer_mode(),
            'random_seed': str(self.random_seed_var.get()).strip(),
            'stability_protocol': self.stability_protocol_var.get(),
            'stability_repeats': int(self.stability_repeats_var.get()),
            'stability_delta_score': float(self.stability_delta_var.get()),
            'find_peaks_max_peaks': int(self.find_peaks_max_var.get()),
            'find_peaks_prominence_pct': float(self.find_peaks_prominence_pct_var.get()),
            'find_peaks_min_distance': float(self.find_peaks_min_distance_var.get()),
            'find_peaks_min_width': float(self.find_peaks_min_width_var.get()),
            'find_peaks_direction': self.find_peaks_direction_var.get(),
            'find_peaks_use_smooth': bool(self.find_peaks_use_smooth_var.get()),
            'residual_suggest_max': int(self.residual_suggest_max_var.get()),
            'residual_suggest_sensitivity': self.residual_suggest_sensitivity_var.get(),
            'residual_suggest_direction': self.residual_suggest_direction_var.get(),
            'residual_suggest_use_smooth': bool(self.residual_suggest_use_smooth_var.get()),
            'autoprefit_sampling': self._current_autoprefit_sampling_mode(),
            'batch_fit_mode': self.batch_fit_mode_var.get(),
            'derived_quantities': copy.deepcopy(self.derived_quantity_definitions),
            'auxiliary_parameters': copy.deepcopy(self.auxiliary_parameter_definitions),
            'parameter_constraints': copy.deepcopy(self.parameter_constraint_definitions),
            'peak_count': int(self.peak_count_var.get()),
        }

        return build_session_state(
            current_file=self.current_file,
            loader=loader,
            fit_settings=fit_settings,
            custom_profiles=list(self.custom_profiles.values()),
            custom_background_profiles=list(self.custom_background_profiles.values()),
            peaks=[self._row_state_from_row(row) for row in self.peak_rows],
        )

    def apply_session_state(self, state: dict):
        normalized_session = normalize_session_payload(state)
        state = normalized_session['state']
        self.custom_profiles = normalized_session['custom_profiles']
        self.custom_background_profiles = normalized_session['custom_background_profiles']
        self.derived_quantity_definitions = normalized_session['derived_quantity_definitions']

        loader = normalized_session['loader']
        self.delimiter_var.set(loader.get('delimiter', self.delimiter_var.get()))
        self.skiprows_var.set(loader.get('skiprows', self.skiprows_var.get()))
        self.xcol_var.set(loader.get('x_col', self.xcol_var.get()))
        self.ycol_var.set(loader.get('y_col', self.ycol_var.get()))

        settings = normalized_session['fit_settings']
        self.auxiliary_parameter_definitions = settings.get('auxiliary_parameters', [])
        self.parameter_constraint_definitions = settings.get('parameter_constraints', [])
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
        if 'optimizer_mode' in settings:
            opt_mode = settings.get('optimizer_mode', self.optimizer_mode_var.get())
            legacy_optimizer_aliases = {
                'Differential Evolution + Levenberg-Marquardt': 'Robust: compare LM and DE+LM',
                'Differential Evolution + LM': 'Robust: compare LM and DE+LM',
                'Robust Global + LM': 'Robust: compare LM and DE+LM',
            }
            opt_mode = legacy_optimizer_aliases.get(opt_mode, opt_mode)
            if opt_mode in FIT_OPTIMIZER_MODES:
                self.optimizer_mode_var.set(opt_mode)

        if 'random_seed' in settings:
            self.random_seed_var.set(str(settings.get('random_seed', self.random_seed_var.get())).strip())

        protocol = settings.get('stability_protocol', self.stability_protocol_var.get())
        if protocol in STABILITY_TEST_PROTOCOLS:
            self.stability_protocol_var.set(protocol)
        try:
            self.stability_repeats_var.set(max(1, int(settings.get('stability_repeats', self.stability_repeats_var.get()))))
            delta_score = float(settings.get('stability_delta_score', self.stability_delta_var.get()))
            if np.isfinite(delta_score) and delta_score >= 0:
                self.stability_delta_var.set(delta_score)
        except Exception:
            pass

        if 'autoprefit_sampling' in settings:
            sampling_mode = settings.get('autoprefit_sampling', self.autoprefit_sampling_var.get())
            if sampling_mode in AUTO_PREFIT_SAMPLING_MODES:
                self.autoprefit_sampling_var.set(sampling_mode)

        if 'batch_fit_mode' in settings:
            batch_mode = settings.get('batch_fit_mode', self.batch_fit_mode_var.get())
            if batch_mode in BATCH_FIT_MODES:
                self.batch_fit_mode_var.set(batch_mode)

        try:
            self.find_peaks_max_var.set(int(settings.get('find_peaks_max_peaks', self.find_peaks_max_var.get())))
            self.find_peaks_prominence_pct_var.set(float(settings.get('find_peaks_prominence_pct', self.find_peaks_prominence_pct_var.get())))
            self.find_peaks_min_distance_var.set(float(settings.get('find_peaks_min_distance', self.find_peaks_min_distance_var.get())))
            self.find_peaks_min_width_var.set(float(settings.get('find_peaks_min_width', self.find_peaks_min_width_var.get())))

            direction = settings.get('find_peaks_direction', self.find_peaks_direction_var.get())
            if direction in PEAK_DETECTION_DIRECTIONS:
                self.find_peaks_direction_var.set(direction)

            self.find_peaks_use_smooth_var.set(bool(settings.get('find_peaks_use_smooth', self.find_peaks_use_smooth_var.get())))
        except Exception:
            pass

        try:
            self.residual_suggest_max_var.set(
                int(settings.get('residual_suggest_max', self.residual_suggest_max_var.get()))
            )

            sensitivity = settings.get(
                'residual_suggest_sensitivity',
                self.residual_suggest_sensitivity_var.get()
            )
            if sensitivity in RESIDUAL_SUGGESTION_SENSITIVITIES:
                self.residual_suggest_sensitivity_var.set(sensitivity)

            direction = settings.get(
                'residual_suggest_direction',
                self.residual_suggest_direction_var.get()
            )
            if direction in RESIDUAL_SUGGESTION_DIRECTIONS:
                self.residual_suggest_direction_var.set(direction)

            self.residual_suggest_use_smooth_var.set(
                bool(settings.get('residual_suggest_use_smooth', self.residual_suggest_use_smooth_var.get()))
            )
        except Exception:
            pass

        self._refresh_background_controls()

        peaks = normalized_session['peaks']
        self.peak_count_var.set(settings.get('peak_count', len(peaks) or self.peak_count_var.get()))
        self._draw_peak_rows(peaks if peaks else [self._default_peak_state(i, int(self.peak_count_var.get())) for i in range(int(self.peak_count_var.get()))])

        state_file = normalized_session['current_file']
        if state_file:
            maybe_file = Path(state_file)
            if maybe_file.exists():
                self.current_file = maybe_file
                self.reload_current_file()
        
                # reload_current_file() resets ROI to the full data range,
                # so restore the session ROI again after reloading the spectrum.
                self.roi_min_var.set(settings.get('roi_min', self.roi_min_var.get()))
                self.roi_max_var.set(settings.get('roi_max', self.roi_max_var.get()))
        
                self._draw_peak_rows(
                    peaks if peaks else [
                        self._default_peak_state(i, int(self.peak_count_var.get()))
                        for i in range(int(self.peak_count_var.get()))
                    ]
                )
                #self.preview()
            else:
                self.status_var.set(
                    'Session loaded, but the saved spectrum file path was not found. '
                    'Load the data file manually.'
                )


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
        
    def _show_fit_report_dialog(self, result, title='Fit report', report_key='fit', history_id=None):
        if result is None:
            return
    
        try:
            report = self._full_fited_fit_report(result)
        except Exception as exc:
            report = f'Could not generate fit report:\n{exc}'

        if report_key is not None:
            self._set_report_document(report_key, report, title=title, history_id=history_id)
    
        win = tk.Toplevel(self.root)
        win.title(title)
        win.geometry('920x680')
        win.minsize(700, 450)
    
        frame = ttk.Frame(win, padding=10)
        frame.pack(fill=tk.BOTH, expand=True)
    
        header = ttk.Label(
            frame,
            text=title,
            font=('Segoe UI', 11, 'bold')
        )
        header.pack(anchor='w', pady=(0, 8))
    
        text_frame = ttk.Frame(frame)
        text_frame.pack(fill=tk.BOTH, expand=True)
    
        report_text = tk.Text(
            text_frame,
            wrap='none',
            height=28,
            width=110
        )
        report_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    
        y_scroll = ttk.Scrollbar(text_frame, orient='vertical', command=report_text.yview)
        y_scroll.pack(side=tk.RIGHT, fill=tk.Y)
    
        x_scroll = ttk.Scrollbar(frame, orient='horizontal', command=report_text.xview)
        x_scroll.pack(fill=tk.X)
    
        report_text.configure(
            yscrollcommand=y_scroll.set,
            xscrollcommand=x_scroll.set
        )
    
        report_text.insert('1.0', report)
        report_text.configure(state='disabled')
    
        btns = ttk.Frame(frame)
        btns.pack(fill=tk.X, pady=(10, 0))
    
        def copy_report():
            self.root.clipboard_clear()
            self.root.clipboard_append(report)
            self.status_var.set('Fit report copied to clipboard.')
    
        ttk.Button(btns, text='Copy report', command=copy_report).pack(side=tk.LEFT)
        ttk.Button(btns, text='Close', command=win.destroy).pack(side=tk.RIGHT)
    
        win.transient(self.root)
        win.lift()
        win.focus_force()
    
    def _full_fited_fit_report(self, result):
        optimizer_mode = (
            getattr(result, 'fited_optimizer_mode', None)
            or self._current_optimizer_mode()
        )
        fit_criterion = (
            getattr(result, 'fited_selection_criterion', None)
            or self._current_fit_criterion()
        )
    
        return full_fited_fit_report(
            result,
            optimizer_mode,
            fit_criterion,
        )
    
    def _safe_derived_expression_names(self, expression):
        """Delegate derived-expression name parsing to the backend."""
        return safe_derived_expression_names(expression)
    
    
    def _validate_derived_expression(self, expression, allowed_parameter_names):
        """Delegate derived-expression validation to the backend."""
        return validate_derived_expression(expression, allowed_parameter_names)
    
    
    def _params_to_value_dict(self, params):
        """Delegate lmfit-parameter value conversion to the backend."""
        return params_to_value_dict(params)
    
    
    def _eval_derived_expression(self, expression, values):
        """Delegate derived-expression evaluation to the backend."""
        return eval_derived_expression(expression, values)
    
    
    def _eval_derived_expression_for_params(self, expression, params):
        """Delegate derived-expression evaluation for lmfit params to the backend."""
        return eval_derived_expression_for_params(expression, params)
    
    
    def _finite_difference_derivative_for_var(self, expression, result, var_name, base_value):
        """Delegate finite-difference derivative calculation to the backend."""
        return finite_difference_derivative_for_var(expression, result, var_name, base_value)
    
    
    def _compute_one_derived_quantity(self, name, expression):
        """Delegate derived-quantity computation and uncertainty propagation."""
        return compute_one_derived_quantity(self.fit_result, name, expression)
    
    
    def _parse_derived_quantity_lines(self, text):
        """Delegate derived-quantity line parsing to the backend."""
        return parse_derived_quantity_lines(text) 
    
    def _format_derived_quantities_report(self, rows):
        """Delegate derived-quantity report formatting to the backend."""
        return format_derived_quantities_report(rows)
    
    def _parse_auxiliary_parameter_lines(self, text):
        """Delegate auxiliary-parameter parsing to the backend."""
        return parse_auxiliary_parameter_lines(text)
    
    
    def _parse_parameter_constraint_lines(self, text):
        """Delegate parameter-constraint parsing to the backend."""
        return parse_parameter_constraint_lines(text)
        
    

    def _parameter_correlation_dataframe(self, payload):
        """Convert backend correlation-matrix payload into a DataFrame."""
        if not payload:
            return pd.DataFrame()

        names = payload.get("parameter_names", [])
        matrix = payload.get("correlation_matrix", [])

        if not names or not matrix:
            return pd.DataFrame()

        return pd.DataFrame(matrix, index=names, columns=names)



    def _format_parameter_correlation_report(self, payload, corr_df):
        """Create a readable text report for the full parameter correlation matrix."""
        warnings_list = payload.get("warnings", []) or []
        pair_rows = payload.get("strongest_pairs", []) or []

        lines = [
            "[[FitED parameter correlation matrix]]",
            "Correlation coefficients are covariance-based local estimates from the fitted parameter covariance matrix.",
            "Values near +1 or -1 indicate strong coupling; values near 0 indicate weak local linear coupling.",
            "These correlations do not prove physical dependence; they diagnose numerical/fit-parameter coupling.",
            "",
        ]

        if warnings_list:
            lines.append("Warnings: " + " | ".join(str(w) for w in warnings_list))
            lines.append("")

        lines.append("[[Correlation matrix]]")
        try:
            lines.append(corr_df.to_string(float_format=lambda v: f"{v: .4f}"))
        except Exception:
            lines.append(str(corr_df))

        lines.append("")
        lines.append("[[Strongest parameter-pair correlations]]")
        pairs_df = pd.DataFrame(pair_rows)
        if pairs_df.empty:
            lines.append("No finite parameter-pair correlations are available.")
        else:
            try:
                lines.append(pairs_df.to_string(index=False))
            except Exception:
                lines.append(str(pair_rows))

        return "\n".join(lines)


    def _build_parameter_correlation_heatmap_figure(self, corr_df):
        """Build a Matplotlib heatmap for the full fitted-parameter correlation matrix."""
        n = max(int(corr_df.shape[0]), 1)

        fig_width = max(7.5, 0.55 * n + 3.0)
        fig_height = max(6.0, 0.55 * n + 2.6)

        fig = Figure(figsize=(fig_width, fig_height), dpi=110)
        ax = fig.add_subplot(111)

        values = corr_df.to_numpy(dtype=float)

        image = ax.imshow(
            values,
            aspect="auto",
            cmap="coolwarm",
            vmin=-1.0,
            vmax=1.0,
        )

        ax.set_xticks(np.arange(n))
        ax.set_xticklabels(
            [str(c) for c in corr_df.columns],
            rotation=60,
            ha="right",
            fontsize=8,
        )

        ax.set_yticks(np.arange(n))
        ax.set_yticklabels(
            [str(i) for i in corr_df.index],
            fontsize=8,
        )

        ax.set_title("Fitted parameter correlation matrix")
        ax.set_xlabel("Fitted parameters")
        ax.set_ylabel("Fitted parameters")

        if values.size <= 225:
            for i in range(values.shape[0]):
                for j in range(values.shape[1]):
                    val = values[i, j]
                    if np.isfinite(val):
                        ax.text(
                            j,
                            i,
                            f"{val:.2f}",
                            ha="center",
                            va="center",
                            fontsize=7,
                        )

        colorbar = fig.colorbar(image, ax=ax, fraction=0.035, pad=0.02)
        colorbar.set_label("Correlation coefficient")

        fig.tight_layout()
        return fig


    def open_parameter_correlation_matrix_dialog(self):
        """Show the full covariance-based fitted-parameter correlation matrix."""
        if self.fit_result is None:
            messagebox.showinfo(
                "No fit result",
                "Run fit, Auto pre-fit, or Refine before opening the correlation matrix."
            )
            return

        try:
            payload = compute_parameter_correlation_matrix(self.fit_result)
            corr_df = self._parameter_correlation_dataframe(payload)
        except Exception as exc:
            messagebox.showerror("Correlation matrix error", str(exc))
            return

        if corr_df.empty:
            messagebox.showinfo(
                "No correlation matrix",
                "No finite parameter correlation matrix is available for this fit."
            )
            return

        report = self._format_parameter_correlation_report(payload, corr_df)
        self._set_report_document(
            'correlation',
            report,
            title='Parameter correlation matrix report',
        )

        win = tk.Toplevel(self.root)
        win.title("Parameter correlation matrix")
        win.geometry("1250x760")
        win.minsize(900, 560)

        frame = ttk.Frame(win, padding=10)
        frame.pack(fill=tk.BOTH, expand=True)

        ttk.Label(
            frame,
            text=(
                "This matrix shows covariance-based correlations between independently "
                "varying fitted parameters. Values near +1 or -1 indicate strong parameter coupling."
            ),
            wraplength=1180,
        ).pack(anchor="w", pady=(0, 8))

        warnings_text = " | ".join(payload.get("warnings", []))
        if warnings_text:
            ttk.Label(
                frame,
                text=f"Warning: {warnings_text}",
                wraplength=1180,
                foreground="#8a4b00",
            ).pack(anchor="w", pady=(0, 8))

        notebook = ttk.Notebook(frame)
        notebook.pack(fill=tk.BOTH, expand=True)

        heatmap_tab = ttk.Frame(notebook, padding=6)
        table_tab = ttk.Frame(notebook, padding=6)
        pairs_tab = ttk.Frame(notebook, padding=6)

        notebook.add(heatmap_tab, text="Heatmap")
        notebook.add(table_tab, text="Matrix table")
        notebook.add(pairs_tab, text="Strongest pairs")

        fig = self._build_parameter_correlation_heatmap_figure(corr_df)
        canvas = FigureCanvasTkAgg(fig, master=heatmap_tab)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        matrix_text_frame = ttk.Frame(table_tab)
        matrix_text_frame.pack(fill=tk.BOTH, expand=True)

        matrix_text = tk.Text(matrix_text_frame, wrap="none")
        matrix_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        matrix_y = ttk.Scrollbar(matrix_text_frame, orient="vertical", command=matrix_text.yview)
        matrix_y.pack(side=tk.RIGHT, fill=tk.Y)
        matrix_x = ttk.Scrollbar(table_tab, orient="horizontal", command=matrix_text.xview)
        matrix_x.pack(fill=tk.X)
        matrix_text.configure(yscrollcommand=matrix_y.set, xscrollcommand=matrix_x.set)
        matrix_text.insert("1.0", corr_df.to_string(float_format=lambda v: f"{v: .4f}"))
        matrix_text.configure(state="disabled")

        pair_rows = payload.get("strongest_pairs", [])
        pairs_df = pd.DataFrame(pair_rows)
        pairs_text_frame = ttk.Frame(pairs_tab)
        pairs_text_frame.pack(fill=tk.BOTH, expand=True)

        pairs_text = tk.Text(pairs_text_frame, wrap="none")
        pairs_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        pairs_y = ttk.Scrollbar(pairs_text_frame, orient="vertical", command=pairs_text.yview)
        pairs_y.pack(side=tk.RIGHT, fill=tk.Y)
        pairs_x = ttk.Scrollbar(pairs_tab, orient="horizontal", command=pairs_text.xview)
        pairs_x.pack(fill=tk.X)
        pairs_text.configure(yscrollcommand=pairs_y.set, xscrollcommand=pairs_x.set)
        if pairs_df.empty:
            pairs_text.insert("1.0", "No finite parameter-pair correlations are available.")
        else:
            pairs_text.insert("1.0", pairs_df.to_string(index=False))
        pairs_text.configure(state="disabled")

        btns = ttk.Frame(frame)
        btns.pack(fill=tk.X, pady=(8, 0))

        def copy_matrix_csv():
            self.root.clipboard_clear()
            self.root.clipboard_append(corr_df.to_csv())
            self.status_var.set("Parameter correlation matrix CSV copied to clipboard.")

        def copy_report():
            self.root.clipboard_clear()
            self.root.clipboard_append(report)
            self.status_var.set("Parameter correlation matrix report copied to clipboard.")

        ttk.Button(btns, text="Copy matrix CSV", command=copy_matrix_csv).pack(side=tk.LEFT)
        ttk.Button(btns, text="Copy report", command=copy_report).pack(side=tk.LEFT, padx=(6, 0))
        ttk.Button(btns, text="Close", command=win.destroy).pack(side=tk.RIGHT)

        win.transient(self.root)
        win.lift()
        win.focus_force()


    def _residual_diagnostics_to_dataframes(self, payload):
        """Convert residual-diagnostics payload into exportable DataFrames."""
        residual_df = pd.DataFrame(payload.get("residual_rows", []))
        autocorr_df = pd.DataFrame(payload.get("autocorrelation_rows", []))
        qq_df = pd.DataFrame(payload.get("qq_rows", []))

        summary = payload.get("summary", {}) or {}
        qq_fit = payload.get("qq_fit", {}) or {}
        warnings_list = payload.get("warnings", []) or []

        summary_rows = []
        for key, value in summary.items():
            summary_rows.append({"field": key, "value": value})
        for key, value in qq_fit.items():
            summary_rows.append({"field": f"qq_{key}", "value": value})
        if warnings_list:
            summary_rows.append({"field": "warnings", "value": " | ".join(warnings_list)})

        summary_df = pd.DataFrame(summary_rows)
        return residual_df, autocorr_df, qq_df, summary_df


    def _format_residual_diagnostics_report(self, payload):
        """Create a readable residual-diagnostics text summary."""
        summary = payload.get("summary", {}) or {}
        warnings_list = payload.get("warnings", []) or []

        lines = [
            "[[FitED residual diagnostics]]",
            "Residual diagnostics help test whether the remaining residual structure is random.",
            "They do not prove that a model is physically correct.",
            "",
        ]

        for key in [
            "n",
            "residual_kind",
            "mean",
            "std",
            "median",
            "mad",
            "rmse",
            "max_abs_residual",
            "durbin_watson",
            "lag1_autocorrelation",
            "qq_r_squared",
        ]:
            if key in summary:
                lines.append(f"{key:<28} = {summary[key]}")

        if warnings_list:
            lines.extend(["", "Warnings:"])
            for warning in warnings_list:
                lines.append(f"- {warning}")

        return "\n".join(lines)


    def _build_residual_diagnostics_figure(self, payload):
        """Build a 2x2 residual-diagnostics figure."""
        residual_df, autocorr_df, qq_df, _ = self._residual_diagnostics_to_dataframes(payload)
        summary = payload.get("summary", {}) or {}
        qq_fit = payload.get("qq_fit", {}) or {}

        fig = Figure(figsize=(10.5, 8.0), dpi=110)

        ax1 = fig.add_subplot(221)
        ax2 = fig.add_subplot(222)
        ax3 = fig.add_subplot(223)
        ax4 = fig.add_subplot(224)

        if not residual_df.empty:
            x = residual_df["x"].to_numpy(dtype=float)
            resid = residual_df["residual"].to_numpy(dtype=float)
            diag = residual_df["diagnostic_residual"].to_numpy(dtype=float)

            ax1.axhline(0.0, linestyle="--", linewidth=1.0)
            ax1.plot(x, resid, ".", markersize=3)
            ax1.set_title("Residuals vs x")
            ax1.set_xlabel("X")
            ax1.set_ylabel("Residual")

            ax2.hist(diag[np.isfinite(diag)], bins="auto")
            ax2.set_title("Residual histogram")
            ax2.set_xlabel(str(summary.get("residual_kind", "residual")))
            ax2.set_ylabel("Count")
        else:
            ax1.text(0.5, 0.5, "No residual rows", transform=ax1.transAxes, ha="center", va="center")
            ax2.text(0.5, 0.5, "No residual rows", transform=ax2.transAxes, ha="center", va="center")

        if not autocorr_df.empty:
            lags = autocorr_df["lag"].to_numpy(dtype=float)
            ac = autocorr_df["autocorrelation"].to_numpy(dtype=float)
            ax3.axhline(0.0, linestyle="--", linewidth=1.0)
            ax3.vlines(lags, 0.0, ac, linewidth=1.0)
            ax3.plot(lags, ac, ".", markersize=4)
            ax3.set_title("Residual autocorrelation")
            ax3.set_xlabel("Lag")
            ax3.set_ylabel("Autocorrelation")
        else:
            ax3.text(0.5, 0.5, "No autocorrelation data", transform=ax3.transAxes, ha="center", va="center")

        if not qq_df.empty:
            tq = qq_df["theoretical_normal_quantile"].to_numpy(dtype=float)
            oq = qq_df["ordered_standardized_residual"].to_numpy(dtype=float)
            ax4.plot(tq, oq, ".", markersize=3)

            slope = qq_fit.get("slope", np.nan)
            intercept = qq_fit.get("intercept", np.nan)
            if np.isfinite(slope) and np.isfinite(intercept):
                xx = np.linspace(np.nanmin(tq), np.nanmax(tq), 100)
                ax4.plot(xx, intercept + slope * xx, linewidth=1.0)

            ax4.set_title("Q-Q plot of residuals")
            ax4.set_xlabel("Theoretical normal quantile")
            ax4.set_ylabel("Ordered standardized residual")
        else:
            ax4.text(0.5, 0.5, "No Q-Q data", transform=ax4.transAxes, ha="center", va="center")

        fig.tight_layout()
        return fig


    def open_residual_diagnostics_dialog(self):
        """Show residual autocorrelation, Q-Q plot, histogram, and summary."""
        if self.fit_result is None or self.last_roi is None or self.last_best_fit is None:
            messagebox.showinfo(
                "No fit result",
                "Run fit, Auto pre-fit, or Refine before opening residual diagnostics."
            )
            return

        try:
            x, y_raw = self.last_roi
            weights = self._weights(y_raw)
            payload = compute_residual_diagnostics(
                x,
                y_raw,
                self.last_best_fit,
                weights=weights,
            )
        except Exception as exc:
            messagebox.showerror("Residual diagnostics error", str(exc))
            return

        win = tk.Toplevel(self.root)
        win.title("Residual diagnostics")
        win.geometry("1250x820")
        win.minsize(900, 600)

        frame = ttk.Frame(win, padding=10)
        frame.pack(fill=tk.BOTH, expand=True)

        ttk.Label(
            frame,
            text=(
                "Residual diagnostics help detect structured residuals, autocorrelation, "
                "and deviations from an approximately Gaussian-noise assumption."
            ),
            wraplength=1180,
        ).pack(anchor="w", pady=(0, 8))

        warnings_text = " | ".join(payload.get("warnings", []))
        if warnings_text:
            ttk.Label(
                frame,
                text=f"Warning: {warnings_text}",
                wraplength=1180,
                foreground="#8a4b00",
            ).pack(anchor="w", pady=(0, 8))

        notebook = ttk.Notebook(frame)
        notebook.pack(fill=tk.BOTH, expand=True)

        plots_tab = ttk.Frame(notebook, padding=6)
        summary_tab = ttk.Frame(notebook, padding=6)

        notebook.add(plots_tab, text="Plots")
        notebook.add(summary_tab, text="Summary / tables")

        fig = self._build_residual_diagnostics_figure(payload)
        canvas = FigureCanvasTkAgg(fig, master=plots_tab)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        report = self._format_residual_diagnostics_report(payload)
        self._set_report_document(
            'residual_diagnostics',
            report,
            title='Residual diagnostics report',
        )

        text_frame = ttk.Frame(summary_tab)
        text_frame.pack(fill=tk.BOTH, expand=True)

        summary_text = tk.Text(text_frame, wrap="none")
        summary_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        y_scroll = ttk.Scrollbar(text_frame, orient="vertical", command=summary_text.yview)
        y_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        x_scroll = ttk.Scrollbar(summary_tab, orient="horizontal", command=summary_text.xview)
        x_scroll.pack(fill=tk.X)
        summary_text.configure(yscrollcommand=y_scroll.set, xscrollcommand=x_scroll.set)
        summary_text.insert("1.0", report)
        summary_text.configure(state="disabled")

        btns = ttk.Frame(frame)
        btns.pack(fill=tk.X, pady=(8, 0))

        def copy_report():
            self.root.clipboard_clear()
            self.root.clipboard_append(report)
            self.status_var.set("Residual diagnostics report copied to clipboard.")

        ttk.Button(btns, text="Copy report", command=copy_report).pack(side=tk.LEFT)
        ttk.Button(btns, text="Close", command=win.destroy).pack(side=tk.RIGHT)

        win.transient(self.root)
        win.lift()
        win.focus_force()



    def _format_confidence_ellipse_report(self, payload):
        """Create a readable report for the 2D covariance confidence ellipse."""
        if not payload:
            return "No confidence ellipse data are available."
    
        cov_mat = payload.get(
            "covariance_matrix",
            [[np.nan, np.nan], [np.nan, np.nan]]
        )
    
        try:
            c00 = float(cov_mat[0][0])
            c01 = float(cov_mat[0][1])
            c11 = float(cov_mat[1][1])
        except Exception:
            c00, c01, c11 = np.nan, np.nan, np.nan
    
        stderr_x = float(np.sqrt(c00)) if np.isfinite(c00) and c00 >= 0 else np.nan
        stderr_y = float(np.sqrt(c11)) if np.isfinite(c11) and c11 >= 0 else np.nan
        covariance_xy = c01 if np.isfinite(c01) else np.nan
    
        eigenvalues = payload.get("eigenvalues", [np.nan, np.nan])
        try:
            eig0 = float(eigenvalues[0])
            eig1 = float(eigenvalues[1])
        except Exception:
            eig0, eig1 = np.nan, np.nan
    
        lines = [
            "[[FitED 2D covariance confidence ellipse]]",
            f"x parameter       = {payload.get('parameter_x', payload.get('x_param', ''))}",
            f"y parameter       = {payload.get('parameter_y', payload.get('y_param', ''))}",
            f"x value           = {payload.get('center_x', payload.get('x_value'))}",
            f"y value           = {payload.get('center_y', payload.get('y_value'))}",
            f"stderr_x          = {stderr_x}",
            f"stderr_y          = {stderr_y}",
            f"covariance_xy     = {covariance_xy}",
            f"correlation       = {payload.get('correlation')}",
            f"covariance matrix = {cov_mat}",
            f"eigenvalues       = {eigenvalues}",
            "",
            "Ellipse levels:",
        ]
    
        for ellipse in payload.get("ellipses", []):
            sigma = float(ellipse.get("sigma", 1.0))
    
            semi_axis_major = (
                sigma * float(np.sqrt(eig0))
                if np.isfinite(eig0) and eig0 >= 0
                else np.nan
            )
            semi_axis_minor = (
                sigma * float(np.sqrt(eig1))
                if np.isfinite(eig1) and eig1 >= 0
                else np.nan
            )
    
            lines.append(
                f"  sigma = {sigma:g}, "
                f"semi_axis_major = {semi_axis_major}, "
                f"semi_axis_minor = {semi_axis_minor}"
            )
    
        warning = payload.get("warning")
        if warning:
            lines.extend(["", f"Warning: {warning}"])
    
        return "\n".join(lines)


    def _build_confidence_ellipse_figure(self, payload):
        """Build a covariance confidence-ellipse plot for one parameter pair."""
        fig = Figure(figsize=(8.2, 6.4), dpi=110)
        ax = fig.add_subplot(111)

        cx = float(payload.get("center_x", np.nan))
        cy = float(payload.get("center_y", np.nan))

        ax.plot([cx], [cy], marker="o", markersize=5, linestyle="None", label="Best fit")

        for ellipse in payload.get("ellipses", []):
            ex = np.asarray(ellipse.get("x", []), dtype=float)
            ey = np.asarray(ellipse.get("y", []), dtype=float)
            sigma = float(ellipse.get("sigma", 1.0))
            if ex.size and ey.size:
                ax.plot(ex, ey, linewidth=1.4, label=f"{sigma:g}σ covariance ellipse")

        px = payload.get("parameter_x", "parameter x")
        py = payload.get("parameter_y", "parameter y")
        corr = payload.get("correlation", np.nan)
        angle = payload.get("ellipse_angle_deg", np.nan)

        ax.set_xlabel(str(px))
        ax.set_ylabel(str(py))
        if np.isfinite(corr) and np.isfinite(angle):
            ax.set_title(
                f"2D covariance ellipse: {px} vs {py}\n"
                f"correlation={corr:.4g}, angle={angle:.3g}°"
            )
        else:
            ax.set_title(f"2D covariance ellipse: {px} vs {py}")

        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.25)
        fig.tight_layout()
        return fig


    def open_confidence_ellipse_dialog(self):
        """
        Show a local covariance-based 2D confidence ellipse for a selected
        pair of independently varying fitted parameters.
        """
        if self.fit_result is None:
            messagebox.showinfo(
                "No fit result",
                "Run fit, Auto pre-fit, or Refine before opening confidence ellipses."
            )
            return

        try:
            corr_payload = compute_parameter_correlation_matrix(self.fit_result)
            names = list(corr_payload.get("parameter_names", []))
        except Exception as exc:
            messagebox.showerror("Confidence ellipse error", str(exc))
            return

        if len(names) < 2:
            messagebox.showinfo(
                "Not enough parameters",
                "At least two independently varying covariance parameters are needed."
            )
            return

        win = tk.Toplevel(self.root)
        win.title("2D parameter confidence ellipse")
        win.geometry("1100x760")
        win.minsize(850, 560)

        frame = ttk.Frame(win, padding=10)
        frame.pack(fill=tk.BOTH, expand=True)

        ttk.Label(
            frame,
            text=(
                "This plot shows local covariance ellipses for two independently varying "
                "parameters. It visualizes parameter coupling near the best fit. "
                "It is not a profile-likelihood confidence contour."
            ),
            wraplength=1040,
        ).pack(anchor="w", pady=(0, 8))

        controls = ttk.Frame(frame)
        controls.pack(fill=tk.X, pady=(0, 8))

        x_var = tk.StringVar(value=names[0])
        y_var = tk.StringVar(value=names[1])

        ttk.Label(controls, text="X parameter").pack(side=tk.LEFT)
        ttk.Combobox(
            controls,
            textvariable=x_var,
            values=names,
            state="readonly",
            width=28,
        ).pack(side=tk.LEFT, padx=(6, 14))

        ttk.Label(controls, text="Y parameter").pack(side=tk.LEFT)
        ttk.Combobox(
            controls,
            textvariable=y_var,
            values=names,
            state="readonly",
            width=28,
        ).pack(side=tk.LEFT, padx=(6, 14))

        plot_area = ttk.Frame(frame)
        plot_area.pack(fill=tk.BOTH, expand=True)

        current = {"canvas": None, "payload": None}

        def redraw():
            try:
                payload = compute_confidence_ellipse_data(
                    self.fit_result,
                    x_var.get(),
                    y_var.get(),
                    sigmas=(1.0, 2.0),
                )
            except Exception as exc:
                messagebox.showerror("Confidence ellipse error", str(exc), parent=win)
                return

            for child in plot_area.winfo_children():
                child.destroy()

            fig = self._build_confidence_ellipse_figure(payload)
            try:
                existing = getattr(self, "last_confidence_ellipse_payloads", [])
                key = (
                    payload.get("parameter_x"),
                    payload.get("parameter_y"),
                )
            
                # Replace old same-pair payload instead of duplicating every click
                existing = [
                    p for p in existing
                    if (p.get("parameter_x"), p.get("parameter_y")) != key
                ]
                existing.append(copy.deepcopy(payload))
                self.last_confidence_ellipse_payloads = existing
            
                active_id = getattr(self, "active_result_package_id", None)
                if active_id is not None:
                    for entry in reversed(getattr(self, "result_package_history", [])):
                        if entry.get("id") == active_id:
                            entry["confidence_ellipse_payloads"] = copy.deepcopy(existing)
                            break
            
            except Exception:
                pass
            canvas = FigureCanvasTkAgg(fig, master=plot_area)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

            current["canvas"] = canvas
            current["payload"] = payload
            report = self._format_confidence_ellipse_report(payload)
            current["report"] = report
            self._set_report_document(
                'confidence_ellipse',
                report,
                title=f"2D confidence ellipse: {payload.get('parameter_x')} vs {payload.get('parameter_y')}",
            )
            self.status_var.set(
                f"Confidence ellipse updated for {payload.get('parameter_x')} vs {payload.get('parameter_y')}."
            )

        ttk.Button(controls, text="Update plot", command=redraw).pack(side=tk.LEFT)

        btns = ttk.Frame(frame)
        btns.pack(fill=tk.X, pady=(8, 0))

        def copy_payload():
            payload = current.get("payload")
            if not payload:
                return
            report = current.get("report") or self._format_confidence_ellipse_report(payload)
            self.root.clipboard_clear()
            self.root.clipboard_append(report)
            self.status_var.set("Confidence ellipse summary copied to clipboard.")

        ttk.Button(btns, text="Copy selected-pair summary", command=copy_payload).pack(side=tk.LEFT)
        ttk.Button(btns, text="Close", command=win.destroy).pack(side=tk.RIGHT)

        redraw()

        win.transient(self.root)
        win.lift()
        win.focus_force()



    def _derived_uncertainty_heatmap_dataframe(self, map_payload):
        """
        Convert backend uncertainty-map rows into a 2D matrix suitable for
        heatmap display and export.
    
        Rows: derived quantities
        Columns: varying fitted parameters
        Values: signed percentage contribution to propagated variance
        """
        if not map_payload:
            return pd.DataFrame()
    
        rows = map_payload.get("rows", [])
        if not rows:
            return pd.DataFrame()
    
        df = pd.DataFrame(rows)
        required = {
            "derived_quantity",
            "parameter",
            "signed_variance_contribution_percent",
        }
    
        if not required.issubset(df.columns):
            return pd.DataFrame()
    
        heatmap_df = df.pivot(
            index="derived_quantity",
            columns="parameter",
            values="signed_variance_contribution_percent",
        )
    
        derived_order = [
            name for name in map_payload.get("derived_names", [])
            if name in heatmap_df.index
        ]
        parameter_order = [
            name for name in map_payload.get("parameter_names", [])
            if name in heatmap_df.columns
        ]
    
        if derived_order:
            remaining_rows = [
                name for name in heatmap_df.index
                if name not in derived_order
            ]
            heatmap_df = heatmap_df.reindex(derived_order + remaining_rows)
    
        if parameter_order:
            remaining_cols = [
                name for name in heatmap_df.columns
                if name not in parameter_order
            ]
            heatmap_df = heatmap_df.reindex(columns=parameter_order + remaining_cols)
    
        return heatmap_df
    
    
    def _build_derived_uncertainty_heatmap_figure(self, heatmap_df):
        """
        Build the Matplotlib figure for the derived-uncertainty contribution map.
        """
        n_rows = max(int(heatmap_df.shape[0]), 1)
        n_cols = max(int(heatmap_df.shape[1]), 1)
    
        fig_width = max(8.0, 0.55 * n_cols + 3.5)
        fig_height = max(4.8, 0.42 * n_rows + 2.5)
    
        fig = Figure(figsize=(fig_width, fig_height), dpi=110)
        ax = fig.add_subplot(111)
    
        values = heatmap_df.to_numpy(dtype=float)
    
        finite_values = values[np.isfinite(values)]
        if finite_values.size == 0:
            ax.text(
                0.5,
                0.5,
                "No finite uncertainty-contribution map values are available.",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            ax.set_axis_off()
            fig.tight_layout()
            return fig
    
        vmax = float(np.max(np.abs(finite_values)))
        if not np.isfinite(vmax) or vmax <= 0:
            vmax = 1.0
    
        image = ax.imshow(
            values,
            aspect="auto",
            cmap="coolwarm",
            vmin=-vmax,
            vmax=vmax,
        )
    
        ax.set_xticks(np.arange(n_cols))
        ax.set_xticklabels(
            [str(c) for c in heatmap_df.columns],
            rotation=60,
            ha="right",
            fontsize=8,
        )
    
        ax.set_yticks(np.arange(n_rows))
        ax.set_yticklabels(
            [str(i) for i in heatmap_df.index],
            fontsize=9,
        )
    
        ax.set_xlabel("Varying fitted parameters")
        ax.set_ylabel("Derived quantities")
        ax.set_title(
            "Derived uncertainty contribution map\n"
            "Signed contribution to propagated variance (%)"
        )
    
        # Annotate cells only when the map is not too dense.
        if values.size <= 180:
            for i in range(n_rows):
                for j in range(n_cols):
                    val = values[i, j]
                    if np.isfinite(val):
                        ax.text(
                            j,
                            i,
                            f"{val:.1f}",
                            ha="center",
                            va="center",
                            fontsize=7,
                        )
    
        colorbar = fig.colorbar(image, ax=ax, fraction=0.035, pad=0.02)
        colorbar.set_label("Signed variance contribution (%)")
    
        fig.tight_layout()
        return fig
    
    
    def _show_derived_uncertainty_heatmap_dialog(self, map_payload):
        """
        Open the uncertainty-contribution heatmap automatically after the user
        computes derived quantities.
        """
        heatmap_df = self._derived_uncertainty_heatmap_dataframe(map_payload)
    
        if heatmap_df.empty:
            return
    
        win = tk.Toplevel(self.root)
        win.title("Derived uncertainty contribution map")
        win.geometry("1250x720")
        win.minsize(850, 520)
    
        frame = ttk.Frame(win, padding=10)
        frame.pack(fill=tk.BOTH, expand=True)
    
        ttk.Label(
            frame,
            text=(
                "This map shows how each varying fitted parameter contributes to "
                "the propagated variance of each derived quantity."
            ),
            wraplength=1180,
        ).pack(anchor="w", pady=(0, 6))
    
        ttk.Label(
            frame,
            text=(
                "Positive values increase the propagated variance. "
                "Negative values indicate covariance cancellation that reduces it."
            ),
            wraplength=1180,
            foreground="#8a4b00",
        ).pack(anchor="w", pady=(0, 8))
    
        warnings_text = " | ".join(map_payload.get("warnings", []))
        if warnings_text:
            ttk.Label(
                frame,
                text=f"Warning: {warnings_text}",
                wraplength=1180,
                foreground="#8a4b00",
            ).pack(anchor="w", pady=(0, 8))
    
        fig = self._build_derived_uncertainty_heatmap_figure(heatmap_df)
    
        canvas = FigureCanvasTkAgg(fig, master=frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    
        btns = ttk.Frame(frame)
        btns.pack(fill=tk.X, pady=(8, 0))
        ttk.Button(btns, text="Close", command=win.destroy).pack(side=tk.RIGHT)
    
        win.transient(self.root)
        win.lift()
        win.focus_force()
    
    def _default_derived_quantity_text(self):
        """Delegate default derived-quantity examples to the backend."""
        return default_derived_quantity_text(self.fit_result)
    
    
    def open_derived_quantities_dialog(self):
        """Open a post-fit popup for user-defined derived quantities."""
        if self.fit_result is None:
            messagebox.showinfo(
                'No fit result',
                'Run fit, Auto pre-fit, or Refine before computing derived quantities.'
            )
            return
    
        win = tk.Toplevel(self.root)
        win.title('Derived quantities')
        win.geometry('1050x720')
        win.minsize(820, 560)
    
        frame = ttk.Frame(win, padding=10)
        frame.pack(fill=tk.BOTH, expand=True)
    
        ttk.Label(
            frame,
            text='Derived quantities from fitted parameters',
            font=('Segoe UI', 11, 'bold')
        ).pack(anchor='w', pady=(0, 6))
    
        ttk.Label(
            frame,
            text=(
                'Write one derived quantity per line as: Name = expression. '
                'Example: Area ratio = p1_amplitude / p2_amplitude'
            ),
            wraplength=980
        ).pack(anchor='w', pady=(0, 8))
    
        main = ttk.Panedwindow(frame, orient=tk.HORIZONTAL)
        main.pack(fill=tk.BOTH, expand=True)
    
        left = ttk.Frame(main, padding=(0, 0, 8, 0))
        right = ttk.Frame(main)
        main.add(left, weight=1)
        main.add(right, weight=2)
    
        ttk.Label(left, text='Available fitted parameters').pack(anchor='w')
    
        param_text = tk.Text(left, wrap='none', width=36, height=28)
        param_text.pack(fill=tk.BOTH, expand=True, pady=(4, 0))
    
        for pname in self.fit_result.params.keys():
            par = self.fit_result.params[pname]
            try:
                val_txt = f"{float(par.value):.8g}"
            except Exception:
                val_txt = str(par.value)
            expr_txt = f"    expr={par.expr}" if getattr(par, 'expr', None) else ''
            param_text.insert(tk.END, f"{pname} = {val_txt}{expr_txt}\n")
    
        param_text.configure(state='disabled')
    
        ttk.Label(right, text='Definitions').pack(anchor='w')
    
        input_text = tk.Text(right, wrap='none', height=10)
        input_text.pack(fill=tk.X, pady=(4, 8))
    
        if self.derived_quantity_definitions:
            existing = '\n'.join(
                f"{d.get('name', '')} = {d.get('expression', '')}"
                for d in self.derived_quantity_definitions
            )
            input_text.insert('1.0', existing)
        else:
            input_text.insert('1.0', self._default_derived_quantity_text())
    
        ttk.Label(right, text='Results').pack(anchor='w')
        results_text = tk.Text(right, wrap='none', height=18)
        results_text.pack(fill=tk.BOTH, expand=True, pady=(4, 0))
        results_text.configure(state='disabled')
    
        def compute():
            try:
                definitions = self._parse_derived_quantity_lines(input_text.get('1.0', tk.END))
                rows = compute_derived_quantities(self.fit_result, definitions)
                uncertainty_map = compute_derived_uncertainty_contribution_map(
                    self.fit_result,
                    definitions,
                )
                
                self.derived_quantity_definitions = definitions
                self.last_derived_quantities = rows
                self.last_derived_uncertainty_map = uncertainty_map
    
                report = self._format_derived_quantities_report(rows)
                derived_history_id = self._new_session_history_id()
                self._set_report_document(
                    'derived',
                    report,
                    title='Derived quantities report',
                    history_id=derived_history_id,
                )
                self._attach_derived_quantities_to_active_result_package(
                    rows,
                    report,
                    uncertainty_map,
                )
    
                results_text.configure(state='normal')
                results_text.delete('1.0', tk.END)
                results_text.insert('1.0', report)
                results_text.configure(state='disabled')
                
                self._show_derived_uncertainty_heatmap_dialog(uncertainty_map)
    
                self.status_var.set(f'Computed {len(rows)} derived quantity/quantities.')
    
            except Exception as exc:
                messagebox.showerror('Derived quantities error', str(exc), parent=win)
    
        def copy_results():
            text = results_text.get('1.0', tk.END).strip()
            if not text:
                return
            self.root.clipboard_clear()
            self.root.clipboard_append(text)
            self.status_var.set('Derived quantities copied to clipboard.')
    
        btns = ttk.Frame(frame)
        btns.pack(fill=tk.X, pady=(10, 0))
    
        ttk.Button(btns, text='Compute', command=compute).pack(side=tk.LEFT)
        ttk.Button(btns, text='Copy results', command=copy_results).pack(side=tk.LEFT, padx=(6, 0))
        ttk.Button(btns, text='Close', command=win.destroy).pack(side=tk.RIGHT)
    
        win.transient(self.root)
        win.lift()
        win.focus_force()
    
    def _weights_from_mode(self, y, mode):
        """Calculate weights from a saved weighting mode without reading Tk variables."""
        return compute_weights(y, mode)
    
    
    def _split_batch_patterns(self, pattern_text):
        """Delegate batch pattern splitting to the backend."""
        return split_batch_patterns(pattern_text)
    
    
    def _collect_batch_files(self, input_folder, pattern_text, recursive=False):
        """Delegate batch file collection to the backend."""
        return collect_batch_files(input_folder, pattern_text, recursive=recursive)
    
    
    def _collect_batch_template(self, batch_mode):
        """
        Collect current GUI/session settings into a thread-safe batch template.
    
        This is done before the worker thread starts so the worker does not need
        to read Tk variables while fitting.
        """
        peak_defs = self._collect_peak_defs(require_active_peak=False)
        background_kind = self.background_var.get()
    
        if not peak_defs and background_kind.strip().lower() == 'none':
            raise ValueError('Batch fitting needs at least one active peak unless Background is not none.')
    
        active_rows = [row for row in self.peak_rows if row['active'].get()]
        all_custom_no_center = (
            len(active_rows) > 0 and
            all(
                row['kind'].get() == 'Custom' and not self._custom_profile_has_center(row)
                for row in active_rows
            )
        )
    
        return {
            'loader': {
                'delimiter': self._resolve_delimiter(),
                'skiprows': int(self.skiprows_var.get()),
                'x_col': int(self.xcol_var.get()),
                'y_col': int(self.ycol_var.get()),
            },
            'roi_min': float(self.roi_min_var.get()),
            'roi_max': float(self.roi_max_var.get()),
            'peak_defs': copy.deepcopy(peak_defs),
            'background_kind': background_kind,
            'poly_order': int(self.poly_order_var.get()),
            'custom_profiles': copy.deepcopy(self.custom_profiles),
            'custom_background_profiles': copy.deepcopy(self.custom_background_profiles),
            'custom_background_profile_name': self.background_profile_var.get(),
            'background_params': copy.deepcopy(self._collect_background_params()),
            'auxiliary_parameters': copy.deepcopy(self.auxiliary_parameter_definitions),
            'parameter_constraints': copy.deepcopy(self.parameter_constraint_definitions),
            'weighting': self.weighting_var.get(),
            'criterion': self._current_fit_criterion(),
            'n_trials': max(1, int(self.autofit_trials_var.get())),
            'autoprefit_sampling_mode': self._current_autoprefit_sampling_mode(),
            'max_nfev': self._max_nfev(),
            'optimizer_mode': self._current_optimizer_mode(),
            'random_seed': self._current_random_seed(),
            'batch_mode': batch_mode,
            'all_custom_no_center': all_custom_no_center,
            'session_state': self.get_session_state(),
        }
    
    
    def _batch_context_for_file(self, filepath, template):
        """Delegate per-file batch context creation to the backend."""
        return batch_context_for_file(filepath, template)
    
    
    def _copy_fit_result_values_into_params(self, params, result):
        """Delegate fitted-value transfer into new Parameters to the backend."""
        return copy_fit_result_values_into_params(params, result)
    
    
    def _batch_fit_one_file(self, filepath, template, cancel_event):
        """Delegate one-file batch fitting to the backend."""
        return batch_fit_one_file(filepath, template, cancel_event)
    
    
    def _batch_result_to_row(self, filepath, result):
        """Delegate batch success-row generation to the backend."""
        return batch_result_to_row(filepath, result)
    
    
    def _batch_failed_row(self, filepath, error):
        """Delegate batch failure-row generation to the backend."""
        return batch_failed_row(filepath, error)
    
    
    def _write_batch_outputs(self, output_folder, rows, template):
        """Delegate batch summary-file writing to the backend."""
        return write_batch_outputs(output_folder, rows, template)
        #try:
            #return str(value)
        #except Exception:
            #return ''
    
    def open_batch_fit_dialog(self):
        """Open a dialog for batch fitting a folder using the current FitED setup."""
        if self.x_full is None or self.y_full is None:
            messagebox.showinfo(
                'No template spectrum',
                'Load and configure one representative spectrum before batch fitting.'
            )
            return
    
        win = tk.Toplevel(self.root)
        win.title('Batch fit folder')
        win.geometry('760x520')
        win.minsize(680, 460)
    
        frame = ttk.Frame(win, padding=12)
        frame.pack(fill=tk.BOTH, expand=True)
    
        ttk.Label(
            frame,
            text='Batch fit folder',
            font=('Segoe UI', 12, 'bold')
        ).pack(anchor='w', pady=(0, 8))
    
        warning = (
            'Warning: batch fitting applies the current FitED model/settings to every file. '
            'Use it only for spectra that reasonably share the same peak model, ROI, background, '
            'bounds, and fitting assumptions. Inspect representative fits before trusting the full table.'
        )
        ttk.Label(
            frame,
            text=warning,
            wraplength=700,
            foreground='#8a4b00'
        ).pack(anchor='w', pady=(0, 12))
    
        form = ttk.Frame(frame)
        form.pack(fill=tk.X)
    
        input_folder_var = tk.StringVar(value='')
        output_folder_var = tk.StringVar(value='')
        pattern_var = tk.StringVar(value='*.txt;*.csv;*.dat;*.asc')
        recursive_var = tk.BooleanVar(value=False)
    
        def choose_input():
            folder = filedialog.askdirectory(title='Choose input folder')
            if folder:
                input_folder_var.set(folder)
                if not output_folder_var.get():
                    output_folder_var.set(str(Path(folder) / 'FitED_batch_results'))
    
        def choose_output():
            folder = filedialog.askdirectory(title='Choose output folder')
            if folder:
                output_folder_var.set(folder)
    
        ttk.Label(form, text='Input folder').grid(row=0, column=0, sticky='w', pady=4)
        ttk.Entry(form, textvariable=input_folder_var, width=58).grid(row=0, column=1, sticky='ew', padx=(6, 6), pady=4)
        ttk.Button(form, text='Browse', command=choose_input).grid(row=0, column=2, sticky='ew', pady=4)
    
        ttk.Label(form, text='Output folder').grid(row=1, column=0, sticky='w', pady=4)
        ttk.Entry(form, textvariable=output_folder_var, width=58).grid(row=1, column=1, sticky='ew', padx=(6, 6), pady=4)
        ttk.Button(form, text='Browse', command=choose_output).grid(row=1, column=2, sticky='ew', pady=4)
    
        ttk.Label(form, text='File patterns').grid(row=2, column=0, sticky='w', pady=4)
        ttk.Entry(form, textvariable=pattern_var, width=58).grid(row=2, column=1, sticky='ew', padx=(6, 6), pady=4)
        ttk.Checkbutton(form, text='Recursive', variable=recursive_var).grid(row=2, column=2, sticky='w', pady=4)
    
        ttk.Label(form, text='Batch mode').grid(row=3, column=0, sticky='w', pady=4)
        ttk.Combobox(
            form,
            textvariable=self.batch_fit_mode_var,
            values=BATCH_FIT_MODES,
            state='readonly',
            width=36
        ).grid(row=3, column=1, sticky='w', padx=(6, 6), pady=4)
    
        form.columnconfigure(1, weight=1)
    
        explanation = (
            'Mode 1 is fastest and uses the current peak-table values as starting values for every file.\n'
            'Mode 2 is slower and runs Auto pre-fit for each file before the final fit. '
            'Find peaks is not re-run during batch mode, so the parameter columns remain consistent.'
        )
        ttk.Label(
            frame,
            text=explanation,
            wraplength=700
        ).pack(anchor='w', pady=(12, 8))
    
        btns = ttk.Frame(frame)
        btns.pack(fill=tk.X, pady=(12, 0))
    
        def start_batch():
            input_folder = input_folder_var.get().strip()
            output_folder = output_folder_var.get().strip()
    
            if not input_folder:
                messagebox.showerror('Batch fit error', 'Choose an input folder.', parent=win)
                return
            if not output_folder:
                messagebox.showerror('Batch fit error', 'Choose an output folder.', parent=win)
                return
    
            files = self._collect_batch_files(
                input_folder,
                pattern_var.get(),
                recursive=bool(recursive_var.get())
            )
    
            if not files:
                messagebox.showerror(
                    'Batch fit error',
                    'No files were found with the selected patterns.',
                    parent=win
                )
                return
    
            confirmed = messagebox.askyesno(
                'Confirm batch fit',
                (
                    f'FitED will fit {len(files)} file(s) using the current model/settings.\n\n'
                    'This can take time. Failed files will be recorded in the summary table.\n\n'
                    'Continue?'
                ),
                parent=win
            )
    
            if not confirmed:
                return
    
            try:
                template = self._collect_batch_template(self.batch_fit_mode_var.get())
            except Exception as exc:
                messagebox.showerror('Batch template error', str(exc), parent=win)
                return
    
            payload = {
                'files': files,
                'output_folder': Path(output_folder),
                'template': template,
            }
    
            win.destroy()
            self._start_batch_fit_background(payload)
    
        ttk.Button(btns, text='Start batch', command=start_batch).pack(side=tk.LEFT)
        ttk.Button(btns, text='Close', command=win.destroy).pack(side=tk.RIGHT)
    
        win.transient(self.root)
        win.lift()
        win.focus_force()
    
    
    def _start_batch_fit_background(self, payload):
        """Start batch fitting in the existing worker-thread system."""
        files = payload['files']
        output_folder = payload['output_folder']
        template = payload['template']
    
        def _worker(cancel_event, progress_queue):
            rows = []
            total = len(files)
    
            for idx, filepath in enumerate(files, start=1):
                self._raise_if_cancelled(cancel_event)
    
                progress_queue.put((
                    'progress',
                    (idx - 1, total, f'Batch fit: {idx}/{total} — {Path(filepath).name}'),
                    None
                ))
    
                try:
                    result = self._batch_fit_one_file(filepath, template, cancel_event)
                    row = self._batch_result_to_row(filepath, result)
                except Exception as exc:
                    row = self._batch_failed_row(filepath, exc)
    
                rows.append(row)
    
            progress_queue.put((
                'progress',
                (total, total, 'Batch fit: writing summary files...'),
                None
            ))
    
            output_info = self._write_batch_outputs(output_folder, rows, template)
    
            return {
                'rows': rows,
                'output_info': output_info,
                'output_folder': output_folder,
            }
    
        self._start_fit_worker(
            'Batch fit',
            _worker,
            self._display_batch_fit_payload
        )
    
    
    def _display_batch_fit_payload(self, payload):
        """Show completion message after batch fitting."""
        output_info = payload.get('output_info', {})
        output_folder = payload.get('output_folder', '')
    
        n_rows = output_info.get('n_rows', 0)
        n_ok = output_info.get('n_ok', 0)
        n_failed = output_info.get('n_failed', 0)
    
        self.status_var.set(
            f'Batch fit complete: {n_ok} succeeded, {n_failed} failed. '
            f'Results saved to {output_folder}'
        )
    
        message = (
            f'Batch fit complete.\n\n'
            f'Files processed: {n_rows}\n'
            f'Succeeded: {n_ok}\n'
            f'Failed: {n_failed}\n\n'
            f'Output folder:\n{output_folder}\n\n'
            f'Files:\n'
            f'- batch_summary.csv\n'
        )
    
        if output_info.get('xlsx_path') is not None:
            message += '- batch_summary.xlsx\n'
    
        message += '- batch_template_session.json\n'
    
        messagebox.showinfo('Batch fit complete', message)

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
            peak_defs = self._collect_peak_defs(require_active_peak=False)
            if not peak_defs and self.background_var.get().strip().lower() == 'none':
                raise ValueError('At least one active peak is required when Background is none.')
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
                auxiliary_parameters=self.auxiliary_parameter_definitions,
                parameter_constraints=self.parameter_constraint_definitions,
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

                derived_df = None
                derived_csv_path = None
                if self.last_derived_quantities:
                    derived_df = pd.DataFrame(self.last_derived_quantities)
                    derived_csv_path = tmpdir / f'{base}_derived_quantities.csv'
                    derived_df.to_csv(derived_csv_path, index=False)
                confidence_ellipse_paths = self._write_confidence_ellipse_exports(
                    tmpdir,
                    base,
                    getattr(self, "last_confidence_ellipse_payloads", []),
                )

                derived_uncertainty_map_df = self._derived_uncertainty_heatmap_dataframe(
                    self.last_derived_uncertainty_map
                )

                if derived_uncertainty_map_df.empty:
                    derived_uncertainty_map_df = None

                derived_map_csv_path = None
                derived_map_png_path = None

                if derived_uncertainty_map_df is not None:
                    map_export_df = derived_uncertainty_map_df.copy()
                    map_export_df.index.name = 'derived_quantity'

                    derived_map_csv_path = tmpdir / f'{base}_derived_uncertainty_map.csv'
                    map_export_df.to_csv(derived_map_csv_path)

                    derived_map_png_path = tmpdir / f'{base}_derived_uncertainty_map.png'
                    map_fig = self._build_derived_uncertainty_heatmap_figure(map_export_df)
                    map_fig.savefig(
                        derived_map_png_path,
                        dpi=220,
                        bbox_inches='tight',
                    )

                correlation_csv_path = None
                correlation_png_path = None
                correlation_matrix_df = None
                try:
                    correlation_payload = compute_parameter_correlation_matrix(self.fit_result)
                    correlation_matrix_df = self._parameter_correlation_dataframe(correlation_payload)
                    if not correlation_matrix_df.empty:
                        corr_export_df = correlation_matrix_df.copy()
                        corr_export_df.index.name = 'parameter'
                        correlation_csv_path = tmpdir / f'{base}_parameter_correlation_matrix.csv'
                        corr_export_df.to_csv(correlation_csv_path)

                        correlation_png_path = tmpdir / f'{base}_parameter_correlation_heatmap.png'
                        corr_fig = self._build_parameter_correlation_heatmap_figure(corr_export_df)
                        corr_fig.savefig(correlation_png_path, dpi=220, bbox_inches='tight')
                except Exception:
                    correlation_matrix_df = None

                residual_csv_path = None
                residual_autocorr_csv_path = None
                residual_qq_csv_path = None
                residual_summary_path = None
                residual_png_path = None
                residual_df = autocorr_df = qq_df = residual_summary_df = None
                try:
                    residual_payload = compute_residual_diagnostics(
                        x,
                        y,
                        self.last_best_fit,
                        weights=self._weights(y),
                    )
                    residual_df, autocorr_df, qq_df, residual_summary_df = self._residual_diagnostics_to_dataframes(residual_payload)

                    if residual_df is not None and not residual_df.empty:
                        residual_csv_path = tmpdir / f'{base}_residual_diagnostics.csv'
                        residual_df.to_csv(residual_csv_path, index=False)

                    if autocorr_df is not None and not autocorr_df.empty:
                        residual_autocorr_csv_path = tmpdir / f'{base}_residual_autocorrelation.csv'
                        autocorr_df.to_csv(residual_autocorr_csv_path, index=False)

                    if qq_df is not None and not qq_df.empty:
                        residual_qq_csv_path = tmpdir / f'{base}_residual_qq_plot_data.csv'
                        qq_df.to_csv(residual_qq_csv_path, index=False)

                    residual_summary_path = tmpdir / f'{base}_residual_diagnostics_summary.txt'
                    residual_summary_path.write_text(
                        self._format_residual_diagnostics_report(residual_payload),
                        encoding='utf-8',
                    )

                    residual_png_path = tmpdir / f'{base}_residual_diagnostics.png'
                    residual_fig = self._build_residual_diagnostics_figure(residual_payload)
                    residual_fig.savefig(residual_png_path, dpi=220, bbox_inches='tight')
                except Exception:
                    pass
                
                confidence_ellipse_summary_df = None
                try:
                    ellipse_rows = confidence_ellipse_pair_summary(self.fit_result)
                    confidence_ellipse_summary_df = pd.DataFrame(ellipse_rows)
                    if confidence_ellipse_summary_df.empty:
                        confidence_ellipse_summary_df = None
                except Exception:
                    confidence_ellipse_summary_df = None
            

                report_path = tmpdir / f'{base}_fit_report.txt'
                report_path.write_text(self._full_fited_fit_report(self.fit_result), encoding='utf-8')

                session_path = tmpdir / f'{base}_session.json'
                session_path.write_text(json.dumps(self.get_session_state(), indent=2), encoding='utf-8')

                meta_df = pd.DataFrame([
                    {'field': 'source_file', 'value': str(self.current_file) if self.current_file else ''},
                    {'field': 'background', 'value': self.background_var.get()},
                    {'field': 'poly_order', 'value': int(self.poly_order_var.get())},
                    {'field': 'weighting', 'value': self.weighting_var.get()},
                    {'field': 'fit_criterion', 'value': self._current_fit_criterion()},
                    {'field': 'optimizer_mode', 'value': self._current_optimizer_mode()},
                    {'field': 'random_seed', 'value': getattr(self.fit_result, 'fited_random_seed', '')},
                    {'field': 'selected_optimizer_candidate', 'value': getattr(self.fit_result, 'fited_selected_candidate', '')},
                    {'field': 'optimizer_candidate_scores', 'value': json.dumps(getattr(self.fit_result, 'fited_candidate_scores', {}))},
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
                        if derived_df is not None:
                            derived_df.to_excel(writer, sheet_name='derived_quantities', index=False)
                        if derived_uncertainty_map_df is not None:
                            map_excel_df = derived_uncertainty_map_df.copy()
                            map_excel_df.index.name = 'derived_quantity'
                            map_excel_df.to_excel(
                                writer,
                                sheet_name='derived_uncertainty_map',
                                index=True,
                            )
                        if correlation_matrix_df is not None and not correlation_matrix_df.empty:
                            corr_excel_df = correlation_matrix_df.copy()
                            corr_excel_df.index.name = 'parameter'
                            corr_excel_df.to_excel(
                                writer,
                                sheet_name='parameter_correlations',
                                index=True,
                            )
                        if confidence_ellipse_summary_df is not None and not confidence_ellipse_summary_df.empty:
                            confidence_ellipse_summary_df.to_excel(writer, sheet_name='confidence_ellipses', index=False)
                        if residual_df is not None and not residual_df.empty:
                            residual_df.to_excel(writer, sheet_name='residuals', index=False)
                        if autocorr_df is not None and not autocorr_df.empty:
                            autocorr_df.to_excel(writer, sheet_name='residual_autocorr', index=False)
                        if qq_df is not None and not qq_df.empty:
                            qq_df.to_excel(writer, sheet_name='residual_qq', index=False)
                        if residual_summary_df is not None and not residual_summary_df.empty:
                            residual_summary_df.to_excel(writer, sheet_name='residual_summary', index=False)
                        
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
                    if derived_csv_path is not None and derived_csv_path.exists():
                        zf.write(derived_csv_path, derived_csv_path.name)
                    for path in confidence_ellipse_paths:
                        if path is not None and path.exists():
                            zf.write(path, path.name)
                    if derived_map_csv_path is not None and derived_map_csv_path.exists():
                        zf.write(derived_map_csv_path, derived_map_csv_path.name)
                    if derived_map_png_path is not None and derived_map_png_path.exists():
                        zf.write(derived_map_png_path, derived_map_png_path.name)
                    for path in [
                        correlation_csv_path,
                        correlation_png_path,
                        residual_csv_path,
                        residual_autocorr_csv_path,
                        residual_qq_csv_path,
                        residual_summary_path,
                        residual_png_path,
                    ]:
                        if path is not None and path.exists():
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



# ============================================================================
# FitED Decay / IRF GUI extension
# This block adds a parallel time-domain fitting route without modifying the
# original GUI method bodies above. It wraps selected methods so standard peak
# fitting continues through the original code path, while Decay / IRF fitting
# enters the new backend model route.
# ============================================================================

import FitED_Backend_v4 as _fited_irf_backend

_FITED_ORIG_build_state = DesktopPLFitterApp._build_state
_FITED_ORIG_build_ui = DesktopPLFitterApp._build_ui
_FITED_ORIG_prepare_fit_context = DesktopPLFitterApp._prepare_fit_context
_FITED_ORIG_preview = DesktopPLFitterApp.preview
_FITED_ORIG_display_run_fit_payload = DesktopPLFitterApp._display_run_fit_payload
_FITED_ORIG_display_autofit_payload = DesktopPLFitterApp._display_autofit_payload
_FITED_ORIG_display_refine_payload = DesktopPLFitterApp._display_refine_payload
_FITED_ORIG_refine_with_added_peaks_background = DesktopPLFitterApp.refine_with_added_peaks_background
_FITED_ORIG_get_session_state = DesktopPLFitterApp.get_session_state
_FITED_ORIG_apply_session_state = DesktopPLFitterApp.apply_session_state
_FITED_ORIG_collect_batch_template = DesktopPLFitterApp._collect_batch_template


def _fited_irf_is_active(self):
    return str(getattr(self, 'analysis_mode_var', tk.StringVar(value=_fited_irf_backend.DECAY_IRF_STANDARD_MODE)).get()).strip() == _fited_irf_backend.DECAY_IRF_ANALYSIS_MODE


def _fited_parse_float(text, field_name='value'):
    raw = str(text).strip()
    low = raw.lower()
    if low in {'inf', '+inf', 'infinity', '+infinity'}:
        return float('inf')
    if low in {'-inf', '-infinity'}:
        return float('-inf')
    try:
        return float(raw)
    except Exception as exc:
        raise ValueError(f"Could not parse {field_name}: {raw}") from exc


def _fited_irf_default_param_table():
    return {
        'baseline': {'value': '0.0', 'min': '-inf', 'max': 'inf', 'vary': True},
        't0': {'value': '0.0', 'min': '-inf', 'max': 'inf', 'vary': True},
        'irf_shift': {'value': '0.0', 'min': '-inf', 'max': 'inf', 'vary': True},
        'A1': {'value': '1.0', 'min': '-inf', 'max': 'inf', 'vary': True},
        'tau1': {'value': '1.0', 'min': '1e-12', 'max': 'inf', 'vary': True},
        'A2': {'value': '0.3', 'min': '-inf', 'max': 'inf', 'vary': False},
        'tau2': {'value': '5.0', 'min': '1e-12', 'max': 'inf', 'vary': False},
        'A3': {'value': '0.1', 'min': '-inf', 'max': 'inf', 'vary': False},
        'tau3': {'value': '20.0', 'min': '1e-12', 'max': 'inf', 'vary': False},
        'beta': {'value': '0.75', 'min': '0.05', 'max': '2.0', 'vary': False},
        'tau_rise': {'value': '0.1', 'min': '1e-12', 'max': 'inf', 'vary': False},
        'tau_decay': {'value': '1.0', 'min': '1e-12', 'max': 'inf', 'vary': False},
    }


def _fited_irf_build_state(self):
    _FITED_ORIG_build_state(self)
    self.analysis_mode_var = tk.StringVar(value=_fited_irf_backend.DECAY_IRF_STANDARD_MODE)
    self.decay_data_type_var = tk.StringVar(value=_fited_irf_backend.DECAY_IRF_DATA_TYPES[0])
    self.decay_model_var = tk.StringVar(value=_fited_irf_backend.DECAY_IRF_DECAY_KINDS[0])
    self.decay_irf_mode_var = tk.StringVar(value=_fited_irf_backend.DECAY_IRF_MODES[0])
    self.decay_signal_sign_var = tk.StringVar(value=_fited_irf_backend.DECAY_IRF_SIGNAL_SIGNS[1])
    self.irf_delimiter_var = tk.StringVar(value='tab')
    self.irf_skiprows_var = tk.IntVar(value=0)
    self.irf_xcol_var = tk.IntVar(value=0)
    self.irf_ycol_var = tk.IntVar(value=1)
    self.irf_baseline_mode_var = tk.StringVar(value='edge median')
    self.irf_zero_mode_var = tk.StringVar(value='peak maximum')
    self.irf_clip_negative_var = tk.BooleanVar(value=True)
    self.gaussian_irf_fwhm_var = tk.StringVar(value='0.1')
    self.irf_current_file = None
    self.irf_x_full = None
    self.irf_y_full = None
    self.decay_param_vars = {}
    for name, cfg in _fited_irf_default_param_table().items():
        self.decay_param_vars[name] = {
            'value': tk.StringVar(value=str(cfg['value'])),
            'min': tk.StringVar(value=str(cfg['min'])),
            'max': tk.StringVar(value=str(cfg['max'])),
            'vary': tk.BooleanVar(value=bool(cfg['vary'])),
        }


def _fited_irf_resolve_delimiter_from_var(var):
    value = str(var.get()).strip().lower()
    if value == 'tab':
        return '\t'
    if value == 'comma':
        return ','
    if value == 'semicolon':
        return ';'
    if value == 'space':
        return r'\s+'
    return None


def _fited_irf_load_irf_from_path(self, path):
    delimiter = _fited_irf_resolve_delimiter_from_var(self.irf_delimiter_var)
    x, y = _fited_irf_backend.load_irf_file(
        path,
        x_col=int(self.irf_xcol_var.get()),
        y_col=int(self.irf_ycol_var.get()),
        delimiter=delimiter,
        skiprows=int(self.irf_skiprows_var.get()),
    )
    self.irf_current_file = Path(path)
    self.irf_x_full = np.asarray(x, dtype=float)
    self.irf_y_full = np.asarray(y, dtype=float)
    if hasattr(self, 'irf_file_label'):
        self.irf_file_label.configure(text=self.irf_current_file.name)
    self.status_var.set(f'IRF loaded: {self.irf_current_file.name} ({len(x)} points).')


def _fited_irf_open_irf_file(self):
    path = filedialog.askopenfilename(
        title='Open measured IRF file',
        filetypes=[
            ('Text data', '*.txt *.csv *.dat *.asc'),
            ('All files', '*.*'),
        ],
    )
    if not path:
        return
    try:
        _fited_irf_load_irf_from_path(self, path)
    except Exception as exc:
        messagebox.showerror('IRF load error', str(exc))


def _fited_irf_reload_irf_file(self):
    if self.irf_current_file is None:
        messagebox.showinfo('No IRF file', 'Load an IRF file first.')
        return
    try:
        _fited_irf_load_irf_from_path(self, self.irf_current_file)
    except Exception as exc:
        messagebox.showerror('IRF reload error', str(exc))


def _fited_irf_collect_config(self):
    params = {}
    for name, vars_dict in self.decay_param_vars.items():
        params[name] = {
            'value': _fited_parse_float(vars_dict['value'].get(), f'{name} value'),
            'min': _fited_parse_float(vars_dict['min'].get(), f'{name} min'),
            'max': _fited_parse_float(vars_dict['max'].get(), f'{name} max'),
            'vary': bool(vars_dict['vary'].get()),
        }

    cfg = {
        'data_type': self.decay_data_type_var.get(),
        'decay_kind': self.decay_model_var.get(),
        'irf_mode': self.decay_irf_mode_var.get(),
        'signal_sign': self.decay_signal_sign_var.get(),
        'gaussian_irf_fwhm': _fited_parse_float(self.gaussian_irf_fwhm_var.get(), 'Gaussian IRF FWHM'),
        'parameters': params,
        'irf_file': str(self.irf_current_file) if self.irf_current_file is not None else '',
        'irf_loader': {
            'delimiter': self.irf_delimiter_var.get(),
            'skiprows': int(self.irf_skiprows_var.get()),
            'x_col': int(self.irf_xcol_var.get()),
            'y_col': int(self.irf_ycol_var.get()),
        },
        'irf_preprocess': {
            'baseline_mode': self.irf_baseline_mode_var.get(),
            'zero_mode': self.irf_zero_mode_var.get(),
            'clip_negative': bool(self.irf_clip_negative_var.get()),
        },
    }

    if self.decay_irf_mode_var.get() == 'Measured IRF reconvolution':
        if self.irf_x_full is None or self.irf_y_full is None:
            raise ValueError('Measured IRF reconvolution requires an uploaded IRF file.')
        cfg['prepared_irf'] = _fited_irf_backend.prepare_irf_kernel(
            self.irf_x_full,
            self.irf_y_full,
            baseline_mode=self.irf_baseline_mode_var.get(),
            zero_mode=self.irf_zero_mode_var.get(),
            clip_negative=bool(self.irf_clip_negative_var.get()),
        )
    return cfg


def _fited_irf_apply_config_to_ui(self, cfg):
    if not isinstance(cfg, dict):
        return
    if cfg.get('data_type') in _fited_irf_backend.DECAY_IRF_DATA_TYPES:
        self.decay_data_type_var.set(cfg.get('data_type'))
    decay_kind = _fited_irf_backend.canonical_decay_irf_decay_kind(cfg.get('decay_kind')) if hasattr(_fited_irf_backend, 'canonical_decay_irf_decay_kind') else cfg.get('decay_kind')
    if decay_kind in _fited_irf_backend.DECAY_IRF_DECAY_KINDS:
        self.decay_model_var.set(decay_kind)
    if cfg.get('irf_mode') in _fited_irf_backend.DECAY_IRF_MODES:
        self.decay_irf_mode_var.set(cfg.get('irf_mode'))
    if cfg.get('signal_sign') in _fited_irf_backend.DECAY_IRF_SIGNAL_SIGNS:
        self.decay_signal_sign_var.set(cfg.get('signal_sign'))
    if 'gaussian_irf_fwhm' in cfg:
        self.gaussian_irf_fwhm_var.set(str(cfg.get('gaussian_irf_fwhm')))

    preprocess = cfg.get('irf_preprocess', {}) if isinstance(cfg.get('irf_preprocess', {}), dict) else {}
    if preprocess.get('baseline_mode') in _fited_irf_backend.DECAY_IRF_BASELINE_MODES:
        self.irf_baseline_mode_var.set(preprocess.get('baseline_mode'))
    if preprocess.get('zero_mode') in _fited_irf_backend.DECAY_IRF_ZERO_MODES:
        self.irf_zero_mode_var.set(preprocess.get('zero_mode'))
    if 'clip_negative' in preprocess:
        self.irf_clip_negative_var.set(bool(preprocess.get('clip_negative')))

    loader = cfg.get('irf_loader', {}) if isinstance(cfg.get('irf_loader', {}), dict) else {}
    if loader:
        self.irf_delimiter_var.set(loader.get('delimiter', self.irf_delimiter_var.get()))
        self.irf_skiprows_var.set(loader.get('skiprows', self.irf_skiprows_var.get()))
        self.irf_xcol_var.set(loader.get('x_col', self.irf_xcol_var.get()))
        self.irf_ycol_var.set(loader.get('y_col', self.irf_ycol_var.get()))

    for name, pcfg in (cfg.get('parameters', {}) or {}).items():
        if name in self.decay_param_vars and isinstance(pcfg, dict):
            self.decay_param_vars[name]['value'].set(str(pcfg.get('value', self.decay_param_vars[name]['value'].get())))
            self.decay_param_vars[name]['min'].set(str(pcfg.get('min', self.decay_param_vars[name]['min'].get())))
            self.decay_param_vars[name]['max'].set(str(pcfg.get('max', self.decay_param_vars[name]['max'].get())))
            self.decay_param_vars[name]['vary'].set(bool(pcfg.get('vary', self.decay_param_vars[name]['vary'].get())))

    irf_file = str(cfg.get('irf_file', '')).strip()
    if irf_file:
        maybe = Path(irf_file)
        if maybe.exists():
            try:
                _fited_irf_load_irf_from_path(self, maybe)
            except Exception:
                pass


def _fited_irf_apply_fit_params_to_ui(self, result):
    if result is None:
        return
    for name, vars_dict in self.decay_param_vars.items():
        if name in result.params:
            par = result.params[name]
            vars_dict['value'].set(f"{float(par.value):.12g}")
            if np.isfinite(par.min):
                vars_dict['min'].set(f"{float(par.min):.12g}")
            if np.isfinite(par.max):
                vars_dict['max'].set(f"{float(par.max):.12g}")
            vars_dict['vary'].set(bool(getattr(par, 'vary', False)))


def _fited_irf_seed_from_loaded_data(self):
    if self.x_full is None or self.y_full is None:
        messagebox.showinfo('No data', 'Load a decay/kinetic trace first.')
        return
    try:
        x, y_raw, y_plot = self._get_roi_data()
        cfg = _fited_irf_collect_config(self) if self.decay_irf_mode_var.get() != 'Measured IRF reconvolution' or self.irf_x_full is not None else {
            'decay_kind': self.decay_model_var.get(),
            'irf_mode': self.decay_irf_mode_var.get(),
            'signal_sign': self.decay_signal_sign_var.get(),
        }
        cfg = _fited_irf_backend.seed_decay_irf_config_from_data(x, y_raw, cfg)
        _fited_irf_apply_config_to_ui(self, cfg)
        self.status_var.set('Decay / IRF parameter guesses were seeded from the current ROI.')
    except Exception as exc:
        messagebox.showerror('Decay / IRF seed error', str(exc))


def _fited_irf_build_tab(self, parent):
    parent.configure(padding=10)

    mode_box = ttk.LabelFrame(parent, text='Fit mode used by Actions', padding=8)
    mode_box.pack(fill=tk.X, pady=(0, 8))
    ttk.Radiobutton(mode_box, text='Peak/profile mode (use Peaks tab)', variable=self.analysis_mode_var, value=_fited_irf_backend.DECAY_IRF_STANDARD_MODE).pack(anchor='w')
    ttk.Radiobutton(mode_box, text='Decay/IRF mode (use this tab)', variable=self.analysis_mode_var, value=_fited_irf_backend.DECAY_IRF_ANALYSIS_MODE).pack(anchor='w')

    model_box = ttk.LabelFrame(parent, text='Decay model', padding=8)
    model_box.pack(fill=tk.X, pady=(0, 8))
    ttk.Label(model_box, text='Data type').grid(row=0, column=0, sticky='w', pady=3)
    ttk.Combobox(model_box, textvariable=self.decay_data_type_var, values=_fited_irf_backend.DECAY_IRF_DATA_TYPES, state='readonly', width=28).grid(row=0, column=1, sticky='ew', padx=(8, 0), pady=3)
    ttk.Label(model_box, text='Decay function').grid(row=1, column=0, sticky='w', pady=3)
    ttk.Combobox(model_box, textvariable=self.decay_model_var, values=_fited_irf_backend.DECAY_IRF_DECAY_KINDS, state='readonly', width=28).grid(row=1, column=1, sticky='ew', padx=(8, 0), pady=3)
    ttk.Label(model_box, text='Signal sign').grid(row=2, column=0, sticky='w', pady=3)
    ttk.Combobox(model_box, textvariable=self.decay_signal_sign_var, values=_fited_irf_backend.DECAY_IRF_SIGNAL_SIGNS, state='readonly', width=28).grid(row=2, column=1, sticky='ew', padx=(8, 0), pady=3)
    ttk.Label(model_box, text='IRF treatment').grid(row=3, column=0, sticky='w', pady=3)
    ttk.Combobox(model_box, textvariable=self.decay_irf_mode_var, values=_fited_irf_backend.DECAY_IRF_MODES, state='readonly', width=28).grid(row=3, column=1, sticky='ew', padx=(8, 0), pady=3)
    ttk.Label(model_box, text='Gaussian IRF FWHM').grid(row=4, column=0, sticky='w', pady=3)
    ttk.Entry(model_box, textvariable=self.gaussian_irf_fwhm_var, width=14).grid(row=4, column=1, sticky='w', padx=(8, 0), pady=3)
    model_box.grid_columnconfigure(1, weight=1)

    irf_box = ttk.LabelFrame(parent, text='Measured IRF file and preprocessing', padding=8)
    irf_box.pack(fill=tk.X, pady=(0, 8))
    ttk.Button(irf_box, text='Open IRF file', command=lambda: _fited_irf_open_irf_file(self)).grid(row=0, column=0, sticky='ew', pady=3)
    ttk.Button(irf_box, text='Reload IRF', command=lambda: _fited_irf_reload_irf_file(self)).grid(row=0, column=1, sticky='ew', padx=(6, 0), pady=3)
    self.irf_file_label = ttk.Label(irf_box, text='No IRF file selected', width=34)
    self.irf_file_label.grid(row=0, column=2, columnspan=2, sticky='w', padx=(8, 0), pady=3)
    ttk.Label(irf_box, text='Delimiter').grid(row=1, column=0, sticky='w', pady=3)
    ttk.Combobox(irf_box, textvariable=self.irf_delimiter_var, values=['tab', 'comma', 'semicolon', 'space', 'auto'], state='readonly', width=10).grid(row=1, column=1, sticky='w', pady=3)
    ttk.Label(irf_box, text='Skip rows').grid(row=1, column=2, sticky='e', pady=3)
    ttk.Spinbox(irf_box, from_=0, to=1000, textvariable=self.irf_skiprows_var, width=8).grid(row=1, column=3, sticky='w', padx=(6, 0), pady=3)
    ttk.Label(irf_box, text='X col').grid(row=2, column=0, sticky='w', pady=3)
    ttk.Spinbox(irf_box, from_=0, to=20, textvariable=self.irf_xcol_var, width=8).grid(row=2, column=1, sticky='w', pady=3)
    ttk.Label(irf_box, text='Y col').grid(row=2, column=2, sticky='e', pady=3)
    ttk.Spinbox(irf_box, from_=0, to=20, textvariable=self.irf_ycol_var, width=8).grid(row=2, column=3, sticky='w', padx=(6, 0), pady=3)
    ttk.Label(irf_box, text='IRF baseline').grid(row=3, column=0, sticky='w', pady=3)
    ttk.Combobox(irf_box, textvariable=self.irf_baseline_mode_var, values=_fited_irf_backend.DECAY_IRF_BASELINE_MODES, state='readonly', width=14).grid(row=3, column=1, sticky='w', pady=3)
    ttk.Label(irf_box, text='IRF zero').grid(row=3, column=2, sticky='e', pady=3)
    ttk.Combobox(irf_box, textvariable=self.irf_zero_mode_var, values=_fited_irf_backend.DECAY_IRF_ZERO_MODES, state='readonly', width=18).grid(row=3, column=3, sticky='w', padx=(6, 0), pady=3)
    ttk.Checkbutton(irf_box, text='Clip negative IRF after baseline subtraction', variable=self.irf_clip_negative_var).grid(row=4, column=0, columnspan=4, sticky='w', pady=(5, 0))

    param_box = ttk.LabelFrame(parent, text='Decay / IRF parameters', padding=8)
    param_box.pack(fill=tk.X, pady=(0, 8))
    ttk.Button(param_box, text='Seed guesses from current ROI', command=lambda: _fited_irf_seed_from_loaded_data(self)).grid(row=0, column=0, columnspan=5, sticky='ew', pady=(0, 6))
    headers = ['Parameter', 'Value', 'Min', 'Max', 'Vary']
    for col, header in enumerate(headers):
        ttk.Label(param_box, text=header, font=('Segoe UI', 9, 'bold')).grid(row=1, column=col, sticky='w', padx=(0, 5), pady=2)
    row_index = 2
    for name in ['baseline', 't0', 'irf_shift', 'A1', 'tau1', 'A2', 'tau2', 'A3', 'tau3', 'beta', 'tau_rise', 'tau_decay']:
        vars_dict = self.decay_param_vars[name]
        ttk.Label(param_box, text=name).grid(row=row_index, column=0, sticky='w', pady=2)
        ttk.Entry(param_box, textvariable=vars_dict['value'], width=12).grid(row=row_index, column=1, sticky='w', padx=(0, 5), pady=2)
        ttk.Entry(param_box, textvariable=vars_dict['min'], width=12).grid(row=row_index, column=2, sticky='w', padx=(0, 5), pady=2)
        ttk.Entry(param_box, textvariable=vars_dict['max'], width=12).grid(row=row_index, column=3, sticky='w', padx=(0, 5), pady=2)
        ttk.Checkbutton(param_box, variable=vars_dict['vary']).grid(row=row_index, column=4, sticky='w', pady=2)
        row_index += 1

    note = (
        'Use the existing Actions tab for Preview, Run fit, Auto pre-fit, stability test, residual diagnostics, '
        'correlation matrix, confidence ellipse, derived quantities, batch fitting, and ZIP export. In Decay/IRF mode, '
        'FitED builds the selected time-domain model in the backend, including optional common-rise models and IRF reconvolution, '
        'then sends it through the same optimizer/diagnostics route.'
    )
    ttk.Label(parent, text=note, wraplength=360, foreground='#444').pack(fill=tk.X, pady=(0, 8))


def _fited_irf_build_ui(self):
    _FITED_ORIG_build_ui(self)
    try:
        decay_tab_wrap = ScrollableFrame(self.workflow_notebook)
        self.workflow_notebook.insert(3, decay_tab_wrap, text='Decay / IRF')
        _fited_irf_build_tab(self, decay_tab_wrap.inner)
    except Exception as exc:
        messagebox.showerror('Decay / IRF UI error', str(exc))


def _fited_irf_prepare_fit_context(self, require_active_peak=True):
    if not _fited_irf_is_active(self):
        return _FITED_ORIG_prepare_fit_context(self, require_active_peak=require_active_peak)

    x, y_raw, y_plot = self._get_roi_data()
    cfg = _fited_irf_collect_config(self)
    weights = self._weights(y_raw)
    return {
        'analysis_mode': _fited_irf_backend.DECAY_IRF_ANALYSIS_MODE,
        'x': x.copy(),
        'y_raw': y_raw.copy(),
        'y_plot': y_plot.copy(),
        'peak_defs': [],
        'background_kind': 'none',
        'poly_order': int(self.poly_order_var.get()),
        'custom_profiles': copy.deepcopy(self.custom_profiles),
        'custom_background_profiles': copy.deepcopy(self.custom_background_profiles),
        'custom_background_profile_name': self.background_profile_var.get(),
        'background_params': copy.deepcopy(self._collect_background_params()),
        'auxiliary_parameters': copy.deepcopy(self.auxiliary_parameter_definitions),
        'parameter_constraints': copy.deepcopy(self.parameter_constraint_definitions),
        'weights': None if weights is None else np.asarray(weights, dtype=float).copy(),
        'criterion': self._current_fit_criterion(),
        'n_trials': max(1, int(self.autofit_trials_var.get())),
        'autoprefit_sampling_mode': self._current_autoprefit_sampling_mode(),
        'max_nfev': self._max_nfev(),
        'optimizer_mode': self._current_optimizer_mode(),
        'random_seed': self._current_random_seed(),
        'active_count': 0,
        'all_custom_no_center': False,
        'decay_irf_config': cfg,
    }


def _fited_irf_plot_decay_result(self, payload, main_label='Best fit', title='Decay / IRF fit'):
    context = payload['context']
    x = context['x']
    y_raw = context['y_raw']
    result = payload['result']
    comps = payload.get('components') or _fited_irf_backend.evaluate_time_irf_components(context, result.params, x=x)
    best = payload.get('best_fit', result.best_fit)

    self.fit_result = result
    self.last_derived_quantities = None
    self.last_derived_uncertainty_map = None
    self.last_confidence_ellipse_payloads = []
    self.last_components = comps
    self.last_best_fit = best
    self.last_roi = (x.copy(), y_raw.copy())
    self.last_fit_peak_count = 0
    _fited_irf_apply_fit_params_to_ui(self, result)

    self.ax_main.clear()
    self.ax_resid.clear()
    self._refresh_hover_axis()
    self.ax_main.plot(x, y_raw, 'k.', ms=3, alpha=0.6, label='Data')
    self.ax_main.plot(x, best, linewidth=2.1, label=main_label)
    for name, comp in comps.items():
        if name == 'reconvolved_decay_model':
            continue
        label = name.replace('_', ' ')
        if name == 'irf_display_scaled':
            label = 'IRF (scaled display)'
        elif name == 'intrinsic_decay_plus_baseline':
            label = 'Intrinsic decay + baseline'
        self.ax_main.plot(x, comp, '--', linewidth=1.0, alpha=0.85, label=label)
    self.ax_main.set_title(title)
    self.ax_main.set_ylabel('Signal')
    self.ax_main.legend(fontsize=8, ncol=2)

    resid = y_raw - best
    self.ax_resid.axhline(0.0, linestyle='--', linewidth=1.0)
    self.ax_resid.plot(x, resid, linewidth=1.0)
    self.ax_resid.set_xlabel('Time')
    self.ax_resid.set_ylabel('Residual')
    self.fig.tight_layout()
    self.canvas.draw_idle()


def _fited_irf_display_run_fit_payload(self, payload):
    if not _fited_irf_backend._decay_irf_context_active(payload.get('context', {})):
        return _FITED_ORIG_display_run_fit_payload(self, payload)
    _fited_irf_plot_decay_result(self, payload, main_label='Reconvolved fit', title='Decay / IRF best fit')
    result = payload['result']
    self.status_var.set(f'Decay / IRF fit complete. {self._fit_metric_summary(result)}')
    history_id = self._new_session_history_id()
    self._capture_result_package_history(result, 'Decay / IRF run fit result', history_id=history_id)
    self._show_fit_report_dialog(result, title='Decay / IRF run fit report', history_id=history_id)


def _fited_irf_display_autofit_payload(self, payload):
    if not _fited_irf_backend._decay_irf_context_active(payload.get('context', {})):
        return _FITED_ORIG_display_autofit_payload(self, payload)
    _fited_irf_plot_decay_result(self, payload, main_label='Auto pre-fit reconvolved fit', title='Decay / IRF automatic pre-fit')
    result = payload['result']
    criterion = payload['context']['criterion']
    best_metric = fit_selection_score(result, criterion)
    self.status_var.set(f'Decay / IRF Auto pre-fit complete. Best {criterion}: {best_metric:.6g}. {self._fit_metric_summary(result)}')
    history_id = self._new_session_history_id()
    self._capture_result_package_history(result, 'Decay / IRF automatic pre-fit result', history_id=history_id)
    self._show_fit_report_dialog(result, title='Decay / IRF automatic pre-fit report', history_id=history_id)


def _fited_irf_display_refine_payload(self, payload):
    if not _fited_irf_backend._decay_irf_context_active(payload.get('context', {})):
        return _FITED_ORIG_display_refine_payload(self, payload)
    _fited_irf_plot_decay_result(self, payload, main_label='Refined reconvolved fit', title='Decay / IRF refined fit')


def _fited_irf_refine_with_added_peaks_background(self):
    if _fited_irf_is_active(self):
        messagebox.showinfo('Not used in Decay / IRF mode', 'Refine with added peaks is specific to peak-profile fitting. For Decay / IRF, use Run fit, Auto pre-fit, or the stability test.')
        return
    return _FITED_ORIG_refine_with_added_peaks_background(self)


def _fited_irf_preview(self):
    if not _fited_irf_is_active(self):
        return _FITED_ORIG_preview(self)
    try:
        context = self._prepare_fit_context(require_active_peak=False)
        model, params = build_model_from_context(context, [])
        x = context['x']
        y_raw = context['y_raw']
        y_plot = context['y_plot']
        preview = model.eval(params=params, x=x)
        comps = _fited_irf_backend.evaluate_time_irf_components(context, params, x=x)
    except Exception as exc:
        messagebox.showerror('Decay / IRF preview error', str(exc))
        return

    self.ax_main.clear()
    self.ax_resid.clear()
    self._refresh_hover_axis()
    self.ax_main.plot(x, y_raw, 'k.', ms=3, alpha=0.6, label='Raw')
    if bool(self.smooth_enabled_var.get()):
        self.ax_main.plot(x, y_plot, linewidth=1.2, alpha=0.9, label='Smoothed preview')
    self.ax_main.plot(x, preview, linewidth=2.0, label='Reconvolved preview')
    for name, comp in comps.items():
        if name == 'reconvolved_decay_model':
            continue
        label = name.replace('_', ' ')
        if name == 'irf_display_scaled':
            label = 'IRF (scaled display)'
        elif name == 'intrinsic_decay_plus_baseline':
            label = 'Intrinsic decay + baseline'
        self.ax_main.plot(x, comp, '--', linewidth=1.0, alpha=0.85, label=label)
    self.ax_main.set_title('Decay / IRF preview')
    self.ax_main.set_ylabel('Signal')
    self.ax_main.legend(fontsize=8, ncol=2)

    resid = y_raw - preview
    self.ax_resid.axhline(0.0, linestyle='--', linewidth=1.0)
    self.ax_resid.plot(x, resid, linewidth=1.0)
    self.ax_resid.set_xlabel('Time')
    self.ax_resid.set_ylabel('Residual')
    self.ax_main.relim()
    self.ax_main.autoscale_view()
    self.ax_resid.relim()
    self.ax_resid.autoscale_view()
    self.fig.tight_layout()
    self.canvas.draw_idle()
    self.status_var.set('Decay / IRF preview updated.')


def _fited_irf_get_session_state(self):
    state = _FITED_ORIG_get_session_state(self)
    try:
        settings = state.setdefault('fit_settings', {})
        settings['analysis_mode'] = self.analysis_mode_var.get()
        settings['decay_irf'] = _fited_irf_collect_config(self) if self.irf_x_full is not None or self.decay_irf_mode_var.get() != 'Measured IRF reconvolution' else {
            'data_type': self.decay_data_type_var.get(),
            'decay_kind': self.decay_model_var.get(),
            'irf_mode': self.decay_irf_mode_var.get(),
            'signal_sign': self.decay_signal_sign_var.get(),
            'gaussian_irf_fwhm': _fited_parse_float(self.gaussian_irf_fwhm_var.get(), 'Gaussian IRF FWHM'),
            'parameters': {
                name: {
                    'value': _fited_parse_float(v['value'].get(), name),
                    'min': _fited_parse_float(v['min'].get(), name),
                    'max': _fited_parse_float(v['max'].get(), name),
                    'vary': bool(v['vary'].get()),
                }
                for name, v in self.decay_param_vars.items()
            },
            'irf_file': str(self.irf_current_file) if self.irf_current_file else '',
        }
        # Prepared IRF arrays can be large; store the file path/settings rather than embedding arrays.
        settings['decay_irf'].pop('prepared_irf', None)
    except Exception:
        pass
    return state


def _fited_irf_apply_session_state(self, state):
    _FITED_ORIG_apply_session_state(self, state)
    try:
        settings = state.get('fit_settings', {}) if isinstance(state, dict) else {}
        mode = settings.get('analysis_mode', self.analysis_mode_var.get())
        if mode in [_fited_irf_backend.DECAY_IRF_STANDARD_MODE, _fited_irf_backend.DECAY_IRF_ANALYSIS_MODE]:
            self.analysis_mode_var.set(mode)
        _fited_irf_apply_config_to_ui(self, settings.get('decay_irf', {}))
    except Exception:
        pass


def _fited_irf_collect_batch_template(self, batch_mode):
    if not _fited_irf_is_active(self):
        return _FITED_ORIG_collect_batch_template(self, batch_mode)
    cfg = _fited_irf_collect_config(self)
    return {
        'analysis_mode': _fited_irf_backend.DECAY_IRF_ANALYSIS_MODE,
        'loader': {
            'delimiter': self._resolve_delimiter(),
            'skiprows': int(self.skiprows_var.get()),
            'x_col': int(self.xcol_var.get()),
            'y_col': int(self.ycol_var.get()),
        },
        'roi_min': float(self.roi_min_var.get()),
        'roi_max': float(self.roi_max_var.get()),
        'peak_defs': [],
        'background_kind': 'none',
        'poly_order': int(self.poly_order_var.get()),
        'custom_profiles': copy.deepcopy(self.custom_profiles),
        'custom_background_profiles': copy.deepcopy(self.custom_background_profiles),
        'custom_background_profile_name': self.background_profile_var.get(),
        'background_params': copy.deepcopy(self._collect_background_params()),
        'auxiliary_parameters': copy.deepcopy(self.auxiliary_parameter_definitions),
        'parameter_constraints': copy.deepcopy(self.parameter_constraint_definitions),
        'weighting': self.weighting_var.get(),
        'criterion': self._current_fit_criterion(),
        'n_trials': max(1, int(self.autofit_trials_var.get())),
        'autoprefit_sampling_mode': self._current_autoprefit_sampling_mode(),
        'max_nfev': self._max_nfev(),
        'optimizer_mode': self._current_optimizer_mode(),
        'random_seed': self._current_random_seed(),
        'batch_mode': batch_mode,
        'all_custom_no_center': False,
        'decay_irf_config': cfg,
        'session_state': self.get_session_state(),
    }




_FITED_ORIG_display_stability_test_payload = DesktopPLFitterApp._display_stability_test_payload


def _fited_irf_display_stability_test_payload(self, payload):
    try:
        best_record = payload.get('best_record', {})
        context = best_record.get('context', {})
    except Exception:
        context = {}
    if not _fited_irf_backend._decay_irf_context_active(context):
        return _FITED_ORIG_display_stability_test_payload(self, payload)

    result = best_record['result']
    context = best_record['context']
    x = context['x']
    comps = _fited_irf_backend.evaluate_time_irf_components(context, result.params, x=x)
    best_record['components'] = comps
    payload['best_record'] = best_record

    _fited_irf_plot_decay_result(
        self,
        {
            'context': context,
            'result': result,
            'components': comps,
            'best_fit': best_record['best_fit'],
        },
        main_label='Best stability-test reconvolved fit',
        title='Decay / IRF stability test: best repeated solution',
    )
    self.last_stability_test_payload = payload

    criterion = payload.get('criterion', self._current_fit_criterion())
    self.status_var.set(
        f"Decay / IRF stability test complete. Best {criterion}: {payload.get('best_score', np.nan):.6g}; "
        f"near-best repeated solutions: {len(payload.get('near_best_records', []))}/"
        f"{payload.get('successful_repeats', 0)}."
    )

    history_id = self._new_session_history_id()
    self._capture_result_package_history(
        result,
        'Decay / IRF stability best-fit result',
        history_id=history_id,
    )
    self._show_stability_test_report_dialog(payload, history_id=history_id)
    self._show_fit_report_dialog(
        result,
        title='Decay / IRF stability test: best repeated solution report',
        report_key='stability_best_fit',
        history_id=history_id,
    )

DesktopPLFitterApp._build_state = _fited_irf_build_state
DesktopPLFitterApp._build_ui = _fited_irf_build_ui
DesktopPLFitterApp._prepare_fit_context = _fited_irf_prepare_fit_context
DesktopPLFitterApp.preview = _fited_irf_preview
DesktopPLFitterApp._display_run_fit_payload = _fited_irf_display_run_fit_payload
DesktopPLFitterApp._display_autofit_payload = _fited_irf_display_autofit_payload
DesktopPLFitterApp._display_refine_payload = _fited_irf_display_refine_payload
DesktopPLFitterApp.refine_with_added_peaks_background = _fited_irf_refine_with_added_peaks_background
DesktopPLFitterApp.get_session_state = _fited_irf_get_session_state
DesktopPLFitterApp.apply_session_state = _fited_irf_apply_session_state
DesktopPLFitterApp._collect_batch_template = _fited_irf_collect_batch_template
DesktopPLFitterApp._display_stability_test_payload = _fited_irf_display_stability_test_payload


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
