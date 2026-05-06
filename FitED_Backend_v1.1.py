"""
Author: Mustafa Mahmoud Aboulsaad
Email: mustafa.aboulsaad@outlook.com
"""

import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
import ast
import csv
import keyword
import os
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from scipy.special import voigt_profile

from lmfit import Model, Parameters
from lmfit.models import (
    GaussianModel,
    LorentzianModel,
    PseudoVoigtModel,
    ConstantModel,
    LinearModel,
    PolynomialModel,
)

try:
    import ipywidgets as widgets
    from IPython.display import display, clear_output
except ImportError:
    widgets = None
    def display(*args, **kwargs):
        raise ImportError("ipywidgets is required for the notebook UI components in pl_fitting_backend.py")
    def clear_output(*args, **kwargs):
        return None

plt.rcParams["figure.dpi"] = 130
plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.spines.right"] = False

def _detect_delimiter(filepath, sample_size=8192):
   
    filepath = Path(filepath)
    try:
        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
            sample = f.read(sample_size)
        dialect = csv.Sniffer().sniff(sample, delimiters=",;	 ")
        return dialect.delimiter
    except Exception:
        return "," if filepath.suffix.lower() == ".csv" else "	"


def _split_line(line, delimiter):
    if delimiter is None:
        delimiter = ","
    if delimiter == r"\s+":
        return re.split(r"\s+", line.strip())
    return [part.strip() for part in line.rstrip("\n\r").split(delimiter)]


def load_spectrum(filepath, x_col=0, y_col=1, delimiter=None, skiprows=0):
    """
    Load x/y numeric columns from a text spectrum file.

    This loader is intentionally tolerant:
    - accepts CSV, TXT, DAT, ASC and similar delimited text files
    - supports explicit delimiters or automatic delimiter detection
    - allows leading header/metadata rows through ``skiprows``
    - ignores any non-numeric rows that remain after skipping
    """
    filepath = Path(filepath)
    x_col = int(x_col)
    y_col = int(y_col)
    skiprows = max(int(skiprows), 0)
    sep = _detect_delimiter(filepath) if delimiter is None else delimiter

    rows = []
    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        for line_number, raw_line in enumerate(f):
            if line_number < skiprows:
                continue

            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue

            parts = _split_line(raw_line, sep)
            max_col = max(x_col, y_col)
            if len(parts) <= max_col:
                continue

            try:
                x_val = pd.to_numeric(parts[x_col], errors="coerce")
                y_val = pd.to_numeric(parts[y_col], errors="coerce")
            except Exception:
                continue

            if pd.isna(x_val) or pd.isna(y_val):
                continue
            rows.append((float(x_val), float(y_val)))

    if not rows:
        delim_name = "auto" if delimiter is None else repr(delimiter)
        raise ValueError(
            "No numeric data rows were found. Check delimiter, X/Y columns, and skip rows. "
            f"Current settings: delimiter={delim_name}, x_col={x_col}, y_col={y_col}, skiprows={skiprows}."
        )

    data = np.asarray(rows, dtype=float)
    x = data[:, 0]
    y = data[:, 1]

    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]

    if x.size == 0:
        raise ValueError("No finite numeric x/y pairs remained after parsing the file.")

    order = np.argsort(x)
    return x[order], y[order]


def crop_roi(x, y, xmin=None, xmax=None):
    if xmin is None:
        xmin = np.min(x)
    if xmax is None:
        xmax = np.max(x)
    mask = (x >= xmin) & (x <= xmax)
    return x[mask], y[mask]


def smooth_if_requested(y, window=9, polyorder=2, enabled=False):
    if not enabled:
        return y.copy()
    window = int(window)
    if window % 2 == 0:
        window += 1
    window = max(window, polyorder + 3)
    if window >= len(y):
        return y.copy()
    return savgol_filter(y, window_length=window, polyorder=polyorder)


def sigma_from_fwhm_gaussian(fwhm):
    return max(float(fwhm), 1e-12) / 2.354820045


def gamma_from_fwhm_lorentzian(fwhm):
    return max(float(fwhm), 1e-12) / 2.0


def exact_voigt_area_normalized(x, amplitude, center, sigma, gamma):
    sigma = max(float(sigma), 1e-15)
    gamma = max(float(gamma), 1e-15)
    return amplitude * voigt_profile(x - center, sigma, gamma)


ExactVoigtModel = Model(exact_voigt_area_normalized, independent_vars=["x"])


BUILTIN_PEAK_KINDS = ["Gaussian", "Lorentzian", "Pseudo-Voigt", "Exact Voigt", "Custom"]


FIT_SELECTION_CRITERIA = ["AIC", "BIC", "chi-square", "reduced chi-square"]


def canonical_fit_criterion(criterion="AIC"):
    """Normalize user-facing fit criterion names to lmfit result attribute names."""
    key = str(criterion or "AIC").strip().lower()
    return {
        "aic": "aic",
        "akaike": "aic",
        "akaike information criterion": "aic",
        "bic": "bic",
        "bayesian information criterion": "bic",
        "chi-square": "chisqr",
        "chisqr": "chisqr",
        "chi square": "chisqr",
        "reduced chi-square": "redchi",
        "reduced chi square": "redchi",
        "redchi": "redchi",
    }.get(key, "aic")


def fit_result_metric(result, criterion="AIC"):
    """Return one numeric metric from an lmfit result; lower is better."""
    attr = canonical_fit_criterion(criterion)
    value = getattr(result, attr, np.inf)
    try:
        value = float(value)
    except Exception:
        return np.inf
    if not np.isfinite(value):
        return np.inf
    return value


def fit_selection_score(result, criterion="AIC"):
    """Score a fitted candidate result for model/trial selection. Lower is better."""
    return fit_result_metric(result, criterion)


def fit_result_metrics(result):
    """Return common lmfit goodness/model-selection metrics as plain values."""
    fields = ["chisqr", "redchi", "aic", "bic", "nfev", "nvarys", "ndata"]
    metrics = {}
    for field in fields:
        value = getattr(result, field, np.nan)
        try:
            if isinstance(value, (np.integer, int)):
                metrics[field] = int(value)
            else:
                metrics[field] = float(value)
        except Exception:
            metrics[field] = value
    return metrics


def fit_metric_summary(result):
    """Compact text summary of the most useful fit metrics."""
    labels = [("chi-square", "chisqr"), ("redchi", "redchi"), ("AIC", "aic"), ("BIC", "bic")]
    parts = []
    for label, attr in labels:
        value = getattr(result, attr, None)
        try:
            value = float(value)
        except Exception:
            continue
        if np.isfinite(value):
            parts.append(f"{label}: {value:.6g}")
    return ", ".join(parts)

SAFE_EXPRESSION_FUNCTIONS = {
    "exp": np.exp,
    "log": np.log,
    "log10": np.log10,
    "sqrt": np.sqrt,
    "sin": np.sin,
    "cos": np.cos,
    "tan": np.tan,
    "abs": np.abs,
    "where": np.where,
    "minimum": np.minimum,
    "maximum": np.maximum,
    "clip": np.clip,
}

_ALLOWED_AST_NODES = (
    ast.Expression,
    ast.BinOp,
    ast.UnaryOp,
    ast.Call,
    ast.Name,
    ast.Load,
    ast.Constant,
    ast.Compare,
    ast.BoolOp,
    ast.And,
    ast.Or,
    ast.Add,
    ast.Sub,
    ast.Mult,
    ast.Div,
    ast.Pow,
    ast.Mod,
    ast.USub,
    ast.UAdd,
    ast.Lt,
    ast.LtE,
    ast.Gt,
    ast.GtE,
    ast.Eq,
    ast.NotEq,
)

def _normalize_parameter_definition(param):
    name = str(param.get("name", "")).strip()
    if not name:
        raise ValueError("Each custom parameter needs a name.")
    if not name.isidentifier() or keyword.iskeyword(name):
        raise ValueError(f"Invalid parameter name: {name}")
    if name == "x" or name in SAFE_EXPRESSION_FUNCTIONS:
        raise ValueError(f"Parameter name '{name}' is reserved.")
    return {
        "name": name,
        "default": float(param.get("default", 1.0)),
        "min": float(param.get("min", -np.inf)),
        "max": float(param.get("max", np.inf)),
    }

def validate_custom_expression(expression, parameter_names):
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as exc:
        raise ValueError(f"Invalid custom expression syntax: {exc}") from exc

    allowed_names = set(parameter_names) | {"x"} | set(SAFE_EXPRESSION_FUNCTIONS)
    for node in ast.walk(tree):
        if not isinstance(node, _ALLOWED_AST_NODES):
            raise ValueError(f"Unsupported syntax in custom expression: {type(node).__name__}")
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name) or node.func.id not in SAFE_EXPRESSION_FUNCTIONS:
                raise ValueError("Only approved math functions are allowed in custom expressions.")
        elif isinstance(node, ast.Name):
            if node.id not in allowed_names:
                raise ValueError(f"Unknown symbol in custom expression: {node.id}")

def normalize_custom_profile_definition(profile_def):
    name = str(profile_def.get("name", "")).strip()
    expression = str(profile_def.get("expression", "")).strip()
    parameters = profile_def.get("parameters", [])

    if not name:
        raise ValueError("Custom profile must have a name.")
    if not expression:
        raise ValueError(f"Custom profile '{name}' needs an expression.")
    if not isinstance(parameters, list) or not parameters:
        raise ValueError(f"Custom profile '{name}' needs at least one parameter.")

    normalized_params = [_normalize_parameter_definition(p) for p in parameters]
    names = [p["name"] for p in normalized_params]
    if len(set(names)) != len(names):
        raise ValueError(f"Custom profile '{name}' has duplicate parameter names.")

    validate_custom_expression(expression, names)
    return {"name": name, "expression": expression, "parameters": normalized_params}

def evaluate_custom_expression(expression, x, param_values):
    scope = {"x": np.asarray(x, dtype=float), **SAFE_EXPRESSION_FUNCTIONS, **param_values}
    try:
        y = eval(expression, {"__builtins__": {}}, scope)
    except Exception as exc:
        raise ValueError(f"Could not evaluate custom expression: {exc}") from exc

    y = np.asarray(y, dtype=float)
    if y.shape != np.asarray(x).shape:
        raise ValueError("Custom expression must return an array with the same shape as x.")
    if not np.all(np.isfinite(y)):
        raise ValueError("Custom expression produced non-finite values.")
    return y

def _make_custom_model_function(function_name, expression, param_names):
    """
    Build a real Python function with explicit parameter names.

    lmfit works better with explicit function signatures such as:
        f(x, amplitude, center, sigma_left, sigma_right)
    than with:
        f(x, **kwargs)

    The parameter names are already validated by normalize_custom_profile_definition().
    """
    args = ", ".join(param_names)
    values = ", ".join([f"{name!r}: {name}" for name in param_names])

    source = (
        f"def {function_name}(x, {args}):\n"
        f"    param_values = {{{values}}}\n"
        f"    return evaluate_custom_expression(expression, x, param_values)\n"
    )

    namespace = {
        "evaluate_custom_expression": evaluate_custom_expression,
        "expression": expression,
    }
    exec(source, namespace)
    return namespace[function_name]


def make_custom_peak_model(profile_def, prefix):
    profile_def = normalize_custom_profile_definition(profile_def)
    expression = profile_def["expression"]
    param_names = [p["name"] for p in profile_def["parameters"]]

    func = _make_custom_model_function("_custom_peak", expression, param_names)

    return Model(
        func,
        independent_vars=["x"],
        prefix=prefix,
    )


def make_custom_background_model(profile_def, prefix):
    profile_def = normalize_custom_profile_definition(profile_def)
    expression = profile_def["expression"]
    param_names = [p["name"] for p in profile_def["parameters"]]

    func = _make_custom_model_function("_custom_background", expression, param_names)

    return Model(
        func,
        independent_vars=["x"],
        prefix=prefix,
    )

def _resolve_profile_from_collection(profile_name, profiles, kind_label="profile"):
    if not profile_name:
        raise ValueError(f"A custom {kind_label} requires a selected custom {kind_label}.")
    if profiles is None:
        raise ValueError(f"No custom {kind_label}s are available.")
    if isinstance(profiles, dict):
        profile = profiles.get(profile_name)
    else:
        profile = next((p for p in profiles if p.get("name") == profile_name), None)
    if profile is None:
        raise ValueError(f"Unknown custom {kind_label}: {profile_name}")
    return normalize_custom_profile_definition(profile)


def _resolve_custom_profile(profile_name, custom_profiles):
    return _resolve_profile_from_collection(profile_name, custom_profiles, kind_label="profile")


def _resolve_custom_background_profile(profile_name, custom_background_profiles):
    return _resolve_profile_from_collection(profile_name, custom_background_profiles, kind_label="background profile")


def make_peak_model(kind, prefix, custom_profiles=None, custom_profile_name=None):
    kind = str(kind).strip().lower()
    if kind == "gaussian":
        return GaussianModel(prefix=prefix)
    elif kind == "lorentzian":
        return LorentzianModel(prefix=prefix)
    elif kind == "pseudo-voigt":
        return PseudoVoigtModel(prefix=prefix)
    elif kind == "exact voigt":
        return Model(exact_voigt_area_normalized, prefix=prefix)
    elif kind == "custom":
        return make_custom_peak_model(_resolve_custom_profile(custom_profile_name, custom_profiles), prefix=prefix)
    else:
        raise ValueError(f"Unknown peak kind: {kind}")


def add_background_model(background_kind, poly_order=2, custom_background_profiles=None, custom_background_profile_name=None):
    background_kind = background_kind.lower()

    if background_kind == "none":
        return None
    elif background_kind == "constant":
        return ConstantModel(prefix="bkg_")
    elif background_kind == "linear":
        return LinearModel(prefix="bkg_")
    elif background_kind == "polynomial":
        poly_order = int(poly_order)
        if poly_order < 1:
            poly_order = 1
        if poly_order > 7:
            poly_order = 7
        return PolynomialModel(prefix="bkg_", degree=poly_order)
    elif background_kind == "custom":
        return make_custom_background_model(
            _resolve_custom_background_profile(custom_background_profile_name, custom_background_profiles),
            prefix="bkg_",
        )
    else:
        raise ValueError(f"Unknown background kind: {background_kind}")



def build_composite_model(
    peak_defs,
    background_kind="linear",
    poly_order=2,
    x=None,
    y=None,
    custom_profiles=None,
    custom_background_profiles=None,
    custom_background_profile_name=None,
    background_params=None,
):
    params = Parameters()
    model = None

    bkg = add_background_model(
        background_kind,
        poly_order=poly_order,
        custom_background_profiles=custom_background_profiles,
        custom_background_profile_name=custom_background_profile_name,
    )
    if bkg is not None:
        model = bkg
        params.update(bkg.make_params())

        bk = background_kind.lower()

        if bk == "constant":
            if y is not None and len(y) > 0:
                params["bkg_c"].set(value=float(np.median(y)))
            else:
                params["bkg_c"].set(value=0.0)

        elif bk == "linear":
            if x is not None and y is not None and len(x) > 1:
                slope_guess = (y[-1] - y[0]) / max((x[-1] - x[0]), 1e-12)
                intercept_guess = np.median(y) - slope_guess * np.median(x)
                params["bkg_slope"].set(value=float(slope_guess))
                params["bkg_intercept"].set(value=float(intercept_guess))
            else:
                params["bkg_slope"].set(value=0.0)
                params["bkg_intercept"].set(value=0.0)

        elif bk == "polynomial":
            if x is not None and y is not None and len(x) > poly_order:
                coeffs = np.polyfit(x, y, deg=int(poly_order))
                coeffs = coeffs[::-1]
                for i, c in enumerate(coeffs):
                    pname = f"bkg_c{i}"
                    if pname in params:
                        params[pname].set(value=float(c))
            else:
                for i in range(int(poly_order) + 1):
                    pname = f"bkg_c{i}"
                    if pname in params:
                        params[pname].set(value=0.0)

        elif bk == "custom":
            profile = _resolve_custom_background_profile(custom_background_profile_name, custom_background_profiles)
            background_params = background_params or {}
            for param_def in profile["parameters"]:
                name = param_def["name"]
                full_name = f"bkg_{name}"
                cfg = background_params.get(name, {}) if isinstance(background_params, dict) else {}
                value = float(cfg.get("value", param_def["default"]))
                pmin = float(cfg.get("min", param_def["min"]))
                pmax = float(cfg.get("max", param_def["max"]))
                params[full_name].set(value=value, min=pmin, max=pmax)

    for i, p in enumerate(peak_defs, start=1):
        prefix = f"p{i}_"
        kind = str(p["kind"]).strip()
        pk = make_peak_model(kind, prefix=prefix, custom_profiles=custom_profiles, custom_profile_name=p.get("custom_profile"))

        if model is None:
            model = pk
        else:
            model = model + pk

        params.update(pk.make_params())

        center = float(p.get("center", 0.0))
        amplitude = max(float(p.get("amplitude", 1.0)), 1e-12)
        fwhm = max(float(p.get("fwhm", 1.0)), 1e-12)
        fwhm_min = max(float(p.get("fwhm_min", 1e-8)), 1e-12)
        fwhm_max = max(float(p.get("fwhm_max", np.inf)), fwhm_min + 1e-12)

        center_min = float(p.get("center_min", center - 2 * fwhm))
        center_max = float(p.get("center_max", center + 2 * fwhm))
        amp_min = float(p.get("amplitude_min", 0.0))
        amp_max = float(p.get("amplitude_max", np.inf))

        kind_l = kind.lower()

        if kind_l in ["gaussian", "lorentzian", "pseudo-voigt"]:
            params[f"{prefix}center"].set(value=center, min=center_min, max=center_max)
            params[f"{prefix}amplitude"].set(value=amplitude, min=amp_min, max=amp_max)

            sigma_guess = sigma_from_fwhm_gaussian(fwhm)
            sigma_min = sigma_from_fwhm_gaussian(fwhm_min)
            sigma_max = sigma_from_fwhm_gaussian(fwhm_max)

            params[f"{prefix}sigma"].set(
                value=sigma_guess,
                min=max(sigma_min, 1e-8),
                max=sigma_max
            )

            if kind_l == "pseudo-voigt" and f"{prefix}fraction" in params:
                frac = float(p.get("fraction", 0.5))
                params[f"{prefix}fraction"].set(value=frac, min=0.0, max=1.0)

        elif kind_l == "exact voigt":
            sigma = float(p.get("sigma", sigma_from_fwhm_gaussian(fwhm)))
            gamma = float(p.get("gamma", gamma_from_fwhm_lorentzian(fwhm)))

            sigma_min = sigma_from_fwhm_gaussian(fwhm_min)
            sigma_max = sigma_from_fwhm_gaussian(fwhm_max)
            gamma_min = gamma_from_fwhm_lorentzian(fwhm_min)
            gamma_max = gamma_from_fwhm_lorentzian(fwhm_max)

            params[f"{prefix}center"].set(value=center, min=center_min, max=center_max)
            params[f"{prefix}amplitude"].set(value=amplitude, min=amp_min, max=amp_max)
            params[f"{prefix}sigma"].set(
                value=max(sigma, 1e-8),
                min=max(sigma_min, 1e-8),
                max=sigma_max
            )
            params[f"{prefix}gamma"].set(
                value=max(gamma, 1e-8),
                min=max(gamma_min, 1e-8),
                max=gamma_max
            )

        elif kind_l == "custom":
            profile = _resolve_custom_profile(p.get("custom_profile"), custom_profiles)
            for param_def in profile["parameters"]:
                name = param_def["name"]
                full_name = f"{prefix}{name}"
                value = float(p.get(name, param_def["default"]))
                pmin = float(p.get(f"{name}_min", param_def["min"]))
                pmax = float(p.get(f"{name}_max", param_def["max"]))
                params[full_name].set(value=value, min=pmin, max=pmax)

        else:
            raise ValueError(f"Unsupported kind: {kind}")

    return model, params


class InteractivePLFitter:
    def __init__(self, x, y):
        self.x_full = np.asarray(x, dtype=float)
        self.y_full = np.asarray(y, dtype=float)

        self.fit_result = None
        self.last_components = None
        self.last_best_fit = None
        self.last_roi = None
        self.current_file = None

        xmin_default = float(np.min(self.x_full))
        xmax_default = float(np.max(self.x_full))

        self.roi_min = widgets.FloatText(value=xmin_default, description="ROI min", layout=widgets.Layout(width="220px"))
        self.roi_max = widgets.FloatText(value=xmax_default, description="ROI max", layout=widgets.Layout(width="220px"))

        self.n_peaks = widgets.IntSlider(value=5, min=1, max=15, step=1, description="Peaks", continuous_update=False)

        self.background_kind = widgets.Dropdown(
            options=["none", "constant", "linear", "polynomial"],
            value="linear",
            description="Background",
            layout=widgets.Layout(width="240px")
        )

        self.poly_order = widgets.IntSlider(value=2, min=1, max=6, step=1, description="Poly order", continuous_update=False)

        self.smooth_enabled = widgets.Checkbox(value=False, description="Preview smoothing")
        self.smooth_window = widgets.IntSlider(value=9, min=5, max=51, step=2, description="SG window", continuous_update=False)
        self.smooth_poly = widgets.IntSlider(value=2, min=1, max=5, step=1, description="SG poly", continuous_update=False)

        self.weighting = widgets.Dropdown(
            options=["none", "poisson-like", "sqrt(y)", "1/y"],
            value="none",
            description="Weights",
            layout=widgets.Layout(width="220px")
        )

        self.preview_button = widgets.Button(description="Update preview")
        self.fit_button = widgets.Button(description="Run fit", button_style="success")
        self.export_button = widgets.Button(description="Export results", button_style="info")

        self.out = widgets.Output()
        self.peak_boxes = []
        self.peaks_vbox = widgets.VBox()

        self._build_peak_controls()

        self.n_peaks.observe(self._on_peak_count_change, names="value")
        self.preview_button.on_click(self._on_preview_click)
        self.fit_button.on_click(self._on_fit_click)
        self.export_button.on_click(self._on_export_click)

    def _default_center_for_peak(self, idx):
        xmin = np.min(self.x_full)
        xmax = np.max(self.x_full)
        return xmin + (idx + 1) * (xmax - xmin) / (self.n_peaks.value + 1)

    def _build_peak_controls(self):
        old_values = []
        if hasattr(self, "peak_boxes"):
            for box in self.peak_boxes:
                old_values.append({
                    "active": box["active"].value,
                    "kind": box["kind"].value,
                    "center": box["center"].value,
                    "amplitude": box["amplitude"].value,
                    "fwhm": box["fwhm"].value,
                    "fwhm_min": box["fwhm_min"].value,
                    "fwhm_max": box["fwhm_max"].value,
                    "center_min": box["center_min"].value,
                    "center_max": box["center_max"].value,
                    "amplitude_min": box["amplitude_min"].value,
                    "amplitude_max": box["amplitude_max"].value,
                    "fraction": box["fraction"].value,
                    "sigma": box["sigma"].value,
                    "gamma": box["gamma"].value,
                })
    
        self.peak_boxes = []
        children = []
    
        for i in range(self.n_peaks.value):
            center_guess = float(self._default_center_for_peak(i))
            y_span = float(np.max(self.y_full) - np.min(self.y_full))
            x_span = float(np.max(self.x_full) - np.min(self.x_full))
            amp_guess = max(y_span * 0.5, 1.0)
            fwhm_guess = max(x_span / 40, 0.2)
    
            if i < len(old_values):
                vals = old_values[i]
                active_value = vals["active"]
                center_guess = vals["center"]
                amp_guess = vals["amplitude"]
                fwhm_guess = vals["fwhm"]
                fwhm_min_val = vals["fwhm_min"]
                fwhm_max_val = vals["fwhm_max"]
                kind_value = vals["kind"]
                center_min_val = vals["center_min"]
                center_max_val = vals["center_max"]
                amplitude_min_val = vals["amplitude_min"]
                amplitude_max_val = vals["amplitude_max"]
                fraction_val = vals["fraction"]
                sigma_val = vals["sigma"]
                gamma_val = vals["gamma"]
            else:
                active_value = True
                kind_value = "Pseudo-Voigt"
                fwhm_min_val = max(x_span / 1000, 1e-3)
                fwhm_max_val = max(x_span / 4, fwhm_guess * 3)
                center_min_val = center_guess - 2 * fwhm_guess
                center_max_val = center_guess + 2 * fwhm_guess
                amplitude_min_val = 0.0
                amplitude_max_val = max(amp_guess * 10, 1.0)
                fraction_val = 0.5
                sigma_val = sigma_from_fwhm_gaussian(fwhm_guess)
                gamma_val = gamma_from_fwhm_lorentzian(fwhm_guess)
                
    
            kind = widgets.Dropdown(
                options=["Gaussian", "Lorentzian", "Pseudo-Voigt", "Exact Voigt"],
                value=kind_value,
                description=f"Peak {i+1}",
                layout=widgets.Layout(width="230px")
            )
    
            center = widgets.FloatSlider(
                value=center_guess,
                min=float(np.min(self.x_full)),
                max=float(np.max(self.x_full)),
                step=(float(np.max(self.x_full)) - float(np.min(self.x_full))) / 500,
                description="center",
                continuous_update=True,
                layout=widgets.Layout(width="300px")
            )
    
            amplitude = widgets.FloatSlider(
                value=amp_guess,
                min=0.0,
                max=max(amplitude_max_val, amp_guess * 2, 1.0),
                step=max(amp_guess / 200, 1e-3),
                description="area",
                continuous_update=True,
                layout=widgets.Layout(width="300px")
            )
    
            fwhm = widgets.FloatSlider(
                value=fwhm_guess,
                min=max(x_span / 1000, 1e-3),
                max=max(x_span / 4, fwhm_guess * 3),
                step=max(x_span / 1000, 1e-4),
                description="FWHM",
                continuous_update=True,
                layout=widgets.Layout(width="300px")
            )

            fwhm_min = widgets.FloatText(value=fwhm_min_val, description="w min", layout=widgets.Layout(width="200px"))
            fwhm_max = widgets.FloatText(value=fwhm_max_val, description="w max", layout=widgets.Layout(width="200px"))
    
            center_min = widgets.FloatText(value=center_min_val, description="c min", layout=widgets.Layout(width="200px"))
            center_max = widgets.FloatText(value=center_max_val, description="c max", layout=widgets.Layout(width="200px"))
            amplitude_min = widgets.FloatText(value=amplitude_min_val, description="a min", layout=widgets.Layout(width="200px"))
            amplitude_max = widgets.FloatText(value=amplitude_max_val, description="a max", layout=widgets.Layout(width="200px"))
    
            fraction = widgets.FloatSlider(
                value=fraction_val,
                min=0.0,
                max=1.0,
                step=0.01,
                description="G/L mix",
                continuous_update=True
            )
    
            sigma = widgets.FloatText(value=sigma_val, description="sigma", layout=widgets.Layout(width="200px"))
            gamma = widgets.FloatText(value=gamma_val, description="gamma", layout=widgets.Layout(width="200px"))
            #active = widgets.Checkbox(value=active_value, description="use", layout=widgets.Layout(width="80px"))
            active = widgets.Checkbox(
                value=active_value,
                description=f"use {i+1}",
                indent=False,
                layout=widgets.Layout(width="90px")
            )
    
            box = {
                "active": active,
                "kind": kind,
                "center": center,
                "amplitude": amplitude,
                "fwhm": fwhm,
                "fwhm_min": fwhm_min,
                "fwhm_max": fwhm_max,
                "center_min": center_min,
                "center_max": center_max,
                "amplitude_min": amplitude_min,
                "amplitude_max": amplitude_max,
                "fraction": fraction,
                "sigma": sigma,
                "gamma": gamma,
            }
    
            panel = widgets.VBox([
                widgets.HTML(f"<b>Peak {i+1}</b>"),
                widgets.HBox([active, kind, center]),
                widgets.HBox([amplitude, fwhm]),
                widgets.HBox([center_min, center_max, amplitude_min, amplitude_max, fwhm_min, fwhm_max]),
                widgets.HBox([fraction, sigma, gamma]),
            ])
    
            self.peak_boxes.append(box)
            children.append(panel)
    
        self.peaks_vbox.children = children
        self._attach_live_updates()


    def _attach_live_updates(self):
        live_widgets = [
            self.roi_min, self.roi_max,
            self.n_peaks,
            self.background_kind, self.poly_order,
            self.smooth_enabled, self.smooth_window, self.smooth_poly,
            self.weighting
        ]
    
        for w in live_widgets:
            w.observe(self._live_preview, names="value")
    
        for box in self.peak_boxes:
            for key in ["active", "kind", "center", "amplitude", "fwhm", "fwhm_min", "fwhm_max", "center_min", "center_max",
                        "amplitude_min", "amplitude_max", "fraction", "sigma", "gamma"]:
                box[key].observe(self._live_preview, names="value")


    def _live_preview(self, change=None):
        with self.out:
            clear_output(wait=True)
            try:
                self.preview()
            except Exception as e:
                print(f"Preview error: {e}")

    def _on_peak_count_change(self, change):
        old_boxes = []
        for box in self.peak_boxes:
            old_boxes.append({
                "active": box["active"].value,
                "kind": box["kind"].value,
                "center": box["center"].value,
                "amplitude": box["amplitude"].value,
                "fwhm": box["fwhm"].value,
                "fwhm_min": box["fwhm_min"].value if "fwhm_min" in box else None,
                "fwhm_max": box["fwhm_max"].value if "fwhm_max" in box else None,
                "center_min": box["center_min"].value,
                "center_max": box["center_max"].value,
                "amplitude_min": box["amplitude_min"].value,
                "amplitude_max": box["amplitude_max"].value,
                "fraction": box["fraction"].value,
                "sigma": box["sigma"].value,
                "gamma": box["gamma"].value,
            })
    
        self._build_peak_controls()
    
        for i, vals in enumerate(old_boxes):
            if i >= len(self.peak_boxes):
                break
            box = self.peak_boxes[i]
            box["active"].value = vals["active"]
            box["kind"].value = vals["kind"]
            box["center"].value = vals["center"]
            box["amplitude"].value = vals["amplitude"]
            box["fwhm"].value = vals["fwhm"]
            if "fwhm_min" in box and vals["fwhm_min"] is not None:
                box["fwhm_min"].value = vals["fwhm_min"]
            if "fwhm_max" in box and vals["fwhm_max"] is not None:
                box["fwhm_max"].value = vals["fwhm_max"]
            box["center_min"].value = vals["center_min"]
            box["center_max"].value = vals["center_max"]
            box["amplitude_min"].value = vals["amplitude_min"]
            box["amplitude_max"].value = vals["amplitude_max"]
            box["fraction"].value = vals["fraction"]
            box["sigma"].value = vals["sigma"]
            box["gamma"].value = vals["gamma"]
    
        self._live_preview()

    def _collect_peak_defs(self):
        peak_defs = []
    
        for box in self.peak_boxes:
            if not box["active"].value:
                continue
    
            center = float(box["center"].value)
            amplitude = float(box["amplitude"].value)
            fwhm = float(box["fwhm"].value)
    
            center_min = float(box["center_min"].value)
            center_max = float(box["center_max"].value)
            amplitude_min = float(box["amplitude_min"].value)
            amplitude_max = float(box["amplitude_max"].value)
            fwhm_min = float(box["fwhm_min"].value)
            fwhm_max = float(box["fwhm_max"].value)
    
            # sanitize center bounds
            if center_min > center_max:
                center_min, center_max = center_max, center_min
            if center_min == center_max:
                eps = max(abs(center) * 1e-9, 1e-9)
                center_min -= eps
                center_max += eps
    
            # sanitize amplitude bounds
            if amplitude_min > amplitude_max:
                amplitude_min, amplitude_max = amplitude_max, amplitude_min
            if amplitude_min == amplitude_max:
                eps = max(abs(amplitude) * 1e-9, 1e-9)
                amplitude_min -= eps
                amplitude_max += eps

            if fwhm_min > fwhm_max:
                fwhm_min, fwhm_max = fwhm_max, fwhm_min
            if fwhm_min == fwhm_max:
                eps = max(abs(fwhm) * 1e-9, 1e-9)
                fwhm_min -= eps
                fwhm_max += eps
    
            peak_defs.append({
                "kind": box["kind"].value,
                "center": center,
                "amplitude": amplitude,
                "fwhm": fwhm,
                "fwhm_min": fwhm_min,
                "fwhm_max": fwhm_max,
                "center_min": center_min,
                "center_max": center_max,
                "amplitude_min": amplitude_min,
                "amplitude_max": amplitude_max,
                "fraction": float(box["fraction"].value),
                "sigma": float(box["sigma"].value),
                "gamma": float(box["gamma"].value),
            })
    
        return peak_defs

    def _get_roi_data(self):
        x, y = crop_roi(self.x_full, self.y_full, self.roi_min.value, self.roi_max.value)

        if len(x) == 0:
            raise ValueError("No data points inside ROI. Adjust ROI min/max.")

        y_plot = smooth_if_requested(
            y,
            window=self.smooth_window.value,
            polyorder=self.smooth_poly.value,
            enabled=self.smooth_enabled.value,
        )
        return x, y, y_plot

    def _weights(self, y):
        eps = 1e-12
        mode = self.weighting.value

        if mode == "none":
            return None
        elif mode == "poisson-like":
            return 1.0 / np.sqrt(np.clip(np.abs(y), 1.0, None))
        elif mode == "sqrt(y)":
            return np.sqrt(np.clip(np.abs(y), eps, None))
        elif mode == "1/y":
            return 1.0 / np.clip(np.abs(y), eps, None)
        return None

    def preview(self):
        x, y_raw, y_plot = self._get_roi_data()
        peak_defs = self._collect_peak_defs()

        model, params = build_composite_model(
            peak_defs,
            background_kind=self.background_kind.value,
            poly_order=self.poly_order.value,
            x=x,
            y=y_raw
        )

        preview = model.eval(params=params, x=x)
        comps = model.eval_components(params=params, x=x)

        fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True, gridspec_kw={"height_ratios": [3, 1]})

        ax = axes[0]
        ax.plot(x, y_raw, "k.", ms=2.5, alpha=0.55, label="Raw")
        if self.smooth_enabled.value:
            ax.plot(x, y_plot, lw=1.2, alpha=0.9, label="Smoothed preview")
        ax.plot(x, preview, lw=2.0, label="Current model preview")

        for name, comp in comps.items():
            ax.plot(x, comp, "--", lw=1.0, alpha=0.85, label=name)

        ax.set_ylabel("Intensity (a.u.)")
        ax.set_title("Preview")
        ax.legend(loc="best", fontsize=8, ncol=2)

        resid = y_raw - preview
        axes[1].axhline(0, ls="--", lw=1)
        axes[1].plot(x, resid, lw=1.0)
        axes[1].set_xlabel("Wavelength")
        axes[1].set_ylabel("Residual")

        plt.tight_layout()
        plt.show()

    def fit(self):
        x, y_raw, _ = self._get_roi_data()
        peak_defs = self._collect_peak_defs()

        model, params = build_composite_model(
            peak_defs,
            background_kind=self.background_kind.value,
            poly_order=self.poly_order.value,
            x=x,
            y=y_raw
        )

        weights = self._weights(y_raw)
        result = model.fit(y_raw, params=params, x=x, weights=weights, nan_policy="raise")

        comps = result.eval_components(x=x)
        best = result.best_fit

        self.fit_result = result
        self.last_components = comps
        self.last_best_fit = best
        self.last_roi = (x.copy(), y_raw.copy())

        fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True, gridspec_kw={"height_ratios": [3, 1]})

        ax = axes[0]
        ax.plot(x, y_raw, "k.", ms=2.5, alpha=0.55, label="Data")
        ax.plot(x, best, lw=2.2, label="Best fit")

        for name, comp in comps.items():
            ax.plot(x, comp, "--", lw=1.0, alpha=0.85, label=name)

        ax.set_ylabel("Intensity (a.u.)")
        ax.set_title("Best fit and components")
        ax.legend(loc="best", fontsize=8, ncol=2)

        resid = y_raw - best
        axes[1].axhline(0, ls="--", lw=1)
        axes[1].plot(x, resid, lw=1.0)
        axes[1].set_xlabel("Wavelength")
        axes[1].set_ylabel("Residual")

        plt.tight_layout()
        plt.show()

        print(result.fit_report(min_correl=0.5))

    def export_results(self, basename="pl_fit_export", folder="."):
        
    
        if self.fit_result is None or self.last_roi is None:
            print("Run a fit first.")
            return
    
        os.makedirs(folder, exist_ok=True)
    
        # curves file
        x, y = self.last_roi
        out = pd.DataFrame({
            "x": x,
            "y_data": y,
            "y_fit": self.last_best_fit,
            "residual": y - self.last_best_fit,
        })
    
        for name, comp in self.last_components.items():
            out[name] = comp
    
        curves_path = os.path.join(folder, f"{basename}_curves.csv")
        out.to_csv(curves_path, index=False)
    
        # parameters file
        rows = []
        for name, par in self.fit_result.params.items():
            rows.append({
                "parameter": name,
                "value": par.value,
                "stderr": par.stderr,
                "min": par.min,
                "max": par.max,
                "vary": par.vary,
                "expr": par.expr,
            })
    
        params_df = pd.DataFrame(rows)
        params_path = os.path.join(folder, f"{basename}_parameters.csv")
        params_df.to_csv(params_path, index=False)
    
        # summary / model-selection metrics
        summary_path = os.path.join(folder, f"{basename}_summary.csv")
        summary_df = pd.DataFrame([
            {"field": key, "value": value}
            for key, value in fit_result_metrics(self.fit_result).items()
        ])
        summary_df.to_csv(summary_path, index=False)
    
        # fit report
        report_path = os.path.join(folder, f"{basename}_fit_report.txt")
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(self.fit_result.fit_report(min_correl=0.5))
    
        print("Saved:")
        print(curves_path)
        print(params_path)
        print(summary_path)
        print(report_path)

    def _on_preview_click(self, b):
        with self.out:
            clear_output(wait=True)
            self.preview()

    def _on_fit_click(self, b):
        with self.out:
            clear_output(wait=True)
            self.fit()

    def _on_export_click(self, b):
        with self.out:
            clear_output(wait=True)
            if getattr(self, "current_file", None) is not None:
                self.export_results(basename=Path(self.current_file).stem, folder="fit_exports")
            else:
                self.export_results(basename="pl_fit_export", folder="fit_exports")

    def show(self):
        top = widgets.VBox([
            widgets.HTML("<h3>Interactive PL peak fitting</h3>"),
            widgets.HBox([self.roi_min, self.roi_max, self.n_peaks]),
            widgets.HBox([self.background_kind, self.poly_order, self.weighting]),
            widgets.HBox([self.smooth_enabled, self.smooth_window, self.smooth_poly]),
            widgets.HBox([self.preview_button, self.fit_button, self.export_button]),
            self.peaks_vbox,
            self.out
        ])
        display(top)

    def get_state(self):
        return {
            "roi_min": self.roi_min.value,
            "roi_max": self.roi_max.value,
            "n_peaks": self.n_peaks.value,
            "background_kind": self.background_kind.value,
            "poly_order": self.poly_order.value,
            "smooth_enabled": self.smooth_enabled.value,
            "smooth_window": self.smooth_window.value,
            "smooth_poly": self.smooth_poly.value,
            "weighting": self.weighting.value,
            "peaks": [
                {
                    "active": box["active"].value,
                    "kind": box["kind"].value,
                    "center": box["center"].value,
                    "amplitude": box["amplitude"].value,
                    "fwhm": box["fwhm"].value,
                    "fwhm_min": box["fwhm_min"].value,
                    "fwhm_max": box["fwhm_max"].value,
                    "center_min": box["center_min"].value,
                    "center_max": box["center_max"].value,
                    "amplitude_min": box["amplitude_min"].value,
                    "amplitude_max": box["amplitude_max"].value,
                    "fraction": box["fraction"].value,
                    "sigma": box["sigma"].value,
                    "gamma": box["gamma"].value,
                }
                for box in self.peak_boxes
            ]
        }

    def apply_state(self, state):
        if state is None:
            return
    
        self.roi_min.value = state["roi_min"]
        self.roi_max.value = state["roi_max"]
        self.background_kind.value = state["background_kind"]
        self.poly_order.value = state["poly_order"]
        self.smooth_enabled.value = state["smooth_enabled"]
        self.smooth_window.value = state["smooth_window"]
        self.smooth_poly.value = state["smooth_poly"]
        self.weighting.value = state["weighting"]
    
        self.n_peaks.value = state["n_peaks"]
        self._build_peak_controls()
    
        for i, vals in enumerate(state["peaks"]):
            if i >= len(self.peak_boxes):
                break
            box = self.peak_boxes[i]
            box["active"].value = vals.get("active", True)
            box["kind"].value = vals["kind"]
            box["center"].value = vals["center"]
            box["amplitude"].value = vals["amplitude"]
            box["fwhm"].value = vals["fwhm"]
            box["fwhm_min"].value = vals.get("fwhm_min", box["fwhm_min"].value)
            box["fwhm_max"].value = vals.get("fwhm_max", box["fwhm_max"].value)
            box["center_min"].value = vals["center_min"]
            box["center_max"].value = vals["center_max"]
            box["amplitude_min"].value = vals["amplitude_min"]
            box["amplitude_max"].value = vals["amplitude_max"]
            box["fraction"].value = vals["fraction"]
            box["sigma"].value = vals["sigma"]
            box["gamma"].value = vals["gamma"]

def pick_peak_centers(x, y, n_clicks=3):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(x, y, lw=1.2)
    ax.set_title(f"Click {n_clicks} peak centers, then close the figure")
    pts = plt.ginput(n_clicks, timeout=-1)
    plt.close(fig)
    centers = [p[0] for p in pts]
    return centers

# Example:
# centers = pick_peak_centers(x_data, y_data, n_clicks=3)
# print(centers)


def list_spectrum_files(data_folder, pattern="*.txt"):
    """Return a sorted list of files matching the provided pattern."""
    data_folder = Path(data_folder)
    return sorted(data_folder.glob(pattern))


def create_file_browser(data_folder, pattern="*.txt", x_col=0, y_col=1, delimiter="\t", skiprows=0):
    """Create the notebook UI for selecting a file and launching the fitter.

    Returns
    -------
    widgets.VBox
        The top-level UI widget.
    dict
        A small state dict containing current_file, x_data, y_data, app,
        and last_fit_state.
    """
    files = list_spectrum_files(data_folder, pattern=pattern)
    if not files:
        raise FileNotFoundError(f"No files matching {pattern!r} found in {Path(data_folder)}")

    state = {
        "current_file": None,
        "x_data": None,
        "y_data": None,
        "app": None,
        "last_fit_state": None,
    }

    file_dropdown = widgets.Dropdown(
        options=[(f.name, str(f)) for f in files],
        description="File:",
        layout=widgets.Layout(width="800px")
    )
    load_button = widgets.Button(description="Load selected file", button_style="success")
    file_out = widgets.Output()
    app_out = widgets.Output()

    def load_selected_file(_):
        state["current_file"] = Path(file_dropdown.value)
        state["x_data"], state["y_data"] = load_spectrum(
            state["current_file"],
            x_col=x_col,
            y_col=y_col,
            delimiter=delimiter,
            skiprows=skiprows,
        )

        with file_out:
            clear_output(wait=True)
            print("Loaded:", state["current_file"].name)
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.plot(state["x_data"], state["y_data"], lw=1.2)
            ax.set_xlabel("Wavelength")
            ax.set_ylabel("PL intensity (a.u.)")
            ax.set_title(state["current_file"].name)
            plt.show()

        with app_out:
            clear_output(wait=True)
            print("Creating fitter...")

            if state["app"] is not None:
                try:
                    state["last_fit_state"] = state["app"].get_state()
                except Exception:
                    pass

            state["app"] = InteractivePLFitter(state["x_data"], state["y_data"])
            state["app"].current_file = state["current_file"]
            state["app"].apply_state(state["last_fit_state"])
            state["app"].show()
            print("Fitter should appear above.")

    load_button.on_click(load_selected_file)

    ui = widgets.VBox([
        widgets.HBox([file_dropdown, load_button]),
        file_out,
        app_out
    ])
    return ui, state
