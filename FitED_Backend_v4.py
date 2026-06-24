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

# FitED NumPy compatibility helper.
# NumPy 2.x uses np.trapezoid; older NumPy versions used np.trapz.
def _fited_trapezoid(y, x=None):
    if hasattr(np, "trapezoid"):
        return np.trapezoid(y, x)
    return _fited_trapezoid(y, x)

from scipy.signal import savgol_filter
from scipy.special import voigt_profile
from scipy import stats

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

# User-facing optimizer modes used by the desktop GUI.
# Keep Levenberg-Marquardt as the default because it is fast for normal spectra.
# The robust mode keeps LM as a baseline candidate, then also tries a bounded
# Differential Evolution global-search stage followed by LM polishing. The final
# returned result is whichever candidate has the lower selected criterion.
FIT_OPTIMIZER_MODES = [
    "Levenberg-Marquardt",
    "Robust: compare LM and DE+LM",
]
DEFAULT_FIT_OPTIMIZER_MODE = FIT_OPTIMIZER_MODES[0]


def canonical_optimizer_mode(mode=None):
    """Normalize user-facing optimizer names used by the GUI/session files."""
    key = str(mode or DEFAULT_FIT_OPTIMIZER_MODE).strip().lower()
    key = key.replace("–", "-").replace("—", "-")
    aliases = {
        "lm": "lm",
        "leastsq": "lm",
        "least squares": "lm",
        "levenberg-marquardt": "lm",
        "levenberg marquardt": "lm",
        "leastsq/leastsquares": "lm",
        "robust: compare lm and de+lm": "robust_lm_vs_de_lm",
        "robust compare lm and de+lm": "robust_lm_vs_de_lm",
        "robust: lm vs de+lm": "robust_lm_vs_de_lm",
        "robust lm vs de+lm": "robust_lm_vs_de_lm",
        "lm vs de+lm": "robust_lm_vs_de_lm",
        "compare lm and de+lm": "robust_lm_vs_de_lm",
        "differential evolution + lm": "robust_lm_vs_de_lm",
        "differential evolution + levenberg-marquardt": "robust_lm_vs_de_lm",
        "differential evolution + levenberg marquardt": "robust_lm_vs_de_lm",
        "de + lm": "robust_lm_vs_de_lm",
        "de_lm": "robust_lm_vs_de_lm",
        "global + lm": "robust_lm_vs_de_lm",
        "robust global + lm": "robust_lm_vs_de_lm",
        "robust": "robust_lm_vs_de_lm",
    }
    return aliases.get(key, "lm")


def _finite_array_limits(values, default_min=0.0, default_max=1.0):
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float(default_min), float(default_max), max(float(default_max - default_min), 1.0)
    amin = float(np.min(arr))
    amax = float(np.max(arr))
    span = max(amax - amin, 1e-12)
    return amin, amax, span


def _auto_global_bounds_for_parameter(name, value, x=None, y=None):
    """Return conservative finite bounds for Differential Evolution when a parameter is unbounded.

    Differential Evolution requires every varied parameter to have finite min/max values.
    The normal LM fit can work with unbounded parameters, so this helper is used only on
    a copied Parameters object for the global-search stage. User-provided finite bounds
    always have priority over these automatic bounds.
    """
    lname = str(name).lower()
    val = float(value) if np.isfinite(value) else 0.0
    x_min, x_max, x_span = _finite_array_limits(x if x is not None else [0.0, 1.0])
    y_min, y_max, y_span = _finite_array_limits(y if y is not None else [0.0, 1.0])
    value_scale = max(abs(val), 1.0)
    y_scale = max(abs(y_min), abs(y_max), y_span, 1.0)

    if "fraction" in lname or lname.endswith("_eta") or lname.endswith("eta"):
        return 0.0, 1.0

    if any(token in lname for token in ["center", "centre", "position", "x0", "mu"]):
        pad = 0.10 * x_span
        return x_min - pad, x_max + pad

    if any(token in lname for token in ["sigma", "gamma", "fwhm", "width", "tau", "lifetime", "decay", "rate"]):
        lo = 1e-12
        hi = max(value_scale * 100.0, x_span * 10.0, 1.0)
        return lo, hi

    if any(token in lname for token in ["amplitude", "height", "area", "scale", "intensity"]):
        amp_scale = max(y_span * x_span, y_scale, value_scale, 1.0)
        if val >= 0:
            return 0.0, amp_scale * 20.0
        return -amp_scale * 20.0, amp_scale * 20.0

    if "slope" in lname:
        slope_scale = max(y_span / max(x_span, 1e-12), abs(val), 1.0)
        return -slope_scale * 20.0, slope_scale * 20.0

    if "intercept" in lname or lname.endswith("_c") or re.search(r"_c\d+$", lname):
        return y_min - 5.0 * y_span - value_scale, y_max + 5.0 * y_span + value_scale

    # Generic custom parameter: search around the current value without assuming sign.
    span = max(value_scale * 20.0, 1.0)
    return val - span, val + span


def prepare_params_for_global_search(params, x=None, y=None):
    """Copy lmfit Parameters and ensure all varied parameters have finite bounds for DE."""
    bounded = params.copy()
    for name, par in bounded.items():
        if not getattr(par, "vary", True):
            continue
        if getattr(par, "expr", None):
            continue

        value = float(par.value) if np.isfinite(par.value) else 0.0
        pmin = float(par.min) if np.isfinite(par.min) else -np.inf
        pmax = float(par.max) if np.isfinite(par.max) else np.inf

        auto_min, auto_max = _auto_global_bounds_for_parameter(name, value, x=x, y=y)
        if not np.isfinite(pmin):
            pmin = auto_min
        if not np.isfinite(pmax):
            pmax = auto_max

        if not np.isfinite(pmin) or not np.isfinite(pmax) or pmin >= pmax:
            span = max(abs(value), 1.0)
            pmin = value - 10.0 * span
            pmax = value + 10.0 * span
            if pmin >= pmax:
                pmin, pmax = value - 1.0, value + 1.0

        value = float(np.clip(value, pmin, pmax))
        par.set(value=value, min=float(pmin), max=float(pmax))
    return bounded


def _fit_lm_candidate(model, y, params, x=None, weights=None, nan_policy="raise", max_nfev=10000):
    """Run one Levenberg-Marquardt/lmfit-leastsq candidate fit."""
    return model.fit(
        y,
        params=params,
        x=x,
        weights=weights,
        nan_policy=nan_policy,
        method="leastsq",
        max_nfev=max_nfev,
    )


def _fit_de_lm_candidate(model, y, params, x=None, weights=None, nan_policy="raise", max_nfev=10000, random_seed=None):
    """Run Differential Evolution global search, then polish from that result with LM."""
    de_params = prepare_params_for_global_search(params, x=x, y=y)
    de_result = model.fit(
        y,
        params=de_params,
        x=x,
        weights=weights,
        nan_policy=nan_policy,
        method="differential_evolution",
        max_nfev=max_nfev,
        seed=random_seed,
    )
    lm_result = _fit_lm_candidate(
        model,
        y,
        de_result.params.copy(),
        x=x,
        weights=weights,
        nan_policy=nan_policy,
        max_nfev=max_nfev,
    )
    try:
        lm_result.fited_global_result = de_result
    except Exception:
        pass
    return lm_result


def _result_is_usable(result):
    """Return True only for finite lmfit results that can safely be compared."""
    try:
        best_fit = np.asarray(result.best_fit, dtype=float)
        return best_fit.size > 0 and np.all(np.isfinite(best_fit))
    except Exception:
        return False


def fit_model_with_optimizer(
    model,
    y,
    params,
    x=None,
    weights=None,
    nan_policy="raise",
    max_nfev=10000,
    optimizer_mode=DEFAULT_FIT_OPTIMIZER_MODE,
    selection_criterion="AIC",
    random_seed=None,
):
    """Fit an lmfit Model using the selected FitED optimizer mode.

    Modes:
    - Levenberg-Marquardt: current default behavior, using lmfit method='leastsq'.
    - Robust: compare LM and DE+LM: run normal LM from the user's current
      parameters, also run bounded Differential Evolution followed by LM polishing,
      then return the candidate with the lower selected criterion.
    - random_seed: optional integer seed forwarded to Differential Evolution so
      stochastic global-search candidates can be reproduced.

    In robust mode, Differential Evolution is an extra candidate, not a replacement
    for the user's current LM start. This prevents the global search from making a
    good LM result worse simply because the DE stage stopped in a poorer region.
    """
    mode = canonical_optimizer_mode(optimizer_mode)
    max_nfev = max(1, int(max_nfev))

    if mode == "robust_lm_vs_de_lm":
        candidates = []
        errors = []

        try:
            lm_result = _fit_lm_candidate(
                model,
                y,
                params.copy(),
                x=x,
                weights=weights,
                nan_policy=nan_policy,
                max_nfev=max_nfev,
            )
            if _result_is_usable(lm_result):
                candidates.append(("Levenberg-Marquardt", lm_result))
        except Exception as exc:
            errors.append(f"LM failed: {exc}")

        try:
            de_lm_result = _fit_de_lm_candidate(
                model,
                y,
                params.copy(),
                x=x,
                weights=weights,
                nan_policy=nan_policy,
                max_nfev=max_nfev,
                random_seed=random_seed,
            )
            if _result_is_usable(de_lm_result):
                candidates.append(("Differential Evolution + Levenberg-Marquardt", de_lm_result))
        except Exception as exc:
            errors.append(f"DE+LM failed: {exc}")

        if not candidates:
            detail = "; ".join(errors) if errors else "No usable result was produced."
            raise RuntimeError(f"Robust optimizer failed. {detail}")

        best_label, best_result = min(
            candidates,
            key=lambda item: fit_selection_score(item[1], selection_criterion),
        )

        try:
            best_result.fited_optimizer_mode = "Robust: compare LM and DE+LM"
            best_result.fited_selected_candidate = best_label
            best_result.fited_candidate_scores = {
                label: fit_selection_score(result, selection_criterion)
                for label, result in candidates
            }
            best_result.fited_selection_criterion = selection_criterion
            best_result.fited_candidate_errors = errors
            best_result.fited_random_seed = random_seed
        except Exception:
            pass
        return best_result

    result = _fit_lm_candidate(
        model,
        y,
        params=params,
        x=x,
        weights=weights,
        nan_policy=nan_policy,
        max_nfev=max_nfev,
    )
    try:
        result.fited_optimizer_mode = "Levenberg-Marquardt"
        result.fited_selected_candidate = "Levenberg-Marquardt"
        result.fited_selection_criterion = selection_criterion
        result.fited_candidate_scores = {}
        result.fited_candidate_errors = []
        result.fited_random_seed = random_seed
    except Exception:
        pass
    return result


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
    selected = getattr(result, "fited_selected_candidate", None)
    if selected:
        parts.append(f"selected optimizer: {selected}")
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

def full_fited_fit_report(result, optimizer_mode=None, fit_criterion=None):
    """
    Build the full FitED fit report from the stored lmfit result metadata.

    Parameters
    ----------
    result
        lmfit ModelResult object returned by FitED.
    optimizer_mode
        Optional explicit optimizer mode fallback for legacy results that do not
        yet contain result.fited_optimizer_mode.
    fit_criterion
        Optional explicit criterion fallback for legacy results that do not yet
        contain result.fited_selection_criterion.
    """
    if result is None:
        return ""

    stored_optimizer_mode = getattr(result, "fited_optimizer_mode", None)
    stored_fit_criterion = getattr(result, "fited_selection_criterion", None)

    optimizer_text = (
        str(stored_optimizer_mode)
        if stored_optimizer_mode not in (None, "")
        else str(optimizer_mode) if optimizer_mode not in (None, "")
        else "not recorded"
    )

    criterion_text = (
        str(stored_fit_criterion)
        if stored_fit_criterion not in (None, "")
        else str(fit_criterion) if fit_criterion not in (None, "")
        else "not recorded"
    )

    selected_candidate = getattr(result, "fited_selected_candidate", "")
    candidate_scores = getattr(result, "fited_candidate_scores", {})
    candidate_errors = getattr(result, "fited_candidate_errors", [])
    result_seed = getattr(result, "fited_random_seed", None)

    result_seed_text = (
        str(result_seed)
        if result_seed is not None
        else "not fixed / not recorded"
    )

    header_lines = [
        "[[FitED optimizer summary]]",
        f"    FitED optimizer mode        = {optimizer_text}",
        f"    FitED selection criterion   = {criterion_text}",
        f"    FitED random seed           = {result_seed_text}",
        f"    Selected optimizer candidate = {selected_candidate if selected_candidate else 'not recorded'}",
        f"    Candidate scores (lower is better) = {candidate_scores if candidate_scores else 'not recorded'}",
    ]

    if candidate_errors:
        header_lines.append(
            f"    Candidate errors             = {candidate_errors}"
        )

    header_lines.extend([
        "",
        "[[lmfit report]]",
    ])

    try:
        lmfit_report = result.fit_report(show_correl=True, min_correl=0.5)
    except Exception as exc:
        lmfit_report = f"Could not generate lmfit report:\n{exc}"

    return "\n".join(header_lines) + "\n" + lmfit_report

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

def _parse_constraint_bound(value):
    text = str(value).strip().lower()
    if text in {"inf", "+inf", "infinity", "+infinity"}:
        return np.inf
    if text in {"-inf", "-infinity"}:
        return -np.inf
    return float(value)


def parse_auxiliary_parameter_lines(text):
    """
    Parse auxiliary fit parameters used by lmfit constraint expressions.

    Format:
        name, value, min, max, vary

    Example:
        delta, 0.026, 0, 0.1, vary
        ratio, 1.0, 0, inf, vary
        offset, 0.0, -inf, inf, fixed
    """
    definitions = []

    for raw in str(text or "").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue

        parts = [p.strip() for p in line.split(",")]
        if len(parts) != 5:
            raise ValueError(
                "Each auxiliary parameter must be: name, value, min, max, vary/fixed"
            )

        name, value, pmin, pmax, vary_text = parts

        if not name.isidentifier() or keyword.iskeyword(name):
            raise ValueError(f"Invalid auxiliary parameter name: {name}")

        vary_key = vary_text.lower()
        if vary_key not in {"vary", "free", "true", "1", "yes", "fixed", "false", "0", "no"}:
            raise ValueError(
                f"Auxiliary parameter '{name}' has invalid vary/fixed flag: {vary_text}"
            )

        vary = vary_key in {"vary", "free", "true", "1", "yes"}

        definitions.append({
            "name": name,
            "value": float(value),
            "min": _parse_constraint_bound(pmin),
            "max": _parse_constraint_bound(pmax),
            "vary": bool(vary),
        })

    return definitions


def parse_parameter_constraint_lines(text):
    """
    Parse lmfit-style parameter constraints.

    Format:
        target_parameter = expression

    Example:
        p2_center = p1_center + delta
        p2_sigma = p1_sigma
        p2_amplitude = p1_amplitude * ratio
    """
    definitions = []

    for raw in str(text or "").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue

        if "=" not in line:
            raise ValueError(
                "Each constraint must be written as: target_parameter = expression"
            )

        target, expression = line.split("=", 1)
        target = target.strip()
        expression = expression.strip()

        if not target:
            raise ValueError("Constraint target cannot be empty.")
        if not expression:
            raise ValueError(f"Constraint for '{target}' has an empty expression.")

        definitions.append({
            "target": target,
            "expression": expression,
        })

    return definitions


def apply_parameter_constraints(
    params,
    auxiliary_parameters=None,
    parameter_constraints=None,
):
    """
    Add auxiliary parameters and apply lmfit expr constraints.

    This is general and works for built-in peaks, custom peaks, backgrounds,
    and any parameter present in the assembled lmfit Parameters object.
    """
    auxiliary_parameters = auxiliary_parameters or []
    parameter_constraints = parameter_constraints or []

    if not auxiliary_parameters and not parameter_constraints:
        return params

    for aux in auxiliary_parameters:
        name = str(aux.get("name", "")).strip()
        if not name:
            continue

        if name in params:
            raise ValueError(
                f"Auxiliary parameter '{name}' conflicts with an existing fit parameter."
            )

        pmin = float(aux.get("min", -np.inf))
        pmax = float(aux.get("max", np.inf))
        value = float(aux.get("value", 0.0))

        if pmin > pmax:
            raise ValueError(f"Auxiliary parameter '{name}' has min > max.")

        if np.isfinite(pmin) or np.isfinite(pmax):
            value = float(np.clip(value, pmin, pmax))

        params.add(
            name,
            value=value,
            min=pmin,
            max=pmax,
            vary=bool(aux.get("vary", True)),
        )

    for con in parameter_constraints:
        target = str(con.get("target", "")).strip()
        expression = str(con.get("expression", "")).strip()

        if not target or not expression:
            continue

        if target not in params:
            raise ValueError(
                f"Constraint target '{target}' is not an existing fit parameter."
            )

        try:
            params[target].set(expr=expression)
        except Exception as exc:
            raise ValueError(
                f"Could not apply constraint '{target} = {expression}': {exc}"
            ) from exc

    try:
        params.update_constraints()
    except Exception as exc:
        raise ValueError(
            f"Could not evaluate parameter constraints. Check names, expressions, and cycles: {exc}"
        ) from exc

    return params

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
    auxiliary_parameters=None,
    parameter_constraints=None,
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
            
    params = apply_parameter_constraints(
        params,
        auxiliary_parameters=auxiliary_parameters,
        parameter_constraints=parameter_constraints,
    )

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
            options=["none", "poisson-like", "sqrt(y) emphasis", "1/y"],
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
        elif mode == "sqrt(y) emphasis":
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
        result = fit_model_with_optimizer(
            model,
            y_raw,
            params=params,
            x=x,
            weights=weights,
            nan_policy="raise",
            optimizer_mode="Robust: compare LM and DE+LM"
        )

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

# =============================================================================
# Desktop-independent FitED logic helpers
# =============================================================================
#
# The desktop application imports the helpers in this section so Tk/Matplotlib
# code can remain focused on GUI construction, plotting, dialogs, and worker
# orchestration only. These functions intentionally preserve the behavior of
# the pre-separation desktop implementation.

import copy
import json

from scipy.signal import find_peaks, peak_prominences


AUTO_PREFIT_SAMPLING_MODES = [
    "Fast Jitter",
    "Latin Hypercube Sampling",
    "Hybrid: Fast Jitter + Latin Hypercube",
]
DEFAULT_AUTO_PREFIT_SAMPLING_MODE = AUTO_PREFIT_SAMPLING_MODES[0]

PEAK_DETECTION_DIRECTIONS = [
    "positive",
    "negative",
]

BATCH_FIT_MODES = [
    "Run fit using current parameters",
    "Auto pre-fit then final fit",
]

STABILITY_TEST_PROTOCOLS = [
    "Repeat Run fit",
    "Repeat Auto pre-fit",
]

# Keep seeds within the conventional unsigned 32-bit range accepted by
# SciPy/NumPy random-number interfaces used here.
MAX_RANDOM_SEED = 2**32 - 1

RESIDUAL_SUGGESTION_DIRECTIONS = [
    "positive",
    "negative",
    "both",
]

RESIDUAL_SUGGESTION_SENSITIVITIES = [
    "Conservative",
    "Normal",
    "Aggressive",
]

DERIVED_QUANTITY_FUNCTIONS = {
    "abs": abs,
    "sqrt": np.sqrt,
    "exp": np.exp,
    "log": np.log,
    "log10": np.log10,
    "sin": np.sin,
    "cos": np.cos,
    "tan": np.tan,
    "minimum": np.minimum,
    "maximum": np.maximum,
}

DERIVED_QUANTITY_CONSTANTS = {
    "pi": np.pi,
    "e": np.e,
}


def compute_weights(y, mode):
    """Calculate FitED weighting arrays from a plain weighting-mode string."""
    eps = 1e-12
    y = np.asarray(y, dtype=float)

    if mode == "none":
        return None
    if mode == "poisson-like":
        return 1.0 / np.sqrt(np.clip(np.abs(y), 1.0, None))
    if mode == "sqrt(y) emphasis":
        return np.sqrt(np.clip(np.abs(y), eps, None))
    if mode == "1/y":
        return 1.0 / np.clip(np.abs(y), eps, None)

    return None


def seed_with_offset(seed, offset=0):
    """Create a deterministic repeat/trial seed while staying in 32-bit range."""
    if seed is None:
        return None
    return int((int(seed) + int(offset)) % (MAX_RANDOM_SEED + 1))


def rng_from_seed(seed):
    """Create a local NumPy Generator for stochastic FitED trial sampling."""
    return np.random.default_rng(seed)


def raise_if_cancelled(cancel_event):
    """Raise the worker-level cancellation sentinel used by the desktop GUI."""
    if cancel_event is not None and cancel_event.is_set():
        raise RuntimeError("Fit cancelled.")


def build_model_from_context(context, peak_defs):
    """Build one lmfit model from a GUI-independent FitED context dictionary."""
    return build_composite_model(
        peak_defs,
        background_kind=context["background_kind"],
        poly_order=context["poly_order"],
        x=context["x"],
        y=context["y_raw"],
        custom_profiles=context["custom_profiles"],
        custom_background_profiles=context["custom_background_profiles"],
        custom_background_profile_name=context["custom_background_profile_name"],
        background_params=context["background_params"],
        auxiliary_parameters=context.get("auxiliary_parameters", []),
        parameter_constraints=context.get("parameter_constraints", []),
    )


def worker_fit_once(context, peak_defs, cancel_event, progress_queue=None, message="Fitting..."):
    """Run one fit from a FitED context without reading Tk variables."""
    raise_if_cancelled(cancel_event)
    if progress_queue is not None:
        progress_queue.put(("progress", (0, 1, message), None))

    model, params = build_model_from_context(context, peak_defs)
    result = fit_model_with_optimizer(
        model,
        context["y_raw"],
        params=params,
        x=context["x"],
        weights=context["weights"],
        nan_policy="raise",
        max_nfev=context["max_nfev"],
        optimizer_mode=context.get("optimizer_mode", DEFAULT_FIT_OPTIMIZER_MODE),
        selection_criterion=context.get("criterion", "AIC"),
        random_seed=context.get("random_seed"),
    )

    raise_if_cancelled(cancel_event)
    comps = result.eval_components(x=context["x"])
    return result, comps, result.best_fit


def seed_peak_defs_from_centers(x, y_raw, peak_defs, custom_profiles):
    """Estimate amplitude/FWHM/bounds for manually or automatically supplied centers."""
    x = np.asarray(x, dtype=float)
    y_raw = np.asarray(y_raw, dtype=float)

    x_span = max(float(np.max(x) - np.min(x)), 1e-12)
    y_min = float(np.min(y_raw))
    y_max = float(np.max(y_raw))
    y_span = max(y_max - y_min, 1.0)

    centers = [float(p["center"]) for p in peak_defs]
    centers_sorted = sorted(centers)

    new_defs = copy.deepcopy(peak_defs)

    for p in new_defs:
        c = float(p["center"])
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

        p["amplitude"] = area_guess
        p["fwhm"] = fwhm_guess
        p["center_min"] = center_min
        p["center_max"] = center_max
        p["amplitude_min"] = 0.0
        p["amplitude_max"] = max(area_guess * 12.0, y_span * x_span)
        p["fwhm_min"] = fwhm_min
        p["fwhm_max"] = fwhm_max
        p["sigma"] = max(fwhm_guess / 2.354820045, 1e-8)
        p["gamma"] = max(fwhm_guess / 2.0, 1e-8)

        if p.get("kind", "").strip().lower() == "custom" and p.get("custom_profile"):
            profile = custom_profiles.get(p["custom_profile"])
            if profile:
                for param in profile.get("parameters", []):
                    name = param["name"]
                    if name in {"center", "amplitude", "fwhm"}:
                        continue
                    p.setdefault(name, float(param.get("default", 1.0)))
                    p.setdefault(f"{name}_min", float(param.get("min", float("-inf"))))
                    p.setdefault(f"{name}_max", float(param.get("max", float("inf"))))

    return new_defs


def latin_hypercube_unit_samples(n_samples, n_dimensions, rng=None):
    """Generate Latin Hypercube samples in [0, 1] without adding SciPy dependencies."""
    n_samples = max(1, int(n_samples))
    n_dimensions = max(1, int(n_dimensions))
    rng = rng if rng is not None else np.random.default_rng()

    samples = np.empty((n_samples, n_dimensions), dtype=float)
    for dim in range(n_dimensions):
        values = (np.arange(n_samples, dtype=float) + rng.random(n_samples)) / n_samples
        rng.shuffle(values)
        samples[:, dim] = values

    return samples


def finite_trial_bounds(lower, upper, center, fallback_span=1.0, positive=False):
    """Return safe finite lower/upper bounds for trial generation."""
    try:
        lower = float(lower)
    except Exception:
        lower = np.nan

    try:
        upper = float(upper)
    except Exception:
        upper = np.nan

    try:
        center = float(center)
    except Exception:
        center = 0.0

    fallback_span = max(float(fallback_span), 1e-12)

    if not np.isfinite(lower) or not np.isfinite(upper) or lower >= upper:
        if positive:
            lower = max(1e-12, center / 10.0) if center > 0 else 1e-12
            upper = max(center * 10.0, fallback_span, lower * 1.0001)
        else:
            lower = center - fallback_span
            upper = center + fallback_span

    if positive:
        lower = max(lower, 1e-12)
        upper = max(upper, lower * 1.0001)

    return lower, upper


def sample_linear_from_unit(lower, upper, unit_value):
    """Map one unit value in [0, 1] into a linear parameter interval."""
    unit_value = float(np.clip(unit_value, 0.0, 1.0))
    return float(lower + unit_value * (upper - lower))


def sample_log_from_unit(lower, upper, unit_value):
    """Map one unit value in [0, 1] into a positive log-scaled interval."""
    unit_value = float(np.clip(unit_value, 0.0, 1.0))
    lower = max(float(lower), 1e-12)
    upper = max(float(upper), lower * 1.0001)
    return float(np.exp(np.log(lower) + unit_value * (np.log(upper) - np.log(lower))))


def approx_voigt_fwhm(sigma, gamma):
    """
    Approximate total Voigt FWHM from Gaussian sigma and Lorentzian gamma.

    Uses the Olivero-Longbothum approximation:
    FWHM_V ≈ 0.5346 * FWHM_L + sqrt(0.2166 * FWHM_L^2 + FWHM_G^2)
    """
    sigma = max(float(sigma), 1e-12)
    gamma = max(float(gamma), 1e-12)

    fwhm_g = 2.354820045 * sigma
    fwhm_l = 2.0 * gamma

    return float(
        0.5346 * fwhm_l
        + np.sqrt(0.2166 * fwhm_l ** 2 + fwhm_g ** 2)
    )


def lhs_dimension_count_for_peak_def(peak_def):
    """
    Number of LHS dimensions needed by one peak.

    Normal peaks use:
        center, FWHM, amplitude

    Exact Voigt uses:
        center, sigma, gamma, amplitude
    """
    kind = str(peak_def.get("kind", "")).strip().lower()
    if kind == "exact voigt":
        return 4
    return 3


def lhs_dimension_count_for_peak_defs(peak_defs):
    """Total LHS dimensions needed by a list of peak definitions."""
    total = 0
    for p in peak_defs:
        total += lhs_dimension_count_for_peak_def(p)
    return max(1, int(total))


def use_lhs_for_trial(sampling_mode, trial_index, n_trials):
    """Decide whether the current trial should use LHS or original fast jitter."""
    if sampling_mode == "Latin Hypercube Sampling":
        return True

    if sampling_mode == "Hybrid: Fast Jitter + Latin Hypercube":
        # Keep the first half as the original current method, then use LHS.
        # For one trial only, keep the original method.
        n_fast = max(1, int(np.ceil(float(n_trials) / 2.0)))
        return int(trial_index) >= n_fast

    return False


def apply_fast_jitter_sampling_to_peak_def(p, trial, x_span, rng=None):
    """
    Original Auto pre-fit trial-generation logic.

    This preserves the previous FitED behavior exactly as the default method.
    """
    rng = rng if rng is not None else np.random.default_rng()
    c = float(p["center"])
    cmin = float(p["center_min"])
    cmax = float(p["center_max"])
    fwhm = max(float(p["fwhm"]), 1e-12)
    amp = max(float(p["amplitude"]), 1e-12)

    center_scale = 1.0 + 0.02 * (trial % 3)
    width_scale = [0.8, 1.0, 1.25][trial % 3]
    amp_scale = [0.7, 1.0, 1.4][(trial // 3) % 3]

    jitter = 0.15 * (cmax - cmin)
    c_trial = np.clip(c + rng.uniform(-jitter, jitter), cmin, cmax)
    fwhm_trial = max(fwhm * width_scale * rng.uniform(0.85, 1.15), 1e-12)
    amp_trial = max(amp * amp_scale * rng.uniform(0.85, 1.15), 1e-12)

    half_window = max((cmax - cmin) * 0.30 * center_scale, x_span / 8000.0)
    new_cmin = c - half_window
    new_cmax = c + half_window
    new_fmin = max(float(p["fwhm_min"]) * 0.9, x_span / 6000.0)
    new_fmax = max(float(p["fwhm_max"]) * 1.05, new_fmin * 1.4)
    new_amin = 0.0
    new_amax = max(float(p["amplitude_max"]) * 1.2, amp_trial * 5.0)

    p["center"] = c_trial
    p["center_min"] = new_cmin
    p["center_max"] = new_cmax
    p["fwhm"] = np.clip(fwhm_trial, new_fmin, new_fmax)
    p["fwhm_min"] = new_fmin
    p["fwhm_max"] = new_fmax
    p["amplitude"] = np.clip(amp_trial, 1e-12, new_amax)
    p["amplitude_min"] = new_amin
    p["amplitude_max"] = new_amax
    p["sigma"] = max(float(p["fwhm"]) / 2.354820045, 1e-8)
    p["gamma"] = max(float(p["fwhm"]) / 2.0, 1e-8)


def apply_lhs_sampling_to_peak_def(p, lhs_row, dim_offset, x_span, y_span):
    """
    Latin Hypercube trial generation for one peak.

    For normal peak profiles, sample:
        center, FWHM, amplitude

    For Exact Voigt, sample:
        center, sigma, gamma, amplitude

    Returns the next LHS dimension offset.
    """
    kind = str(p.get("kind", "")).strip().lower()

    c = float(p.get("center", 0.0))
    fwhm = max(float(p.get("fwhm", 1.0)), 1e-12)
    amp = float(p.get("amplitude", 1.0))

    cmin, cmax = finite_trial_bounds(
        p.get("center_min", c - x_span),
        p.get("center_max", c + x_span),
        c,
        fallback_span=max(x_span, 1e-12),
        positive=False,
    )

    amin, amax = finite_trial_bounds(
        p.get("amplitude_min", 0.0),
        p.get("amplitude_max", max(abs(amp) * 10.0, y_span * x_span, 1.0)),
        amp,
        fallback_span=max(abs(amp) * 10.0, y_span * x_span, 1.0),
        positive=False,
    )

    p["center"] = sample_linear_from_unit(cmin, cmax, lhs_row[dim_offset])
    p["center_min"] = cmin
    p["center_max"] = cmax

    if kind == "exact voigt":
        sigma_guess = max(float(p.get("sigma", fwhm / 2.354820045)), 1e-12)
        gamma_guess = max(float(p.get("gamma", fwhm / 2.0)), 1e-12)

        fwhm_min = max(float(p.get("fwhm_min", fwhm / 10.0)), 1e-12)
        fwhm_max = max(float(p.get("fwhm_max", fwhm * 10.0)), fwhm_min * 1.0001)

        sigma_min, sigma_max = finite_trial_bounds(
            fwhm_min / 2.354820045,
            fwhm_max / 2.354820045,
            sigma_guess,
            fallback_span=max(sigma_guess * 10.0, x_span / 10.0, 1e-12),
            positive=True,
        )

        gamma_min, gamma_max = finite_trial_bounds(
            fwhm_min / 2.0,
            fwhm_max / 2.0,
            gamma_guess,
            fallback_span=max(gamma_guess * 10.0, x_span / 10.0, 1e-12),
            positive=True,
        )

        sigma = sample_log_from_unit(sigma_min, sigma_max, lhs_row[dim_offset + 1])
        gamma = sample_log_from_unit(gamma_min, gamma_max, lhs_row[dim_offset + 2])
        amplitude = sample_linear_from_unit(amin, amax, lhs_row[dim_offset + 3])

        p["sigma"] = max(float(sigma), 1e-8)
        p["gamma"] = max(float(gamma), 1e-8)
        p["fwhm"] = approx_voigt_fwhm(p["sigma"], p["gamma"])

        p["fwhm_min"] = fwhm_min
        p["fwhm_max"] = fwhm_max
        p["amplitude"] = amplitude
        p["amplitude_min"] = amin
        p["amplitude_max"] = amax

        return dim_offset + 4

    fmin, fmax = finite_trial_bounds(
        p.get("fwhm_min", fwhm / 10.0),
        p.get("fwhm_max", fwhm * 10.0),
        fwhm,
        fallback_span=max(x_span / 2.0, fwhm, 1e-12),
        positive=True,
    )

    sampled_fwhm = sample_log_from_unit(fmin, fmax, lhs_row[dim_offset + 1])
    amplitude = sample_linear_from_unit(amin, amax, lhs_row[dim_offset + 2])

    p["fwhm"] = sampled_fwhm
    p["fwhm_min"] = fmin
    p["fwhm_max"] = fmax
    p["amplitude"] = amplitude
    p["amplitude_min"] = amin
    p["amplitude_max"] = amax

    p["sigma"] = max(float(p["fwhm"]) / 2.354820045, 1e-8)
    p["gamma"] = max(float(p["fwhm"]) / 2.0, 1e-8)

    return dim_offset + 3


def randomize_custom_param_fast(cfg, rng=None):
    """Original custom-parameter randomization used for custom Auto pre-fit."""
    rng = rng if rng is not None else np.random.default_rng()
    val = float(cfg["value"])
    pmin = float(cfg["min"])
    pmax = float(cfg["max"])

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
        trial_val = np.exp(rng.uniform(np.log(lo), np.log(hi)))
    else:
        # linear randomization otherwise
        trial_val = rng.uniform(pmin, pmax)

    cfg["value"] = float(np.clip(trial_val, pmin, pmax))


def randomize_custom_param_lhs(cfg, unit_value):
    """Latin Hypercube randomization for one custom parameter."""
    val = float(cfg["value"])
    pmin = float(cfg["min"])
    pmax = float(cfg["max"])

    if not np.isfinite(pmin):
        pmin = val * 0.2 if val != 0 else -10.0
    if not np.isfinite(pmax):
        pmax = val * 5.0 if val != 0 else 10.0

    if pmin > pmax:
        pmin, pmax = pmax, pmin

    if pmin >= 0 and pmax > 0:
        trial_val = sample_log_from_unit(pmin, pmax, unit_value)
    else:
        trial_val = sample_linear_from_unit(pmin, pmax, unit_value)

    cfg["value"] = float(np.clip(trial_val, pmin, pmax))


def run_autoprefit_search_worker(context, base_peak_defs, cancel_event, progress_queue):
    """Run the standard Auto pre-fit repeated-search worker without GUI dependencies."""
    n_trials = context["n_trials"]
    sampling_mode = context.get("autoprefit_sampling_mode", DEFAULT_AUTO_PREFIT_SAMPLING_MODE)
    rng = rng_from_seed(context.get("random_seed"))

    best_result = None
    best_defs = None
    best_score = np.inf
    last_error = None

    x = np.asarray(context["x"], dtype=float)
    y_raw = np.asarray(context["y_raw"], dtype=float)
    x_span = max(float(np.max(x) - np.min(x)), 1e-12)
    y_span = max(float(np.max(y_raw) - np.min(y_raw)), 1.0)

    lhs_samples = None
    if sampling_mode in ["Latin Hypercube Sampling", "Hybrid: Fast Jitter + Latin Hypercube"]:
        lhs_dimensions = lhs_dimension_count_for_peak_defs(base_peak_defs)
        lhs_samples = latin_hypercube_unit_samples(n_trials, lhs_dimensions, rng=rng)

    for trial in range(n_trials):
        raise_if_cancelled(cancel_event)
        progress_queue.put((
            "progress",
            (trial, n_trials, f"Auto pre-fit [{sampling_mode}]: trial {trial + 1}/{n_trials} ..."),
            None
        ))

        trial_defs = copy.deepcopy(base_peak_defs)
        use_lhs = (
            lhs_samples is not None and
            use_lhs_for_trial(sampling_mode, trial, n_trials)
        )

        dim_offset = 0

        for p in trial_defs:
            if use_lhs:
                dim_offset = apply_lhs_sampling_to_peak_def(
                    p,
                    lhs_samples[trial],
                    dim_offset,
                    x_span=x_span,
                    y_span=y_span,
                )
            else:
                apply_fast_jitter_sampling_to_peak_def(
                    p,
                    trial=trial,
                    x_span=x_span,
                    rng=rng,
                )

        try:
            model, params = build_model_from_context(context, trial_defs)
            result = fit_model_with_optimizer(
                model,
                y_raw,
                params=params,
                x=x,
                weights=context["weights"],
                nan_policy="raise",
                max_nfev=context["max_nfev"],
                optimizer_mode=context.get("optimizer_mode", DEFAULT_FIT_OPTIMIZER_MODE),
                selection_criterion=context.get("criterion", "AIC"),
                random_seed=seed_with_offset(context.get("random_seed"), trial),
            )
            raise_if_cancelled(cancel_event)

            if not np.all(np.isfinite(result.best_fit)):
                continue

            score = fit_selection_score(result, context["criterion"])
            if np.isfinite(score) and score < best_score:
                best_score = score
                best_result = result
                best_defs = copy.deepcopy(trial_defs)
        except Exception as exc:
            last_error = exc
            continue

    if best_result is None and last_error is not None:
        raise RuntimeError(f"Automatic pre-fit failed in all trials. Last error: {last_error}")
    if best_result is None:
        raise RuntimeError("Automatic pre-fit failed for all attempts.")

    progress_queue.put(("progress", (n_trials, n_trials, "Auto pre-fit: finished."), None))
    return best_result, best_defs


def run_autoprefit_search_custom_worker(context, peak_defs, cancel_event, progress_queue):
    """Run the custom-profile Auto pre-fit repeated-search worker without GUI dependencies."""
    n_trials = context["n_trials"]
    sampling_mode = context.get("autoprefit_sampling_mode", DEFAULT_AUTO_PREFIT_SAMPLING_MODE)
    rng = rng_from_seed(context.get("random_seed"))

    best_result = None
    best_defs = None
    best_score = np.inf
    last_error = None
    x = context["x"]
    y_raw = context["y_raw"]

    custom_param_count = 0
    for p in peak_defs:
        if p.get("kind") == "Custom":
            custom_param_count += len(p.get("custom_params", {}))

    lhs_samples = None
    if sampling_mode in ["Latin Hypercube Sampling", "Hybrid: Fast Jitter + Latin Hypercube"]:
        lhs_samples = latin_hypercube_unit_samples(n_trials, max(1, custom_param_count), rng=rng)

    for trial in range(n_trials):
        raise_if_cancelled(cancel_event)
        progress_queue.put((
            "progress",
            (trial, n_trials, f"Custom auto pre-fit [{sampling_mode}]: trial {trial + 1}/{n_trials} ..."),
            None
        ))

        trial_defs = copy.deepcopy(peak_defs)
        use_lhs = (
            lhs_samples is not None and
            use_lhs_for_trial(sampling_mode, trial, n_trials)
        )

        lhs_dim = 0
        for p in trial_defs:
            if p.get("kind") != "Custom":
                continue

            custom_params = p.get("custom_params", {})
            for pname, cfg in custom_params.items():
                if use_lhs:
                    randomize_custom_param_lhs(cfg, lhs_samples[trial, lhs_dim])
                else:
                    randomize_custom_param_fast(cfg, rng=rng)
                lhs_dim += 1

        try:
            model, params = build_model_from_context(context, trial_defs)
            result = fit_model_with_optimizer(
                model,
                y_raw,
                params=params,
                x=x,
                weights=context["weights"],
                nan_policy="raise",
                max_nfev=context["max_nfev"],
                optimizer_mode=context.get("optimizer_mode", DEFAULT_FIT_OPTIMIZER_MODE),
                selection_criterion=context.get("criterion", "AIC"),
                random_seed=seed_with_offset(context.get("random_seed"), trial),
            )
            raise_if_cancelled(cancel_event)

            if not np.all(np.isfinite(result.best_fit)):
                continue

            score = fit_selection_score(result, context["criterion"])
            if np.isfinite(score) and score < best_score:
                best_score = score
                best_result = result
                best_defs = copy.deepcopy(trial_defs)
        except Exception as exc:
            last_error = exc
            continue

    if best_result is None and last_error is not None:
        raise RuntimeError(f"Custom auto pre-fit failed. Last error: {last_error}")
    if best_result is None:
        raise RuntimeError("Automatic pre-fit failed for all attempts.")

    progress_queue.put(("progress", (n_trials, n_trials, "Custom auto pre-fit: finished."), None))
    return best_result, best_defs


def copy_fit_result_values_into_params(params, result):
    """
    Copy fitted parameter values into a new Parameters object.

    Bounds are expanded if needed, matching the safe logic already used in
    Refine with added peaks.
    """
    for name, par in params.items():
        if name not in result.params:
            continue

        try:
            val = float(result.params[name].value)
            current_min = par.min
            current_max = par.max

            if np.isfinite(current_min) and val < current_min:
                margin = max(abs(val) * 0.1, 1e-12)
                params[name].set(min=val - margin)

            if np.isfinite(current_max) and val > current_max:
                margin = max(abs(val) * 0.1, 1e-12)
                params[name].set(max=val + margin)

            params[name].set(value=val)
        except Exception:
            pass

    return params


def refine_with_added_peaks_worker(context, params_prev, old_count, active_count, cancel_event, progress_queue):
    """Run the two-stage Refine-with-added-peaks search without Tk dependencies."""
    x = context["x"]
    y_raw = context["y_raw"]
    peak_defs = context["peak_defs"]
    staged_defs = copy.deepcopy(peak_defs)

    for i in range(min(old_count, len(staged_defs))):
        p = staged_defs[i]
        prefix = f"p{i+1}_"
        if f"{prefix}center" in params_prev:
            c = float(params_prev[f"{prefix}center"].value)
            p["center"] = c
            halfw = max(0.15 * (float(p["center_max"]) - float(p["center_min"])), 1e-6)
            p["center_min"] = c - halfw
            p["center_max"] = c + halfw
        if f"{prefix}amplitude" in params_prev:
            a = max(float(params_prev[f"{prefix}amplitude"].value), 1e-12)
            p["amplitude"] = a
            p["amplitude_min"] = 0.0
            p["amplitude_max"] = max(a * 2.0, float(p["amplitude_max"]))
        if f"{prefix}gamma" in params_prev:
            p["gamma"] = max(float(params_prev[f"{prefix}gamma"].value), 1e-8)
        if f"{prefix}sigma" in params_prev:
            sigma = max(float(params_prev[f"{prefix}sigma"].value), 1e-8)
            p["sigma"] = sigma
            fwhm = (
                approx_voigt_fwhm(sigma, p["gamma"])
                if str(p.get("kind", "")).strip().lower() == "exact voigt"
                else 2.354820045 * sigma
            )
            p["fwhm"] = fwhm
            p["fwhm_min"] = max(fwhm * 0.7, 1e-8)
            p["fwhm_max"] = max(fwhm * 1.5, p["fwhm_min"] * 1.2)

    if old_count < len(staged_defs):
        new_defs = seed_peak_defs_from_centers(
            x,
            y_raw,
            staged_defs[old_count:],
            context["custom_profiles"],
        )
        staged_defs[old_count:] = new_defs

    best_result = None
    best_score = np.inf
    n_trials = context["n_trials"]
    rng = rng_from_seed(context.get("random_seed"))

    # Use the same Auto pre-fit sampling choice for the newly added peaks.
    # Old peaks are still initialized near the previous fit result.
    sampling_mode = context.get("autoprefit_sampling_mode", DEFAULT_AUTO_PREFIT_SAMPLING_MODE)
    new_peak_count = max(0, len(staged_defs) - old_count)

    x_span = max(float(np.max(x) - np.min(x)), 1e-12)
    y_span = max(float(np.max(y_raw) - np.min(y_raw)), 1.0)

    lhs_samples = None
    if new_peak_count > 0 and sampling_mode in ["Latin Hypercube Sampling", "Hybrid: Fast Jitter + Latin Hypercube"]:
        lhs_dimensions = lhs_dimension_count_for_peak_defs(staged_defs[old_count:])
        lhs_samples = latin_hypercube_unit_samples(
            n_trials,
            lhs_dimensions,
            rng=rng,
        )

    for trial in range(n_trials):
        raise_if_cancelled(cancel_event)
        progress_queue.put((
            "progress",
            (trial, n_trials, f"Refine with added peaks [{sampling_mode}]: trial {trial + 1}/{n_trials} ..."),
            None
        ))
        trial_defs = copy.deepcopy(staged_defs)

        use_lhs = (
            lhs_samples is not None and
            use_lhs_for_trial(sampling_mode, trial, n_trials)
        )

        dim_offset = 0

        for i in range(old_count, len(trial_defs)):
            p = trial_defs[i]

            if use_lhs:
                dim_offset = apply_lhs_sampling_to_peak_def(
                    p,
                    lhs_samples[trial],
                    dim_offset,
                    x_span=x_span,
                    y_span=y_span,
                )
                continue

            # Original Refine-with-added-peaks fast-jitter logic.
            # Keep this unchanged to preserve the current default behavior.
            c = float(p["center"])
            cmin = float(p["center_min"])
            cmax = float(p["center_max"])
            fwhm = max(float(p["fwhm"]), 1e-12)
            amp = max(float(p["amplitude"]), 1e-12)
            width_scale = [0.8, 1.0, 1.25][trial % 3]
            amp_scale = [0.7, 1.0, 1.4][(trial // 3) % 3]
            jitter = 0.12 * (cmax - cmin)

            p["center"] = np.clip(c + rng.uniform(-jitter, jitter), cmin, cmax)
            p["fwhm"] = np.clip(fwhm * width_scale, float(p["fwhm_min"]), float(p["fwhm_max"]))
            p["amplitude"] = np.clip(amp * amp_scale, 1e-12, float(p["amplitude_max"]))
            p["sigma"] = max(float(p["fwhm"]) / 2.354820045, 1e-8)
            p["gamma"] = max(float(p["fwhm"]) / 2.0, 1e-8)

        try:
            model, params = build_model_from_context(context, trial_defs)
            result = fit_model_with_optimizer(
                model,
                y_raw,
                params=params,
                x=x,
                weights=context["weights"],
                nan_policy="raise",
                max_nfev=context["max_nfev"],
                optimizer_mode=context.get("optimizer_mode", DEFAULT_FIT_OPTIMIZER_MODE),
                selection_criterion=context.get("criterion", "AIC"),
                random_seed=seed_with_offset(context.get("random_seed"), trial),
            )
            raise_if_cancelled(cancel_event)
            score = fit_selection_score(result, context["criterion"])
            if np.isfinite(score) and score < best_score:
                best_score = score
                best_result = result
        except Exception:
            continue

    if best_result is None:
        raise RuntimeError("Refinement failed for all trials.")

    model2, params2 = build_model_from_context(context, peak_defs)
    params2 = copy_fit_result_values_into_params(params2, best_result)

    final_result = fit_model_with_optimizer(
        model2,
        y_raw,
        params=params2,
        x=x,
        weights=context["weights"],
        nan_policy="raise",
        max_nfev=context["max_nfev"],
        optimizer_mode=context.get("optimizer_mode", DEFAULT_FIT_OPTIMIZER_MODE),
        selection_criterion=context.get("criterion", "AIC"),
        random_seed=seed_with_offset(context.get("random_seed"), n_trials),
    )
    raise_if_cancelled(cancel_event)
    progress_queue.put(("progress", (n_trials, n_trials, "Refine with added peaks: finished."), None))

    return {
        "context": context,
        "result": final_result,
        "components": final_result.eval_components(x=x),
        "best_fit": final_result.best_fit,
        "active_count": active_count,
        "peak_defs": peak_defs,
    }


class NoopProgressQueue:
    """Small helper used to silence nested trial messages inside larger workers."""
    def put(self, *args, **kwargs):
        return None


def summarize_stability_test(records, failures, protocol, repeats, delta_score, base_seed, base_context):
    """Create score, optimizer-frequency, and near-best parameter-spread summaries."""
    records_sorted = sorted(records, key=lambda item: item["score"])
    best_record = records_sorted[0]

    scores = np.asarray([float(item["score"]) for item in records_sorted], dtype=float)
    best_score = float(scores[0])
    median_score = float(np.median(scores))
    mean_score = float(np.mean(scores))
    std_score = float(np.std(scores, ddof=1)) if scores.size > 1 else 0.0
    worst_score = float(np.max(scores))

    delta_score = max(0.0, float(delta_score))
    near_best_records = [
        item for item in records_sorted
        if float(item["score"]) <= best_score + delta_score
    ]

    optimizer_counts = {}
    for item in records_sorted:
        label = str(item.get("selected_candidate", "") or "not recorded")
        optimizer_counts[label] = optimizer_counts.get(label, 0) + 1

    parameter_order = []
    seen = set()
    for pname in best_record["result"].params.keys():
        if pname not in seen:
            parameter_order.append(pname)
            seen.add(pname)
    for item in near_best_records:
        for pname in item["result"].params.keys():
            if pname not in seen:
                parameter_order.append(pname)
                seen.add(pname)

    parameter_spread = []
    for pname in parameter_order:
        values = []
        for item in near_best_records:
            par = item["result"].params.get(pname)
            if par is None:
                continue
            try:
                value = float(par.value)
            except Exception:
                continue
            if np.isfinite(value):
                values.append(value)

        if not values:
            continue

        arr = np.asarray(values, dtype=float)
        try:
            best_value = float(best_record["result"].params[pname].value)
        except Exception:
            best_value = np.nan

        parameter_spread.append({
            "parameter": pname,
            "best_value": best_value,
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0,
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
            "n": int(arr.size),
        })

    return {
        "protocol": protocol,
        "requested_repeats": int(repeats),
        "successful_repeats": int(len(records_sorted)),
        "failed_repeats": int(len(failures)),
        "failures": failures,
        "base_seed": base_seed,
        "criterion": base_context.get("criterion", "AIC"),
        "optimizer_mode": base_context.get("optimizer_mode", DEFAULT_FIT_OPTIMIZER_MODE),
        "autoprefit_sampling_mode": base_context.get("autoprefit_sampling_mode", DEFAULT_AUTO_PREFIT_SAMPLING_MODE),
        "n_trials": int(base_context.get("n_trials", 0)),
        "delta_score": delta_score,
        "best_score": best_score,
        "median_score": median_score,
        "mean_score": mean_score,
        "std_score": std_score,
        "worst_score": worst_score,
        "records": records_sorted,
        "near_best_records": near_best_records,
        "optimizer_counts": optimizer_counts,
        "parameter_spread": parameter_spread,
        "best_record": best_record,
    }


def run_stability_test_worker(setup, cancel_event, progress_queue):
    """Run repeated Run-fit or Auto-pre-fit stability testing without Tk dependencies."""
    protocol = setup["protocol"]
    repeats = int(setup["repeats"])
    delta_score = float(setup["delta_score"])
    base_seed = setup.get("base_seed")
    context = setup["context"]

    records = []
    failures = []
    quiet_queue = NoopProgressQueue()

    for repeat_index in range(repeats):
        raise_if_cancelled(cancel_event)
        repeat_number = repeat_index + 1
        repeat_seed = seed_with_offset(base_seed, repeat_index)
        repeat_context = copy.deepcopy(context)
        repeat_context["random_seed"] = repeat_seed

        seed_text = "stochastic" if repeat_seed is None else str(repeat_seed)
        progress_queue.put((
            "progress",
            (
                repeat_index,
                repeats,
                f"Fit stability test [{protocol}]: repeat {repeat_number}/{repeats}; seed={seed_text} ..."
            ),
            None
        ))

        try:
            if protocol == "Repeat Run fit":
                result, comps, best = worker_fit_once(
                    repeat_context,
                    repeat_context["peak_defs"],
                    cancel_event,
                    progress_queue=None,
                    message="Stability run fit..."
                )
                peak_defs = copy.deepcopy(repeat_context["peak_defs"])
            elif protocol == "Repeat Auto pre-fit":
                if repeat_context.get("all_custom_no_center", False):
                    result, peak_defs = run_autoprefit_search_custom_worker(
                        repeat_context,
                        repeat_context["peak_defs"],
                        cancel_event,
                        quiet_queue
                    )
                else:
                    seeded_peak_defs = seed_peak_defs_from_centers(
                        repeat_context["x"],
                        repeat_context["y_raw"],
                        repeat_context["peak_defs"],
                        repeat_context["custom_profiles"],
                    )
                    result, peak_defs = run_autoprefit_search_worker(
                        repeat_context,
                        seeded_peak_defs,
                        cancel_event,
                        quiet_queue
                    )
                comps = result.eval_components(x=repeat_context["x"])
                best = result.best_fit
            else:
                raise ValueError(f"Unsupported stability protocol: {protocol}")

            score = fit_selection_score(result, repeat_context["criterion"])
            if not np.isfinite(score):
                raise RuntimeError("The selected stability score is non-finite.")

            records.append({
                "repeat": repeat_number,
                "seed": repeat_seed,
                "score": float(score),
                "result": result,
                "components": comps,
                "best_fit": np.asarray(best, dtype=float).copy(),
                "peak_defs": copy.deepcopy(peak_defs),
                "context": repeat_context,
                "selected_candidate": getattr(result, "fited_selected_candidate", ""),
                "candidate_scores": copy.deepcopy(getattr(result, "fited_candidate_scores", {})),
            })
        except Exception as exc:
            failures.append({
                "repeat": repeat_number,
                "seed": repeat_seed,
                "error": str(exc),
            })
            continue

    if not records:
        if failures:
            detail = failures[-1].get("error", "No successful repeat.")
            raise RuntimeError(f"Fit stability test failed for all repeats. Last error: {detail}")
        raise RuntimeError("Fit stability test did not produce any successful repeat.")

    progress_queue.put((
        "progress",
        (repeats, repeats, "Fit stability test: summarizing repeated solutions..."),
        None
    ))

    return summarize_stability_test(
        records=records,
        failures=failures,
        protocol=protocol,
        repeats=repeats,
        delta_score=delta_score,
        base_seed=base_seed,
        base_context=context,
    )


def format_stability_test_report(payload):
    """Create a text report for the repeated-fit stability test."""
    protocol = payload.get("protocol", "")
    criterion = payload.get("criterion", "")
    base_seed = payload.get("base_seed")
    base_seed_text = "blank / stochastic" if base_seed is None else str(base_seed)
    successful = int(payload.get("successful_repeats", 0))
    failed = int(payload.get("failed_repeats", 0))
    requested = int(payload.get("requested_repeats", successful + failed))
    near_best = payload.get("near_best_records", [])
    delta_score = float(payload.get("delta_score", 0.0))

    lines = [
        "[[FitED fit stability test]]",
        f"Protocol                         = {protocol}",
        f"Selection criterion              = {criterion}",
        f"Optimizer mode                   = {payload.get('optimizer_mode', '')}",
        f"Auto pre-fit sampling            = {payload.get('autoprefit_sampling_mode', '')}",
        f"Auto-fit trials per Auto pre-fit = {payload.get('n_trials', '')}",
        f"Repeated searches requested      = {requested}",
        f"Successful / failed repeats      = {successful} / {failed}",
        f"Base random seed                  = {base_seed_text}",
        "Repeat seed rule                  = base seed + repeat index when a base seed is set",
        f"Near-best threshold               = score <= best score + {delta_score:.8g}",
        f"Near-best repeated solutions      = {len(near_best)}",
        "",
        "[[Score spread]]",
        f"Best {criterion:<24} = {float(payload.get('best_score', np.nan)):.10g}",
        f"Median {criterion:<22} = {float(payload.get('median_score', np.nan)):.10g}",
        f"Mean {criterion:<24} = {float(payload.get('mean_score', np.nan)):.10g}",
        f"Std {criterion:<25} = {float(payload.get('std_score', np.nan)):.10g}",
        f"Worst {criterion:<23} = {float(payload.get('worst_score', np.nan)):.10g}",
        "",
        "[[Selected optimizer candidate frequency]]",
    ]

    optimizer_counts = payload.get("optimizer_counts", {})
    if optimizer_counts:
        for label, count in sorted(optimizer_counts.items(), key=lambda item: (-item[1], item[0])):
            lines.append(f"{label:<48} {int(count):>6}")
    else:
        lines.append("not recorded")

    lines.extend([
        "",
        "[[Repeated solution scores]]",
        f"{'Repeat':>8} {'Seed':>14} {'Score':>18} {'Δ from best':>18}  Selected optimizer",
        "-" * 94,
    ])

    best_score = float(payload.get("best_score", np.nan))
    for item in payload.get("records", []):
        seed = item.get("seed")
        seed_txt = "stochastic" if seed is None else str(seed)
        score = float(item.get("score", np.nan))
        delta = score - best_score
        lines.append(
            f"{int(item.get('repeat', 0)):>8} {seed_txt:>14} {score:>18.10g} {delta:>18.10g}  "
            f"{item.get('selected_candidate', '') or 'not recorded'}"
        )

    lines.extend([
        "",
        "[[Near-best parameter spread]]",
        f"{'Parameter':<28} {'Best':>16} {'Mean':>16} {'Std':>16} {'Min':>16} {'Max':>16} {'N':>6}",
        "-" * 120,
    ])

    spread_rows = payload.get("parameter_spread", [])
    if spread_rows:
        for row in spread_rows:
            lines.append(
                f"{row.get('parameter', ''):<28} "
                f"{float(row.get('best_value', np.nan)):>16.8g} "
                f"{float(row.get('mean', np.nan)):>16.8g} "
                f"{float(row.get('std', np.nan)):>16.8g} "
                f"{float(row.get('min', np.nan)):>16.8g} "
                f"{float(row.get('max', np.nan)):>16.8g} "
                f"{int(row.get('n', 0)):>6}"
            )
    else:
        lines.append("No parameter spread could be computed.")

    failures = payload.get("failures", [])
    if failures:
        lines.extend(["", "[[Failed repeats]]"])
        for item in failures:
            seed = item.get("seed")
            seed_txt = "stochastic" if seed is None else str(seed)
            lines.append(
                f"Repeat {item.get('repeat', '?')} seed={seed_txt}: {item.get('error', '')}"
            )

    lines.extend([
        "",
        "Interpretation note:",
        "- A narrow score spread and narrow near-best parameter spread indicate a more stable numerical decomposition.",
        "- A narrow score spread but wide parameter spread indicates multiple alternative decompositions with similar fit quality.",
        "- This diagnostic complements residual inspection, parameter correlations, and domain knowledge; it does not by itself prove physical uniqueness.",
    ])

    return "\n".join(lines)


def split_batch_patterns(pattern_text):
    """Split user file patterns such as '*.txt;*.csv;*.dat'."""
    raw = str(pattern_text or "").replace(",", ";").split(";")
    patterns = [p.strip() for p in raw if p.strip()]
    if not patterns:
        patterns = ["*.txt", "*.csv", "*.dat", "*.asc"]
    return patterns


def collect_batch_files(input_folder, pattern_text, recursive=False):
    """Collect files for batch fitting from one folder."""
    input_folder = Path(input_folder)
    patterns = split_batch_patterns(pattern_text)

    files = []
    seen = set()

    for pattern in patterns:
        iterator = input_folder.rglob(pattern) if recursive else input_folder.glob(pattern)
        for path in iterator:
            path = Path(path)
            if not path.is_file():
                continue

            key = str(path.resolve())
            if key in seen:
                continue

            seen.add(key)
            files.append(path)

    return sorted(files, key=lambda p: str(p).lower())


def batch_context_for_file(filepath, template):
    """Load one file and build the fit context used by existing fitting helpers."""
    loader = template["loader"]

    x_full, y_full = load_spectrum(
        filepath,
        x_col=int(loader["x_col"]),
        y_col=int(loader["y_col"]),
        delimiter=loader["delimiter"],
        skiprows=int(loader["skiprows"]),
    )

    x, y_raw = crop_roi(
        np.asarray(x_full, dtype=float),
        np.asarray(y_full, dtype=float),
        template["roi_min"],
        template["roi_max"],
    )

    if len(x) == 0:
        raise ValueError("No data points inside ROI.")

    return {
        "x": np.asarray(x, dtype=float).copy(),
        "y_raw": np.asarray(y_raw, dtype=float).copy(),
        "y_plot": np.asarray(y_raw, dtype=float).copy(),
        "peak_defs": copy.deepcopy(template["peak_defs"]),
        "background_kind": template["background_kind"],
        "poly_order": int(template["poly_order"]),
        "custom_profiles": copy.deepcopy(template["custom_profiles"]),
        "custom_background_profiles": copy.deepcopy(template["custom_background_profiles"]),
        "custom_background_profile_name": template["custom_background_profile_name"],
        "background_params": copy.deepcopy(template["background_params"]),
        'auxiliary_parameters': copy.deepcopy(template.get('auxiliary_parameters', [])),
        'parameter_constraints': copy.deepcopy(template.get('parameter_constraints', [])),
        "weights": compute_weights(y_raw, template["weighting"]),
        "criterion": template["criterion"],
        "n_trials": int(template["n_trials"]),
        "autoprefit_sampling_mode": template["autoprefit_sampling_mode"],
        "max_nfev": int(template["max_nfev"]),
        "optimizer_mode": template["optimizer_mode"],
        "random_seed": template.get("random_seed"),
        "all_custom_no_center": bool(template.get("all_custom_no_center", False)),
    }


def batch_fit_one_file(filepath, template, cancel_event):
    """Fit one spectrum using the selected batch mode."""
    context = batch_context_for_file(filepath, template)

    batch_mode = template.get("batch_mode", BATCH_FIT_MODES[0])

    # Mode 1: direct fit using the current peak table/session values.
    if batch_mode == "Run fit using current parameters":
        result, comps, best = worker_fit_once(
            context,
            context["peak_defs"],
            cancel_event,
            progress_queue=None,
            message="Batch fitting..."
        )
        return result

    # Mode 2: Auto pre-fit first, then final fit from the best pre-fit result.
    peak_defs = copy.deepcopy(context["peak_defs"])

    # Background-only case: Auto pre-fit has no peaks to seed, so just run fit.
    if not peak_defs:
        result, comps, best = worker_fit_once(
            context,
            peak_defs,
            cancel_event,
            progress_queue=None,
            message="Batch fitting..."
        )
        return result

    dummy_queue = NoopProgressQueue()

    if context.get("all_custom_no_center", False):
        best_result, best_peak_defs = run_autoprefit_search_custom_worker(
            context,
            peak_defs,
            cancel_event,
            dummy_queue
        )
    else:
        seeded_peak_defs = seed_peak_defs_from_centers(
            context["x"],
            context["y_raw"],
            peak_defs,
            context["custom_profiles"],
        )
        best_result, best_peak_defs = run_autoprefit_search_worker(
            context,
            seeded_peak_defs,
            cancel_event,
            dummy_queue
        )

    # Final fit/polish from the best Auto pre-fit solution.
    model, params = build_model_from_context(context, best_peak_defs)
    params = copy_fit_result_values_into_params(params, best_result)

    final_result = fit_model_with_optimizer(
        model,
        context["y_raw"],
        params=params,
        x=context["x"],
        weights=context["weights"],
        nan_policy="raise",
        max_nfev=context["max_nfev"],
        optimizer_mode=context.get("optimizer_mode", DEFAULT_FIT_OPTIMIZER_MODE),
        selection_criterion=context.get("criterion", "AIC"),
        random_seed=seed_with_offset(context.get("random_seed"), int(context.get("n_trials", 0))),
    )

    raise_if_cancelled(cancel_event)
    return final_result


def batch_result_to_row(filepath, result):
    """Convert one successful lmfit result into one summary-table row."""
    row = {
        "file": Path(filepath).name,
        "path": str(filepath),
        "status": "ok",
        "error": "",
        "chisqr": float(getattr(result, "chisqr", np.nan)),
        "redchi": float(getattr(result, "redchi", np.nan)),
        "aic": float(getattr(result, "aic", np.nan)),
        "bic": float(getattr(result, "bic", np.nan)),
        "nfev": int(getattr(result, "nfev", -1)),
        "ndata": int(getattr(result, "ndata", -1)),
        "nvarys": int(getattr(result, "nvarys", -1)),
        "optimizer_mode": str(getattr(result, "fited_optimizer_mode", "")),
        "selected_optimizer_candidate": getattr(result, "fited_selected_candidate", ""),
        "random_seed": getattr(result, "fited_random_seed", ""),
    }

    for name, par in result.params.items():
        try:
            row[name] = float(par.value)
        except Exception:
            row[name] = par.value

        try:
            row[f"{name}_stderr"] = float(par.stderr) if par.stderr is not None else np.nan
        except Exception:
            row[f"{name}_stderr"] = np.nan

    return row


def batch_failed_row(filepath, error):
    """Create one summary row for a failed file."""
    return {
        "file": Path(filepath).name,
        "path": str(filepath),
        "status": "failed",
        "error": str(error),
    }


def write_batch_outputs(output_folder, rows, template):
    """Write batch summary files."""
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(rows)

    csv_path = output_folder / "batch_summary.csv"
    df.to_csv(csv_path, index=False)

    xlsx_path = output_folder / "batch_summary.xlsx"
    excel_created = False
    try:
        import openpyxl  # noqa: F401
        with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
            df.to_excel(writer, sheet_name="batch_summary", index=False)

            meta_df = pd.DataFrame([
                {"field": "batch_mode", "value": template.get("batch_mode", "")},
                {"field": "roi_min", "value": template.get("roi_min", "")},
                {"field": "roi_max", "value": template.get("roi_max", "")},
                {"field": "background", "value": template.get("background_kind", "")},
                {"field": "poly_order", "value": template.get("poly_order", "")},
                {"field": "weighting", "value": template.get("weighting", "")},
                {"field": "fit_criterion", "value": template.get("criterion", "")},
                {"field": "optimizer_mode", "value": template.get("optimizer_mode", "")},
                {"field": "random_seed", "value": template.get("random_seed", "")},
                {"field": "autoprefit_sampling_mode", "value": template.get("autoprefit_sampling_mode", "")},
                {"field": "max_nfev", "value": template.get("max_nfev", "")},
                {"field": "n_trials", "value": template.get("n_trials", "")},
            ])
            meta_df.to_excel(writer, sheet_name="batch_settings", index=False)

        excel_created = True
    except Exception:
        excel_created = False

    session_path = output_folder / "batch_template_session.json"
    try:
        session_path.write_text(
            json.dumps(template.get("session_state", {}), indent=2),
            encoding="utf-8"
        )
    except Exception:
        pass

    return {
        "csv_path": csv_path,
        "xlsx_path": xlsx_path if excel_created else None,
        "session_path": session_path,
        "n_rows": len(rows),
        "n_ok": int((df.get("status") == "ok").sum()) if "status" in df else 0,
        "n_failed": int((df.get("status") == "failed").sum()) if "status" in df else 0,
    }


def default_peak_state_for_range(idx, count, x_min=0.0, x_max=100.0):
    """Build the default editable peak-state dictionary used by desktop workflows."""
    x_min = float(x_min)
    x_max = float(x_max)
    x_span = max(x_max - x_min, 1.0)
    center_guess = x_min + (idx + 1) * x_span / (count + 1)
    amp_guess = 0.01
    fwhm_guess = max(x_span / 80.0, 1e-4)
    return {
        "active": True,
        "kind": "Pseudo-Voigt",
        "custom_profile": "",
        "center": center_guess,
        "amplitude": amp_guess,
        "fwhm": fwhm_guess,
        "center_min": center_guess - 2 * fwhm_guess,
        "center_max": center_guess + 2 * fwhm_guess,
        "amplitude_min": 0.0,
        "amplitude_max": max(amp_guess * 5, 1.0),
        "fwhm_min": max(x_span / 1000.0, 1e-3),
        "fwhm_max": max(fwhm_guess * 5, x_span / 50.0),
        "fraction": 0.5,
        "sigma": max(fwhm_guess / 2.354820045, 1e-8),
        "gamma": max(fwhm_guess / 2.0, 1e-8),
        "custom_params": {},
    }


def x_units_to_samples(x, value):
    """Convert a distance/width in x-units to sample units for scipy.find_peaks."""
    try:
        value = float(value)
    except Exception:
        return None

    if not np.isfinite(value) or value <= 0:
        return None

    x = np.asarray(x, dtype=float)
    if x.size < 2:
        return None

    dx = np.median(np.abs(np.diff(x)))
    if not np.isfinite(dx) or dx <= 0:
        return None

    return max(1, int(np.ceil(value / dx)))


def negative_peak_seed_from_centers(x, y_raw, peak_defs):
    """
    Convert positive-peak initial guesses into negative-peak guesses.

    This is used only when the Find peaks direction is set to 'negative'.
    It allows downward dips to be fitted with negative amplitudes.
    """
    x = np.asarray(x, dtype=float)
    y_raw = np.asarray(y_raw, dtype=float)

    y_min = float(np.min(y_raw))
    y_max = float(np.max(y_raw))
    y_span = max(y_max - y_min, 1.0)
    x_span = max(float(np.max(x) - np.min(x)), 1e-12)

    new_defs = copy.deepcopy(peak_defs)

    for p in new_defs:
        center = float(p.get("center", 0.0))
        idx = int(np.argmin(np.abs(x - center)))
        local_y = float(y_raw[idx])

        fwhm = max(float(p.get("fwhm", x_span / 100.0)), 1e-12)
        depth_guess = max(y_max - local_y, 0.03 * y_span)
        area_guess = max(depth_guess * fwhm, 1e-9)

        p["amplitude"] = -area_guess
        p["amplitude_min"] = -max(area_guess * 12.0, y_span * x_span)
        p["amplitude_max"] = 0.0

    return new_defs


def detect_peaks_auto(
    x,
    y_raw,
    y_detection_display,
    *,
    direction="positive",
    prominence_pct=5.0,
    min_distance_x=0.0,
    min_width_x=0.0,
    max_peaks=10,
    custom_profiles=None,
    default_x_min=None,
    default_x_max=None,
):
    """
    Detect candidate positive/negative peak centers and create seeded peak definitions.

    This backend routine preserves the desktop Find-peaks proposal logic. It
    returns an empty `centers` list when no peaks satisfy the supplied settings.
    """
    x = np.asarray(x, dtype=float)
    y_raw = np.asarray(y_raw, dtype=float)
    y_detection_display = np.asarray(y_detection_display, dtype=float)
    custom_profiles = custom_profiles or {}

    if direction not in PEAK_DETECTION_DIRECTIONS:
        direction = "positive"

    if direction == "negative":
        y_for_detection = -y_detection_display
    else:
        y_for_detection = y_detection_display

    y_range = max(float(np.nanmax(y_for_detection) - np.nanmin(y_for_detection)), 1e-12)

    prominence_pct = max(0.0, float(prominence_pct))
    prominence_abs = None
    if prominence_pct > 0:
        prominence_abs = (prominence_pct / 100.0) * y_range

    distance_samples = x_units_to_samples(x, min_distance_x)
    width_samples = x_units_to_samples(x, min_width_x)

    find_kwargs = {}
    if prominence_abs is not None:
        find_kwargs["prominence"] = prominence_abs
    if distance_samples is not None:
        find_kwargs["distance"] = distance_samples
    if width_samples is not None:
        find_kwargs["width"] = width_samples

    peak_indices, properties = find_peaks(y_for_detection, **find_kwargs)

    if peak_indices.size == 0:
        return {
            "centers": [],
            "seeded_defs": [],
            "direction": direction,
            "selected_indices": np.asarray([], dtype=int),
            "selected_prominences": np.asarray([], dtype=float),
        }

    try:
        prominences = properties.get("prominences", None)
        if prominences is None or len(prominences) != len(peak_indices):
            prominences = peak_prominences(y_for_detection, peak_indices)[0]
        prominences = np.asarray(prominences, dtype=float)
    except Exception:
        prominences = y_for_detection[peak_indices]

    max_peaks = max(1, int(max_peaks))
    max_peaks = min(max_peaks, 15)

    # Keep the strongest peaks by prominence, then sort them by x-position for the table.
    strongest_order = np.argsort(prominences)[::-1][:max_peaks]
    selected_indices = peak_indices[strongest_order]
    selected_prominences = prominences[strongest_order]

    x_order = np.argsort(x[selected_indices])
    selected_indices = selected_indices[x_order]
    selected_prominences = selected_prominences[x_order]

    centers = [float(x[idx]) for idx in selected_indices]
    n_found = len(centers)

    # Build normal default peak states, then replace the centers with detected centers.
    x_min = float(default_x_min) if default_x_min is not None else (float(np.min(x)) if x.size else 0.0)
    x_max = float(default_x_max) if default_x_max is not None else (float(np.max(x)) if x.size else 100.0)
    candidate_defs = []
    for i, center in enumerate(centers):
        state = default_peak_state_for_range(i, n_found, x_min=x_min, x_max=x_max)
        state["active"] = True
        state["kind"] = "Pseudo-Voigt"
        state["center"] = center
        candidate_defs.append(state)

    # Reuse FitED center-seeding logic to estimate amplitude/FWHM/bounds.
    seeded_defs = seed_peak_defs_from_centers(
        x,
        y_raw,
        candidate_defs,
        custom_profiles,
    )

    if direction == "negative":
        seeded_defs = negative_peak_seed_from_centers(x, y_raw, seeded_defs)

    return {
        "centers": centers,
        "seeded_defs": seeded_defs,
        "direction": direction,
        "selected_indices": selected_indices,
        "selected_prominences": selected_prominences,
    }


def residual_noise_sigma(residual):
    """Robust residual noise estimate using MAD."""
    residual = np.asarray(residual, dtype=float)
    residual = residual[np.isfinite(residual)]
    if residual.size == 0:
        return 0.0

    med = float(np.median(residual))
    mad = float(np.median(np.abs(residual - med)))
    return 1.4826 * mad


def residual_suggestion_threshold(residual, y_raw, sensitivity="Normal"):
    """
    Convert Conservative/Normal/Aggressive into a residual-prominence threshold.

    The threshold combines:
    - a fraction of the original data range
    - a multiple of robust residual noise

    This avoids suggesting tiny noise wiggles as missing peaks.
    """
    settings = {
        "Conservative": (3.0, 3.5),
        "Normal": (1.5, 2.5),
        "Aggressive": (0.5, 1.5),
    }
    pct, sigma_mult = settings.get(sensitivity, settings["Normal"])

    y_raw = np.asarray(y_raw, dtype=float)
    residual = np.asarray(residual, dtype=float)

    y_span = float(np.nanmax(y_raw) - np.nanmin(y_raw))
    y_span = max(y_span, 1e-12)

    sigma = residual_noise_sigma(residual)

    threshold = max((pct / 100.0) * y_span, sigma_mult * sigma)

    if not np.isfinite(threshold) or threshold <= 0:
        threshold = 1e-12

    return float(threshold)


def detect_residual_peak_candidates(
    x,
    y_raw,
    best_fit,
    *,
    smooth_enabled=False,
    smooth_window=9,
    smooth_poly=2,
    sensitivity="Normal",
    direction="positive",
    max_suggestions=5,
    min_distance_x=0.0,
    min_width_x=0.0,
    existing_centers=None,
):
    """
    Detect candidate missing peak components from a residual signal.

    Returns:
        selected_candidates, residual, residual_for_detection
    """
    x = np.asarray(x, dtype=float)
    y_raw = np.asarray(y_raw, dtype=float)
    best = np.asarray(best_fit, dtype=float)

    if x.size != y_raw.size or best.size != y_raw.size:
        raise ValueError("Stored fit arrays are inconsistent. Run the fit again.")

    residual = y_raw - best

    residual_for_detection = smooth_if_requested(
        residual,
        window=int(smooth_window),
        polyorder=int(smooth_poly),
        enabled=bool(smooth_enabled),
    )

    threshold = residual_suggestion_threshold(residual_for_detection, y_raw, sensitivity=sensitivity)

    distance_samples = x_units_to_samples(x, min_distance_x)
    if distance_samples is None:
        distance_samples = max(1, int(len(x) / 200))

    width_samples = x_units_to_samples(x, min_width_x)

    find_kwargs = {
        "prominence": threshold,
        "distance": max(1, int(distance_samples)),
    }
    if width_samples is not None:
        find_kwargs["width"] = width_samples

    if direction not in RESIDUAL_SUGGESTION_DIRECTIONS:
        direction = "positive"

    search_signals = []
    if direction in ("positive", "both"):
        search_signals.append(("positive", residual_for_detection, 1.0))
    if direction in ("negative", "both"):
        search_signals.append(("negative", -residual_for_detection, -1.0))

    dx = np.median(np.abs(np.diff(x))) if x.size > 1 else 1.0
    if not np.isfinite(dx) or dx <= 0:
        dx = 1.0

    x_span = max(float(np.nanmax(x) - np.nanmin(x)), 1e-12)

    # Avoid exact duplicates at almost the same center.
    # This still allows real shoulders close to an existing peak.
    existing_centers = list(existing_centers or [])
    duplicate_tol = max(2.0 * dx, 0.003 * x_span)

    candidates = []

    for cand_direction, signal, amp_sign in search_signals:
        peak_indices, properties = find_peaks(signal, **find_kwargs)

        if peak_indices.size == 0:
            continue

        prominences = properties.get("prominences", np.full(len(peak_indices), np.nan))
        widths = properties.get("widths", np.full(len(peak_indices), np.nan))

        for j, idx in enumerate(peak_indices):
            center = float(x[idx])

            if any(abs(center - old_center) < duplicate_tol for old_center in existing_centers):
                continue

            try:
                prominence = float(prominences[j])
            except Exception:
                prominence = float(abs(signal[idx]))

            try:
                width_x = float(widths[j]) * dx
            except Exception:
                width_x = np.nan

            if not np.isfinite(prominence) or prominence <= 0:
                continue

            candidates.append({
                "center": center,
                "index": int(idx),
                "direction": cand_direction,
                "amplitude_sign": float(amp_sign),
                "prominence": prominence,
                "width_x": width_x,
                "residual_value": float(residual[idx]),
            })

    if not candidates:
        return [], residual, residual_for_detection

    # Rank by prominence but avoid duplicate suggestions very close to each other.
    max_suggestions = max(1, int(max_suggestions))
    max_suggestions = min(max_suggestions, 5)

    ranked = sorted(candidates, key=lambda c: c["prominence"], reverse=True)
    selected = []

    for cand in ranked:
        if any(abs(cand["center"] - old["center"]) < duplicate_tol for old in selected):
            continue
        selected.append(cand)
        if len(selected) >= max_suggestions:
            break

    selected = sorted(selected, key=lambda c: c["center"])

    return selected, residual, residual_for_detection


def residual_candidate_peak_defs(
    x,
    y_raw,
    current_states,
    candidates,
    custom_profiles=None,
    default_x_min=None,
    default_x_max=None,
):
    """
    Convert selected residual candidates into new peak-row states.

    Existing rows are returned unchanged; new rows are returned separately.
    """
    x = np.asarray(x, dtype=float)
    y_raw = np.asarray(y_raw, dtype=float)
    current_states = copy.deepcopy(list(current_states))
    custom_profiles = custom_profiles or {}

    active_states_for_spacing = [
        state for state in current_states
        if bool(state.get("active", True))
    ]

    total_count = len(current_states) + len(candidates)
    new_defaults = []

    x_min = float(default_x_min) if default_x_min is not None else (float(np.nanmin(x)) if x.size else 0.0)
    x_max = float(default_x_max) if default_x_max is not None else (float(np.nanmax(x)) if x.size else 100.0)

    for i, cand in enumerate(candidates):
        state = default_peak_state_for_range(
            len(current_states) + i,
            max(total_count, 1),
            x_min=x_min,
            x_max=x_max,
        )
        state["active"] = True
        state["kind"] = "Pseudo-Voigt"
        state["custom_profile"] = ""
        state["center"] = float(cand["center"])

        if np.isfinite(cand.get("width_x", np.nan)) and cand["width_x"] > 0:
            state["fwhm"] = float(cand["width_x"])
            state["sigma"] = max(float(cand["width_x"]) / 2.354820045, 1e-8)
            state["gamma"] = max(float(cand["width_x"]) / 2.0, 1e-8)

        new_defaults.append(state)

    # Reuse existing FitED center-seeding logic so bounds and widths are
    # consistent with Auto pre-fit and Find peaks.
    spacing_context = active_states_for_spacing + new_defaults
    seeded_context = seed_peak_defs_from_centers(
        x,
        y_raw,
        spacing_context,
        custom_profiles,
    )
    seeded_new = seeded_context[-len(new_defaults):] if new_defaults else []

    x_span = max(float(np.nanmax(x) - np.nanmin(x)), 1e-12)

    for p, cand in zip(seeded_new, candidates):
        if np.isfinite(cand.get("width_x", np.nan)) and cand["width_x"] > 0:
            fwhm = max(float(cand["width_x"]), 1e-12)
            p["fwhm"] = fwhm
            p["fwhm_min"] = max(fwhm * 0.25, x_span / 5000.0)
            p["fwhm_max"] = max(fwhm * 5.0, p["fwhm_min"] * 1.5)
            p["sigma"] = max(fwhm / 2.354820045, 1e-8)
            p["gamma"] = max(fwhm / 2.0, 1e-8)
        else:
            fwhm = max(float(p.get("fwhm", x_span / 100.0)), 1e-12)

        amp_abs = max(float(cand["prominence"]) * fwhm, 1e-12)

        if cand["direction"] == "negative":
            p["amplitude"] = -amp_abs
            p["amplitude_min"] = -max(amp_abs * 12.0, abs(cand["prominence"]) * x_span, 1e-12)
            p["amplitude_max"] = 0.0
        else:
            p["amplitude"] = amp_abs
            p["amplitude_min"] = 0.0
            p["amplitude_max"] = max(amp_abs * 12.0, abs(cand["prominence"]) * x_span, 1e-12)

    return current_states, seeded_new


def safe_derived_expression_names(expression):
    """Return variable/function names used in a derived-quantity expression."""
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as exc:
        raise ValueError(f"Invalid expression syntax: {exc}") from exc

    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            names.add(node.id)
    return names


def validate_derived_expression(expression, allowed_parameter_names):
    """Validate a derived-quantity expression before evaluating it."""
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as exc:
        raise ValueError(f"Invalid expression syntax: {exc}") from exc

    allowed_names = (
        set(allowed_parameter_names)
        | set(DERIVED_QUANTITY_FUNCTIONS)
        | set(DERIVED_QUANTITY_CONSTANTS)
    )

    allowed_nodes = (
        ast.Expression,
        ast.BinOp,
        ast.UnaryOp,
        ast.Call,
        ast.Name,
        ast.Load,
        ast.Constant,
        ast.Add,
        ast.Sub,
        ast.Mult,
        ast.Div,
        ast.Pow,
        ast.Mod,
        ast.USub,
        ast.UAdd,
    )

    for node in ast.walk(tree):
        if not isinstance(node, allowed_nodes):
            raise ValueError(f"Unsupported syntax in expression: {type(node).__name__}")

        if isinstance(node, ast.Name):
            if node.id not in allowed_names:
                raise ValueError(f"Unknown symbol in expression: {node.id}")

        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name):
                raise ValueError("Only direct function calls are allowed.")
            if node.func.id not in DERIVED_QUANTITY_FUNCTIONS:
                raise ValueError(f"Function not allowed: {node.func.id}")

    return tree


def params_to_value_dict(params):
    """Convert lmfit Parameters to a plain dictionary of parameter values."""
    try:
        params.update_constraints()
    except Exception:
        pass

    values = {}
    for name, par in params.items():
        try:
            values[name] = float(par.value)
        except Exception:
            values[name] = np.nan
    return values


def eval_derived_expression(expression, values):
    """Safely evaluate one derived-quantity expression."""
    scope = {}
    scope.update(DERIVED_QUANTITY_FUNCTIONS)
    scope.update(DERIVED_QUANTITY_CONSTANTS)
    scope.update(values)

    try:
        value = eval(expression, {"__builtins__": {}}, scope)
    except Exception as exc:
        raise ValueError(f"Could not evaluate expression: {exc}") from exc

    try:
        value = float(value)
    except Exception as exc:
        raise ValueError("Derived expression did not return a scalar number.") from exc

    if not np.isfinite(value):
        raise ValueError("Derived expression returned a non-finite value.")

    return value


def eval_derived_expression_for_params(expression, params):
    """Evaluate expression using an lmfit Parameters object."""
    values = params_to_value_dict(params)
    return eval_derived_expression(expression, values)


def finite_difference_derivative_for_var(expression, result, var_name, base_value):
    """
    Numerical derivative d(expression)/d(var_name).

    This perturbs independent fitted variables from result.var_names.
    Expression-constrained parameters such as p1_fwhm are updated through
    lmfit's parameter constraints when possible.
    """
    params0 = result.params
    par0 = params0[var_name]

    h = 1e-6 * max(abs(float(base_value)), 1.0)
    if not np.isfinite(h) or h <= 0:
        h = 1e-6

    pmin = float(par0.min) if np.isfinite(par0.min) else -np.inf
    pmax = float(par0.max) if np.isfinite(par0.max) else np.inf

    can_forward = np.isfinite(base_value + h) and (base_value + h <= pmax)
    can_backward = np.isfinite(base_value - h) and (base_value - h >= pmin)

    def eval_at(new_value):
        pcopy = params0.copy()
        pcopy[var_name].set(value=float(new_value))
        try:
            pcopy.update_constraints()
        except Exception:
            pass
        return eval_derived_expression_for_params(expression, pcopy)

    if can_forward and can_backward:
        fp = eval_at(base_value + h)
        fm = eval_at(base_value - h)
        return (fp - fm) / (2.0 * h)

    if can_forward:
        f0 = eval_derived_expression_for_params(expression, params0.copy())
        fp = eval_at(base_value + h)
        return (fp - f0) / h

    if can_backward:
        f0 = eval_derived_expression_for_params(expression, params0.copy())
        fm = eval_at(base_value - h)
        return (f0 - fm) / h

    return np.nan


def propagate_uncertainty(expression, result):
    """
    Propagate uncertainty from result.covar by first-order covariance propagation.

    Returns:
        stderr, variance, warnings
    """
    warnings = []
    covar = getattr(result, "covar", None)
    var_names = list(getattr(result, "var_names", []))

    stderr = np.nan
    variance = np.nan

    if covar is None:
        warnings.append("No covariance matrix is available; propagated stderr cannot be calculated.")
        return stderr, variance, warnings

    if not var_names:
        warnings.append("No varying parameters are available; propagated stderr cannot be calculated.")
        return stderr, variance, warnings

    covar = np.asarray(covar, dtype=float)
    if covar.shape != (len(var_names), len(var_names)):
        warnings.append("Covariance matrix shape does not match result.var_names.")
        return stderr, variance, warnings

    gradient = []
    derivative_failed = False

    for vname in var_names:
        try:
            base_value = float(result.params[vname].value)
            deriv = finite_difference_derivative_for_var(
                expression,
                result,
                vname,
                base_value
            )
        except Exception:
            deriv = np.nan

        if not np.isfinite(deriv):
            derivative_failed = True
            deriv = 0.0

        gradient.append(float(deriv))

    gradient = np.asarray(gradient, dtype=float)

    if derivative_failed:
        warnings.append(
            "At least one numerical derivative failed; propagated stderr may be incomplete."
        )

    try:
        variance = float(gradient @ covar @ gradient)

        diag_scale = max(
            float(np.max(np.abs(np.diag(covar)))),
            1.0,
        )
        
        if variance < 0 and abs(variance) < 1e-10 * diag_scale:
            variance = 0.0
        
        if variance >= 0 and np.isfinite(variance):
            stderr = float(np.sqrt(variance))
        else:
            warnings.append("Calculated propagated variance is negative or non-finite.")
    except Exception as exc:
        warnings.append(f"Could not propagate uncertainty: {exc}")

    return stderr, variance, warnings

def compute_derived_uncertainty_contribution_map(result, definitions):
    """
    Compute a signed uncertainty-contribution map for user-defined derived quantities.

    For each derived quantity g, propagated variance is:
        variance = gradient.T @ covariance @ gradient

    The per-parameter signed contribution is:
        contribution_i = gradient_i * (covariance @ gradient)_i

    The displayed/exported map value is:
        100 * contribution_i / total_variance

    Notes:
    - Contributions are defined with respect to result.var_names, i.e. the
      independent varying fit parameters covered by result.covar.
    - Negative contributions are physically meaningful: they indicate covariance
      cancellation that reduces the propagated variance.
    """
    if result is None:
        raise ValueError("Run a fit before computing the derived uncertainty map.")

    covar = getattr(result, "covar", None)
    var_names = list(getattr(result, "var_names", []))

    payload = {
        "rows": [],
        "derived_names": [],
        "parameter_names": list(var_names),
        "warnings": [],
    }

    if covar is None:
        payload["warnings"].append(
            "No covariance matrix is available; the derived uncertainty contribution map cannot be calculated."
        )
        return payload

    if not var_names:
        payload["warnings"].append(
            "No varying parameters are available; the derived uncertainty contribution map cannot be calculated."
        )
        return payload

    covar = np.asarray(covar, dtype=float)

    if covar.shape != (len(var_names), len(var_names)):
        payload["warnings"].append(
            "Covariance matrix shape does not match result.var_names; the derived uncertainty contribution map cannot be calculated."
        )
        return payload

    allowed_names = list(result.params.keys())

    for item in definitions:
        name = str(item.get("name", "")).strip()
        expression = str(item.get("expression", "")).strip()

        if not name or not expression:
            continue

        validate_derived_expression(expression, allowed_names)
        payload["derived_names"].append(name)

        warnings = []
        gradient = []
        derivative_failed = False

        for vname in var_names:
            try:
                base_value = float(result.params[vname].value)
                deriv = finite_difference_derivative_for_var(
                    expression,
                    result,
                    vname,
                    base_value
                )
            except Exception:
                deriv = np.nan

            if not np.isfinite(deriv):
                derivative_failed = True
                deriv = 0.0

            gradient.append(float(deriv))

        gradient = np.asarray(gradient, dtype=float)

        if derivative_failed:
            warnings.append(
                "At least one numerical derivative failed; map contributions may be incomplete."
            )

        try:
            covar_times_gradient = covar @ gradient
            variance_contributions = gradient * covar_times_gradient

            total_variance = float(np.sum(variance_contributions))

            diag_scale = max(
                float(np.max(np.abs(np.diag(covar)))),
                1.0,
            )
            
            if total_variance < 0 and abs(total_variance) < 1e-10 * diag_scale:
                total_variance = 0.0

            if total_variance >= 0 and np.isfinite(total_variance):
                propagated_stderr = float(np.sqrt(total_variance))

                if total_variance > 0:
                    signed_percentages = 100.0 * variance_contributions / total_variance
                else:
                    signed_percentages = np.full_like(
                        variance_contributions,
                        np.nan,
                        dtype=float
                    )
                    warnings.append(
                        "Propagated variance is zero; percentage contributions are undefined."
                    )
            else:
                propagated_stderr = np.nan
                signed_percentages = np.full_like(
                    variance_contributions,
                    np.nan,
                    dtype=float
                )
                warnings.append(
                    "Calculated propagated variance is negative or non-finite; map percentages are unavailable."
                )

        except Exception as exc:
            covar_times_gradient = np.full_like(gradient, np.nan, dtype=float)
            variance_contributions = np.full_like(gradient, np.nan, dtype=float)
            signed_percentages = np.full_like(gradient, np.nan, dtype=float)
            total_variance = np.nan
            propagated_stderr = np.nan
            warnings.append(f"Could not calculate uncertainty contribution map: {exc}")

        warning_text = " | ".join(warnings)
        if warning_text:
            payload["warnings"].append(f"{name}: {warning_text}")

        for (
            vname,
            grad_i,
            cgrad_i,
            contribution_i,
            percent_i,
        ) in zip(
            var_names,
            gradient,
            covar_times_gradient,
            variance_contributions,
            signed_percentages,
        ):
            payload["rows"].append({
                "derived_quantity": name,
                "expression": expression,
                "parameter": vname,
                "gradient": float(grad_i) if np.isfinite(grad_i) else np.nan,
                "covariance_weighted_gradient": (
                    float(cgrad_i) if np.isfinite(cgrad_i) else np.nan
                ),
                "variance_contribution": (
                    float(contribution_i) if np.isfinite(contribution_i) else np.nan
                ),
                "signed_variance_contribution_percent": (
                    float(percent_i) if np.isfinite(percent_i) else np.nan
                ),
                "total_variance": total_variance,
                "propagated_stderr": propagated_stderr,
                "warning": warning_text,
            })

    return payload

def parse_derived_quantity_lines(text):
    """
    Parse user-defined derived quantity definitions.

    Expected format:
        Name = expression

    Example:
        Area ratio = p1_amplitude / p2_amplitude
        Splitting = p2_center - p1_center
    """
    definitions = []

    for raw in str(text).splitlines():
        line = raw.strip()
        if not line or line.startswith('#'):
            continue

        if '=' not in line:
            raise ValueError(
                'Each derived quantity must be written as: Name = expression'
            )

        name, expression = line.split('=', 1)
        name = name.strip()
        expression = expression.strip()

        if not name:
            raise ValueError('Derived quantity name cannot be empty.')
        if not expression:
            raise ValueError(f'Derived quantity "{name}" has an empty expression.')

        definitions.append({
            'name': name,
            'expression': expression,
        })

    if not definitions:
        raise ValueError('No derived quantities were defined.')

    return definitions


def format_derived_quantities_report(rows):
    """
    Create a readable text report for derived quantities and their
    propagated uncertainties.
    """
    lines = [
        '[[FitED derived quantities]]',
        'Uncertainty method: first-order covariance propagation',
        'Formula: variance = gradient.T @ covariance @ gradient',
        'Note: propagated stderr is covariance-based and local/linear.',
        '',
        f"{'Name':<28} {'Value':>16} {'Propagated stderr':>20}  Expression",
        '-' * 95,
    ]

    for row in rows:
        val = row.get('value', np.nan)
        err = row.get('propagated_stderr', np.nan)

        try:
            val_txt = f"{float(val):.8g}"
        except Exception:
            val_txt = str(val)

        try:
            err_txt = f"{float(err):.8g}" if np.isfinite(float(err)) else 'nan'
        except Exception:
            err_txt = 'nan'

        lines.append(
            f"{row.get('name', ''):<28} {val_txt:>16} {err_txt:>20}  {row.get('expression', '')}"
        )

        if row.get('warning'):
            lines.append(f"    warning: {row['warning']}")

    return '\n'.join(lines)


def default_derived_quantity_text(result):
    """
    Suggest common derived-quantity examples based on available fitted
    parameter names.
    """
    if result is None:
        return ''

    names = set(result.params.keys())
    examples = []

    if 'p1_amplitude' in names and 'p2_amplitude' in names:
        examples.append('Area ratio p1/p2 = p1_amplitude / p2_amplitude')
        examples.append(
            'Peak 1 fraction = p1_amplitude / (p1_amplitude + p2_amplitude)'
        )

    if 'p1_center' in names and 'p2_center' in names:
        examples.append('Center splitting p2-p1 = p2_center - p1_center')

    if 'p1_fwhm' in names and 'p2_fwhm' in names:
        examples.append('FWHM ratio p1/p2 = p1_fwhm / p2_fwhm')

    return '\n'.join(examples)

def compute_one_derived_quantity(result, name, expression):
    """
    Compute one derived quantity and propagate uncertainty from result.covar.

    Uses first-order covariance propagation:
        variance = gradient.T @ covariance @ gradient
    """
    if result is None:
        raise ValueError("Run a fit before computing derived quantities.")

    allowed_names = list(result.params.keys())
    validate_derived_expression(expression, allowed_names)

    base_params = result.params.copy()
    value = eval_derived_expression_for_params(expression, base_params)

    used_names = safe_derived_expression_names(expression)
    used_parameters = sorted([n for n in used_names if n in result.params])

    warnings = []

    fixed_used = []
    for pname in used_parameters:
        par = result.params[pname]
        if not getattr(par, "vary", False) and getattr(par, "expr", None) is None:
            fixed_used.append(pname)

    if fixed_used:
        warnings.append(
            "Expression uses fixed parameter(s): "
            + ", ".join(fixed_used)
            + ". Their uncertainty is treated as zero."
        )

    stderr, variance, propagation_warnings = propagate_uncertainty(expression, result)
    warnings.extend(propagation_warnings)

    return {
        "name": name,
        "expression": expression,
        "value": value,
        "propagated_stderr": stderr,
        "variance": variance,
        "used_parameters": ", ".join(used_parameters),
        "warning": " | ".join(warnings),
    }


def compute_derived_quantities(result, definitions):
    """Compute multiple user-defined derived quantities from one fit result."""
    return [
        compute_one_derived_quantity(result, item["name"], item["expression"])
        for item in definitions
    ]


def build_session_state(
    *,
    current_file,
    loader,
    fit_settings,
    custom_profiles,
    custom_background_profiles,
    peaks,
    app_version="step16-custom-merged",
):
    """Compose one JSON-serializable FitED session payload from plain state values."""
    return {
        "app_version": app_version,
        "current_file": str(current_file) if current_file else None,
        "loader": loader,
        "fit_settings": fit_settings,
        "custom_profiles": list(custom_profiles),
        "custom_background_profiles": list(custom_background_profiles),
        "peaks": peaks,
    }


def normalize_derived_quantity_definitions(derived_defs):
    """Normalize session-loaded derived-quantity definitions."""
    if not isinstance(derived_defs, list):
        return []

    return [
        {
            "name": str(d.get("name", "")).strip(),
            "expression": str(d.get("expression", "")).strip(),
        }
        for d in derived_defs
        if isinstance(d, dict)
        and str(d.get("name", "")).strip()
        and str(d.get("expression", "")).strip()
    ]


def compute_parameter_correlation_matrix(result):
    """
    Return the full covariance-based correlation matrix for independently
    varying fitted parameters.

    The matrix is built from result.covar and result.var_names, so expression-
    constrained and fixed parameters are not included as independent variables.
    """
    if result is None:
        raise ValueError("No fit result is available.")

    covar = getattr(result, "covar", None)
    if covar is None:
        raise ValueError("The fit result has no covariance matrix.")

    covar = np.asarray(covar, dtype=float)
    if covar.ndim != 2 or covar.shape[0] != covar.shape[1]:
        raise ValueError("The covariance matrix is not square.")

    names = list(getattr(result, "var_names", []) or [])
    if not names:
        names = [
            name for name, par in result.params.items()
            if getattr(par, "vary", False) and not getattr(par, "expr", None)
        ]

    if len(names) != covar.shape[0]:
        names = names[:covar.shape[0]]
        if len(names) != covar.shape[0]:
            names = [f"var_{i+1}" for i in range(covar.shape[0])]

    diag = np.diag(covar).astype(float)
    scale = np.sqrt(np.where(diag > 0, diag, np.nan))

    with np.errstate(divide="ignore", invalid="ignore"):
        corr = covar / np.outer(scale, scale)

    corr = np.asarray(corr, dtype=float)
    corr[np.eye(corr.shape[0], dtype=bool)] = 1.0
    corr = np.clip(corr, -1.0, 1.0, out=corr, where=np.isfinite(corr))

    warnings_list = []
    if np.any(~np.isfinite(corr)):
        warnings_list.append(
            "Some correlation coefficients are non-finite because one or more covariance diagonal values are zero, negative, or unavailable."
        )

    strongest_pairs = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            value = corr[i, j]
            if np.isfinite(value):
                strongest_pairs.append({
                    "parameter_1": names[i],
                    "parameter_2": names[j],
                    "correlation": float(value),
                    "abs_correlation": float(abs(value)),
                })

    strongest_pairs.sort(key=lambda row: row["abs_correlation"], reverse=True)

    return {
        "parameter_names": names,
        "correlation_matrix": corr.tolist(),
        "covariance_matrix": covar.tolist(),
        "strongest_pairs": strongest_pairs,
        "warnings": warnings_list,
    }


def compute_residual_diagnostics(x, y, best_fit, weights=None, max_lag=None):
    """
    Compute residual diagnostics used to assess whether residuals are random
    and approximately consistent with the assumed noise model.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    best_fit = np.asarray(best_fit, dtype=float)

    if x.shape != y.shape or y.shape != best_fit.shape:
        raise ValueError("x, y, and best_fit must have the same shape.")

    residual = y - best_fit
    finite = np.isfinite(x) & np.isfinite(y) & np.isfinite(best_fit) & np.isfinite(residual)

    x = x[finite]
    y = y[finite]
    best_fit = best_fit[finite]
    residual = residual[finite]

    if residual.size < 3:
        raise ValueError("At least three finite residual points are required for residual diagnostics.")

    if weights is not None:
        weights = np.asarray(weights, dtype=float)
        if weights.shape == finite.shape:
            weights = weights[finite]
        if weights.shape == residual.shape and np.all(np.isfinite(weights)):
            diagnostic_residual = residual * weights
            residual_kind = "weighted_residual"
        else:
            diagnostic_residual = residual.copy()
            residual_kind = "residual"
    else:
        diagnostic_residual = residual.copy()
        residual_kind = "residual"

    diagnostic_residual = np.asarray(diagnostic_residual, dtype=float)
    center = float(np.mean(diagnostic_residual))
    spread = float(np.std(diagnostic_residual, ddof=1)) if diagnostic_residual.size > 1 else np.nan

    if np.isfinite(spread) and spread > 0:
        standardized = (diagnostic_residual - center) / spread
    else:
        standardized = diagnostic_residual - center

    n = int(diagnostic_residual.size)
    if max_lag is None:
        max_lag = min(50, max(1, n // 4))
    max_lag = max(1, min(int(max_lag), n - 1))

    centered = diagnostic_residual - center
    denom = float(np.sum(centered ** 2))
    autocorr_rows = []
    for lag in range(1, max_lag + 1):
        if denom > 0:
            ac = float(np.sum(centered[:-lag] * centered[lag:]) / denom)
        else:
            ac = np.nan
        autocorr_rows.append({"lag": int(lag), "autocorrelation": ac})

    diff = np.diff(diagnostic_residual)
    dw_denom = float(np.sum(diagnostic_residual ** 2))
    durbin_watson = float(np.sum(diff ** 2) / dw_denom) if dw_denom > 0 else np.nan
    lag1 = autocorr_rows[0]["autocorrelation"] if autocorr_rows else np.nan

    try:
        (theoretical_q, ordered_values), (slope, intercept, r_value) = stats.probplot(
            standardized,
            dist="norm",
            fit=True,
        )
        qq_rows = [
            {
                "theoretical_normal_quantile": float(tq),
                "ordered_standardized_residual": float(rv),
            }
            for tq, rv in zip(theoretical_q, ordered_values)
        ]
        qq_fit = {
            "slope": float(slope),
            "intercept": float(intercept),
            "r_value": float(r_value),
            "r_squared": float(r_value ** 2),
        }
    except Exception:
        qq_rows = []
        qq_fit = {
            "slope": np.nan,
            "intercept": np.nan,
            "r_value": np.nan,
            "r_squared": np.nan,
        }

    summary = {
        "n": n,
        "residual_kind": residual_kind,
        "mean": float(np.mean(diagnostic_residual)),
        "std": spread,
        "median": float(np.median(diagnostic_residual)),
        "mad": float(np.median(np.abs(diagnostic_residual - np.median(diagnostic_residual)))),
        "rmse": float(np.sqrt(np.mean(residual ** 2))),
        "max_abs_residual": float(np.max(np.abs(residual))),
        "durbin_watson": durbin_watson,
        "lag1_autocorrelation": float(lag1) if np.isfinite(lag1) else np.nan,
        "qq_r_squared": qq_fit.get("r_squared", np.nan),
    }

    residual_rows = [
        {
            "x": float(xi),
            "y_data": float(yi),
            "y_fit": float(fi),
            "residual": float(ri),
            "diagnostic_residual": float(di),
            "standardized_diagnostic_residual": float(si),
        }
        for xi, yi, fi, ri, di, si in zip(
            x,
            y,
            best_fit,
            residual,
            diagnostic_residual,
            standardized,
        )
    ]

    warnings_list = []
    if np.isfinite(lag1) and abs(lag1) > 0.3:
        warnings_list.append(
            "Lag-1 residual autocorrelation is large; residuals may contain systematic structure."
        )
    if np.isfinite(qq_fit.get("r_squared", np.nan)) and qq_fit["r_squared"] < 0.98:
        warnings_list.append(
            "Q-Q plot linearity is imperfect; residuals may deviate from a Gaussian-noise assumption."
        )

    return {
        "summary": summary,
        "residual_rows": residual_rows,
        "autocorrelation_rows": autocorr_rows,
        "qq_rows": qq_rows,
        "qq_fit": qq_fit,
        "warnings": warnings_list,
    }


def confidence_ellipse_pair_summary(result):
    """
    Return a compact table describing covariance ellipses for all parameter pairs.
    Width and height are full axis lengths for 1-sigma and 2-sigma covariance ellipses.
    """
    corr_payload = compute_parameter_correlation_matrix(result)
    names = corr_payload["parameter_names"]
    covar = np.asarray(corr_payload["covariance_matrix"], dtype=float)
    corr = np.asarray(corr_payload["correlation_matrix"], dtype=float)

    rows = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            sub = covar[np.ix_([i, j], [i, j])]
            if sub.shape != (2, 2) or not np.all(np.isfinite(sub)):
                continue

            vals, vecs = np.linalg.eigh(sub)
            vals = np.maximum(vals, 0.0)
            order = np.argsort(vals)[::-1]
            vals = vals[order]
            vecs = vecs[:, order]

            angle = float(np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0])))
            rows.append({
                "parameter_x": names[i],
                "parameter_y": names[j],
                "covariance": float(sub[0, 1]),
                "correlation": float(corr[i, j]) if np.isfinite(corr[i, j]) else np.nan,
                "ellipse_angle_deg": angle,
                "width_1sigma": float(2.0 * np.sqrt(vals[0])),
                "height_1sigma": float(2.0 * np.sqrt(vals[1])),
                "width_2sigma": float(4.0 * np.sqrt(vals[0])),
                "height_2sigma": float(4.0 * np.sqrt(vals[1])),
            })

    rows.sort(
        key=lambda row: abs(row.get("correlation", 0.0))
        if np.isfinite(row.get("correlation", np.nan)) else -1.0,
        reverse=True,
    )
    return rows


def compute_confidence_ellipse_data(result, parameter_x, parameter_y, sigmas=(1.0, 2.0), n_points=240):
    """
    Compute covariance-ellipse coordinates for a selected parameter pair.

    This is a local covariance ellipse, not a profile-likelihood contour.
    It visualizes joint parameter uncertainty and correlation around the best fit.
    """
    if result is None:
        raise ValueError("No fit result is available.")

    corr_payload = compute_parameter_correlation_matrix(result)
    names = corr_payload["parameter_names"]
    covar = np.asarray(corr_payload["covariance_matrix"], dtype=float)

    parameter_x = str(parameter_x).strip()
    parameter_y = str(parameter_y).strip()

    if parameter_x not in names:
        raise ValueError(f"Parameter '{parameter_x}' is not an independent covariance parameter.")
    if parameter_y not in names:
        raise ValueError(f"Parameter '{parameter_y}' is not an independent covariance parameter.")
    if parameter_x == parameter_y:
        raise ValueError("Choose two different parameters.")

    ix = names.index(parameter_x)
    iy = names.index(parameter_y)
    sub = covar[np.ix_([ix, iy], [ix, iy])]

    if sub.shape != (2, 2) or not np.all(np.isfinite(sub)):
        raise ValueError("The selected parameter pair does not have a finite 2D covariance submatrix.")

    vals, vecs = np.linalg.eigh(sub)
    vals = np.maximum(vals, 0.0)
    order = np.argsort(vals)[::-1]
    vals = vals[order]
    vecs = vecs[:, order]

    theta = np.linspace(0.0, 2.0 * np.pi, int(max(n_points, 24)))

    center_x = float(result.params[parameter_x].value)
    center_y = float(result.params[parameter_y].value)

    ellipses = []
    for sigma in sigmas:
        sigma = float(sigma)
        circle = np.vstack([
            np.sqrt(vals[0]) * np.cos(theta),
            np.sqrt(vals[1]) * np.sin(theta),
        ])
        coords = np.array([[center_x], [center_y]]) + sigma * (vecs @ circle)
        ellipses.append({
            "sigma": sigma,
            "x": coords[0].astype(float).tolist(),
            "y": coords[1].astype(float).tolist(),
        })

    corr = np.nan
    if sub[0, 0] > 0 and sub[1, 1] > 0:
        corr = float(sub[0, 1] / np.sqrt(sub[0, 0] * sub[1, 1]))

    angle = float(np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0])))

    return {
        "parameter_x": parameter_x,
        "parameter_y": parameter_y,
        "center_x": center_x,
        "center_y": center_y,
        "value_x": center_x,
        "value_y": center_y,
        "covariance_matrix": sub.tolist(),
        "correlation": corr,
        "ellipse_angle_deg": angle,
        "eigenvalues": vals.astype(float).tolist(),
        "ellipses": ellipses,
    }



def normalize_session_payload(state):
    """
    Normalize profile collections and derived-quantity definitions in a saved session.

    The desktop still owns widget updates; this helper keeps the validation and
    compatibility cleanup outside Tk code.
    """
    state = state if isinstance(state, dict) else {}
    settings = state.get("fit_settings", {})
    settings = settings if isinstance(settings, dict) else {}

    custom_profiles = {}
    for prof in state.get("custom_profiles", []):
        try:
            normalized = normalize_custom_profile_definition(prof)
            custom_profiles[normalized["name"]] = normalized
        except Exception:
            pass

    custom_background_profiles = {}
    for prof in state.get("custom_background_profiles", []):
        try:
            normalized = normalize_custom_profile_definition(prof)
            custom_background_profiles[normalized["name"]] = normalized
        except Exception:
            pass

    return {
        "state": state,
        "loader": state.get("loader", {}) if isinstance(state.get("loader", {}), dict) else {},
        "fit_settings": settings,
        "peaks": state.get("peaks", []) if isinstance(state.get("peaks", []), list) else [],
        "current_file": state.get("current_file"),
        "custom_profiles": custom_profiles,
        "custom_background_profiles": custom_background_profiles,
        "derived_quantity_definitions": normalize_derived_quantity_definitions(
            settings.get("derived_quantities", [])
        ),
        
        # New defaults for parameter constraints
        "auxiliary_parameter_definitions": settings.get("auxiliary_parameters", [])
            if isinstance(settings.get("auxiliary_parameters", []), list) else [],
        "parameter_constraint_definitions": settings.get("parameter_constraints", [])
            if isinstance(settings.get("parameter_constraints", []), list) else [],
    }




# ============================================================================
# FitED Decay / IRF extension
# Added as a standalone backend route so the original peak-profile route remains
# available unchanged. The functions below intentionally reuse the same lmfit
# optimizer, trial search, stability, batch, diagnostics, and reporting pipeline.
# ============================================================================

try:
    from scipy.signal import fftconvolve
except Exception:  # pragma: no cover - scipy is already a FitED dependency
    fftconvolve = None

DECAY_IRF_ANALYSIS_MODE = "Decay / IRF"
DECAY_IRF_STANDARD_MODE = "Standard peaks"
DECAY_IRF_DECAY_KINDS = [
    "Single exponential",
    "Bi-exponential",
    "Triple exponential",
    "Stretched exponential",
    "Single rise + single decay",
    "Common rise + bi-exponential decay",
    "Common rise + triple-exponential decay",
    "Common rise + stretched exponential",
]


def canonical_decay_irf_decay_kind(decay_kind=None):
    """Normalize Decay / IRF model names, keeping old session labels compatible."""
    text = str(decay_kind or DECAY_IRF_DECAY_KINDS[0]).strip()
    key = text.lower().replace("–", "-").replace("—", "-")
    aliases = {
        "single exponential": "Single exponential",
        "mono exponential": "Single exponential",
        "mono-exponential": "Single exponential",
        "bi-exponential": "Bi-exponential",
        "biexponential": "Bi-exponential",
        "double exponential": "Bi-exponential",
        "double-exponential": "Bi-exponential",
        "triple exponential": "Triple exponential",
        "triple-exponential": "Triple exponential",
        "tri-exponential": "Triple exponential",
        "stretched exponential": "Stretched exponential",
        "single rise + single decay": "Single rise + single decay",
        "rise + decay": "Single rise + single decay",
        "finite-rise single decay": "Single rise + single decay",
        "finite rise single decay": "Single rise + single decay",
        "single rise + single exponential decay": "Single rise + single decay",
        "common rise + bi-exponential decay": "Common rise + bi-exponential decay",
        "common rise + biexponential decay": "Common rise + bi-exponential decay",
        "common rise + double exponential decay": "Common rise + bi-exponential decay",
        "common rise + triple-exponential decay": "Common rise + triple-exponential decay",
        "common rise + triple exponential decay": "Common rise + triple-exponential decay",
        "common rise + tri-exponential decay": "Common rise + triple-exponential decay",
        "common rise + stretched exponential": "Common rise + stretched exponential",
        "common rise + stretched-exponential decay": "Common rise + stretched exponential",
    }
    return aliases.get(key, text if text in DECAY_IRF_DECAY_KINDS else DECAY_IRF_DECAY_KINDS[0])
DECAY_IRF_MODES = [
    "No IRF",
    "Measured IRF reconvolution",
    "Gaussian synthetic IRF",
]
DECAY_IRF_DATA_TYPES = [
    "TRPL / TCSPC",
    "Transient absorption kinetic trace",
    "General decay",
]
DECAY_IRF_SIGNAL_SIGNS = [
    "positive amplitudes",
    "signed amplitudes",
    "negative amplitudes",
]
DECAY_IRF_BASELINE_MODES = ["none", "minimum", "edge median"]
DECAY_IRF_ZERO_MODES = ["peak maximum", "center of mass", "keep IRF time axis"]


def _decay_irf_context_active(context):
    return str(context.get("analysis_mode", "")).strip().lower() in {
        "decay / irf",
        "decay_irf",
        "time_irf",
        "time-domain / irf",
        "time domain / irf",
    }


def load_irf_file(filepath, x_col=0, y_col=1, delimiter=None, skiprows=0):
    """Load an experimentally measured IRF from a text file using FitED's tolerant loader."""
    return load_spectrum(
        filepath,
        x_col=x_col,
        y_col=y_col,
        delimiter=delimiter,
        skiprows=skiprows,
    )


def _median_dt_from_x(x):
    x = np.asarray(x, dtype=float)
    if x.ndim != 1 or x.size < 2:
        raise ValueError("At least two time points are required for decay/IRF fitting.")
    diffs = np.diff(x)
    finite = diffs[np.isfinite(diffs)]
    if finite.size == 0:
        raise ValueError("Could not determine the time spacing from the decay data.")
    dt = float(np.median(finite))
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError("The time axis must be increasing for decay/IRF fitting.")
    return dt


def _baseline_correct_irf(irf_y, mode="edge median", edge_fraction=0.10):
    y = np.asarray(irf_y, dtype=float).copy()
    mode = str(mode or "edge median").strip().lower()
    if y.size == 0:
        raise ValueError("The IRF array is empty.")
    if mode == "none":
        return y, 0.0
    if mode == "minimum":
        baseline = float(np.nanmin(y))
    else:
        n_edge = max(1, int(round(float(edge_fraction) * y.size)))
        edge = np.concatenate([y[:n_edge], y[-n_edge:]])
        baseline = float(np.nanmedian(edge[np.isfinite(edge)])) if np.any(np.isfinite(edge)) else 0.0
    y = y - baseline
    return y, baseline


def _irf_reference_time(irf_x, irf_y, zero_mode="peak maximum"):
    x = np.asarray(irf_x, dtype=float)
    y = np.asarray(irf_y, dtype=float)
    mode = str(zero_mode or "peak maximum").strip().lower()
    if x.size == 0 or y.size == 0:
        return 0.0
    if mode == "keep irf time axis":
        return 0.0
    if mode == "center of mass":
        weights = np.clip(y, 0.0, None)
        denom = float(np.sum(weights))
        if np.isfinite(denom) and denom > 0:
            return float(np.sum(x * weights) / denom)
    idx = int(np.nanargmax(y))
    return float(x[idx])


def prepare_irf_kernel(
    irf_x,
    irf_y,
    baseline_mode="edge median",
    zero_mode="peak maximum",
    clip_negative=True,
    edge_fraction=0.10,
):
    """
    Prepare an experimental IRF for reconvolution.

    The returned IRF time axis is shifted according to zero_mode, the baseline is
    subtracted, negative values can be clipped, and the area is normalized to 1.
    """
    x = np.asarray(irf_x, dtype=float)
    y = np.asarray(irf_y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if x.size < 2:
        raise ValueError("The IRF file must contain at least two finite x/y points.")
    order = np.argsort(x)
    x = x[order]
    y = y[order]

    y, removed_baseline = _baseline_correct_irf(y, baseline_mode, edge_fraction=edge_fraction)
    if bool(clip_negative):
        y = np.clip(y, 0.0, None)

    reference = _irf_reference_time(x, y, zero_mode=zero_mode)
    x = x - reference

    area = float(_fited_trapezoid(y, x))
    if not np.isfinite(area) or abs(area) <= 1e-30:
        area = float(np.sum(y) * max(_median_dt_from_x(x), 1e-30))
    if not np.isfinite(area) or abs(area) <= 1e-30:
        raise ValueError("The prepared IRF has zero or non-finite area after preprocessing.")
    y = y / area

    return {
        "x": x,
        "y": y,
        "removed_baseline": float(removed_baseline),
        "reference_time": float(reference),
        "area_before_normalization": float(area),
        "baseline_mode": str(baseline_mode),
        "zero_mode": str(zero_mode),
        "clip_negative": bool(clip_negative),
    }


def _interp_irf_on_data_grid(x, irf_payload, irf_shift=0.0):
    x = np.asarray(x, dtype=float)
    irf_x = np.asarray(irf_payload["x"], dtype=float) + float(irf_shift)
    irf_y = np.asarray(irf_payload["y"], dtype=float)
    grid = np.interp(x, irf_x, irf_y, left=0.0, right=0.0)
    area = float(_fited_trapezoid(grid, x))
    if np.isfinite(area) and abs(area) > 1e-30:
        grid = grid / area
    return grid


def _gaussian_irf_on_data_grid(x, fwhm, center):
    x = np.asarray(x, dtype=float)
    sigma = max(float(fwhm), 1e-15) / 2.354820045
    irf = np.exp(-0.5 * ((x - float(center)) / sigma) ** 2)
    area = float(_fited_trapezoid(irf, x))
    if np.isfinite(area) and abs(area) > 1e-30:
        irf = irf / area
    return irf


def _decay_response_array(
    x,
    decay_kind,
    t0,
    A1,
    tau1,
    A2=0.0,
    tau2=1.0,
    A3=0.0,
    tau3=1.0,
    beta=1.0,
    tau_rise=1.0,
    tau_decay=10.0,
):
    x = np.asarray(x, dtype=float)
    u = x - float(t0)
    h = (u >= 0.0).astype(float)
    up = np.clip(u, 0.0, None)
    kind = canonical_decay_irf_decay_kind(decay_kind).strip().lower()

    tau1 = max(float(tau1), 1e-30)
    tau2 = max(float(tau2), 1e-30)
    tau3 = max(float(tau3), 1e-30)
    beta = max(float(beta), 1e-12)
    tau_rise = max(float(tau_rise), 1e-30)
    tau_decay = max(float(tau_decay), 1e-30)

    single = float(A1) * np.exp(-up / tau1)
    bi = single + float(A2) * np.exp(-up / tau2)
    triple = bi + float(A3) * np.exp(-up / tau3)
    stretched = float(A1) * np.exp(-((up / tau1) ** beta))
    rise_factor = 1.0 - np.exp(-up / tau_rise)

    if kind == "single exponential":
        response = single
    elif kind == "bi-exponential":
        response = bi
    elif kind == "triple exponential":
        response = triple
    elif kind == "stretched exponential":
        response = stretched
    elif kind == "single rise + single decay":
        response = float(A1) * rise_factor * np.exp(-up / tau_decay)
    elif kind == "common rise + bi-exponential decay":
        response = rise_factor * bi
    elif kind == "common rise + triple-exponential decay":
        response = rise_factor * triple
    elif kind == "common rise + stretched exponential":
        response = rise_factor * stretched
    else:
        raise ValueError(f"Unknown decay model: {decay_kind}")

    response = response * h
    response[~np.isfinite(response)] = 0.0
    return response


def _linear_reconvolution_on_grid(x, response, irf_grid):
    x = np.asarray(x, dtype=float)
    response = np.asarray(response, dtype=float)
    irf_grid = np.asarray(irf_grid, dtype=float)
    dt = _median_dt_from_x(x)
    if response.shape != x.shape or irf_grid.shape != x.shape:
        raise ValueError("response, IRF, and time axis must have identical shapes.")
    if fftconvolve is not None:
        full = fftconvolve(response, irf_grid, mode="full") * dt
    else:
        full = np.convolve(response, irf_grid, mode="full") * dt
    conv_t = float(x[0] + x[0]) + np.arange(full.size, dtype=float) * dt
    return np.interp(x, conv_t, full, left=0.0, right=0.0)


def _decay_irf_model_function_factory(context):
    cfg = copy.deepcopy(context.get("decay_irf_config", {}))
    decay_kind = canonical_decay_irf_decay_kind(cfg.get("decay_kind", DECAY_IRF_DECAY_KINDS[0]))
    irf_mode = cfg.get("irf_mode", DECAY_IRF_MODES[0])
    irf_payload = copy.deepcopy(cfg.get("prepared_irf"))
    gaussian_fwhm = float(cfg.get("gaussian_irf_fwhm", 0.1))

    def _decay_irf_model(
        x,
        baseline,
        t0,
        irf_shift,
        A1,
        tau1,
        A2,
        tau2,
        A3,
        tau3,
        beta,
        tau_rise,
        tau_decay,
    ):
        response = _decay_response_array(
            x,
            decay_kind=decay_kind,
            t0=t0,
            A1=A1,
            tau1=tau1,
            A2=A2,
            tau2=tau2,
            A3=A3,
            tau3=tau3,
            beta=beta,
            tau_rise=tau_rise,
            tau_decay=tau_decay,
        )

        mode = str(irf_mode or DECAY_IRF_MODES[0]).strip().lower()
        if mode == "measured irf reconvolution":
            if irf_payload is None:
                raise ValueError("Measured IRF reconvolution was selected, but no IRF has been loaded.")
            irf_grid = _interp_irf_on_data_grid(x, irf_payload, irf_shift=irf_shift)
            signal = _linear_reconvolution_on_grid(x, response, irf_grid)
        elif mode == "gaussian synthetic irf":
            irf_grid = _gaussian_irf_on_data_grid(x, gaussian_fwhm, center=float(t0) + float(irf_shift))
            signal = _linear_reconvolution_on_grid(x, response, irf_grid)
        else:
            signal = response

        return float(baseline) + signal

    return _decay_irf_model


def _decay_param_cfg(config, name, default_value, default_min=-np.inf, default_max=np.inf, default_vary=True):
    p = config.get("parameters", {}).get(name, {}) if isinstance(config.get("parameters", {}), dict) else {}
    value = float(p.get("value", config.get(name, default_value)))
    pmin = float(p.get("min", config.get(f"{name}_min", default_min)))
    pmax = float(p.get("max", config.get(f"{name}_max", default_max)))
    vary = bool(p.get("vary", config.get(f"{name}_vary", default_vary)))
    if pmin > pmax:
        raise ValueError(f"Decay/IRF parameter '{name}' has min > max.")
    if np.isfinite(pmin) or np.isfinite(pmax):
        value = float(np.clip(value, pmin, pmax))
    return value, pmin, pmax, vary


def _add_decay_param(params, config, name, default_value, default_min=-np.inf, default_max=np.inf, default_vary=True):
    value, pmin, pmax, vary = _decay_param_cfg(
        config,
        name,
        default_value,
        default_min=default_min,
        default_max=default_max,
        default_vary=default_vary,
    )
    params.add(name, value=value, min=pmin, max=pmax, vary=vary)


def _default_decay_amplitude_bounds(config, y=None):
    signal_sign = str(config.get("signal_sign", "signed amplitudes")).strip().lower()
    y_min, y_max, y_span = _finite_array_limits(y if y is not None else [0.0, 1.0])
    amp_scale = max(abs(y_min), abs(y_max), y_span, 1.0)
    if signal_sign == "positive amplitudes":
        return 0.0, amp_scale * 20.0
    if signal_sign == "negative amplitudes":
        return -amp_scale * 20.0, 0.0
    return -amp_scale * 20.0, amp_scale * 20.0


def build_time_irf_reconvolution_model(context):
    """Build the lmfit Model/Parameters pair for the Decay / IRF route."""
    x = np.asarray(context["x"], dtype=float)
    y = np.asarray(context["y_raw"], dtype=float)
    cfg = copy.deepcopy(context.get("decay_irf_config", {}))
    decay_kind = canonical_decay_irf_decay_kind(cfg.get("decay_kind", DECAY_IRF_DECAY_KINDS[0]))

    model_func = _decay_irf_model_function_factory(context)
    model = Model(model_func, independent_vars=["x"], prefix="")
    params = Parameters()

    x_min, x_max, x_span = _finite_array_limits(x)
    y_min, y_max, y_span = _finite_array_limits(y)
    amp_min, amp_max = _default_decay_amplitude_bounds(cfg, y=y)
    amp_guess = float(y[0] - np.nanmedian(y[-max(3, len(y)//10):])) if len(y) else y_span
    if cfg.get("signal_sign", "signed amplitudes") == "positive amplitudes":
        amp_guess = max(abs(amp_guess), 0.2 * y_span)
    elif cfg.get("signal_sign", "signed amplitudes") == "negative amplitudes":
        amp_guess = -max(abs(amp_guess), 0.2 * y_span)
    elif abs(amp_guess) < 1e-30:
        amp_guess = 0.5 * y_span

    _add_decay_param(params, cfg, "baseline", float(np.nanmedian(y[-max(3, len(y)//10):])), default_min=y_min - 5*y_span, default_max=y_max + 5*y_span, default_vary=True)
    _add_decay_param(params, cfg, "t0", float(x_min), default_min=x_min - 0.25*x_span, default_max=x_max, default_vary=True)
    _add_decay_param(params, cfg, "irf_shift", 0.0, default_min=-0.25*x_span, default_max=0.25*x_span, default_vary=True)

    tau_default = max(0.20 * x_span, _median_dt_from_x(x))
    tau_min = max(_median_dt_from_x(x) * 0.05, 1e-30)
    tau_max = max(x_span * 20.0, tau_min * 10.0)

    uses_tau1 = decay_kind in [
        "Single exponential",
        "Bi-exponential",
        "Triple exponential",
        "Stretched exponential",
        "Common rise + bi-exponential decay",
        "Common rise + triple-exponential decay",
        "Common rise + stretched exponential",
    ]
    uses_A2_tau2 = decay_kind in [
        "Bi-exponential",
        "Triple exponential",
        "Common rise + bi-exponential decay",
        "Common rise + triple-exponential decay",
    ]
    uses_A3_tau3 = decay_kind in [
        "Triple exponential",
        "Common rise + triple-exponential decay",
    ]
    uses_beta = decay_kind in [
        "Stretched exponential",
        "Common rise + stretched exponential",
    ]
    uses_rise = decay_kind in [
        "Single rise + single decay",
        "Common rise + bi-exponential decay",
        "Common rise + triple-exponential decay",
        "Common rise + stretched exponential",
    ]
    uses_tau_decay = decay_kind == "Single rise + single decay"

    _add_decay_param(params, cfg, "A1", amp_guess, default_min=amp_min, default_max=amp_max, default_vary=True)
    _add_decay_param(params, cfg, "tau1", tau_default, default_min=tau_min, default_max=tau_max, default_vary=uses_tau1)
    _add_decay_param(params, cfg, "A2", amp_guess * 0.3, default_min=amp_min, default_max=amp_max, default_vary=uses_A2_tau2)
    _add_decay_param(params, cfg, "tau2", tau_default * 5.0, default_min=tau_min, default_max=tau_max, default_vary=uses_A2_tau2)
    _add_decay_param(params, cfg, "A3", amp_guess * 0.1, default_min=amp_min, default_max=amp_max, default_vary=uses_A3_tau3)
    _add_decay_param(params, cfg, "tau3", tau_default * 20.0, default_min=tau_min, default_max=tau_max, default_vary=uses_A3_tau3)
    _add_decay_param(params, cfg, "beta", 0.75, default_min=0.05, default_max=2.0, default_vary=uses_beta)
    _add_decay_param(params, cfg, "tau_rise", tau_default * 0.1, default_min=tau_min, default_max=tau_max, default_vary=uses_rise)
    _add_decay_param(params, cfg, "tau_decay", tau_default, default_min=tau_min, default_max=tau_max, default_vary=uses_tau_decay)

    inactive = []
    if not uses_tau1:
        inactive.append("tau1")
    if not uses_A2_tau2:
        inactive.extend(["A2", "tau2"])
    if not uses_A3_tau3:
        inactive.extend(["A3", "tau3"])
    if not uses_beta:
        inactive.append("beta")
    if not uses_rise:
        inactive.append("tau_rise")
    if not uses_tau_decay:
        inactive.append("tau_decay")
    for name in inactive:
        if name in params:
            params[name].set(vary=False)

    if str(cfg.get("irf_mode", DECAY_IRF_MODES[0])).strip().lower() == "no irf":
        params["irf_shift"].set(value=0.0, vary=False)

    params = apply_parameter_constraints(
        params,
        auxiliary_parameters=context.get("auxiliary_parameters", []),
        parameter_constraints=context.get("parameter_constraints", []),
    )
    return model, params


def evaluate_time_irf_components(context, params, x=None):
    """Return display/export component curves for the Decay / IRF route."""
    if x is None:
        x = context["x"]
    x = np.asarray(x, dtype=float)
    values = {name: float(par.value) for name, par in params.items() if name in params}
    cfg = context.get("decay_irf_config", {})
    decay_kind = canonical_decay_irf_decay_kind(cfg.get("decay_kind", DECAY_IRF_DECAY_KINDS[0]))
    response = _decay_response_array(
        x,
        decay_kind=decay_kind,
        t0=values.get("t0", 0.0),
        A1=values.get("A1", 0.0),
        tau1=values.get("tau1", 1.0),
        A2=values.get("A2", 0.0),
        tau2=values.get("tau2", 1.0),
        A3=values.get("A3", 0.0),
        tau3=values.get("tau3", 1.0),
        beta=values.get("beta", 1.0),
        tau_rise=values.get("tau_rise", 1.0),
        tau_decay=values.get("tau_decay", 10.0),
    )
    baseline = values.get("baseline", 0.0)
    model, _ = build_time_irf_reconvolution_model(context)
    best = model.eval(params=params, x=x)
    comps = {
        "intrinsic_decay_plus_baseline": baseline + response,
    }

    mode = str(cfg.get("irf_mode", DECAY_IRF_MODES[0])).strip().lower()
    try:
        if mode == "measured irf reconvolution" and cfg.get("prepared_irf") is not None:
            irf_grid = _interp_irf_on_data_grid(x, cfg["prepared_irf"], irf_shift=values.get("irf_shift", 0.0))
        elif mode == "gaussian synthetic irf":
            irf_grid = _gaussian_irf_on_data_grid(
                x,
                cfg.get("gaussian_irf_fwhm", 0.1),
                center=values.get("t0", 0.0) + values.get("irf_shift", 0.0),
            )
        else:
            irf_grid = None
        if irf_grid is not None:
            y = np.asarray(context.get("y_raw", best), dtype=float)
            _, _, y_span = _finite_array_limits(y)
            irf_span = max(float(np.nanmax(irf_grid) - np.nanmin(irf_grid)), 1e-30)
            comps["irf_display_scaled"] = baseline + (irf_grid - np.nanmin(irf_grid)) / irf_span * 0.35 * y_span
    except Exception:
        pass
    comps["reconvolved_decay_model"] = best
    return comps


def seed_decay_irf_config_from_data(x, y, config=None):
    """Return a conservative first-guess Decay/IRF configuration from one kinetic trace."""
    cfg = copy.deepcopy(config or {})
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.size < 2:
        return cfg
    x_min, x_max, x_span = _finite_array_limits(x)
    y_min, y_max, y_span = _finite_array_limits(y)
    tail = y[-max(3, y.size // 10):]
    baseline = float(np.nanmedian(tail))
    amp = float(y[0] - baseline)
    if abs(amp) < 1e-30:
        amp = float(y_max - baseline) if abs(y_max - baseline) >= abs(y_min - baseline) else float(y_min - baseline)
    tau = max(0.20 * x_span, _median_dt_from_x(x))
    p = cfg.setdefault("parameters", {})
    def _set(name, value, pmin, pmax, vary=True):
        p[name] = {"value": float(value), "min": float(pmin), "max": float(pmax), "vary": bool(vary)}
    amp_bound = max(abs(y_min), abs(y_max), y_span, abs(amp), 1.0) * 20.0
    _set("baseline", baseline, y_min - 5*y_span, y_max + 5*y_span, True)
    _set("t0", x_min, x_min - 0.25*x_span, x_max, True)
    _set("irf_shift", 0.0, -0.25*x_span, 0.25*x_span, True)
    decay_kind = canonical_decay_irf_decay_kind(cfg.get("decay_kind", DECAY_IRF_DECAY_KINDS[0]))
    cfg["decay_kind"] = decay_kind
    uses_tau1 = decay_kind in [
        "Single exponential",
        "Bi-exponential",
        "Triple exponential",
        "Stretched exponential",
        "Common rise + bi-exponential decay",
        "Common rise + triple-exponential decay",
        "Common rise + stretched exponential",
    ]
    uses_A2_tau2 = decay_kind in [
        "Bi-exponential",
        "Triple exponential",
        "Common rise + bi-exponential decay",
        "Common rise + triple-exponential decay",
    ]
    uses_A3_tau3 = decay_kind in [
        "Triple exponential",
        "Common rise + triple-exponential decay",
    ]
    uses_beta = decay_kind in [
        "Stretched exponential",
        "Common rise + stretched exponential",
    ]
    uses_rise = decay_kind in [
        "Single rise + single decay",
        "Common rise + bi-exponential decay",
        "Common rise + triple-exponential decay",
        "Common rise + stretched exponential",
    ]
    uses_tau_decay = decay_kind == "Single rise + single decay"

    _set("A1", amp, -amp_bound, amp_bound, True)
    _set("tau1", tau, max(_median_dt_from_x(x)*0.05, 1e-30), max(x_span*20.0, tau*10.0), uses_tau1)
    _set("A2", amp*0.3, -amp_bound, amp_bound, uses_A2_tau2)
    _set("tau2", tau*5.0, max(_median_dt_from_x(x)*0.05, 1e-30), max(x_span*20.0, tau*20.0), uses_A2_tau2)
    _set("A3", amp*0.1, -amp_bound, amp_bound, uses_A3_tau3)
    _set("tau3", tau*20.0, max(_median_dt_from_x(x)*0.05, 1e-30), max(x_span*20.0, tau*50.0), uses_A3_tau3)
    _set("beta", 0.75, 0.05, 2.0, uses_beta)
    _set("tau_rise", tau*0.1, max(_median_dt_from_x(x)*0.05, 1e-30), max(x_span*20.0, tau*10.0), uses_rise)
    _set("tau_decay", tau, max(_median_dt_from_x(x)*0.05, 1e-30), max(x_span*20.0, tau*10.0), uses_tau_decay)
    return cfg


def _sample_decay_param_fast(cfg, name, trial, rng, scale_log=False):
    p = cfg.setdefault("parameters", {}).setdefault(name, {})
    if not bool(p.get("vary", True)):
        return
    val = float(p.get("value", 1.0))
    pmin = float(p.get("min", -np.inf))
    pmax = float(p.get("max", np.inf))
    if not np.isfinite(pmin) or not np.isfinite(pmax) or pmin >= pmax:
        span = max(abs(val), 1.0)
        pmin, pmax = val - span, val + span
        if scale_log:
            pmin = max(1e-30, abs(val) / 20.0 if val != 0 else 1e-30)
            pmax = max(abs(val) * 20.0, pmin * 10.0)
    if scale_log:
        lo = max(pmin, 1e-30)
        hi = max(pmax, lo * 1.0001)
        value = sample_log_from_unit(lo, hi, rng.random())
    else:
        value = sample_linear_from_unit(pmin, pmax, rng.random())
    p["value"] = float(value)


def _random_decay_irf_trial_config(context, trial, rng):
    cfg = copy.deepcopy(context.get("decay_irf_config", {}))
    cfg = seed_decay_irf_config_from_data(context["x"], context["y_raw"], cfg)
    for name in ["baseline", "t0", "irf_shift", "A1", "A2", "A3"]:
        _sample_decay_param_fast(cfg, name, trial, rng, scale_log=False)
    for name in ["tau1", "tau2", "tau3", "tau_rise", "tau_decay", "beta"]:
        _sample_decay_param_fast(cfg, name, trial, rng, scale_log=True)
    return cfg


# Keep references to the original standard-route functions before overriding their names.
_FITED_STANDARD_build_model_from_context = build_model_from_context
_FITED_STANDARD_worker_fit_once = worker_fit_once
_FITED_STANDARD_run_autoprefit_search_worker = run_autoprefit_search_worker
_FITED_STANDARD_batch_context_for_file = batch_context_for_file


def build_model_from_context(context, peak_defs):
    """Build one lmfit model from a FitED context dictionary, including Decay / IRF mode."""
    if _decay_irf_context_active(context):
        return build_time_irf_reconvolution_model(context)
    return _FITED_STANDARD_build_model_from_context(context, peak_defs)


def worker_fit_once(context, peak_defs, cancel_event, progress_queue=None, message="Fitting..."):
    """Run one fit from a FitED context, including the Decay / IRF route."""
    if not _decay_irf_context_active(context):
        return _FITED_STANDARD_worker_fit_once(context, peak_defs, cancel_event, progress_queue, message)

    raise_if_cancelled(cancel_event)
    if progress_queue is not None:
        progress_queue.put(("progress", (0, 1, message), None))
    model, params = build_time_irf_reconvolution_model(context)
    result = fit_model_with_optimizer(
        model,
        context["y_raw"],
        params=params,
        x=context["x"],
        weights=context["weights"],
        nan_policy="raise",
        max_nfev=context["max_nfev"],
        optimizer_mode=context.get("optimizer_mode", DEFAULT_FIT_OPTIMIZER_MODE),
        selection_criterion=context.get("criterion", "AIC"),
        random_seed=context.get("random_seed"),
    )
    try:
        result.fited_analysis_mode = DECAY_IRF_ANALYSIS_MODE
        result.fited_decay_irf_config = copy.deepcopy(context.get("decay_irf_config", {}))
    except Exception:
        pass
    raise_if_cancelled(cancel_event)
    comps = evaluate_time_irf_components(context, result.params, x=context["x"])
    return result, comps, result.best_fit


def run_autoprefit_search_worker(context, base_peak_defs, cancel_event, progress_queue):
    """Run Auto pre-fit for standard peaks or the Decay / IRF route."""
    if not _decay_irf_context_active(context):
        return _FITED_STANDARD_run_autoprefit_search_worker(context, base_peak_defs, cancel_event, progress_queue)

    n_trials = max(1, int(context.get("n_trials", 1)))
    rng = rng_from_seed(context.get("random_seed"))
    best_result = None
    best_score = np.inf
    last_error = None

    for trial in range(n_trials):
        raise_if_cancelled(cancel_event)
        if progress_queue is not None:
            progress_queue.put((
                "progress",
                (trial, n_trials, f"Decay/IRF Auto pre-fit: trial {trial + 1}/{n_trials} ..."),
                None,
            ))
        trial_context = copy.deepcopy(context)
        trial_context["decay_irf_config"] = _random_decay_irf_trial_config(context, trial, rng)
        try:
            result, comps, best = worker_fit_once(
                trial_context,
                [],
                cancel_event,
                progress_queue=None,
                message="Decay/IRF fitting...",
            )
            score = fit_selection_score(result, context.get("criterion", "AIC"))
            if np.isfinite(score) and score < best_score:
                best_score = score
                best_result = result
        except Exception as exc:
            last_error = exc
            continue

    if best_result is None and last_error is not None:
        raise RuntimeError(f"Decay/IRF Auto pre-fit failed in all trials. Last error: {last_error}")
    if best_result is None:
        raise RuntimeError("Decay/IRF Auto pre-fit failed for all attempts.")
    if progress_queue is not None:
        progress_queue.put(("progress", (n_trials, n_trials, "Decay/IRF Auto pre-fit: finished."), None))
    return best_result, []


def batch_context_for_file(filepath, template):
    """Load one file and build a batch context, including Decay / IRF templates."""
    context = _FITED_STANDARD_batch_context_for_file(filepath, template)
    if str(template.get("analysis_mode", "")).strip().lower() in {"decay / irf", "decay_irf", "time_irf"}:
        context["analysis_mode"] = DECAY_IRF_ANALYSIS_MODE
        context["decay_irf_config"] = copy.deepcopy(template.get("decay_irf_config", {}))
        context["peak_defs"] = []
        context["all_custom_no_center"] = False
    return context

# Batch override added after the Decay / IRF route so batch Auto pre-fit can use
# the time-domain trial search even though this route has no peak table.
_FITED_STANDARD_batch_fit_one_file = batch_fit_one_file


def batch_fit_one_file(filepath, template, cancel_event):
    """Fit one file in batch mode, including Decay / IRF templates."""
    if str(template.get("analysis_mode", "")).strip().lower() not in {"decay / irf", "decay_irf", "time_irf"}:
        return _FITED_STANDARD_batch_fit_one_file(filepath, template, cancel_event)

    context = batch_context_for_file(filepath, template)
    batch_mode = template.get("batch_mode", BATCH_FIT_MODES[0])

    if batch_mode == "Run fit using current parameters":
        result, comps, best = worker_fit_once(
            context,
            [],
            cancel_event,
            progress_queue=None,
            message="Batch Decay/IRF fitting...",
        )
        return result

    dummy_queue = NoopProgressQueue()
    best_result, best_peak_defs = run_autoprefit_search_worker(
        context,
        [],
        cancel_event,
        dummy_queue,
    )

    model, params = build_model_from_context(context, [])
    params = copy_fit_result_values_into_params(params, best_result)
    final_result = fit_model_with_optimizer(
        model,
        context["y_raw"],
        params=params,
        x=context["x"],
        weights=context["weights"],
        nan_policy="raise",
        max_nfev=context["max_nfev"],
        optimizer_mode=context.get("optimizer_mode", DEFAULT_FIT_OPTIMIZER_MODE),
        selection_criterion=context.get("criterion", "AIC"),
        random_seed=seed_with_offset(context.get("random_seed"), int(context.get("n_trials", 0))),
    )
    try:
        final_result.fited_analysis_mode = DECAY_IRF_ANALYSIS_MODE
        final_result.fited_decay_irf_config = copy.deepcopy(context.get("decay_irf_config", {}))
    except Exception:
        pass
    raise_if_cancelled(cancel_event)
    return final_result


# ============================================================================
# FitED Decay / IRF bound-preserving seed override
# This override fixes the Decay / IRF auto pre-fit seeding behavior so that
# automatic trial generation can update starting values without replacing
# user-defined min/max/vary settings from the Decay / IRF parameter table.
# Existing standard peak/profile functions remain untouched.
# ============================================================================
_FITED_ORIG_seed_decay_irf_config_from_data_BOUND_FIX = seed_decay_irf_config_from_data


def _fited_decay_irf_float_or_none(value):
    try:
        value = float(value)
    except Exception:
        return None
    return value


def _fited_decay_irf_bound_was_user_defined(name, field, value):
    """Return True when a Decay/IRF parameter bound should be preserved."""
    value = _fited_decay_irf_float_or_none(value)
    if value is None:
        return False
    if np.isfinite(value):
        return True
    return False


def _fited_decay_irf_clip_to_bounds(value, pmin, pmax):
    value = _fited_decay_irf_float_or_none(value)
    if value is None or not np.isfinite(value):
        value = 0.0
    pmin = _fited_decay_irf_float_or_none(pmin)
    pmax = _fited_decay_irf_float_or_none(pmax)
    if pmin is not None and pmax is not None and pmin > pmax:
        raise ValueError("Decay/IRF parameter has min > max after bound-preserving seeding.")
    if pmin is not None and np.isfinite(pmin) and value < pmin:
        value = pmin
    if pmax is not None and np.isfinite(pmax) and value > pmax:
        value = pmax
    return float(value)


def seed_decay_irf_config_from_data(x, y, config=None):
    """Seed Decay/IRF guesses from data while preserving user bounds and vary flags.

    The first Decay/IRF implementation used this function to create broad automatic
    parameter ranges for the trial search. That was useful for a blank/default
    table, but it could silently replace manually chosen bounds such as a narrow
    baseline range. This override keeps the automatic values as starting guesses,
    while restoring any finite user-provided min/max bounds and all user-provided
    vary flags from the incoming configuration.
    """
    incoming = copy.deepcopy(config or {})
    incoming_params = copy.deepcopy(incoming.get("parameters", {}) or {})

    seeded = _FITED_ORIG_seed_decay_irf_config_from_data_BOUND_FIX(x, y, incoming)
    seeded_params = seeded.setdefault("parameters", {})

    for name, original in incoming_params.items():
        if not isinstance(original, dict):
            continue
        target = seeded_params.setdefault(name, {})

        # Preserve the user's Vary checkbox exactly when present.
        if "vary" in original:
            target["vary"] = bool(original.get("vary"))

        # Preserve finite user bounds. Infinite/missing bounds are allowed to be
        # replaced by the automatic finite defaults, which keeps Auto pre-fit useful
        # for a blank/default table.
        if "min" in original and _fited_decay_irf_bound_was_user_defined(name, "min", original.get("min")):
            target["min"] = float(original.get("min"))
        if "max" in original and _fited_decay_irf_bound_was_user_defined(name, "max", original.get("max")):
            target["max"] = float(original.get("max"))

        # Keep the automatic seed value, but force it inside the preserved bounds.
        target["value"] = _fited_decay_irf_clip_to_bounds(
            target.get("value", original.get("value", 0.0)),
            target.get("min", -np.inf),
            target.get("max", np.inf),
        )

    return seeded



# ============================================================================
# FitED Decay / IRF strict run-fit bound enforcement
# This block is intentionally appended after the previous Decay / IRF route.
# It does not alter the standard peak/profile route. It makes Run fit, Auto
# pre-fit trials, stability tests, and batch Decay/IRF fits rebuild lmfit
# Parameters with the finite min/max values currently present in the Decay/IRF
# parameter table. If a fitted result ever escapes those finite bounds, FitED
# raises an explicit error instead of silently accepting a nonphysical result.
# ============================================================================
_FITED_PREV_build_time_irf_reconvolution_model_STRICT_BOUNDS = build_time_irf_reconvolution_model
_FITED_PREV_worker_fit_once_STRICT_BOUNDS = worker_fit_once


def _fited_decay_irf_param_table_from_context(context):
    cfg = context.get("decay_irf_config", {}) if isinstance(context, dict) else {}
    table = cfg.get("parameters", {}) if isinstance(cfg, dict) else {}
    return table if isinstance(table, dict) else {}


def _fited_decay_irf_float(value, default=None):
    try:
        out = float(value)
    except Exception:
        return default
    return out


def _fited_decay_irf_has_finite_bound(value):
    value = _fited_decay_irf_float(value, default=None)
    return value is not None and np.isfinite(value)


def _fited_decay_irf_enforce_finite_user_bounds(params, context):
    """Force lmfit Parameters to use the finite bounds from Decay/IRF config."""
    table = _fited_decay_irf_param_table_from_context(context)
    for name, cfg in table.items():
        if name not in params or not isinstance(cfg, dict):
            continue
        par = params[name]
        if getattr(par, "expr", None):
            continue

        current_min = float(par.min) if np.isfinite(par.min) else -np.inf
        current_max = float(par.max) if np.isfinite(par.max) else np.inf

        if _fited_decay_irf_has_finite_bound(cfg.get("min")):
            current_min = float(cfg.get("min"))
        if _fited_decay_irf_has_finite_bound(cfg.get("max")):
            current_max = float(cfg.get("max"))

        if current_min > current_max:
            raise ValueError(f"Decay/IRF parameter '{name}' has min > max.")

        value = float(par.value) if np.isfinite(par.value) else _fited_decay_irf_float(cfg.get("value"), 0.0)
        if np.isfinite(current_min) and value < current_min:
            value = current_min
        if np.isfinite(current_max) and value > current_max:
            value = current_max

        par.set(value=float(value), min=float(current_min), max=float(current_max))

    return params


def _fited_decay_irf_assert_result_inside_finite_user_bounds(context, result, tolerance=1e-8):
    """Fail loudly if an optimizer result violates user finite bounds."""
    if result is None or not hasattr(result, "params"):
        return
    table = _fited_decay_irf_param_table_from_context(context)
    violations = []
    for name, cfg in table.items():
        if name not in result.params or not isinstance(cfg, dict):
            continue
        par = result.params[name]
        if getattr(par, "expr", None):
            continue
        value = _fited_decay_irf_float(par.value, default=None)
        if value is None or not np.isfinite(value):
            continue
        if _fited_decay_irf_has_finite_bound(cfg.get("min")):
            pmin = float(cfg.get("min"))
            scale = max(abs(pmin), abs(value), 1.0)
            if value < pmin - tolerance * scale:
                violations.append(f"{name}={value:.6g} below min {pmin:.6g}")
        if _fited_decay_irf_has_finite_bound(cfg.get("max")):
            pmax = float(cfg.get("max"))
            scale = max(abs(pmax), abs(value), 1.0)
            if value > pmax + tolerance * scale:
                violations.append(f"{name}={value:.6g} above max {pmax:.6g}")
    if violations:
        raise RuntimeError(
            "Decay/IRF fitted result violated finite user bounds: " + "; ".join(violations)
        )


def build_time_irf_reconvolution_model(context):
    """Build Decay/IRF model and strictly preserve finite user parameter bounds."""
    model, params = _FITED_PREV_build_time_irf_reconvolution_model_STRICT_BOUNDS(context)
    params = _fited_decay_irf_enforce_finite_user_bounds(params, context)
    return model, params


def worker_fit_once(context, peak_defs, cancel_event, progress_queue=None, message="Fitting..."):
    """Run one fit and verify Decay/IRF finite user bounds are respected."""
    result, comps, best = _FITED_PREV_worker_fit_once_STRICT_BOUNDS(
        context,
        peak_defs,
        cancel_event,
        progress_queue=progress_queue,
        message=message,
    )
    if _decay_irf_context_active(context):
        _fited_decay_irf_assert_result_inside_finite_user_bounds(context, result)
    return result, comps, best
