# FitED

**Author:** Mustafa Mahmoud Ibrahim Aboulsaad
**Version:** v1.4
**Software DOI:** 10.5281/zenodo.19411620

---

## Overview

FitED is a user-centric, extensible desktop application for robust peak-profile, decay, and general functional data fitting. It is designed to help users load experimental one-dimensional data, define a fitting region, construct multi-peak, background, custom analytical, or time-domain decay models, perform nonlinear fitting, visualize fitted components and residuals, evaluate fit quality, and export results for documentation and further analysis.

FitED is mainly developed for spectroscopy and related experimental datasets, but it can also be used for other one-dimensional fitting problems where the user needs an interactive graphical environment rather than writing a new fitting script for every dataset.

In addition to the standard peak/profile fitting workflow, FitED includes a dedicated **Decay / IRF** route for time-domain fitting. This route is designed for **time-resolved photoluminescence (TRPL)**, **time-correlated single-photon counting (TCSPC)**, transient absorption kinetic traces such as **ΔOD versus time**, and other decay-like responses requiring instrument-response-function treatment.

For measured IRF fitting, FitED performs **reconvolution**, not deconvolution. The intrinsic sample response is modeled first and then convolved with the measured instrument response function before comparison with the experimental data.

---

## What the Software Allows the User to Do

FitED allows the user to:

* Load spectral, decay, or other one-dimensional experimental data from text-based files.
* Select X and Y data columns from multi-column files.
* Choose the delimiter manually or use automatic delimiter detection.
* Skip header or metadata rows before numerical parsing.
* Select a fitting region of interest using ROI minimum and maximum values.
* Apply optional Savitzky-Golay smoothing for preview and peak detection only.
* Switch between the standard **Peak/profile fitting** route and the dedicated **Decay / IRF time-domain fitting** route.
* Define multiple peak components with adjustable initial values and fitting bounds.
* Enable or disable individual peak rows without deleting the peak setup.
* Pick peak centers interactively from the plot.
* Detect peaks automatically using configurable peak-finding settings.
* Fit positive peaks, negative peaks, or manually defined components depending on the workflow.
* Choose built-in peak profiles including **Gaussian**, **Lorentzian**, **Pseudo-Voigt**, and **Exact Voigt**.
* Use Exact Voigt profiles with independent Gaussian sigma and Lorentzian gamma parameters.
* Use the Pseudo-Voigt Gaussian/Lorentzian mixing fraction, where `0` corresponds to Gaussian-like behavior and `1` corresponds to Lorentzian-like behavior.
* Define custom analytical peak profiles using user-defined expressions and parameters.
* Define custom analytical fitting models for non-standard data.
* Add background models including none, constant, linear, polynomial, and custom analytical background functions.
* Fit TRPL, TCSPC, transient absorption kinetic traces, and other positive or signed decay-like time-domain data.
* Load a measured IRF file from text-based data with selectable delimiter, skip rows, and X/Y columns.
* Use measured IRF reconvolution to fit decay data while accounting for finite instrumental response.
* Use a Gaussian synthetic IRF when no measured IRF is available.
* Fit built-in decay models including:

  * single exponential,
  * bi-exponential,
  * triple exponential,
  * stretched exponential,
  * common-rise single-exponential decay,
  * common-rise bi-exponential decay,
  * common-rise triple-exponential decay,
  * common-rise stretched-exponential decay.
* Select the signal sign for decay fitting, including positive amplitudes, negative amplitudes, or signed/free amplitudes.
* Apply IRF preprocessing options including baseline handling, negative-value clipping, IRF time-zero selection, and internal IRF normalization.
* Choose whether the IRF time axis is kept as measured or re-zeroed using the IRF peak maximum or center-of-mass position.
* Fit or fix timing parameters such as the sample-response onset time and the IRF alignment shift.
* Display the measured data, reconvolved fit, intrinsic response before IRF convolution, scaled IRF, and residuals in the plot.
* Set fitting weights, including no weighting, Poisson-like weighting, sqrt(y) emphasis, and 1/y-type weighting.
* Preview the current model before running the optimizer.
* Drag peak positions in the plot for visual model adjustment before confirming a final fit.
* Run a normal fit from the current user-defined parameters.
* Run automatic pre-fit searches from selected or detected peak centers.
* Refine a previous fit after adding extra peaks.
* Fit batches of files using the current model setup.
* Choose the model-selection or trial-ranking criterion among **AIC**, **BIC**, **chi-square**, and **reduced chi-square**.
* Choose the optimizer mode, including **Levenberg-Marquardt** and a robust mode that compares Levenberg-Marquardt with Differential Evolution followed by Levenberg-Marquardt polishing.
* Control the maximum number of function evaluations.
* Set an optional random seed for reproducible stochastic searches.
* Choose automatic pre-fit sampling strategies, including Fast Jitter, Latin Hypercube Sampling, and Hybrid Fast Jitter plus Latin Hypercube Sampling.
* Run stability tests by repeating fits or automatic pre-fit searches with different seeds.
* Rank repeated stability-test results using the selected criterion.
* Inspect repeated-solution score spread and near-best parameter spread.
* Inspect fitted curves, individual components, intrinsic decay responses, reconvolved decay fits, IRF display curves, and residuals.
* Generate residual diagnostics to help evaluate systematic errors or model inadequacy.
* Inspect parameter correlations and covariance-related diagnostics.
* Inspect confidence ellipses for selected parameter pairs.
* Define derived quantities from fitted parameters.
* Propagate uncertainty for derived quantities using the fitted covariance information when available.
* Inspect uncertainty contributions for derived quantities.
* Save and load fitting sessions.
* Keep report histories inside the Reports tab during the current session.
* Preserve generated reports after closing popup windows.
* Store temporary result packages during the session and save them later as ZIP files.

---

## Main Interface Structure

The FitED desktop interface is organized into workflow tabs:

* **Load data**
* **Fit settings**
* **Peaks**
* **Decay / IRF**
* **Actions**
* **Reports**

The plot area shows the loaded data, preview curves, best-fit curves, fitted components, IRF curves when applicable, intrinsic decay responses, reconvolved fits, and residuals. The GUI keeps the plotting and user interaction visible while the fitting logic is handled by the backend.

The **Peaks** tab controls the standard peak/profile fitting route.

The **Decay / IRF** tab controls the dedicated time-domain fitting route. In this tab, the user can select whether the **Actions** tab should operate in:

* **Peak/profile mode**, which uses the Peaks tab and standard peak/background model construction.
* **Decay/IRF mode**, which uses the Decay / IRF tab to build a time-domain decay model.

The same **Actions** tab is used for Preview, Run fit, Auto pre-fit, stability testing, diagnostics, batch fitting, and export. This preserves the same FitED workflow while allowing the backend to build either a peak-profile model or an IRF-reconvolved time-domain model.

---

## Main Features

* Desktop graphical user interface based on Tkinter and Matplotlib.
* Interactive loading, previewing, fitting, and exporting of one-dimensional data.
* Multi-peak fitting with adjustable centers, areas, widths, profile types, and bounds.
* Peak activation checkboxes to include or exclude specific components.
* Built-in Gaussian, Lorentzian, Pseudo-Voigt, and Exact Voigt peak profiles.
* Exact Voigt model with independently handled sigma and gamma parameters.
* Custom peak-profile definitions based on user expressions and named parameters.
* Custom background-profile definitions based on user expressions and named parameters.
* Background fitting using none, constant, linear, polynomial, or custom functions.
* ROI-based fitting using the selected data range rather than line numbers.
* Optional Savitzky-Golay smoothing for preview and peak-detection assistance.
* Peak picking directly from the plot.
* Automatic peak detection with configurable peak-finding controls.
* Peak dragging for interactive visual adjustment before final fitting.
* Dedicated Decay / IRF mode for TRPL, TCSPC, transient absorption kinetic traces, and general decay-like data.
* Built-in decay-response functions for single, bi-, triple, stretched, and common-rise decay models.
* Measured IRF reconvolution for time-domain fitting.
* Optional Gaussian synthetic IRF for approximate instrument-response treatment.
* IRF preprocessing with baseline correction, negative clipping, time-zero selection, interpolation, and normalization.
* Support for shared absolute IRF/sample time axes or re-zeroed IRF references.
* Separate timing parameters for sample-response onset and IRF alignment shift.
* Positive, negative, or signed/free amplitude options for different experimental signal conventions.
* Plotting of raw decay data, reconvolved fit, intrinsic response before IRF convolution, scaled IRF, and residuals.
* Weighting options for different fitting emphasis.
* Normal fitting, automatic pre-fitting, refinement with added peaks, and batch fitting.
* Automatic pre-fit trial ranking using AIC, BIC, chi-square, or reduced chi-square.
* Automatic pre-fit sampling by Fast Jitter, Latin Hypercube Sampling, or Hybrid sampling.
* Optimizer selection between Levenberg-Marquardt and robust LM versus DE+LM comparison.
* Optional random seed for reproducible fitting searches.
* Fit stability testing using repeated searches and repeated seed-based trials.
* Fit reports, stability reports, best-stability-fit reports, and derived-quantity reports.
* Residual and fit-quality diagnostics.
* Parameter-correlation and covariance-based diagnostic outputs.
* Confidence-ellipse analysis for selected parameter pairs.
* Derived quantities with uncertainty propagation.
* Session-only report histories inside the Reports tab.
* Temporary result packages that can be saved later as ZIP archives.
* Session save and load support.
* Export of curves, fitted parameters, reports, summaries, session files, Excel outputs, and ZIP packages.

---

## Decay / IRF Route

The **Decay / IRF** route is designed for fitting time-domain data where the measured signal is broadened by the instrument response function.

Typical examples include:

* TRPL decay traces,
* TCSPC fluorescence lifetime data,
* transient absorption kinetic traces such as ΔOD versus time,
* photocurrent decay traces,
* absorbance recovery traces,
* general decay-like time-domain signals.

The measured signal is modeled as:

$$
I_\mathrm{measured}(t) = B + \mathrm{IRF}(t) * R(t)
$$

where:

* (I_\mathrm{measured}(t)) is the measured experimental signal.
* (B) is the fitted baseline.
* (\mathrm{IRF}(t)) is the instrument response function.
* (R(t)) is the intrinsic sample response.
* (*) denotes convolution.

The intrinsic response is the model describing the actual sample behavior before instrumental broadening. The reconvolved fit is the curve that should be compared directly with the measured data.

---

## Built-In Decay Models

The Decay / IRF route includes several built-in decay models. In the equations below, (t_0) is the onset time of the intrinsic response. For (t < t_0), the intrinsic response is normally treated as zero.

For compact notation:

$$
u = t - t_0
$$

and the response is evaluated for (u \geq 0).

---

### Single Exponential

$$
R(t) = A_1 \exp\left(-\frac{u}{\tau_1}\right)
$$

where:

* (A_1) is the amplitude of the decay component.
* (\tau_1) is the lifetime of the decay component.

---

### Bi-Exponential

$$
R(t) =
A_1 \exp\left(-\frac{u}{\tau_1}\right)
+
A_2 \exp\left(-\frac{u}{\tau_2}\right)
$$

where:

* (A_1) and (A_2) are amplitudes.
* (\tau_1) and (\tau_2) are lifetimes.

The labels (\tau_1) and (\tau_2) do not inherently mean first and second physical processes. Users may choose to constrain the bounds so that (\tau_1) represents the fast component and (\tau_2) represents the slow component.

---

### Triple Exponential

$$
R(t) =
A_1 \exp\left(-\frac{u}{\tau_1}\right)
+
A_2 \exp\left(-\frac{u}{\tau_2}\right)
+
A_3 \exp\left(-\frac{u}{\tau_3}\right)
$$

where:

* (A_1), (A_2), and (A_3) are amplitudes.
* (\tau_1), (\tau_2), and (\tau_3) are lifetimes.

---

### Stretched Exponential

$$
R(t) =
A_1 \exp\left[-\left(\frac{u}{\tau_1}\right)^\beta\right]
$$

where:

* (A_1) is the amplitude.
* (\tau_1) is the characteristic lifetime.
* (\beta) is the stretching exponent.

The stretching exponent can describe non-single-exponential or distributed decay behavior. Users should interpret (\tau_1) and (\beta) with care because they may be correlated.

---

### Common Rise + Single-Exponential Decay

$$
R(t) =
\left[1 - \exp\left(-\frac{u}{\tau_\mathrm{rise}}\right)\right]
A_1 \exp\left(-\frac{u}{\tau_\mathrm{decay}}\right)
$$

where:

* (\tau_\mathrm{rise}) is the finite rise time.
* (\tau_\mathrm{decay}) is the decay lifetime.
* (A_1) is the amplitude.

This model should be used only when the sample response itself has a real physical rise. If the observed rise is only due to instrument broadening, measured IRF reconvolution with a direct decay model is usually more appropriate.

---

### Common Rise + Bi-Exponential Decay

$$
R(t) =
\left[1 - \exp\left(-\frac{u}{\tau_\mathrm{rise}}\right)\right]
\left[
A_1 \exp\left(-\frac{u}{\tau_1}\right)
+
A_2 \exp\left(-\frac{u}{\tau_2}\right)
\right]
$$

where:

* (\tau_\mathrm{rise}) is the common finite rise time.
* (A_1) and (A_2) are decay amplitudes.
* (\tau_1) and (\tau_2) are decay lifetimes.

---

### Common Rise + Triple-Exponential Decay

$$
R(t) =
\left[1 - \exp\left(-\frac{u}{\tau_\mathrm{rise}}\right)\right]
\left[
A_1 \exp\left(-\frac{u}{\tau_1}\right)
+
A_2 \exp\left(-\frac{u}{\tau_2}\right)
+
A_3 \exp\left(-\frac{u}{\tau_3}\right)
\right]
$$

where:

* (\tau_\mathrm{rise}) is the common finite rise time.
* (A_1), (A_2), and (A_3) are decay amplitudes.
* (\tau_1), (\tau_2), and (\tau_3) are decay lifetimes.

---

### Common Rise + Stretched Exponential

$$
R(t) =
\left[1 - \exp\left(-\frac{u}{\tau_\mathrm{rise}}\right)\right]
A_1 \exp\left[-\left(\frac{u}{\tau_1}\right)^\beta\right]
$$

where:

* (\tau_\mathrm{rise}) is the finite rise time.
* (A_1) is the amplitude.
* (\tau_1) is the characteristic lifetime.
* (\beta) is the stretching exponent.

---

## IRF Reconvolution

For measured IRF fitting, FitED uses reconvolution. The intrinsic response is first calculated and then convolved with the IRF:

$$
I_\mathrm{fit}(t) = B + \mathrm{IRF}(t) * R(t)
$$

The reconvolved fit is the final curve compared with the measured data.

The IRF is treated as the time-response kernel of the instrument. Its amplitude in the plot may be scaled for visual comparison, while the internal fitting can use a normalized IRF kernel for stable reconvolution.

This means that the plotted IRF intensity should not automatically be interpreted as a physical signal amplitude. It is often shown at a convenient visual scale to compare its width and timing with the measured decay.

---

## IRF Preprocessing

FitED can preprocess the measured IRF before reconvolution.

### IRF Baseline

The **IRF baseline** option controls how the IRF offset is removed before convolution.

Available options include:

* **None**: uses the IRF as loaded. This is suitable only when the IRF has already been baseline-corrected.
* **Minimum**: subtracts the minimum value of the IRF.
* **Edge median**: estimates the IRF baseline from the edge regions of the IRF file and subtracts it.

The IRF baseline is not the same as the fitted sample baseline. The fitted sample baseline is controlled by the decay parameter `baseline`.

### Negative IRF Clipping

After IRF baseline correction, negative IRF values can optionally be clipped to zero. This is useful when small negative values appear only because of noise or baseline subtraction.

### IRF Zero

The **IRF zero** option defines the time reference used for the IRF.

Available options include:

* **Keep IRF time axis**: keeps the IRF on its original x-axis. This is useful when the IRF and sample decay share the same absolute time axis.
* **Peak maximum**: shifts the IRF so that its maximum is at zero time.
* **Center of mass**: shifts the IRF so that its intensity-weighted temporal center is at zero time.

The choice of IRF zero affects the interpretation of `t0` and `irf_shift`.

If **Keep IRF time axis** is used, the IRF already carries its timing relative to the sample data. In this case, `t0` can often be fixed near zero and `irf_shift` can be allowed to vary only slightly.

If **Peak maximum** is used, the IRF is re-zeroed, and `t0` usually represents the visible onset of the sample response in the measured data.

---

## Timing Parameters

The Decay / IRF route contains two timing-related parameters: `t0` and `irf_shift`.

### t0

`t0` is the onset time of the intrinsic sample response.

For a direct exponential decay:

$$
R(t) = 0 \quad \text{for} \quad t < t_0
$$

and:

$$
R(t) = A_1 \exp\left[-\frac{t - t_0}{\tau_1}\right]
\quad \text{for} \quad t \geq t_0
$$

### irf_shift

`irf_shift` is an additional timing correction applied to the IRF alignment. It is not a lifetime. It is used to correct small timing mismatches between the measured IRF and the measured sample decay.

The timing alignment can be thought of as:

$$
I_\mathrm{fit}(t) =
B + \mathrm{IRF}(t + \Delta t_\mathrm{IRF}) * R(t - t_0)
$$

where (\Delta t_\mathrm{IRF}) corresponds to `irf_shift`.

Because both `t0` and `irf_shift` affect the early-time alignment, they can be strongly correlated. Users should avoid letting both parameters vary freely over wide ranges unless there is a clear scientific reason.

---

## Weighting and Residual Interpretation

FitED supports several weighting choices, including no weighting, Poisson-like weighting, sqrt(y) emphasis, and 1/y-type weighting.

For count-based TRPL or TCSPC data, Poisson-like weighting may be more appropriate than ordinary unweighted fitting because the noise level often depends on the number of counts.

Raw residuals are calculated as:

$$
r_i = y_i - y_{\mathrm{fit}, i}
$$

For count-like data, weighted or Pearson-like residuals may be more informative:

$$
r_{\mathrm{weighted}, i}
========================

\frac{y_i - y_{\mathrm{fit}, i}}
{\sqrt{y_{\mathrm{fit}, i}}}
$$

Residual diagnostics should be interpreted together with the selected weighting, the signal level, the model choice, parameter correlations, and the physical plausibility of the fitted parameters.

---

## Typical Outputs

FitED can generate and export:

* Fitted total curve.
* Individual peak-component curves.
* Background component curve when applicable.
* Residual curve.
* Best-fit numerical data.
* Fitted parameter table.
* Parameter bounds and final parameter values.
* Fit metric summary, including chi-square, reduced chi-square, AIC, and BIC.
* Fit report text.
* Stability test report.
* Stability best-fit report.
* Repeated-fit score spread from stability testing.
* Near-best parameter spread from stability testing.
* Derived quantity table.
* Derived quantity uncertainty report.
* Residual diagnostic plots and summaries.
* Parameter correlation matrix output.
* Confidence-ellipse output.
* Covariance-related diagnostic outputs when available.
* Excel output files.
* Session JSON file.
* ZIP result package for sharing, archiving, or later documentation.

For Decay / IRF fitting, FitED can also generate and export:

* Reconvolved fitted decay curve.
* Intrinsic decay response before IRF convolution.
* Scaled IRF curve used for visual display.
* Decay residual curve.
* Fitted decay parameters, including baseline, response onset time, IRF shift, amplitudes, lifetimes, stretching exponent, and rise time when applicable.
* Decay-model fit metrics, including chi-square, reduced chi-square, AIC, and BIC.
* Decay/IRF fit report with fitted parameter values, bounds, uncertainties, and correlations.
* Diagnostic plots for time-domain residuals.
* Session and ZIP outputs preserving the Decay / IRF model settings and fitted results.

---

## Reports and History Behavior

FitED keeps the latest reports and session-only report histories inside the **Reports** tab. Popup report windows can still be opened normally, but closing a popup does not remove the report from the Reports tab.

The Reports tab stores:

* Fit reports.
* Stability test reports.
* Stability best-fit reports.
* Derived quantity reports.
* Correlation and diagnostic reports.
* Temporary result packages.

Temporary result packages can be reviewed during the session and saved later as ZIP files.

---

## Interface Terms

### Data Loading

* **Open spectrum file**: opens a file-selection dialog for text, CSV, DAT, ASC, or generic data files.
* **Delimiter**: selects tab, comma, semicolon, space, or automatic delimiter detection.
* **Skip rows**: ignores leading header or metadata rows before numerical parsing.
* **X col / Y col**: selects the zero-indexed independent and dependent variable columns.
* **Reload file**: reloads the current file using the current parsing settings.
* **Reset peaks / fit state**: resets the region of interest, peak definitions, picked centers, and stored fit results for the current spectrum.

### ROI and Preprocessing

* **ROI min / ROI max**: lower and upper limits of the selected fitting region.
* **Preview smoothing**: enables Savitzky-Golay smoothing for preview and optional peak detection only.
* **SG window**: Savitzky-Golay smoothing window size.
* **SG poly**: polynomial order used for Savitzky-Golay smoothing.

### Background

* **Background**: selected background model used in the fit.
* **Manage custom backgrounds**: opens the editor for user-defined background expressions and parameters.
* **Poly order**: polynomial degree used when the polynomial background is selected.

### Weighting

* **Weights**: weighting scheme applied during fitting.
* **No weighting**: fits all points without an additional weighting scheme.
* **Poisson-like weighting**: applies a count-like weighting behavior useful for intensity-type data.
* **sqrt(y) emphasis**: intentionally emphasizes higher-intensity points and should be interpreted as an emphasis option rather than a standard noise model.
* **1/y weighting**: gives relatively stronger emphasis to low-intensity points.

### Fit Settings

* **Fit criterion**: criterion used to rank automatic pre-fit trials or repeated stability-test results.
* **AIC**: Akaike Information Criterion.
* **BIC**: Bayesian Information Criterion.
* **Chi-square**: weighted or unweighted sum-of-squares fit metric depending on the selected weighting.
* **Reduced chi-square**: chi-square normalized by the effective degrees of freedom.
* **Optimizer mode**: selected fitting strategy used by the backend.
* **Levenberg-Marquardt**: local least-squares optimizer starting from the current parameters.
* **Robust: compare LM and DE+LM**: compares a normal LM fit with a Differential Evolution global-search result polished by LM and keeps the better candidate.
* **Max nfev**: maximum number of function evaluations allowed during fitting.
* **Random seed**: optional integer seed used for reproducible stochastic search behavior.
* **Auto-fit trials**: number of automatic pre-fit trial attempts.
* **Auto pre-fit sampling**: method used to generate automatic pre-fit trial starting conditions.
* **Fast Jitter**: fast trial-generation method based on randomized perturbations around the initial estimates.
* **Latin Hypercube Sampling**: trial-generation method that samples the allowed parameter space more systematically.
* **Hybrid: Fast Jitter + Latin Hypercube**: combines the fast original trial generation with Latin Hypercube trials.

### Peak/Profile Terms

* **Number of peaks**: number of peak rows/components available in the model.
* **Use**: checkbox that includes or excludes a peak row from the active fit.
* **Kind**: peak line-shape model, such as Gaussian, Lorentzian, Pseudo-Voigt, Exact Voigt, or Custom.
* **Profile**: selected user-defined custom profile when Kind is set to Custom.
* **Center**: peak position.
* **Area**: integrated peak amplitude.
* **FWHM**: full width at half maximum.
* **c min / c max**: lower and upper fitting bounds for the peak center.
* **a min / a max**: lower and upper fitting bounds for the peak area.
* **w min / w max**: lower and upper fitting bounds for the peak width.
* **G/L mix**: Gaussian/Lorentzian mixing fraction for the Pseudo-Voigt model.
* **Sigma**: Gaussian width parameter used in the Exact Voigt model.
* **Gamma**: Lorentzian width parameter used in the Exact Voigt model.

### Decay / IRF Terms

* **Fit mode used by Actions**: selects whether the Actions tab uses the standard peak/profile model or the Decay / IRF time-domain model.
* **Peak/profile mode**: uses the Peaks tab, background settings, and normal peak-profile model construction.
* **Decay/IRF mode**: uses the Decay / IRF tab to build and fit a time-domain decay model.
* **Data type**: selects the experimental context, such as TRPL/TCSPC, transient absorption kinetic trace, or general decay.
* **Decay function**: selected built-in decay response model.
* **Single exponential**: one decay component with one amplitude and one lifetime.
* **Bi-exponential**: two decay components with two amplitudes and two lifetimes.
* **Triple exponential**: three decay components with three amplitudes and three lifetimes.
* **Stretched exponential**: non-single-exponential decay described by an amplitude, characteristic lifetime, and stretching exponent.
* **Common rise + single-exponential decay**: a finite-rise response multiplied by one exponential decay component.
* **Common rise + bi-exponential decay**: a shared finite-rise response multiplied by two exponential decay components.
* **Common rise + triple-exponential decay**: a shared finite-rise response multiplied by three exponential decay components.
* **Common rise + stretched exponential**: a shared finite-rise response multiplied by a stretched-exponential decay.
* **Signal sign**: controls whether decay amplitudes are constrained to be positive, negative, or signed/free.
* **Positive amplitudes**: appropriate for normal positive TRPL or TCSPC intensity decays.
* **Negative amplitudes**: useful for negative-going kinetic traces.
* **Signed/free amplitudes**: allows amplitudes to be positive or negative, useful for general kinetic traces such as ΔOD versus time.
* **IRF treatment**: controls whether the decay model is fitted without IRF, with measured IRF reconvolution, or with a Gaussian synthetic IRF.
* **Measured IRF reconvolution**: convolves the intrinsic decay response with the loaded instrument response function before comparing the model with the measured data.
* **Gaussian synthetic IRF**: uses an approximate Gaussian instrument response when a measured IRF is not available.
* **Gaussian IRF FWHM**: full width at half maximum of the synthetic Gaussian IRF.
* **Open IRF file**: loads the measured IRF file used for reconvolution.
* **Reload IRF**: reloads the currently selected IRF file using the current parsing settings.
* **IRF baseline**: controls how the IRF background offset is removed before reconvolution.
* **IRF baseline none**: uses the IRF values as loaded, suitable only when the IRF is already baseline-corrected.
* **IRF baseline minimum**: subtracts the minimum IRF value.
* **IRF baseline edge median**: estimates the IRF baseline from the edge regions of the IRF file.
* **Clip negative IRF after baseline subtraction**: sets negative IRF values to zero after baseline correction.
* **IRF zero**: defines the time reference used for the IRF.
* **Keep IRF time axis**: keeps the IRF on its original x-axis, useful when the IRF and sample decay share the same absolute time axis.
* **Peak maximum**: shifts the IRF so that its maximum is at zero time.
* **Center of mass**: shifts the IRF so that its intensity-weighted temporal center is at zero time.
* **baseline**: fitted constant offset of the measured decay signal.
* **t0**: onset time of the intrinsic sample response.
* **irf_shift**: additional timing correction applied to the IRF alignment.
* **A1, A2, A3**: amplitudes of the decay components.
* **tau1, tau2, tau3**: lifetimes of the decay components.
* **beta**: stretching exponent used in the stretched-exponential model.
* **tau_rise**: finite rise time used in common-rise models.
* **tau_decay**: decay time used in the common-rise single-decay model.
* **Vary**: checkbox controlling whether a decay parameter is optimized or fixed during fitting.
* **Intrinsic response before IRF**: the fitted sample-response function before convolution with the instrument response.
* **Reconvolved fit**: the final fitted curve after convolution with the IRF, directly comparable to the measured data.
* **IRF scaled display**: the IRF curve rescaled for visualization only; the internal fitting may use a normalized IRF kernel.

### Actions and Reports

* **Residual**: difference between experimental data and fitted model.
* **Preview**: displays the current model, components, and residuals without finalizing an optimized fit.
* **Manage custom profiles**: opens the editor for user-defined peak expressions and parameters.
* **Pick centers from plot**: allows the user to select peak centers interactively from the plot.
* **Find peaks**: detects candidate peak positions using configurable detection settings.
* **Peak dragging**: allows interactive movement of selected peak positions on the plot for preview adjustment.
* **Auto pre-fit**: automatic fitting stage that searches for improved starting parameters based on peak centers and trial-generation settings.
* **Refine with added peaks**: refinement stage used after adding additional peaks to a previous fit.
* **Run fit**: manual/current-parameter fitting stage using the current values and bounds.
* **Batch fit**: applies the current setup to multiple files.
* **Fit stability test**: repeats fitting or automatic pre-fit searches with different seeds to assess robustness.
* **Derived quantities**: defines and evaluates quantities calculated from fitted parameters, with uncertainty propagation when covariance information is available.
* **Save session**: saves the current file reference, fitting setup, model settings, bounds, and custom definitions into a JSON session file.
* **Load session**: loads a previously saved FitED session.
* **Save ZIP results**: saves fitted outputs, reports, tables, curves, and session information into a ZIP package.
* **Reports tab**: stores current-session histories of fit reports, stability reports, best-fit reports, derived-quantity reports, diagnostic reports, and temporary result packages.

---

## Intended Use

FitED is intended for academic and research use in spectral analysis, time-domain decay analysis, and related fitting tasks. It is suitable for users who need a straightforward graphical environment for fitting experimental spectra or decay traces without manually writing fitting scripts for each dataset.

FitED is designed to support scientific interpretation, not replace it. The user remains responsible for selecting meaningful models, physically reasonable bounds, appropriate weighting, and proper interpretation of the fitted parameters.

---

## Important Notes

Users are responsible for choosing appropriate peak models, decay models, parameter bounds, and fitting settings. The software is a fitting tool and does not replace scientific judgment. Fit quality, parameter meaning, and physical interpretation must always be evaluated by the user.

For Decay / IRF fitting, users should carefully distinguish between the intrinsic sample response and the reconvolved measured response. The intrinsic response represents the physical decay model before instrumental broadening, while the reconvolved fit is the curve that should be compared directly with the measured data.

The IRF should normally be treated as a timing-response kernel. Its scale in the plot may be adjusted for visual comparison, while the fitting routine can internally normalize the IRF for stable reconvolution. The plotted IRF intensity should therefore not automatically be interpreted as a physical signal amplitude.

The timing parameters `t0` and `irf_shift` can be strongly correlated. If the IRF and sample data share the same time axis, it is often better to keep the IRF time axis, fix `t0` near zero, and allow only a small IRF shift if needed. If the IRF is re-zeroed using its peak maximum, `t0` usually represents the visible onset of the decay in the sample data. Users should avoid letting both `t0` and `irf_shift` vary over wide ranges unless there is a clear reason.

Multi-exponential decay fitting can produce strong correlations between amplitudes and lifetimes. Large parameter correlations or large covariance-based uncertainties do not necessarily mean that the fitted curve is unusable, but they indicate that the individual parameters may not be uniquely determined. Users should evaluate residuals, parameter bounds, stability tests, correlations, and physical plausibility before interpreting fitted lifetimes and amplitudes.

For count-based TRPL or TCSPC data, Poisson-like weighting may be more appropriate than ordinary unweighted residuals, depending on the purpose of the fit. Raw residual diagnostics can appear non-normal because the noise level changes with signal intensity.

---

## Citation

If you use this software in academic work, please cite:

**Aboulsaad, M. M. I. (2026). FitED. Zenodo.**
https://doi.org/10.5281/zenodo.19411620

---

## License and Usage

FitED is source-available non-commercial software.

You may use, copy, redistribute, and modify FitED for non-commercial purposes.

Commercial use is not permitted without prior written permission from the author.

FitED is not open-source software under the OSI definition because commercial use is restricted.

See `LICENSE.txt`, 'LICENSE.cff' for details.

---

## Disclaimer

This software is provided as-is, without warranty of any kind, express or implied. The author shall not be liable for any claim, damages, or other liability arising from, out of, or in connection with the software or the use or other dealings in the software.
