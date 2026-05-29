# FitED

**Author:** Mustafa Mahmoud Ibrahim Aboulsaad  
**Version:** v1.3  
**Software DOI:** 10.5281/zenodo.19411620

---

## Overview

FitED is a user-centric, extensible desktop application for robust peak-profile and general functional data fitting. It is designed to help users load experimental one-dimensional data, define a fitting region, construct multi-peak, background, and custom analytical models, perform nonlinear fitting, visualize fitted components and residuals, evaluate fit quality, and export results for documentation and further analysis.

FitED is mainly developed for spectroscopy and related experimental datasets, but it can also be used for other one-dimensional fitting problems where the user needs an interactive graphical environment rather than writing a new fitting script for every dataset.

---

## What the Software Allows the User to Do

FitED allows the user to:

- Load spectral or one-dimensional experimental data from text-based files
- Select the X and Y columns from multi-column files
- Choose tab, comma, semicolon, space, or automatic delimiter detection
- Skip leading header or metadata rows before numerical parsing
- Select a fitting region of interest using ROI minimum and maximum values
- Apply optional Savitzky-Golay smoothing for preview and peak-detection purposes only
- Define multiple peak components with adjustable starting values and fitting bounds
- Include or exclude individual peak rows using the `Use` checkbox without deleting the peak setup
- Pick peak centers interactively from the plot
- Detect candidate peaks automatically using configurable peak-finding settings
- Fit positive peaks, negative peaks, or manually defined peak components depending on the workflow
- Choose built-in peak profiles, including Gaussian, Lorentzian, Pseudo-Voigt, and Exact Voigt
- Use Exact Voigt profiles with independently handled Gaussian sigma and Lorentzian gamma parameters
- Use the Pseudo-Voigt Gaussian/Lorentzian mixing fraction, where 0 corresponds to Gaussian-like behavior and 1 corresponds to Lorentzian-like behavior
- Define custom analytical peak profiles using user-defined expressions and named parameters
- Define custom fitting models for non-standard data, such as photoluminescence decay or other functional responses
- Add background models, including none, constant, linear, polynomial, and custom analytical backgrounds
- Set fitting weights, including no weighting, Poisson-like weighting, sqrt(y) emphasis, and 1/y-type weighting
- Preview the current model, components, and residuals before running the optimizer
- Drag peak positions in the plot for visual model adjustment before confirming a final fit
- Run a normal fit from the current user-defined values and bounds
- Run automatic pre-fit searches from selected, detected, or manually defined peak centers
- Refine a previous fit after adding extra peaks
- Fit batches of files using the current fitting setup
- Choose the model-selection or trial-ranking criterion among AIC, BIC, chi-square, and reduced chi-square
- Choose the optimizer mode, including Levenberg-Marquardt and a robust mode that compares Levenberg-Marquardt with Differential Evolution followed by Levenberg-Marquardt polishing
- Control the maximum number of function evaluations
- Set an optional random seed for reproducible stochastic searches
- Choose automatic pre-fit sampling strategies, including Fast Jitter, Latin Hypercube Sampling, and Hybrid Fast Jitter plus Latin Hypercube Sampling
- Run stability tests by repeating fits or automatic pre-fit searches with different seeds
- Rank repeated stability-test results using the selected criterion
- Inspect repeated-solution score spread and near-best parameter spread
- Inspect fitted curves, individual components, and residuals
- Generate residual diagnostics to help evaluate systematic fitting errors or model inadequacy
- Inspect parameter correlations and covariance-related diagnostics
- Define derived quantities from fitted parameters
- Propagate uncertainty for derived quantities using covariance information when available
- Inspect uncertainty contributions for derived quantities
- Save and load fitting sessions
- Keep report histories inside the Reports tab during the current session
- Preserve generated reports after closing popup windows
- Store temporary result packages during the session and save them later as ZIP files

---

## Main Interface Structure

The FitED desktop interface is organized into workflow tabs:

- **Load data**
- **Fit settings**
- **Peaks**
- **Actions**
- **Reports**

The plot area displays the loaded data, model preview, fitted curve, fitted components, and residuals. The GUI is used for interaction, plotting, and report display, while the fitting logic is handled by the backend.

---

## Main Features

- Desktop graphical user interface based on Tkinter and Matplotlib
- Interactive loading, previewing, fitting, and exporting of one-dimensional data
- Multi-peak fitting with adjustable centers, areas, widths, profile types, and parameter bounds
- Peak activation checkboxes to include or exclude individual components
- Built-in Gaussian, Lorentzian, Pseudo-Voigt, and Exact Voigt peak profiles
- Exact Voigt model with independent sigma and gamma handling
- Custom peak-profile definitions based on user expressions and named parameters
- Custom background-profile definitions based on user expressions and named parameters
- Background fitting using none, constant, linear, polynomial, or custom functions
- ROI-based fitting using the selected data range
- Optional Savitzky-Golay smoothing for preview and peak-detection assistance
- Interactive peak picking directly from the plot
- Automatic peak detection with configurable peak-finding controls
- Peak dragging for interactive visual adjustment before final fitting
- Weighting options for different fitting emphasis
- Normal fitting, automatic pre-fitting, refinement with added peaks, and batch fitting
- Automatic pre-fit trial ranking using AIC, BIC, chi-square, or reduced chi-square
- Automatic pre-fit sampling by Fast Jitter, Latin Hypercube Sampling, or Hybrid sampling
- Optimizer selection between Levenberg-Marquardt and robust LM versus DE+LM comparison
- Optional random seed for reproducible fitting searches
- Fit stability testing using repeated searches and seed-based trials
- Fit reports, stability reports, best-stability-fit reports, and derived-quantity reports
- Residual and fit-quality diagnostics
- Parameter-correlation and covariance-based diagnostic outputs
- Derived quantities with uncertainty propagation
- Session-only report histories inside the Reports tab
- Temporary result packages that can be saved later as ZIP archives
- Session save and load support
- Export of curves, fitted parameters, reports, summaries, session files, Excel outputs, and ZIP packages

---

## Typical Outputs

FitED can generate and export:

- Fitted total curve
- Individual peak-component curves
- Background component curve when applicable
- Residual curve
- Best-fit numerical data
- Fitted parameter table
- Parameter bounds and final parameter values
- Fit metric summary, including chi-square, reduced chi-square, AIC, and BIC
- Fit report text
- Stability test report
- Stability best-fit report
- Repeated-fit score spread from stability testing
- Near-best parameter spread from stability testing
- Derived quantity table
- Derived quantity uncertainty report
- Residual diagnostic plots and summaries
- Parameter correlation matrix output
- Covariance-related diagnostic outputs when available
- Excel output files
- Session JSON file
- ZIP result package for sharing, archiving, or later documentation

---

## Reports and History Behavior

FitED keeps the latest reports and session-only report histories inside the Reports tab. Popup report windows can still be opened normally, but closing a popup does not remove the report from the Reports tab.

The Reports tab stores:

- Fit reports
- Stability test reports
- Stability best-fit reports
- Derived quantity reports
- Temporary result packages

Temporary result packages can be reviewed during the session and saved later as ZIP files.

---

## Interface Terms

| Term | Meaning |
|---|---|
| **Open spectrum file** | Opens a file-selection dialog for text, CSV, DAT, ASC, or generic data files |
| **Delimiter** | Selects tab, comma, semicolon, space, or automatic delimiter detection |
| **Skip rows** | Ignores leading header or metadata rows before numerical parsing |
| **X col / Y col** | Selects the zero-indexed independent and dependent variable columns |
| **Reload file** | Reloads the current file using the current parsing settings |
| **Reset peaks / fit state** | Resets the region of interest, peak definitions, picked centers, and stored fit results for the current spectrum |
| **ROI min / ROI max** | Lower and upper limits of the selected fitting region |
| **Preview smoothing** | Enables Savitzky-Golay smoothing for preview and optional peak detection only |
| **SG window** | Savitzky-Golay smoothing window size |
| **SG poly** | Polynomial order used for Savitzky-Golay smoothing |
| **Background** | Selected background model used in the fit |
| **Manage custom backgrounds** | Opens the editor for user-defined background expressions and parameters |
| **Poly order** | Polynomial degree used when the polynomial background is selected |
| **Weights** | Weighting scheme applied during fitting |
| **No weighting** | Fits all points without an additional weighting scheme |
| **Poisson-like weighting** | Applies a count-like weighting behavior useful for intensity-type data |
| **sqrt(y) emphasis** | Intentionally emphasizes higher-intensity points and should be interpreted as an emphasis option rather than a standard noise model |
| **1/y weighting** | Gives relatively stronger emphasis to low-intensity points |
| **Fit criterion** | Criterion used to rank automatic pre-fit trials or repeated stability-test results |
| **AIC** | Akaike Information Criterion |
| **BIC** | Bayesian Information Criterion |
| **Chi-square** | Weighted or unweighted sum-of-squares fit metric depending on the selected weighting |
| **Reduced chi-square** | Chi-square normalized by the effective degrees of freedom |
| **Optimizer mode** | Selected fitting strategy used by the backend |
| **Levenberg-Marquardt** | Local least-squares optimizer starting from the current parameters |
| **Robust: compare LM and DE+LM** | Compares a normal LM fit with a Differential Evolution global-search result polished by LM and keeps the better candidate |
| **Max nfev** | Maximum number of function evaluations allowed during fitting |
| **Random seed** | Optional integer seed used for reproducible stochastic search behavior |
| **Auto-fit trials** | Number of automatic pre-fit trial attempts |
| **Auto pre-fit sampling** | Method used to generate automatic pre-fit trial starting conditions |
| **Fast Jitter** | Fast trial-generation method based on randomized perturbations around the initial peak estimates |
| **Latin Hypercube Sampling** | Trial-generation method that samples the allowed parameter space more systematically |
| **Hybrid: Fast Jitter + Latin Hypercube** | Combines the fast original trial generation with Latin Hypercube trials |
| **Number of peaks** | Number of peak rows/components available in the model |
| **Use** | Checkbox that includes or excludes a peak row from the active fit |
| **Kind** | Peak line-shape model, such as Gaussian, Lorentzian, Pseudo-Voigt, Exact Voigt, or Custom |
| **Profile** | Selected user-defined custom profile when Kind is set to Custom |
| **Center** | Peak position |
| **Area** | Integrated peak amplitude |
| **FWHM** | Full width at half maximum |
| **c min / c max** | Lower and upper fitting bounds for the peak center |
| **a min / a max** | Lower and upper fitting bounds for the peak area |
| **w min / w max** | Lower and upper fitting bounds for the peak width |
| **G/L mix** | Gaussian/Lorentzian mixing fraction for the Pseudo-Voigt model |
| **Sigma** | Gaussian width parameter used in the Exact Voigt model |
| **Gamma** | Lorentzian width parameter used in the Exact Voigt model |
| **Residual** | Difference between experimental data and fitted model |
| **Preview** | Displays the current model, components, and residuals without finalizing an optimized fit |
| **Manage custom profiles** | Opens the editor for user-defined peak expressions and parameters |
| **Pick centers from plot** | Allows the user to select peak centers interactively from the plot |
| **Find peaks** | Detects candidate peak positions using configurable detection settings |
| **Peak dragging** | Allows interactive movement of selected peak positions on the plot for preview adjustment |
| **Auto pre-fit** | Automatic fitting stage that searches for improved starting parameters based on peak centers and trial-generation settings |
| **Refine with added peaks** | Refinement stage used after adding additional peaks to a previous fit |
| **Run fit** | Manual/current-parameter fitting stage using the current values and bounds |
| **Batch fit** | Applies the current setup to multiple files |
| **Fit stability test** | Repeats fitting or automatic pre-fit searches with different seeds to assess robustness |
| **Derived quantities** | Defines and evaluates quantities calculated from fitted parameters, with uncertainty propagation when covariance information is available |
| **Save session** | Saves the current file reference, fitting setup, model settings, bounds, and custom definitions into a JSON session file |
| **Load session** | Loads a previously saved FitED session |
| **Save ZIP results** | Saves fitted outputs, reports, tables, curves, and session information into a ZIP package |
| **Reports tab** | Stores current-session histories of fit reports, stability reports, best-fit reports, derived-quantity reports, and temporary result packages |

---

## Intended Use

FitED is intended for academic and research use in spectral analysis and related fitting tasks. It is suitable for users who need a straightforward graphical environment for fitting experimental spectra without manually writing fitting scripts for each dataset.

---

## Important Note

Users are responsible for choosing appropriate peak models, parameter bounds, and fitting settings. The software is a fitting tool and does not replace scientific judgment. Fit quality, parameter meaning, and physical interpretation must always be evaluated by the user.

---

## Citation

If you use this software in academic work, please cite:

Aboulsaad, M. M. I. (2026). FitED. Zenodo. https://doi.org/10.5281/zenodo.19411620

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
