# FitED

**FitED** is a GUI-based scientific fitting and analysis software for experimental spectral data, peak-profile fitting, custom functional fitting, visualization, and export of fitting results.

FitED is **source-available for non-commercial use only**. You may use, copy, modify, and redistribute it for non-commercial academic, educational, and research purposes. Commercial use is not permitted without prior written permission from the author.

**Author:** Mustafa Mahmoud Ibrahim Aboulsaad  
**Version:** v1.0  
**Software DOI:** https://doi.org/10.5281/zenodo.19411621  

---

## Overview

FitED is a user-centric, extensible desktop application for robust peak-profile and general functional data fitting. It is designed to help users load experimental spectral data, define a fitting region, construct multi-peak and custom models, perform nonlinear fitting, visualize the results, and export the fitted outputs for further analysis.

---

## What the Software Does

FitED allows the user to:

- Load spectral data from text-based files
- Select the fitting region of interest
- Apply optional smoothing for preview purposes
- Define multiple peaks and adjust their initial parameters and fitting bounds
- Choose different peak line shapes, including Gaussian, Lorentzian, Pseudo-Voigt, and Exact Voigt
- Choose a custom profile for fitting custom data, such as photoluminescence decay with single-exponential, bi-exponential, or stretched-exponential functions based on a user-defined expression and parameters
- Add background models such as none, constant, linear, or polynomial
- Add a custom background model based on a user-defined expression and parameters
- Run the fit and inspect the fitted curve, peak components, and residuals
- Export fitting results, parameters, and reports for documentation and further processing

---

## Main Features

- Desktop graphical user interface
- Interactive multi-peak fitting
- Adjustable peak positions, amplitudes, widths, and bounds
- Custom fitting profiles
- Support for several background models, including user-defined background models
- Three levels of fitting:
  - Initial automatic fitting
  - Fitting with added peaks
  - Manual fitting
- Selection of different fitting criterions
- Peak dragging
- Result export in multiple formats
- Session save and load support
- ZIP export of fitting outputs for easy sharing and archiving

---

## Typical Outputs

FitED can export:

- Fitted curves
- Peak components
- Residuals
- Fitted parameter tables
- Fit reports
- Session information
- Packaged ZIP output for record keeping

---

## Interface Terms

| Term | Meaning |
|---|---|
| **Open spectrum file** | Opens a file-selection dialog for text, CSV, DAT, ASC, or generic files |
| **Delimiter** | Allows tab, comma, semicolon, space, or automatic delimiter selection |
| **Skip rows** | Removes leading metadata or header rows before numerical parsing |
| **X col / Y col** | Selects the zero-indexed independent and dependent variable columns |
| **Reload file** | Reloads the current file using the current parsing settings |
| **Reset peaks / fit state** | Resets the region of interest, peak definitions, picked centers, and stored fit results for the current spectrum |
| **ROI min / ROI max** | Lower and upper limits of the fitting region |
| **Peaks** | Number of peak components included in the model |
| **Background** | Background model used in the fit |
| **Manage custom backgrounds** | User-defined background model |
| **Poly order** | Polynomial degree for the background model |
| **SG window** | Savitzky-Golay smoothing window size used for preview |
| **SG poly** | Polynomial order used in Savitzky-Golay smoothing |
| **Weights** | Weighting scheme applied during fitting |
| **Auto-fit trial** | Number of iterations used for the automatic pre-fit level |
| **Kind** | Peak line-shape model: Gaussian, Lorentzian, Pseudo-Voigt, Exact Voigt, or Custom |
| **Profile** | User-defined custom profile when Custom is selected |
| **Center** | Peak position |
| **Area** | Integrated peak amplitude |
| **FWHM** | Full width at half maximum |
| **c min / c max** | Lower and upper bounds for peak center |
| **a min / a max** | Lower and upper bounds for peak area |
| **w min / w max** | Lower and upper bounds for peak width |
| **G/L mix** | Gaussian/Lorentzian mixing fraction for the Pseudo-Voigt model |
| **Sigma** | Gaussian width parameter |
| **Gamma** | Lorentzian width parameter |
| **Residual** | Difference between experimental data and fitted model |
| **Preview** | View of the raw data, fitted profiles, and peaks with the current input values |
| **Manage custom profiles** | User-defined expression and parameters for custom fitting profiles |
| **Pick centers from plot** | Allows the user to pick peak positions interactively from the plot |
| **Auto pre-fit** | First level of fitting. It allows `lmfit` to find suitable parameters for reducing residuals based on selected peak centers |
| **Refine with added peaks** | Second level of fitting. It allows additional peaks to be added and refines the fit after applying constraints to initially fitted peak data |
| **Run fit** | Third level of fitting. It allows full manual control of boundaries and parameters |
| **Save session** | Saves the current session, including the raw data file and input parameters, into a `.json` file |
| **Load session** | Loads a previously saved session |
| **Save ZIP results** | Saves fitted parameters and automatically includes the `.json` session file in the ZIP archive |

---

## Intended Use

FitED is intended for academic and research use in spectral analysis and related fitting tasks. It is suitable for users who need a straightforward graphical environment for fitting experimental spectra without manually writing fitting scripts for each dataset.

---

## Important Note

Users are responsible for choosing appropriate peak models, parameter bounds, and fitting settings.

FitED is a fitting tool and does not replace scientific judgment. Fit quality, parameter meaning, and physical interpretation must always be evaluated by the user.

---

## Citation

If you use FitED in academic work, please cite:

> Aboulsaad, M. M. FitED. Zenodo.  
> https://doi.org/10.5281/zenodo.19411621

Software title: **FitED**  
Author: **Mustafa Mahmoud Ibrahim Aboulsaad**  
Software DOI: **10.5281/zenodo.19411621**

---

## License and Usage

FitED is **source-available non-commercial software**.

You may use, copy, redistribute, and modify FitED for non-commercial purposes.

Commercial use is **not permitted** without prior written permission from the author.

FitED is **not open-source software under the OSI definition**, because commercial use is restricted.

See [`LICENSE.txt`](LICENSE.txt) for full license terms.

---

## Disclaimer

This software is provided **as is**, without warranty of any kind, express or implied.

The author shall not be liable for any claim, damages, or other liability arising from, out of, or in connection with the software or the use or other dealings in the software.

Users are solely responsible for validating all fitting results, exported outputs, calculations, plots, parameters, and interpretations produced using FitED.
