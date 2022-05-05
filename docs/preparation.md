# Preparing spectra
Before feeding spectra to the spectral separation routine, they must be appropriately prepared for analysis. The module `spectrum_processing_functions` is convenient to use for this purpose. Key functions relevant will be covered below.

## Spectrum normalization
The function `simple_normalizer()` does a decent job at normalizing a spectrum and reducing emission lines, at least for the purpose at hand. It uses a 3 pass process with a moving median filter and 2 successive polynomial fits.

Input:
- **wavelength** (numpy array)
- **flux** (numpy array)
- **reduce_em_lines** (bool), Default = True. Performs filtering to remove upper outliers.
- **plot** (bool), Default = False. Plots the normalization compared with original.

Returns:
- wavelength, normalized_flux

## Loading template spectra
If using template spectra from [Coelho et al. 2005](https://www.aanda.org/articles/aa/pdf/2005/44/aa3511-05.pdf), the function `load_template_spectrum()` can be used to load from file.

Input:
- **template_spectrum_path** (string)

Returns:
- wavelength (numpy array), flux (numpy array)

## Resampling spectra
The templates and observed spectra must all be resampled to the same wavelength grid, equi-spaced in velocity space. Multiple functions exist for this, but a convenient wrapper performs this for all spectra simultanously. This is `resample_multiple_spectra()`.

Input:
- **delta_v** (float)
- **\*args**. Should be a collection of tuples with (wavelength, flux) for each. This collection should contain all spectra to be resampled. An example: \[(wavelength_1, flux_1), (wavelengt_2, flux_2)\].
- **wavelength_template** (numpy array). Default = None. If supplied, resampling will be done to match this wavelength grid instead of creating a new one.
- **wavelength_a** (float), Default = None. Lower bound for the resampled wavelength.
- **wavelength_b** (float). Default = None. Upper bound for the resampled wavelength.
- **resampled_len_even** (bool). Default = True. Will force the resampled spectra to have an even number of elements.

Returns:
- **wavelength_template** (numpy array).
- **tuple(flux_1, flux_2, ...)**. Resampled fluxes (each numpy arrays of 1 or 2 dim depending on the input shape).

Note: It is also allowed to have one or more of the supplied tuples contain multiple spectra, e.g. 2-dim arrays (wavelength_collection, flux_collection).

## Limit wavelength interval
The function `limit_wavelength_interval_multiple_spectra()` is convenient for removing unneeded data. The spectra must have previously been resampled to the same wavelength grid.

Input:
- **wavelength_limits** (tuple(float, float)). 
- **wavelength** (numpy array).
- **\*args** (collection of numpy arrays). The flux values of all spectra in the form (flux1, flux2, flux3). Each array can be either 1d or 2d (collection).
- **buffer_size** (float). Default = None. If provided, padding is introduced to the edges in the form of a small buffer zone.
- **even_length** (bool). Default = True. Forces output to be even length.

If `buffer_size=None`, returns:
- wavelength, List\[flux1, flux2, ...\]

Else, returns:
- wavelength_unbuffered, List\[flux1_unbuffered, ...\], wavelength_buffered, List\[flux1_buffered, ...\], buffer_mask

## Barycentric corrections
[barycorrpy](https://pypi.org/project/barycorrpy/) is recommended to calculate barycentric corrections. If it is so desired, the spectra can be corrected for them before RV calculation by using the `shift_spectrum()` function from the `spectral_separation_routine` module.

Input:
- **flux** (numpy 1d array)
- **radial_velocity_shift** (float). The radial velocity shift in km/s.
- **delta_v**. The resolution of the resampled wavelength grid in km/s.

Returns:
- **flux_shifted**

## Estimate RV of one or two components
To perform an initial RV calculation before the routine (to have initial guesses), the function `radial_velocities_of_multiple_spectra()` from the module `calculate_radial_velocities` can be used.

Input:
- **inv_flux_collection** (numpy array shape (data_size, n_spectra)). This must be a collection of inverted normalized fluxes (1-flux), all resampled to the same wavelength grid.
- **inv_flux_template** (numpy 1D array). An inverted template flux resampled to the same wavelength grid.
- **options** (RadialVelocityOptions). Options for the fitting routine. See [Routine options](routine_options) for more details.
- **fit_two_compontents** (bool), Default = True. If True, will successively fit two broadening functions. Otherwise, will only fit for component A.
- **number_of_parallel_jobs** (int), Default = 4. Indicates how many parallel spectra should be calculated by joblib.
- **plot** (bool), Default = False. If True, will plot resulting fits.

Returns (1 component):
- **RVs** (numpy array)
- tuple(bf_velocity, bf_vals, bf_vals_smooth, model_vals)

Returns (2 components):
- **RVs_A** (numpy array)
- **RVs_B** (numpy array)
- tuple(bf_velocity, bf_vals, bf_vals_smooth, model_vals_A, model_vals_B)

[Next Page: Evaluating output](evaluate)