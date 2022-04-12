# Accessing output
`spectral_separation_routine` returns one of two sets of results. If `options.return_unbuffered` is True, every flux and wavelength returned will be the unbuffered version, where no padding was introduced. Otherwise, the buffer zones will not be removed in the arrays returned by the function. In both cases, 5 arrays are returned:
- **RV_collection_A** (numpy array shape (n_spectra, ) or (n_orders, n_spectra), [see note in routine description](quickstart.md)). All the found radial velocities for the stellar component A.
- **RV_collection_B** (numpy array shape (n_spectra, ) or (n_orders, n_spectra)). All the found radial velocities for the stellar component B.
- **separated_flux_A** (numpy array shape (data.size, )). The component A flux found using spectral separation.
- **separated_flux_B**. Same, but for component B.
- **wavelength** (numpy array shape (data.size, )). The wavelength values associated with the two separated spectra.

`spectral_separation_routine_multiple_intervals` returns instead a list of the same results as `spectral_separation_routine`, with one tuple of results for each interval.

By specifying a *save_path* and setting *save_all_results* to True in the **options** argument to one of the two functions, it is possible to save a much higher wealth of raw information from the routine. These will be labelled by either the RoutineOptions.filename_bulk parameter, or the min_max wavelengths of the data-set, and includes 
- Final broadening function values for both components for every spectrum (\*bfvals_X.txt and \*bfsmooth_X.txt)
- Fitted rotational broadening function profile values (\*models_X.txt)
- Broadening function velocity values (\*velocities_X.txt)
- Found radial velocities (\*rvX.txt)
- Initial radial velocity guess (\*rv_initial.txt)
- Separated fluxes (\*sep_flux.txt)

These outputs are designed to be conveniently used with the diagnostic plotting functions in the module `evaluate_ssr_results.py`.

[Next Page: Preparing spectra](preparation)
