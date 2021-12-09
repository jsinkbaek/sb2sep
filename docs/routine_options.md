## Routine options
`spectral_separation_routine` requires that you create 3 custom objects from the module `storage_classes` that holds options and extra parameters for the parent and two child routines.

These are:
- a `RoutineOptions` object for the parent routine.
- a `SeparateComponentsOptions` object for the component separation subroutine.
- a `RadialVelocityOptions` object for the radial velocity subroutine.

Below, each parameter in the objects will be described. Some parameters are accessible from multiple objects, making it possible to give different values for each. This is discouraged.

For `RoutineOptions`:
- **time_values** (numpy array shape (n_spectra, )), Default = None. Holds the timestamps of all spectra. Primarily used to feed to output files.
- **convergence_limit** (float), Default = 1E-5. Convergence criteria for the radial velocity RMS needed to exit the parent routine.
- **iteration_limit** (int), Default = 10. Maximum allowed iterations of the parent routine before automatic exit.
- **plot** (bool), Default = True. If True, diagnostic plots are shown and updated between iterations.
- **verbose** (bool), Default = True. Indicates if parent routine should print extra updates during iteration.
- **return_unbuffered** (bool), Default = True. If received spectra are padded around the edges, setting this to True will remove said padding before returning separated spectra of both components.
- **save_plot_path** (string), Default = None. If not None, the final diagnostic plots will be saved as a multipage PDF file in the specified folder.
- **save_all_results** (bool), Default = True. If True, raw data is saved to disk before returning from the function. This data can be used along with the `evaluate_ssr_results` module.
- **save_path** (string), Default = '/.'. Defines the folder that raw data is saved to when `save_all_results` is set to True.
- **buffer_mask** (numpy array shape (:, )), Default = None. If data is padded with a buffer around the edges, this mask should be supplied. It should give the unbuffered data when flipped, e.g. `flux_buffered\[~buffer_mask\] = flux`. This parameter can be left at `None` when using `spectral_separation_routine_multiple_intervals`, as long as `wavelength_buffer_size` is supplied directly to that function, since the routine will manually set it.
