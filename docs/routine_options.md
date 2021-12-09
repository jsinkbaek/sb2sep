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

For `SeparateComponentsOptions`:
- **delta_v (float)**, Default = 1.0. This must be set to be equal to the delta_v used to resample the spectra.
- **convergence_limit** (float), Default = 1E-2. Convergence criteria for the changes induced in the component spectra between iterations.
- **max_iterations (int)**, Default = 10. Maximum allowed iterations for the `separate_component_spectra` subroutine.
- **rv_proximity_limit** (float), Default = 0.0. If 0.0, this parameter is ignored. Otherwise, it is used to exclude spectra for the component calculation based on the calculated RV difference between each component. It is superceded by either `rv_lower_limit` or `use_for_spectral_separation_X` if any of them are provided for the component. See note after.
- **rv_lower_limit** (float), Default = 0.0. If 0.0, this parameter is ignored. Otherwise, it is used to exclude spectra for the component calculation based on the RV of the primary (A) component. This requires that the system RV has been approximately removed from the spectra beforehand. It is superceded by `use_for_spectral_separation_X` if provided for the component. See note after.
- **ignore_component_B** (bool), Default = False. If True, the secondary component (B) will be ignored for ALL provided spectra. This should NEVER be used unless you know with certainty that all the provided spectra does not include light from the secondary.
- **verbose** (bool), Default = False. If True, provides additional prints to the console.
- **use_for_spectral_separation_A** (numpy array shape (n_used_spectra, )), Default = None. If supplied, should be an array of integers indicating which of the provided spectra should be used for the calculation of the separated spectrum of component A. If 11 out of 21 spectra are used, this array should have length 11.
- **use_for_spectral_separation_B**. Same as previous, but for component B instead.
- **weights** (numpy array shape (n_spectra,)), Default = None. If provided, supplies a weight to each spectrum in the calculation of both component spectra. Can be used for a similar purpose as `use_for_spectral_separation_X` by providing weights 1 or 0 to the spectra.

Note: 4 of these parameters can affect which spectra are used in the separated component spectra, depending on which are provided. `rv_proximity_limit` is ranked the worst, and will be ignored if either `rv_lower_limit` or `use_for_spectral_separation_X` is provided. `rv_lower_limit` will also be ignored if `use_for_spectral_separation_X` is provided. Overall, it is recommended to manually select beforehand which spectra should be used, and which should not. The last parameter `weights` affects the spectra independently of the other. It is simply used to supply ANY constant weight to each of the spectra. This can serve a similar purpose as `use_for_spectral_separation_X`, or it can be used in conjunction with if desired. There might be a slight performance boost when using `use_for_spectral_separation_X` to skip calculation for some of the spectra.

