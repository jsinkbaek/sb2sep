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

For `RadialVelocityOptions`:
- **vsini_guess_A** (float), Default = 1.0. Guess for the fit parameter of vsini for component A, for the rotational broadening function profile fit. This will be modified by the parent routine after each iteration and set to the mean of all the spectra.
- **vsini_guess_B** (float), Default = 1.0.
- **vary_vsini_A** (bool), Default = True. If False, vsini will not be a free parameter during RV fit for component A.
- **vary_vsini_B** (bool), Default = True.
- **vsini_vary_limit_A** (float), Default = None. If provided, vsini_A can be changed only by `+- vsini_vary_limit_A*vsini_A` during next iteration. If not provided, parent routine will set it to 0.7 for stability.
- **vsini_vary_limit_B** (float), Default = None.
- **delta_v** (float), Default = 1.0. Must be same as for `SeparateComponentsOptions`.
- **spectral_resolution** (float), Default = 60000. Provides the spectral resolution such that the BF fit can be appropriately broadened.
- **velocity_fit_width_A** (float), Default = 50. How far away to include data during fitting of component A RV (in km/s). The rotational broadening function profile also separately masks the data using (velocity - rv)/vsini, which is worth keeping in mind.
- **velocity_fit_width_B** (float), Default = 20.
- **refit_width_A** (float), Default = None. If either this or `refit_width_B` is provided, will repeat fitting procedure afterwards with a smaller fitting width. This slows computation but can improve final precision on found RVs and might improve convergence.
- **refit_width_B** (float), Default = None.
- **limbd_coef_A** (float), Default = 0.68. Linear limb darkening parameter for fitting profile. Not of great importance for RV calculation.
- **limbd_coef_B** (float), Default = 0.68.
- **vary_limbd_coef_A** (bool), Default = False. If True, linear limb darkening coefficient will be a free parameter for component A fit.
- **vary_limbd_coef_B** (bool), Default = False.
- **RV_A** (numpy array shape (n_spectra, ), Default = None. Fitting routine will mask data using this and vsini with (velocity - rv)/vsini. Parent routine will set this automatically to provided RV guesses (and found RVs on successive iterations), so there is no reason to supply this manually.
- **RV_B** (numpy array shape (n_spectra, ), Default = None.
- **smooth_sigma_A** (float), Default = 4.0. Smoothing value in km/s to use when performing gaussian smoothing on the component A broadening function before fitting.
- **smooth_sigma_B** (float), Default = 4.0
- **bf_velocity_span** (float), Default = 200. The width of the broadening function in velocity space.
- **period** (float), Default = None. Orbital period of the system. Must be supplied if eclipse masking is to be done, e.g. if one of the components should not be subtracted from the spectrum at specific phases.
- **time_values** (numpy array shape (n_spectra, )), Default = None. Timestamps of all the spectra. Must be supplied if eclipse masking is to be done, e.g. if one of the components should not be subtracted from the spectrum at specific phases.
- **ignore_at_phase_A** (tuple(float, float)), Default = None. If supplied, separated component A spectrum will not be subtracted from spectra used for component B RV if they are within specific phase interval (lower, upper). This interval can wrap around, e.g. (0.98, 0.02) is a valid interval. Make sure that the provided `time_values` has the same phase-shift when calculating phase with np.mod(time_values, period)/period.
- **ignore_at_phase_B** (tuple(float, float)), Default = None. Same as previous, but opposite.
- **verbose** (bool), Default = False. If True, extra prints are done.
- **iteration_limit** (int), Default = 6. Maximum allowed iterations per spectrum for RV caculation subroutine. Amount is effectively doubled if `refit_width_X` is provided.
- **convergence_limit** (float), Default = 5E-3. Convergence criterium for RV RMS.
- **rv_lower_limit** (float), Default = 0.0. Used for plotting the limits in broadening function plots if `RoutineOptions.plot = True`.

[`Next Page: Accessing outputs`](results)
