# The Spectral Separation Routine
The package is separated into multiple modules. The functions associated with the spectral separation are located in the module

```
spectral_separation_routine
```

Here, the main two functions that can be used for spectral separation are called `spectral_separation_routine` and `spectral_separation_routine_multiple_intervals`. The last function is a wrapper on the first, that can also split the input spectra into multiple different intervals.

### spectral_separation_routine()
`spectral_separation_routine` requires 8 input arguments:
- **flux_collection** (numpy array shape (data.size, n_spectra) or (data.size, n_orders, n_spectra)). This 2- or 3-dimensional array should include all observed normalized spectra to be analysed.
- **flux_templateA** (numpy array shape (data.size, ). 1-dimensional array that includes normalized spectrum of a stellar template to use for the first stellar component.
- **flux_templateB** (numpy array shape (data.size, ). Same as before, but for the secondary component.
- **wavelength** (numpy array shape (data.size, ). 1-dimensional array of wavelength values for the spectra. All spectra must have been previously resampled to the same wavelength grid, equi-spaced in velocity. This can be done with functions in the [spectral_processing_functions](spectral_processing_functions) module.
- **options** (RoutineOptions object). This object includes all options and some extra parameters needed by the parent routine.
- **sep_comp_options** (SeparateComponentsOptions). This object includes options and extra parameters which should be passed down to the child routine 'separate_component_spectra', which handles the actual spectral separation when radial velocities are supplied.
- **rv_options** (RadialVelocityOptions). This object includes options and extra parameters needed by the other child routine 'recalculate_RVs', which handles radial velocity calculation given separated component spectra.
- **RV_guess_collection** (numpy array shape (n_spectra, 2) or (n_orders, n_spectra, 2)). Initial guesses for radial velocities of both components (component A is (:, 0) if ndim=2 or (:, :, 0) if ndim=3). These must be provided for both components in order to ensure convergence.

Most optional arguments are encapsulated in the *Options objects. The only three direct optional arguments that can be supplied
are
- **mask_collection_orders** (numpy array shape (data.size, n_orders, n_spectra)). This must be a boolean array that indicates the wavelength indices of each spectral order if flux_collection.ndim == 3 (see note below).
- **period** (float), Default = None. Orbital period of the system. Must be supplied along with **time_values** if eclipse masking is to be done during RV calculation, e.g. if one of the components should not be subtracted from the spectrum at specific phases.
- **time_values** (numpy array shape (n_spectra, )), Default = None. Timestamps of all the spectra. np.mod(time_values, period)/period should range from 0 to 1. Must be supplied if eclipse masking is to be done, e.g. if one of the components should not be subtracted from the spectrum at specific phases.


#### Note on ndims and spectral orders
It is possible to supply multiple spectral orders to the function and gain merged separated spectra as output in a convenient way.
To do this, the spectral orders must first be interpolated to the same wavelength grid, filling any elements outside the
range of each with a value of 1.0. Therefore **flux_collection** must have the shape (data.size, n_orders, n_spectra). A
mask array **mask_collection_orders** must then be supplied (same shape). Applying it to each spectral order, for example order 0 spectrum 0 with
`` flux_collection[mask_collection_orders[:, 0, 0], 0, 0]``,
 should then return only the flux values within the given order. The sub-routines using spectral orders are a bit outdated relative
to the traditional use. Generally, I do not expect many cases to have high enough S/N for both components to be able to
utilize this feature properly and recommend that most uses supply merged spectra to the routine and use the 
`spectral_separation_routine_multiple_intervals()` wrapper.

RV_guess_collection can have either shape (n_orders, n_spectra, 2) or (n_spectra, 2) in this case. If the latter, each order is
assumed to have the same initial RV guess, which is generally a decent assumption.


### spectral_separation_routine_multiple_intervals()
The wrapper function `spectral_separation_routine_multiple_intervals` is similar, but requires 9 input arguments instead, and has 3 optional arguments. These are, in order:
- **wavelength** (shape (data.size, )). The full wavelength interval supplied.
- **wavelength_intervals** (list of tuples (float a, float b)). Each tuple in the list defines a single independent instance that the routine should be run, on the wavelength interval (a, b). Providing the argument \[ (4500, 5000)\] would thus tell the function to perform spectral separation once, with data only in the wavelength interval 4500-5000 Ångström.
- **flux_collection** (shape (data.size, n_spectra)).
- **flux_templateA**
- **flux_templateB**
- **options**
- **sep_comp_options**
- **rv_options**
- **RV_guess_collection**
- **buffer_size** (float), Default: None. This will be described in more detail in the [Routine options](routine_options) section with the `buffer_mask` parameter. It defines the size in Ångström of a small buffer of data attached around the edges of each interval, that can be used to avoid effects of data-wrapping on both the separated spectra and the broadening function calculation. None results in no attached buffer.
- **period** (float), Default: None.
- **time_values** (np.ndarray shape (n_spectra, )). Default: None.




[Next page: Routine options](routine_options)

