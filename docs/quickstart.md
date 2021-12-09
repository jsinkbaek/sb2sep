# Quickstart - Spectral Separation
The package is separated into multiple modules. The functions associated with the spectral separation is located in the module

```
spectral_separation_routine
```

Here, the main two functions that can be used for spectral separation are called `spectral_separation_routine` and `spectral_separation_routine_multiple_intervals`. The last function is a wrapper on the first, that can also split the input spectra into multiple different intervals.

`spectral_separation_routine` requires 8 input arguments:
- **inv_flux_collection** (numpy array shape (data.size, n_spectra)). This 2-dimensional array should include all observed normalized and inverted spectra to be analysed.
- **inv_flux_templateA** (numpy array shape (data.size, ). 1-dimensional array that includes normalized inverted spectrum of a stellar template to use for the first stellar component.
- **inv_flux_templateB** (numpy array shape (data.size, ). Same as before, but for the secondary component.
- **wavelength** (numpy array shape (data.size, ). 1-dimensional array of wavelength values for the spectra. The spectra must have been previously resampled to the same wavelength grid, equi-spaced in velocity. This can be done with functions in the [spectral_processing_functions](spectral_processing_functions) module.
- **options** (RoutineOptions object). This object includes all options and some extra parameters needed by the parent routine.
- **sep_comp_options** (SeparateComponentsOptions). This object includes options and extra parameters which should be passed down to the child routine 'separate_component_spectra', which handles the actual spectral separation when radial velocities are supplied.
- **rv_options** (RadialVelocityOptions). This object includes options and extra parameters needed by the other child routine 'recalculate_RVs', which handles radial velocity calculation given separated component spectra.
- **RV_guess_collection** (numpy array shape (n_spectra, 2)). Initial guesses for radial velocities of both components (component A is (:, 0)). These must be provided for both components in order to ensure convergence.

The wrapper function `spectral_separation_routine_multiple_intervals` is similar, but requires 9 input arguments instead, and has 1 optional argument. These are, in order:
- **wavelength**
- **wavelength_intervals** (list of tuples (float a, float b)). Each tuple in the list defines a single independent instance that the routine should be run, on the wavelength interval (a, b). Providing the argument \[ (4500, 5000)\] would thus tell the function to perform spectral separation once, with data only in the wavelength interval 4500-5000 Ångström.
- **inv_flux_collection**
- **inv_flux_templateA**
- **inv_flux_templateB**
- **options**
- **sep_comp_options**
- **rv_options**
- **RV_guess_collection**
- **buffer_size** (float), Default: None. This will be described in more detail in the [Routine options](routine_options) section. It defines the size in Ångström of a small buffer of data attached around the edges of each interval, that can be used to avoid the effects of data-wrapping. None results in no attached buffer.




[```Next page: Routine options```](routine_options)

