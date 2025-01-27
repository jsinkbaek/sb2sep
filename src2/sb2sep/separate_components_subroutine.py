import numpy as np
from .shift_wavelength_spectrum import shift_wavelength_spectrum as shift_spec


def separate_components_subroutine(
        wavelength_grid, flux_collection, config, rv_sets, weight_sets, fitflag_sets, subtractflag_sets,
        output_queue, plot_queue, routine_iteration
):
    """
    Performs spectral separation (spectral disentangling) using the method outlined in Gonzalez & Levato (2006).
    Assumes that first component is the dominant component in the spectrum. Attempts to separate the components using
    RV shifts and averaged spectra.

    :param wavelength_grid:     np.ndarray shape (datasize, ) wavelength of all the spectra
    :param flux_collection:     np.ndarray shape (datasize, nspectra) of all the observed spectra
    :param config:              dictionary with configuration values.
    :param rv_sets:             tuple of RV collections, each of size (nspectra, ), one collection for each star.
                                F.ex. rv_sets=(rvs_A, rvs_B, ...).
    :param weight_sets:         tuple of weight collections, each of size (nspectra, ), one collection for each star.
                                F.ex. weight_sets=(weights_A, weights_B, ...).
    :param fitflag_sets:        Same format. Boolean marks whether RVs are fit (for each component). If not, use
                                behavior described in configuration["RVwhenNotFitted"].
    :param subtractflag_sets:   Same format. Boolean marks if component spectrum should be subtracted when calculating
                                other component spectra.

    :return *separated_fluxes:   the separated and meaned component spectra of all the stellar components provided.
    """
    n_spectra = flux_collection[0, :].size
    n_components = len(rv_sets)
    wl = wavelength_grid
    fl = flux_collection
    maxiterations = config["SeparateComponents"]["MaxIterations"]

    sep_flux = [np.zeros(flux_collection.shape[0]) for i in range(n_components)]
    RMS_values = [np.zeros(flux_collection.shape[0]) for i in range(n_components)]

    iteration_counter = 0
    while True:
        for i in range(n_components):
            RMS_values[i] = -sep_flux[i]

        for i in range(n_components):
            sep_flux[i][:] = 0.0
            n_weight = 0.0
            for k in range(n_spectra):
                if weight_sets[i][k] == 0.0:        # Skip this spectrum
                    continue
                shifted_flux = shift_spec(wl,fl[:, k],-rv_sets[i][k],config["rvInterpolationMethod"],1.0)
                for j in range(n_components):
                    if j == i or subtractflag_sets[j][k] is False:  # Skip iteration
                        continue
                    if fitflag_sets[j][k] is False:
                        if config["RVwhenNotFitted"] == "Skip":     # Skip iteration
                            continue
                    fillval = np.min([np.quantile(sep_flux[j], 0.84), 1.0])
                    shifted_flux -= shift_spec(
                        wl, sep_flux[j], rv_sets[j][k]-rv_sets[i][k], config["rvInterpolationMethod"], fillval
                    )
                sep_flux[i] += weight_sets[i][k] * shifted_flux
                n_weight += weight_sets[i][k]
            sep_flux[i] = sep_flux[i] / n_weight

        convergence = []
        iteration_counter += 1
        for i in range(n_components):
            RMS_values[i] = np.sum((RMS_values[i]+sep_flux[i])**2)/RMS_values[i].size
            convergence.append(RMS_values[i] < config["SeparateComponents"]["Convergence"])
        if np.all(convergence) or iteration_counter >= maxiterations:
            break

    if config["verbose"]:
        print(f"Component separation completed in {iteration_counter} iterations out of {maxiterations}")
        print(f"Target RMS of separated spectra: {config["SeparateComponents"]["Convergence"]}")
        print(f"Final RMS of separated spectra (ordered): ")
        for i in range(n_components):
            print(RMS_values[i])
        print()

    for i in range(1, n_components):        # Add back continuum to all components after the first
        sep_flux[i] += 1.0

    output_queue.put({"subroutine": "sep", "sep_flux": sep_flux, "iteration": routine_iteration})
    plot_queue.put({"subroutine": "sep", "sep_flux": sep_flux, "iteration": routine_iteration})
    return sep_flux


