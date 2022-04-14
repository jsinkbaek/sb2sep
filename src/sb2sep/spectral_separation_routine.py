"""
First edition May 02/2021.
Package version: December 08/2021.
@author Jeppe Sinkbæk Thomsen, Master's thesis studen at Aarhus University.
Supervisor: Karsten Frank Brogaard.

This is a collection of functions that form a routine to perform spectral separation of detached eclipsing binaries
with a giant component and a main sequence component. The routine is adapted from multiple scripts, an IDL script by
Karsten Frank Brogaard named "dis_real_merged_8430105_2021.pro", and a similar python script "spec_8430105_bf.py" by the
same author. Both follows the formula layout of the article:
'Separation of composite spectra: the spectroscopic detecton of an eclipsing binary star'
        by J.F. Gonzalez and H. Levato ( A&A 448, 283-292(2006) )
"""
import matplotlib.pyplot as plt

from sb2sep.calculate_radial_velocities import radial_velocity_single_component
from sb2sep.broadening_function_svd import *
from sb2sep.storage_classes import InitialFitParameters, SeparateComponentsOptions, RadialVelocityOptions, \
    RoutineOptions
from sb2sep.rotational_broadening_function_fitting import get_fit_parameter_values
import sb2sep.spectrum_processing_functions as spf
import numpy as np
from copy import deepcopy
from joblib import Parallel, delayed
import matplotlib
import sb2sep.broadening_function_svd as bfsvd
# from scipy.interpolate import interp1d
import scipy.constants as scc
from matplotlib.backends.backend_pdf import PdfPages
from copy import copy


def shift_spectrum(flux, radial_velocity_shift, delta_v):
    """
    Performs a wavelength shift of a spectrum. Two-step process:
     - First, it performs a low resolution roll of the flux elements. Resolution here is defined by delta_v (which must
       be the velocity spacing of the flux array). Example precision: 1 km/s (with delta_v=1.0)
     - Second, it performs linear interpolation to produce a small shift to correct for the low precision.
     The spectrum must be evenly spaced in velocity space in order for this function to properly work.
    :param flux:                    np.ndarray size (:, ), flux values of the spectrum.
    :param radial_velocity_shift:   float, the shift in velocity units (km/s)
    :param delta_v:                 float, the resolution of the spectrum in km/s.
    :return flux_shifted:           np.ndarray size (:, ), shifted flux values of the spectrum.

    shift = radial_velocity_shift / delta_v
    indices = np.linspace(0, flux.size, flux.size)
    indices_shifted = indices - shift
    flux_shifted = np.interp(indices_shifted, indices, flux)
    """
    # Perform large delta_v precision shift
    large_shift = np.floor(radial_velocity_shift / delta_v)
    flux_ = np.roll(flux, int(large_shift))

    # Perform small < delta_v precision shift using linear interpolation
    small_shift = radial_velocity_shift / delta_v - large_shift
    indices = np.linspace(0, flux.size, flux.size)
    indices_shifted = indices - small_shift  # minus will give same shift direction as np.roll()
    # flux_shifted = interp1d(indices, flux_, fill_value='extrapolate')(indices_shifted)
    flux_shifted = np.interp(indices_shifted, indices, flux_)

    return flux_shifted


def shift_wavelength_spectrum(wavelength, flux, radial_velocity_shift):
    wavelength_shifted = wavelength * (1 + radial_velocity_shift/(scc.speed_of_light / 1000))
    # flux_shifted = interp1d(wavelength_shifted, flux, fill_value='extrapolate')(wavelength)
    flux_shifted = np.interp(wavelength, wavelength_shifted, flux)
    return flux_shifted


def separate_component_spectra(
        flux_collection, radial_velocity_collection_A, radial_velocity_collection_B,
        options: SeparateComponentsOptions
):
    """
    Assumes that component A is the dominant component in the spectrum. Attempts to separate the two components using
    RV shifts and averaged spectra.

    :param flux_collection:               np.ndarray shape (datasize, nspectra) of all the observed spectra
    :param radial_velocity_collection_A:  np.ndarray shape (nspectra, ) of radial velocity values for component A
    :param radial_velocity_collection_B:  np.ndarray shape (nspectra, ) of radial velocity values for component B
    :param options:                       options for the subroutine, including convergence tolerances, spectrum weights
                                            and more

    :return separated_flux_A, separated_flux_B:   the separated and meaned total component spectra of A and B.
    """
    n_spectra = flux_collection[0, :].size
    separated_flux_B = np.zeros((flux_collection[:, 0].size,))  # Set to 0 before iteration
    separated_flux_A = np.zeros((flux_collection[:, 0].size,))

    use_spectra_A = options.use_for_spectral_separation_A
    use_spectra_B = options.use_for_spectral_separation_B

    delta_v = options.delta_v
    weights = options.weights
    if weights is None:
        weights = np.ones(n_spectra)

    iteration_counter = 0
    while True:
        RMS_values_A = -separated_flux_A
        RMS_values_B = -separated_flux_B
        iteration_counter += 1
        separated_flux_A[:] = 0.0
        n_used_spectra = 0
        for i in range(0, n_spectra):
            if (use_spectra_A is None) or (i in use_spectra_A):
                rvA = radial_velocity_collection_A[i]
                rvB = radial_velocity_collection_B[i]
                if options.rv_lower_limit == 0.0 and use_spectra_A is None:
                    condition = np.abs(rvA-rvB) > options.rv_proximity_limit
                elif use_spectra_A is None:
                    condition = np.abs(rvA) > options.rv_lower_limit
                else:
                    condition = True

                if condition:
                    shifted_flux_A = shift_spectrum(flux_collection[:, i], -rvA, delta_v)
                    if options.ignore_component_B is False:
                        separated_flux_A += weights[i] * \
                                            (shifted_flux_A - shift_spectrum(separated_flux_B, rvB - rvA, delta_v))
                    else:
                        separated_flux_A += weights[i] * shifted_flux_A
                    n_used_spectra += weights[i]
            elif use_spectra_A.size != 0:
                pass
            else:
                raise TypeError(f'use_spectra_A is either of wrong type ({type(use_spectra_A)}), empty, or wrong value.\n' +
                                f'Expected type: {type(True)} or np.ndarray. Expected value if bool: True')
        separated_flux_A = separated_flux_A / n_used_spectra

        separated_flux_B[:] = 0.0
        n_used_spectra = 0
        for i in range(0, n_spectra):
            if (use_spectra_B is None) or (i in use_spectra_B):
                rvA = radial_velocity_collection_A[i]
                rvB = radial_velocity_collection_B[i]
                if options.rv_lower_limit == 0.0 and use_spectra_B is None:
                    condition = np.abs(rvA-rvB) > options.rv_proximity_limit
                elif use_spectra_B is None:
                    condition = np.abs(rvA) > options.rv_lower_limit
                else:
                    condition = True

                if condition:
                    shifted_flux_B = shift_spectrum(flux_collection[:, i], -rvB, delta_v)
                    separated_flux_B += weights[i] * \
                                        (shifted_flux_B - shift_spectrum(separated_flux_A, rvA - rvB, delta_v))
                    n_used_spectra += weights[i]
            elif use_spectra_B.size != 0:
                pass
            else:
                raise TypeError(f'use_spectra_B is either of wrong type ({type(use_spectra_B)}), empty, or wrong value.\n' +
                                f'Expected type: {type(True)} or np.ndarray. Expected value if bool: True')
        separated_flux_B = separated_flux_B / n_used_spectra

        RMS_values_A += separated_flux_A
        RMS_values_B += separated_flux_B
        RMS_A = np.sum(RMS_values_A**2)/RMS_values_A.size
        RMS_B = np.sum(RMS_values_B**2)/RMS_values_B.size
        if RMS_A < options.convergence_limit and RMS_B < options.convergence_limit:
            if options.verbose is True:
                print(f'Separate Component Spectra: Convergence limit of {options.convergence_limit} successfully '
                      f'reached in {iteration_counter} iterations. \nReturning last separated spectra.')
            break
        elif iteration_counter >= options.max_iterations:
            warnings.warn(f'Warning: Iteration limit of {options.max_iterations} reached without reaching convergence '
                          f'limit of {options.convergence_limit}. \nCurrent RMS_A: {RMS_A}. RMS_B: {RMS_B} \n'
                          'Returning last separated spectra.')
            break
    if options.verbose is True:
        print('n_spectra vs n_used_spectra: ', n_spectra, ' ', n_used_spectra)
    separated_flux_B += 1.0     # add back continuum to second component
    return separated_flux_A, separated_flux_B


def separate_component_spectra_orders(
        flux_order_collection, radial_velocity_collection_A, radial_velocity_collection_B,
        options: SeparateComponentsOptions, overlap_weights
):
    """
    Assumes flux_order_collection.shape = [wavelength.size, n_orders, n_spectra], where wavelength.size is the amount of
    elements in the equi-velocity-spaced interpolated wavelength grid for a merged spectrum.
    Each individual order should have the same amount of elements, but values outside the range must be = 1
    mask_collection_orders.shape = [wavelength.size, n_orders, n_spectra]. dtype=bool. Each mask must remove values
    outside of order range
    """

    n_spectra = flux_order_collection[0, 0, :].size
    n_orders = flux_order_collection[0, :, 0].size
    separated_flux_A = np.zeros((flux_order_collection[:, 0, 0].size, ))
    separated_flux_B = np.zeros((flux_order_collection[:, 0, 0].size, ))
    use_spectra_A = options.use_for_spectral_separation_A
    use_spectra_B = options.use_for_spectral_separation_B
    delta_v = options.delta_v
    weights = options.weights
    if weights is None:
        weights = np.ones(n_spectra)

    iteration_counter = 0
    while True:
        if iteration_counter == 0:
            RMS_values_A = np.zeros(shape=separated_flux_A.shape); RMS_values_B = np.zeros(shape=separated_flux_B.shape)
        else:
            RMS_values_A = -separated_flux_A
            RMS_values_B = -separated_flux_B
        iteration_counter += 1
        separated_flux_A[:] = 0.0
        n_used_spectra = 0
        for i in range(0, n_spectra):
            if (use_spectra_A is None) or (i in use_spectra_A):
                if radial_velocity_collection_A.ndim == 2:
                    rvA = np.mean(radial_velocity_collection_A[:, i])
                    rvB = np.mean(radial_velocity_collection_B[:, i])
                else:
                    rvA = radial_velocity_collection_A[i]
                    rvB = radial_velocity_collection_B[i]
                if options.rv_lower_limit == 0.0 and use_spectra_A is None:
                    condition = np.abs(rvA - rvB) > options.rv_proximity_limit
                elif use_spectra_A is None:
                    condition = np.abs(rvA) > options.rv_lower_limit
                else:
                    condition = True
                if condition:
                    overlap_mean = np.zeros((separated_flux_A.size, ))
                    for j in range(0, n_orders):
                        if radial_velocity_collection_A.ndim == 2:
                            rvA = radial_velocity_collection_A[j, i]
                            rvB = radial_velocity_collection_B[j, i]
                        shifted_flux_A = shift_spectrum((flux_order_collection[:, j, i])*overlap_weights[:, j, i],
                                                        -rvA, delta_v)
                        if options.ignore_component_B is False:
                            shift_sep_B = shift_spectrum(overlap_weights[:, j, i]*separated_flux_B, rvB-rvA, delta_v)
                            separated_flux_A += (weights[i]*(shifted_flux_A - shift_sep_B))
                        else:
                            separated_flux_A += weights[i] * shifted_flux_A
                        overlap_mean += shift_spectrum(overlap_weights[:, j, i], -rvA, delta_v)
                    n_used_spectra += weights[i] * overlap_mean
            elif use_spectra_A.size != 0:
                pass
            else:
                raise TypeError(
                    f'use_spectra_A is either of wrong type ({type(use_spectra_A)}), empty, or wrong value.\n' +
                    f'Expected type: {type(True)} or np.ndarray. Expected value if bool: True'
                )
        separated_flux_A = separated_flux_A / n_used_spectra

        separated_flux_B[:] = 0.0
        n_used_spectra = 0
        for i in range(0, n_spectra):
            if (use_spectra_B is None) or (i in use_spectra_B):
                if radial_velocity_collection_A.ndim == 2:
                    rvA = np.mean(radial_velocity_collection_A[:, i])
                    rvB = np.mean(radial_velocity_collection_B[:, i])
                else:
                    rvA = radial_velocity_collection_A[i]
                    rvB = radial_velocity_collection_B[i]
                if options.rv_lower_limit == 0.0 and use_spectra_B is None:
                    condition = np.abs(rvA-rvB) > options.rv_proximity_limit
                elif use_spectra_B is None:
                    condition = np.abs(rvA) > options.rv_proximity_limit
                else:
                    condition = True

                if condition:
                    overlap_mean = np.zeros((separated_flux_B.size,))
                    for j in range(0, n_orders):
                        if radial_velocity_collection_A.ndim == 2:
                            rvA = radial_velocity_collection_A[j, i]
                            rvB = radial_velocity_collection_B[j, i]
                        shifted_flux_B = shift_spectrum((flux_order_collection[:, j, i])*overlap_weights[:, j, i],
                                                        -rvB, delta_v)
                        shift_sep_A = shift_spectrum(overlap_weights[:, j, i]*separated_flux_A, rvB-rvA, delta_v)
                        separated_flux_B += weights[i] * (shifted_flux_B - shift_sep_A)
                        overlap_mean += shift_spectrum(overlap_weights[:, j, i], -rvB, delta_v)
                    n_used_spectra += weights[i] * overlap_mean
            elif use_spectra_B.size != 0:
                pass
            else:
                raise TypeError(
                    f'use_spectra_B is either of wrong type ({type(use_spectra_B)}), empty, or wrong value.\n' +
                    f'Expected type: {type(True)} or np.ndarray. Expected value if bool: True')

        separated_flux_B = separated_flux_B/n_used_spectra

        RMS_values_A += separated_flux_A
        RMS_values_B += separated_flux_B
        RMS_values_A = RMS_values_A[~np.isnan(RMS_values_A)]
        RMS_values_B = RMS_values_B[~np.isnan(RMS_values_B)]
        RMS_A = np.sum(RMS_values_A ** 2) / RMS_values_A.size
        RMS_B = np.sum(RMS_values_B ** 2) / RMS_values_B.size
        if RMS_A < options.convergence_limit and RMS_B < options.convergence_limit:
            if options.verbose is True:
                print(f'Separate Component Spectra: Convergence limit of {options.convergence_limit} successfully '
                      f'reached in {iteration_counter} iterations. \nReturning last separated spectra.')
            break
        elif iteration_counter >= options.max_iterations:
            warnings.warn(
                f'Warning: Iteration limit of {options.max_iterations} reached without reaching convergence '
                f'limit of {options.convergence_limit}. \nCurrent RMS_A: {RMS_A}. RMS_B: {RMS_B} \n'
                'Returning last separated spectra.')
            break
    if options.verbose is True:
        print('n_spectra vs n_used_spectra: ', n_spectra, ' ', n_used_spectra)
    separated_flux_B += 1.0     # add back a continuum to second component
    separated_flux_A[np.isnan(separated_flux_A)] = 1.0
    separated_flux_B[np.isnan(separated_flux_B)] = 1.0
    return separated_flux_A, separated_flux_B


def _calculate_overlap(mask_collection_orders):
    overlap_weights = np.zeros(shape=mask_collection_orders.shape, dtype=float)
    n_orders = mask_collection_orders[0, :, 0].size
    n_spectra = mask_collection_orders[0, 0, :].size
    for i in range(0, n_spectra):
        for j in range(0, n_orders):
            other_masks = np.delete(mask_collection_orders[:, :, i], j, axis=1)
            mask = mask_collection_orders[:, j, i]
            overlap_mask = np.zeros(mask.size, dtype=bool)
            for k in range(0, n_orders-1):
                other_mask_temp = other_masks[:, k]
                comb_mask = mask & other_mask_temp
                overlap_mask = overlap_mask + comb_mask
            overlap_weights[(mask & ~overlap_mask), j, i] = 1.0
            mask_idx = np.argwhere(mask)
            mask_lowhalf = np.copy(mask)
            mask_lowhalf[mask_idx[mask_idx.size//2:]] = False       # Mask identifies only first half of data as True
            mask_highhalf = np.copy(mask)
            mask_highhalf[mask_idx[0:mask_idx.size//2]] = False
            overlap_low = overlap_mask & mask_lowhalf
            overlap_high = overlap_mask & mask_highhalf
            overlap_weights[overlap_low, j, i] = np.linspace(0, 1, overlap_low[overlap_low].size)
            overlap_weights[overlap_high, j, i] = np.linspace(1, 0, overlap_high[overlap_high].size)
    return overlap_weights


def _update_bf_plot(plot_ax, model, RV_actual, index):
    fit = model[0]
    model_values = model[1]
    velocity_values = model[2]
    bf_smooth_values = model[4]
    _, RV_measured, _, _, _, _ = get_fit_parameter_values(fit.params)
    RV_offset = RV_actual - RV_measured
    plot_ax.plot(velocity_values + RV_offset, 1+0.02*bf_smooth_values/np.max(bf_smooth_values)-0.05*index)
    plot_ax.plot(velocity_values + RV_offset, 1+0.02*model_values/np.max(bf_smooth_values)-0.05*index, 'k--')
    plot_ax.plot(np.ones(shape=(2,))*RV_actual,
                 [1-0.05*index-0.005, 1+0.025*np.max(model_values)/np.max(bf_smooth_values)-0.05*index],
                 color='grey')


def recalculate_RVs(
        flux_collection: np.ndarray, separated_flux_A: np.ndarray, separated_flux_B: np.ndarray,
        RV_collection_A: np.ndarray, RV_collection_B: np.ndarray, flux_templateA: np.ndarray,
        flux_templateB: np.ndarray, buffer_mask: np.ndarray, options: RadialVelocityOptions,
        plot_ax_A=None, plot_ax_B=None, time_values=None, period=None
):
    """
    This part of the spectral separation routine corrects the spectra for the separated spectra found by
    separate_component_spectra and recalculates RV values for each component using the corrected spectra (with one
    component removed).

    :param flux_collection: np.ndarray shape (:, n_spectra). Collection of normalized fluxes for the program spectra.
    :param separated_flux_A:    np.ndarray shape (:, ). Meaned normalized flux from separate_component_spectra() for A.
    :param separated_flux_B:    np.ndarray shape (:, ). Meaned normalized flux from separate_component_spectra() for B.
    :param RV_collection_A:     np.ndarray shape (n_spectra, ). Current RV values used to remove A from spectrum with.
    :param RV_collection_B:     np.ndarray shape (n_spectra, ). Current RV values used to remove B from spectrum with.
    :param flux_templateA:  np.ndarray shape (:, ). Template spectrum normalized flux for component A.
    :param flux_templateB:  np.ndarray shape (:, ). Template spectrum normalized flux for component B.
    :param buffer_mask:         np.ndarray shape (:, ). Mask used to remove "buffer" (or "padding") from spectrum.
                                    See spectral_separation_routine() for more info.
    :param options:             fitting and broadening function options
    :param plot_ax_A:           matplotlib.axes.axes. Used to update RV plots during iterations.
    :param plot_ax_B:           matplotlib.axes.axes. Used to update RV plots during iterations.

    :return:    RV_collection_A,        RV_collection_B, (fits_A, fits_B)
                RV_collection_A:        updated values for the RV of component A.
                RV_collection_B:        updated values for the RV of component B.
                fits_A, fits_B:         np.ndarrays storing the rotational BF profile fits found for each spectrum.
    """
    RV_collection_A = deepcopy(RV_collection_A)
    RV_collection_B = deepcopy(RV_collection_B)
    n_spectra = flux_collection[0, :].size
    v_span = options.bf_velocity_span
    delta_v = options.delta_v

    if plot_ax_A is not None:
        plot_ax_A.clear()
        plot_ax_A.set_xlim([-v_span/2, +v_span/2])
        plot_ax_A.set_xlabel('Velocity shift [km/s]')
    if plot_ax_B is not None:
        plot_ax_B.clear()
        plot_ax_B.set_xlim([-v_span/2, +v_span/2])
        plot_ax_B.set_xlabel('Velocity shift [km/s]')

    bf_fitres_A = np.empty(shape=(n_spectra,), dtype=tuple)
    bf_fitres_B = np.empty(shape=(n_spectra,), dtype=tuple)

    krange = 1

    for i in range(0, n_spectra):
        if options.refit_width_A is not None or options.refit_width_B is not None:
            krange = 2
        else:
            pass
        for k in range(0, krange):       # perform "burn-in" with wide fit-width, and then refit only with peak data
            if k == 0:
                fit_width_A = copy(options.velocity_fit_width_A)
                fit_width_B = copy(options.velocity_fit_width_B)
            if k == 1:
                options.velocity_fit_width_A = options.refit_width_A
                options.velocity_fit_width_B = options.refit_width_B

            iterations = 0
            while True:
                iterations += 1

                # # Calculate RV_A # #
                corrected_flux_A = (1-flux_collection[:, i]) - \
                                   shift_spectrum(1-separated_flux_B, RV_collection_B[i], delta_v)

                if period is not None and options.ignore_at_phase_B is not None and time_values is not None:
                    if _check_for_total_eclipse(time_values[i], period, options.ignore_at_phase_B) is True:
                        corrected_flux_A = 1-flux_collection[:, i]

                corrected_flux_A = corrected_flux_A[~buffer_mask]

                options.RV_A = RV_collection_A[i]
                # Generate fit parameter object
                ifitparams_A = InitialFitParameters(
                    options.vsini_A, options.spectral_resolution, options.velocity_fit_width_A, options.limbd_coef_A,
                    options.bf_smooth_sigma_A, options.bf_velocity_span, options.vary_vsini_A,
                    options.vsini_vary_limit_A, options.vary_limbd_coef_A, RV=0.0
                )

                # Perform calculation
                template_shifted = shift_spectrum(flux_templateA, RV_collection_A[i], delta_v)
                BRsvd_template_A = BroadeningFunction(
                    1 - flux_collection[~buffer_mask, 0], 1 - template_shifted[~buffer_mask], v_span, delta_v
                )
                BRsvd_template_A.smooth_sigma = options.bf_smooth_sigma_A
                RV_deviation_A, model_A = radial_velocity_single_component(
                    corrected_flux_A, BRsvd_template_A, ifitparams_A
                )
                RV_collection_A[i] = RV_collection_A[i] + RV_deviation_A

                # # Calculate RV_B # #
                corrected_flux_B = (1-flux_collection[:, i]) - \
                                   shift_spectrum(1-separated_flux_A, RV_collection_A[i], delta_v)

                if period is not None and options.ignore_at_phase_A is not None and time_values is not None:
                    if _check_for_total_eclipse(time_values[i], period, options.ignore_at_phase_A) is True:
                        corrected_flux_B = 1-flux_collection[:, i]

                corrected_flux_B = corrected_flux_B[~buffer_mask]

                options.RV_B = RV_collection_B[i]
                ifitparams_B = InitialFitParameters(
                    options.vsini_B, options.spectral_resolution, options.velocity_fit_width_B, options.limbd_coef_B,
                    options.bf_smooth_sigma_B, options.bf_velocity_span, options.vary_vsini_B,
                    options.vsini_vary_limit_B, options.vary_limbd_coef_B, RV=0.0
                )
                template_shifted = shift_spectrum(flux_templateB, RV_collection_B[i], delta_v)
                BRsvd_template_B = BroadeningFunction(
                    1 - flux_collection[~buffer_mask, 0], 1 - template_shifted[~buffer_mask], v_span, delta_v
                )
                BRsvd_template_B.smooth_sigma = options.bf_smooth_sigma_B
                RV_deviation_B, model_B = radial_velocity_single_component(
                    corrected_flux_B, BRsvd_template_B, ifitparams_B
                )
                RV_collection_B[i] = RV_collection_B[i] + RV_deviation_B

                if np.abs(RV_deviation_A) < options.convergence_limit and \
                        np.abs(RV_deviation_B) < options.convergence_limit:
                    if options.verbose:
                        print(f'RV: spectrum {i} successful.')
                    break
                elif iterations > options.iteration_limit:
                    if k == 1 and options.verbose is True:
                        warnings.warn(
                            f'RV: spectrum {i} did not reach convergence limit {options.convergence_limit}.'
                        )
                    break

        bf_fitres_A[i], bf_fitres_B[i] = model_A, model_B

        rv_lower_limit = options.rv_lower_limit
        if plot_ax_A is not None:
            _update_bf_plot(plot_ax_A, model_A, RV_collection_A[i], i)
            if rv_lower_limit != 0.0:
                plot_ax_A.plot([rv_lower_limit, rv_lower_limit], [0, 1.1], 'k', linewidth=0.3)
                plot_ax_A.plot([-rv_lower_limit, -rv_lower_limit], [0, 1.1], 'k', linewidth=0.3)
        if plot_ax_B is not None:
            _update_bf_plot(plot_ax_B, model_B, RV_collection_B[i], i)
            if rv_lower_limit != 0.0:
                plot_ax_B.plot([rv_lower_limit, rv_lower_limit], [0, 1.1], 'k', linewidth=0.3)
                plot_ax_B.plot([-rv_lower_limit, -rv_lower_limit], [0, 1.1], 'k', linewidth=0.3)

    if krange == 2:
        options.velocity_fit_width_A = fit_width_A
        options.velocity_fit_width_B = fit_width_B

    return RV_collection_A, RV_collection_B, (bf_fitres_A, bf_fitres_B)


def recalculate_RVs_orders(
        flux_collection_orders, mask_collection_orders, separated_flux_A, separated_flux_B, RV_collection_orders_A,
        RV_collection_orders_B, flux_templateA, flux_templateB, options: RadialVelocityOptions, plot_ax_A=None,
        plot_ax_B=None, time_values=None, period=None, plot_order=0
):
    """
    Assumes flux_order_collection.shape = [wavelength.size, n_orders, n_spectra], where wavelength.size is the amount of
    elements in the equi-velocity-spaced interpolated wavelength grid for a merged spectrum.
    Each individual order should have the same amount of elements, but values outside the range must be = 1
    mask_collection_orders.shape = [wavelength.size, n_orders, n_spectra]. dtype=bool. Each mask must remove values
    outside of order range
    RV_collection_orders_X.shape = (n_orders, n_spectra)
    """
    RV_collection_orders_A = deepcopy(RV_collection_orders_A)
    RV_collection_orders_B = deepcopy(RV_collection_orders_B)
    n_spectra = flux_collection_orders[0, 0, :].size
    n_orders = flux_collection_orders[0, :, 0].size
    v_span = options.bf_velocity_span
    delta_v = options.delta_v
    krange = 1

    bf_fitres_A = np.empty(shape=(n_orders, n_spectra), dtype=tuple)
    bf_fitres_B = np.empty(shape=(n_orders, n_spectra), dtype=tuple)

    if options.n_parallel_jobs > 1:
        arguments = (
            RV_collection_orders_A, RV_collection_orders_B, flux_collection_orders, separated_flux_A,
            separated_flux_B, flux_templateA, flux_templateB, mask_collection_orders, delta_v, v_span, period,
            time_values, n_orders, krange, bf_fitres_A, bf_fitres_B, options
        )
        res_par = Parallel(n_jobs=options.n_parallel_jobs)(
            delayed(_rv_loop_orders)(*arguments, i) for i in range(0, n_spectra)
        )
        for i in range(0, n_spectra):
            result_temp = res_par[i]
            RV_collection_orders_A[:, i] = result_temp[0]
            RV_collection_orders_B[:, i] = result_temp[1]
            bf_fitres_A[:, i] = result_temp[2][0]
            bf_fitres_B[:, i] = result_temp[2][1]

            rv_lower_limit = options.rv_lower_limit
            if plot_ax_A is not None:
                _update_bf_plot(plot_ax_A, bf_fitres_A[0, i], RV_collection_orders_A[0, i], i)
                if rv_lower_limit != 0.0:
                    plot_ax_A.plot([rv_lower_limit, rv_lower_limit], [0, 1.1], 'k', linewidth=0.3)
                    plot_ax_A.plot([-rv_lower_limit, -rv_lower_limit], [0, 1.1], 'k', linewidth=0.3)
            if plot_ax_B is not None:
                _update_bf_plot(plot_ax_B, bf_fitres_B[0, i], RV_collection_orders_B[0, i], i)
                if rv_lower_limit != 0.0:
                    plot_ax_B.plot([rv_lower_limit, rv_lower_limit], [0, 1.1], 'k', linewidth=0.3)
                    plot_ax_B.plot([-rv_lower_limit, -rv_lower_limit], [0, 1.1], 'k', linewidth=0.3)

    else:
        for i in range(0, n_spectra):
            if options.verbose:
                print(f'RV: spectrum {i} begun.')
            if options.refit_width_A is not None or options.refit_width_B is not None:
                krange = 2
            else:
                pass
            for k in range(0, krange):
                if k == 0:
                    fit_width_A = copy(options.velocity_fit_width_A)
                    fit_width_B = copy(options.velocity_fit_width_B)
                if k == 1:
                    options.velocity_fit_width_A = options.refit_width_A
                    options.velocity_fit_width_B = options.refit_width_B
                iterations = 0
                RV_deviation_A = np.ones((RV_collection_orders_A[:, 0].size,))
                RV_deviation_B = np.ones((RV_collection_orders_B[:, 0].size,))
                while True:
                    iterations += 1
                    for j in range(0, n_orders):
                        # # Calculate RV_A # #
                        corrected_flux_A = (1 - flux_collection_orders[:, j, i]) - \
                                           (1 - shift_spectrum(separated_flux_B, RV_collection_orders_B[j, i], delta_v))
                        if period is not None and options.ignore_at_phase_B is not None and time_values is not None:
                            if _check_for_total_eclipse(time_values[i], period, options.ignore_at_phase_B) is True:
                                corrected_flux_A = 1 - flux_collection_orders[:, j, i]
                        corrected_flux_A = corrected_flux_A[mask_collection_orders[:, j, i]]
                        options.RV_A = RV_collection_orders_A[j, i]
                        ifitparams_A = InitialFitParameters(
                            options.vsini_A, options.spectral_resolution, options.velocity_fit_width_A,
                            options.limbd_coef_A,
                            options.bf_smooth_sigma_A, options.bf_velocity_span, options.vary_vsini_A,
                            options.vsini_vary_limit_A, options.vary_limbd_coef_A, options.RV_A
                        )
                        template_shifted = shift_spectrum(  # shift template to reduce boundary issues
                            flux_templateA, RV_collection_orders_A[j, i], delta_v
                        )[mask_collection_orders[:, j, i]]
                        BRsvd_template_A = BroadeningFunction(
                            corrected_flux_A,
                            1 - template_shifted,
                            v_span, delta_v
                        )
                        BRsvd_template_A.smooth_sigma = options.bf_smooth_sigma_A
                        ifitparams_A.RV = 0.0
                        rva_temp, bf_fitres_A[j, i] = radial_velocity_single_component(
                            corrected_flux_A, BRsvd_template_A, ifitparams_A
                        )
                        RV_deviation_A[j] = rva_temp
                        RV_collection_orders_A[j, i] = rva_temp + RV_collection_orders_A[j, i]

                        # # Calculate RV_B # #
                        corrected_flux_B = (1 - flux_collection_orders[:, j, i]) - \
                                           (1 - shift_spectrum(separated_flux_A, RV_collection_orders_A[j, i], delta_v))
                        if period is not None and options.ignore_at_phase_A is not None and time_values is not None:
                            if _check_for_total_eclipse(time_values[i], period, options.ignore_at_phase_A) is True:
                                corrected_flux_B = 1 - flux_collection_orders[:, j, i]
                        corrected_flux_B = corrected_flux_B[mask_collection_orders[:, j, i]]
                        options.RV_B = RV_collection_orders_B[j, i]
                        ifitparams_B = InitialFitParameters(
                            options.vsini_B, options.spectral_resolution, options.velocity_fit_width_B,
                            options.limbd_coef_B,
                            options.bf_smooth_sigma_B, options.bf_velocity_span, options.vary_vsini_B,
                            options.vsini_vary_limit_B, options.vary_limbd_coef_B, options.RV_B
                        )
                        template_shifted = shift_spectrum(
                            flux_templateB[mask_collection_orders[:, j, i]], RV_collection_orders_B[j, i], delta_v
                        )
                        BRsvd_template_B = BroadeningFunction(
                            corrected_flux_B,
                            1 - template_shifted, v_span, delta_v
                        )
                        BRsvd_template_B.smooth_sigma = options.bf_smooth_sigma_B
                        ifitparams_B.RV = 0.0
                        rvb_temp, bf_fitres_B[j, i] = radial_velocity_single_component(
                            corrected_flux_B, BRsvd_template_B, ifitparams_B
                        )
                        RV_deviation_B[j] = rvb_temp
                        RV_collection_orders_B[j, i] = rvb_temp + RV_collection_orders_B[j, i]
                    if options.verbose:
                        print(
                            f'RV dev spec {i}: {np.abs(np.mean(RV_deviation_A)):.{options.print_prec}f} (A) '
                            f'{np.abs(np.mean(RV_deviation_B)):.{options.print_prec}f} (B)'
                        )
                    if np.abs(np.mean(RV_deviation_A)) < options.convergence_limit and \
                            np.abs(np.mean(RV_deviation_B)) < options.convergence_limit:
                        if options.verbose:
                            print(f'RV: spectrum {i} successful.')
                        break
                    elif iterations > options.iteration_limit:
                        if k == 1 or (options.refit_width_B is None and options.refit_width_A is None):
                            if options.verbose is True:
                                warnings.warn(
                                    f'RV: spectrum {i} did not reach convergence limit {options.convergence_limit}.'
                                )
                        break

            rv_lower_limit = options.rv_lower_limit
            if plot_ax_A is not None:
                _update_bf_plot(plot_ax_A, bf_fitres_A[plot_order, i], RV_collection_orders_A[plot_order, i], i)
                if rv_lower_limit != 0.0:
                    plot_ax_A.plot([rv_lower_limit, rv_lower_limit], [0, 1.1], 'k', linewidth=0.3)
                    plot_ax_A.plot([-rv_lower_limit, -rv_lower_limit], [0, 1.1], 'k', linewidth=0.3)
            if plot_ax_B is not None:
                _update_bf_plot(plot_ax_B, bf_fitres_B[plot_order, i], RV_collection_orders_B[plot_order, i], i)
                if rv_lower_limit != 0.0:
                    plot_ax_B.plot([rv_lower_limit, rv_lower_limit], [0, 1.1], 'k', linewidth=0.3)
                    plot_ax_B.plot([-rv_lower_limit, -rv_lower_limit], [0, 1.1], 'k', linewidth=0.3)

    if krange == 2 and options.n_parallel_jobs == 1:
        options.velocity_fit_width_A = fit_width_A
        options.velocity_fit_width_B = fit_width_B

    return RV_collection_orders_A, RV_collection_orders_B, (bf_fitres_A, bf_fitres_B)


def _rv_loop_orders(
        RV_collection_orders_A, RV_collection_orders_B, flux_collection_orders, separated_flux_A, separated_flux_B,
        flux_templateA, flux_templateB, mask_collection_orders, delta_v, v_span, period, time_values, n_orders, krange,
        bf_fitres_A, bf_fitres_B, options,
        i
):
    # RV_collection_orders_A = np.copy(RV_collection_orders_A)
    # RV_collection_orders_B = np.copy(RV_collection_orders_B)
    # bf_fitres_A = np.copy(bf_fitres_A)
    # bf_fitres_B = np.copy(bf_fitres_B)
    options = deepcopy(options) # probably not needed
    if options.verbose:
        print(f'RV: spectrum {i} begun.')
    if options.refit_width_A is not None or options.refit_width_B is not None:
        krange = 2
    else:
        pass
    for k in range(0, krange):
        if k == 0:
            fit_width_A = copy(options.velocity_fit_width_A)
            fit_width_B = copy(options.velocity_fit_width_B)
        if k == 1:
            options.velocity_fit_width_A = options.refit_width_A
            options.velocity_fit_width_B = options.refit_width_B
        iterations = 0
        RV_deviation_A = np.ones((RV_collection_orders_A[:, 0].size,))
        RV_deviation_B = np.ones((RV_collection_orders_B[:, 0].size,))
        while True:
            iterations += 1
            for j in range(0, n_orders):
                # # Calculate RV_A # #
                corrected_flux_A = (1 - flux_collection_orders[:, j, i]) - \
                                   (1 - shift_spectrum(separated_flux_B, RV_collection_orders_B[j, i], delta_v))
                if period is not None and options.ignore_at_phase_B is not None and time_values is not None:
                    if _check_for_total_eclipse(time_values[i], period, options.ignore_at_phase_B) is True:
                        corrected_flux_A = 1 - flux_collection_orders[:, j, i]
                corrected_flux_A = corrected_flux_A[mask_collection_orders[:, j, i]]
                options.RV_A = RV_collection_orders_A[j, i]
                ifitparams_A = InitialFitParameters(
                    options.vsini_A, options.spectral_resolution, options.velocity_fit_width_A, options.limbd_coef_A,
                    options.bf_smooth_sigma_A, options.bf_velocity_span, options.vary_vsini_A,
                    options.vsini_vary_limit_A, options.vary_limbd_coef_A, options.RV_A
                )
                template_shifted = shift_spectrum(  # shift template to reduce boundary issues
                    flux_templateA, RV_collection_orders_A[j, i], delta_v
                )[mask_collection_orders[:, j, i]]
                BRsvd_template_A = BroadeningFunction(
                    corrected_flux_A,
                    1 - template_shifted,
                    v_span, delta_v
                )
                BRsvd_template_A.smooth_sigma = options.bf_smooth_sigma_A
                ifitparams_A.RV = 0.0
                rva_temp, bf_fitres_A[j, i] = radial_velocity_single_component(
                    corrected_flux_A, BRsvd_template_A, ifitparams_A
                )
                RV_deviation_A[j] = rva_temp
                RV_collection_orders_A[j, i] = rva_temp + RV_collection_orders_A[j, i]

                # # Calculate RV_B # #
                corrected_flux_B = (1 - flux_collection_orders[:, j, i]) - \
                                   (1 - shift_spectrum(separated_flux_A, RV_collection_orders_A[j, i], delta_v))
                if period is not None and options.ignore_at_phase_A is not None and time_values is not None:
                    if _check_for_total_eclipse(time_values[i], period, options.ignore_at_phase_A) is True:
                        corrected_flux_B = 1 - flux_collection_orders[:, j, i]
                corrected_flux_B = corrected_flux_B[mask_collection_orders[:, j, i]]
                options.RV_B = RV_collection_orders_B[j, i]
                ifitparams_B = InitialFitParameters(
                    options.vsini_B, options.spectral_resolution, options.velocity_fit_width_B, options.limbd_coef_B,
                    options.bf_smooth_sigma_B, options.bf_velocity_span, options.vary_vsini_B,
                    options.vsini_vary_limit_B, options.vary_limbd_coef_B, options.RV_B
                )
                template_shifted = shift_spectrum(
                    flux_templateB[mask_collection_orders[:, j, i]], RV_collection_orders_B[j, i], delta_v
                )
                BRsvd_template_B = BroadeningFunction(
                    corrected_flux_B,
                    1 - template_shifted, v_span, delta_v
                )
                BRsvd_template_B.smooth_sigma = options.bf_smooth_sigma_B
                ifitparams_B.RV = 0.0
                rvb_temp, bf_fitres_B[j, i] = radial_velocity_single_component(
                    corrected_flux_B, BRsvd_template_B, ifitparams_B
                )
                RV_deviation_B[j] = rvb_temp
                RV_collection_orders_B[j, i] = rvb_temp + RV_collection_orders_B[j, i]
            if options.verbose:
                print(
                    f'RV dev spec {i}: {np.abs(np.mean(RV_deviation_A)):.{options.print_prec}f} (A) '
                    f'{np.abs(np.mean(RV_deviation_B)):.{options.print_prec}f} (B)'
                )
            if np.abs(np.mean(RV_deviation_A)) < options.convergence_limit and \
                    np.abs(np.mean(RV_deviation_B)) < options.convergence_limit:
                if options.verbose:
                    print(f'RV: spectrum {i} successful.')
                break
            elif iterations > options.iteration_limit:
                if k == 1 or (options.refit_width_B is None and options.refit_width_A is None):
                    if options.verbose is True:
                        warnings.warn(
                            f'RV: spectrum {i} did not reach convergence limit {options.convergence_limit}.'
                        )
                break
    return RV_collection_orders_A[:, i], RV_collection_orders_B[:, i], (bf_fitres_A[:, i], bf_fitres_B[:, i])


def _check_for_total_eclipse(time_value, period, eclipse_phase_area):
    phase = np.mod(time_value, period)/period
    lower = eclipse_phase_area[0]
    upper = eclipse_phase_area[1]
    if lower < upper:
        condition = (phase > lower) & (phase < upper)
    elif lower > upper:
        condition = (phase > lower) | (phase < upper)
    else:
        raise ValueError('eclipse_phase_area must comprise of a lower and an upper value that are separate.')
    return condition


def _initialize_ssr_plots():
    # RVs and separated spectra
    fig_1 = plt.figure(figsize=(16, 9))
    gs_1 = fig_1.add_gridspec(2, 2)
    f1_ax1 = fig_1.add_subplot(gs_1[0, :])
    f1_ax2 = fig_1.add_subplot(gs_1[1, 0])
    f1_ax3 = fig_1.add_subplot(gs_1[1, 1])

    # Broadening function fits A
    fig_3 = plt.figure(figsize=(16, 9))
    gs_3 = fig_3.add_gridspec(1, 1)
    f3_ax1 = fig_3.add_subplot(gs_3[:, :])

    # Broadening function fits B
    fig_4 = plt.figure(figsize=(16, 9))
    gs_4 = fig_4.add_gridspec(1, 1)
    f4_ax1 = fig_4.add_subplot(gs_4[:, :])
    return f1_ax1, f1_ax2, f1_ax3, f3_ax1, f4_ax1


def _plot_ssr_iteration(
        f1_ax1, f1_ax2, f1_ax3, separated_flux_A, separated_flux_B, wavelength, flux_template_A,
        flux_template_B, RV_collection_A, RV_collection_B, time, period, buffer_mask, rv_lower_limit,
        rv_proximity_limit
):
    f1_ax1.clear(); f1_ax2.clear(); f1_ax3.clear()
    separated_flux_A, separated_flux_B = separated_flux_A[~buffer_mask], separated_flux_B[~buffer_mask]
    wavelength = wavelength[~buffer_mask]
    flux_template_A, flux_template_B = flux_template_A[~buffer_mask], flux_template_B[~buffer_mask]

    if RV_collection_A.ndim == 2:
        RV_A = np.mean(RV_collection_A, axis=0)
        RV_B = np.mean(RV_collection_B, axis=0)
    else:
        RV_A = RV_collection_A
        RV_B = RV_collection_B
    if period is None:
        xval = time
        f1_ax1.set_xlabel('BJD - 245000')
    else:
        xval = np.mod(time, period) / period
        f1_ax1.set_xlabel('Phase')
    if rv_lower_limit == 0.0:
        RV_below_limit_mask = np.abs(RV_A - RV_B) < rv_proximity_limit
    else:
        RV_below_limit_mask = np.abs(RV_A) < rv_lower_limit
    f1_ax1.plot(xval[~RV_below_limit_mask], RV_A[~RV_below_limit_mask], 'b*')
    f1_ax1.plot(xval[~RV_below_limit_mask], RV_B[~RV_below_limit_mask], 'r*')
    f1_ax1.plot(xval[RV_below_limit_mask], RV_A[RV_below_limit_mask], 'bx')
    f1_ax1.plot(xval[RV_below_limit_mask], RV_B[RV_below_limit_mask], 'rx')

    f1_ax2.plot(wavelength, flux_template_A, '--', color='grey', linewidth=0.25)
    f1_ax2.plot(wavelength, separated_flux_A, 'b', linewidth=2)

    f1_ax3.plot(wavelength, flux_template_B, '--', color='grey', linewidth=0.25)
    f1_ax3.plot(wavelength, separated_flux_B, 'r', linewidth=2)

    f1_ax1.set_ylabel('Radial Velocity [km/s]')
    f1_ax2.set_ylabel('Normalized Flux')
    f1_ax2.set_xlabel('Wavelength [Å]')
    f1_ax3.set_xlabel('Wavelength [Å]')

    plt.draw_all()
    plt.show(block=False)
    plt.pause(0.1)


def save_multi_image(filename):
    """
    https://www.tutorialspoint.com/saving-all-the-open-matplotlib-figures-in-one-file-at-once
    """
    pp = PdfPages(filename)
    fig_nums = plt.get_fignums()
    figs = [plt.figure(n) for n in fig_nums]
    for fig in figs:
        fig.savefig(pp, format='pdf')
    pp.close()


def spectral_separation_routine(
        flux_collection: np.ndarray, flux_templateA: np.ndarray, flux_templateB: np.ndarray,
        wavelength: np.ndarray,
        options: RoutineOptions, sep_comp_options: SeparateComponentsOptions, rv_options: RadialVelocityOptions,
        RV_guess_collection: np.ndarray, mask_collection_orders=None, time_values: np.ndarray = None,
        period: float = None, buffer_mask: np.ndarray = None
):
    """
    Routine that separates component spectra and calculates radial velocities by iteratively calling
    separate_component_spectra() and recalculate_RVs() to attempt to converge towards the correct RV values for the
    system. Requires good starting guesses on the component RVs. It also plots each iteration (if enabled) to follow
    along the results.

    The routine must be provided with normalized spectra sampled in the same wavelengths equi-spaced in velocity space.
    It is recommended to supply spectra that are buffered (padded) in the ends, and a buffer_mask that indicates this.
    This will limit the effect of loop-back when rolling (shifting) the spectra. spf.limit_wavelength_interval() can be
    used to produce a buffered data-set (and others of the same size).

    The spectra used for calculating the separated spectra should cover a wide range of orbital phases to provide a
    decent averaged spectrum. They should have a high S/N, and should not include significant emission lines. Avoid
    using spectra with RVs of the two components close to each others, or spectra from within eclipses.

    Additional features included to improve either the RV fitting process or the spectral separation:
      - Fitted v*sin(i) values are meaned and used as fit guesses for next iteration. Simultaneously, the parameter
        is bounded to the limits v*sin(i) * [1-0.5, 1+0.5], in order to ensure stable fitting for all spectra.

      - The routine can be provided with an array of indices specifying which spectra to use when creating the separated
        spectra. This is the recommended way to designate. A lower RV limit on the primary component, or a proximity
        limit of the two RVs, can also be provided isntead. However, this does not consider any other important
        factors for why you would leave out a spectrum (low S/N, bad reduction, within eclipses).
        Providing rv_lower_limit (or rv_proximity_limit) while also supplying use_spectra, can be useful since the
        routines avoids using the "bad" component B RVs when calculating RMS values.
        Note that if use_spectra is provided as an array, the two rv limit options will only be used for RMS
        calculation, and not for spectral separation.

    :param flux_collection:   np.ndarray shape (:, n_spectra). Collection of normalized fluxes for each spectrum.
                                        if ndims=3, assumes shape (:, n_orders, n_spectra) containing independent orders
                                        Then mask_collection_orders must be supplied as well
    :param flux_templateA:    np.ndarray shape (:, ). Normalized flux of template spectrum for component A.
    :param flux_templateB:    np.ndarray shape (:, ). Normalized flux of template spectrum for component B.
    :param wavelength:            np.ndarray shape (:, ). Wavelength values for the spectra (both program and template).
    :param options:               Options for this global routine.
    :param sep_comp_options:      Options to pass on to the subroutine separate_component_spectra.
    :param rv_options:            Options to pass on to the subroutine recalculate_RVs.
    :param RV_guess_collection:   np.ndarray shape (n_spectra, 2). Initial RV guesses for each component (A: [:, 0]).
    :param buffer_mask: np.ndarray shape (:, ), Default = None. If data is padded with a buffer around the edges,
                        this mask should be supplied. It should give the unbuffered data when flipped, e.g.
                        `flux_buffered[~buffer_mask] = flux`.

    :return:    RV_collection_A,  RV_collection_B, separated_flux_A, separated_flux_B, wavelength
                RV_collection_A:  np.ndarray shape (n_spectra, ). RV values of component A for each program spectrum.
                RV_collection_B:  same, but for component B (includes values below rv_lower_limit).
                separated_flux_A: np.ndarray shape (:*, ). The found "separated" or "disentangled" spectrum for A.
                                    It is an inverted flux (1-normalized_flux).
                separated_flux_B: np.ndarray shape (:*, ). The found "separated" or "disentangled" spectrum for B.
                wavelength:       np.ndarray shape (:*, ). Wavelength values for the separated spectra.

            Note on :*
                if buffer_mask is provided, the returned spectra will be the un-buffered versions, meaning
                separated_flux_A.size = flux_templateA[buffer_mask].size. Same for the returned wavelength.
                This can be disabled by setting return_unbuffered=False in the options object.
    """

    if RV_guess_collection.ndim == 2:
        RV_collection_A, RV_collection_B = np.copy(RV_guess_collection[:, 0]), np.copy(RV_guess_collection[:, 1])
    elif RV_guess_collection.ndim == 3:
        RV_collection_A = np.copy(RV_guess_collection[:, :, 0])
        RV_collection_B = np.copy(RV_guess_collection[:, :, 1])
    else:
        raise ValueError('RV_guess_collection.ndim != 2 or 3.')

    convergence_limit = options.convergence_limit
    iteration_limit = options.iteration_limit

    if buffer_mask is None:
        buffer_mask = np.zeros(wavelength.shape, dtype=bool)

    # Initialize plot figures
    if options.plot:
        f1_ax1, f1_ax2, f1_ax3, f3_ax1, f4_ax1 = _initialize_ssr_plots()
    else:
        f1_ax1 = None; f1_ax2 = None; f1_ax3 = None; f3_ax1 = None; f4_ax1 = None

    if flux_collection.ndim == 3:
        overlap_weights = _calculate_overlap(mask_collection_orders)

    # Iterative loop that repeatedly separates the spectra from each other in order to calculate new RVs (Gonzales 2005)
    iterations = 0
    print('Spectral Separation: ')
    while True:
        print(f'\nIteration {iterations}.')
        if flux_collection.ndim == 3:
            if RV_collection_A.ndim == 1:
                RV_collection_A = RV_collection_A.reshape((1, RV_collection_A.size))
                RV_collection_B = RV_collection_B.reshape((1, RV_collection_B.size))
                RV_collection_A = np.repeat(RV_collection_A, axis=0, repeats=flux_collection[0, :, 0].size)
                RV_collection_B = np.repeat(RV_collection_B, axis=0, repeats=flux_collection[0, :, 0].size)

        if sep_comp_options.rv_lower_limit == 0.0:
            if RV_collection_A.ndim == 2:
                RV_mask = np.abs(
                    np.mean(RV_collection_A, axis=0)-np.mean(RV_collection_B, axis=0)
                ) > sep_comp_options.rv_proximity_limit
            else:
                RV_mask = np.abs(RV_collection_A-RV_collection_B) > sep_comp_options.rv_proximity_limit
        else:
            if RV_collection_A.ndim == 2:
                RV_mask = np.abs(
                    np.mean(RV_collection_A, axis=0) - np.mean(RV_collection_B, axis=0)
                ) > sep_comp_options.rv_lower_limit
            else:
                RV_mask = np.abs(RV_collection_A) > sep_comp_options.rv_lower_limit

        if RV_collection_A.ndim == 2:
            RMS_A, RMS_B = -np.mean(RV_collection_A, axis=0), -np.mean(RV_collection_B, axis=0)[RV_mask]
        else:
            RMS_A, RMS_B = -RV_collection_A, -RV_collection_B[RV_mask]

        if flux_collection.ndim == 3:
            separated_flux_A, separated_flux_B = separate_component_spectra_orders(
                flux_collection, RV_collection_A, RV_collection_B, sep_comp_options, overlap_weights
            )
        else:
            separated_flux_A, separated_flux_B = separate_component_spectra(
                flux_collection, RV_collection_A, RV_collection_B, sep_comp_options
            )

        if flux_collection.ndim == 3:
            RV_collection_A, RV_collection_B, (bf_fitres_A, bf_fitres_B) = recalculate_RVs_orders(
                flux_collection, mask_collection_orders, separated_flux_A, separated_flux_B, RV_collection_A,
                RV_collection_B, flux_templateA, flux_templateB, rv_options, plot_ax_A=f3_ax1, plot_ax_B=f4_ax1,
                time_values=time_values, period=period, plot_order=options.plot_order
            )
        else:
            RV_collection_A, RV_collection_B, (bf_fitres_A, bf_fitres_B) = recalculate_RVs(
                flux_collection, separated_flux_A, separated_flux_B, RV_collection_A, RV_collection_B,
                flux_templateA, flux_templateB, buffer_mask, rv_options, plot_ax_A=f3_ax1,
                plot_ax_B=f4_ax1, time_values=time_values, period=period
            )

        if options.plot:
            _plot_ssr_iteration(
                f1_ax1, f1_ax2, f1_ax3, separated_flux_A, separated_flux_B, wavelength, flux_templateA,
                flux_templateB, RV_collection_A, RV_collection_B, time_values, period,
                buffer_mask, sep_comp_options.rv_lower_limit, sep_comp_options.rv_proximity_limit
            )

        # Average vsini values for future fit guess and limit allowed fit area
        vsini_A, vsini_B = np.empty(shape=bf_fitres_A.shape), np.empty(shape=bf_fitres_B.shape)
        if vsini_A.ndim == 2:
            for i in range(0, vsini_A[0, :].size):
                for j in range(0, vsini_A[:, 0].size):
                    _, _, vsini_A[j, i], _, _, _ = get_fit_parameter_values(bf_fitres_A[j, i][0].params)
                    _, _, vsini_B[j, i], _, _, _ = get_fit_parameter_values(bf_fitres_B[j, i][0].params)
        else:
            for i in range(0, vsini_A.size):
                _, _, vsini_A[i], _, _, _ = get_fit_parameter_values(bf_fitres_A[i][0].params)
                _, _, vsini_B[i], _, _, _ = get_fit_parameter_values(bf_fitres_B[i][0].params)
        rv_options.vsini_A = np.mean(vsini_A)
        rv_options.vsini_B = np.mean(vsini_B)
        if rv_options.vsini_vary_limit_A is None:
            rv_options.vsini_vary_limit_A = 0.7
        if rv_options.vsini_vary_limit_B is None:
            rv_options.vsini_vary_limit_B = 0.7
        if options.verbose:
            print('vsini A: ', rv_options.vsini_A)
            print('vsini B: ', rv_options.vsini_B)

        iterations += 1
        if RV_collection_A.ndim == 2:
            RMS_A += np.mean(RV_collection_A, axis=0)
            RMS_B += np.mean(RV_collection_B, axis=0)[RV_mask]
            RMS_A = np.sqrt(np.sum(RMS_A**2)/np.mean(RV_collection_A, axis=0).size)
            RMS_B = np.sqrt(np.sum(RMS_B**2)/np.mean(RV_collection_B, axis=0)[RV_mask].size)
        else:
            RMS_A += RV_collection_A
            RMS_B += RMS_B
            RMS_A = np.sqrt(np.sum(RMS_A**2))/RV_collection_A.size
            RMS_B = np.sqrt(np.sum(RMS_B**2))/RV_collection_B[RV_mask].size
        if options.verbose:
            print(f'RV_A RMS: {RMS_A}. ' + f'RV_B RMS: {RMS_B}.')
        if RMS_A < convergence_limit and RMS_B < convergence_limit:
            print(f'Spectral separation routine terminates after reaching convergence limit {convergence_limit}.')
            break
        if iterations >= iteration_limit:
            warnings.warn(f'RV convergence limit of {convergence_limit} not reached in {iterations} iterations.',
                          category=Warning)
            print('Spectral separation routine terminates.')
            break

    if options.save_all_results is True:
        save_separation_data(
            options.save_path, wavelength[~buffer_mask], time_values, RV_collection_A,
            RV_collection_B, RV_guess_collection, separated_flux_A[~buffer_mask], separated_flux_B[~buffer_mask],
            bf_fitres_A, bf_fitres_B, flux_templateA[~buffer_mask], flux_templateB[~buffer_mask], options
        )

    if options.save_plot_path is not None:
        save_multi_image(options.save_plot_path)

    if options.return_unbuffered:
        return RV_collection_A, RV_collection_B, separated_flux_A[~buffer_mask], separated_flux_B[~buffer_mask], \
               wavelength[~buffer_mask]
    else:
        return RV_collection_A, RV_collection_B, separated_flux_A, separated_flux_B, wavelength


def save_separation_data(
        location, wavelength, time_values, RVs_A, RVs_B, RVs_initial, separated_flux_A, separated_flux_B, bf_fitres_A,
        bf_fitres_B, template_flux_A, template_flux_B, options: RoutineOptions
):
    if options.filename_bulk is None:
        filename_bulk = str(int(np.round(np.min(wavelength)))) + '_' + str(int(np.round(np.max(wavelength))))
    else:
        filename_bulk = options.filename_bulk

    if RVs_A.ndim == 2:
        rvA_array = np.empty((RVs_A[0, :].size, 1+RVs_A[:, 0].size))
        rvA_array[:, 0] = time_values
        rvA_array[:, 1:] = RVs_A.T
    else:
        rvA_array = np.empty((RVs_A.size, 2))
        rvA_array[:, 0], rvA_array[:, 1] = time_values, RVs_A
    if RVs_B.ndim == 2:
        rvB_array = np.empty((RVs_B[0, :].size, 1+RVs_B[:, 0].size))
        rvB_array[:, 0] = time_values
        rvB_array[:, 1:] = RVs_B.T
    else:
        rvB_array = np.empty((RVs_B.size, 2))
        rvB_array[:, 0], rvB_array[:, 1] = time_values, RVs_B
    
    sep_array = np.empty((wavelength.size, 5))
    sep_array[:, 0], sep_array[:, 1], sep_array[:, 2] = wavelength, separated_flux_A, separated_flux_B
    sep_array[:, 3], sep_array[:, 4] = template_flux_A, template_flux_B

    if RVs_A.ndim == 2:
        fmt_tuple_A = ['%.6f' for x in range(0, RVs_A[:, 0].size)]
        fmt_tuple_A = ['%.9f'] + fmt_tuple_A
        fmt_tuple_A = tuple(fmt_tuple_A)
    else:
        fmt_tuple_A = ('%.9f', '%.6f')
    if RVs_B.ndim == 2:
        fmt_tuple_B = ['%.6f' for x in range(0, RVs_B[:, 0].size)]
        fmt_tuple_B = ['%.9f'] + fmt_tuple_B
        fmt_tuple_B = tuple(fmt_tuple_B)
    else:
        fmt_tuple_B = ('%.9f', '%.6f')
    np.savetxt(location + filename_bulk + '_rvA.txt', rvA_array, header='Time [input units] \t RV_A [km/s] orders',
               fmt=fmt_tuple_A)
    np.savetxt(location + filename_bulk + '_rv_initial.txt', RVs_initial, header='RV_A [km/s] \t RV_B [km/s]',
               fmt='%.6f')
    np.savetxt(location + filename_bulk + '_rvB.txt', rvB_array, header='Time [input units] \t RV_B [km/s] orders',
               fmt=fmt_tuple_B)
    np.savetxt(location + filename_bulk + '_sep_flux.txt', sep_array, header='flux_A \t flux_B', fmt='%.6f')

    if bf_fitres_A.ndim == 2:
        for j in range(0, bf_fitres_A[:, 0].size):
            vel_array = np.empty((bf_fitres_A[j, :].size, bf_fitres_A[j, 0][1].size))
            bf_array = np.empty((bf_fitres_A[j, :].size, bf_fitres_A[j, 0][1].size))
            bf_smooth_array = np.empty((bf_fitres_A[j, :].size, bf_fitres_A[j, 0][1].size))
            model_array = np.empty((bf_fitres_A[j, :].size, bf_fitres_A[j, 0][1].size))
            for i in range(0, bf_fitres_A[j, :].size):
                model_vals_A, bf_vel_A, bf_vals_A, smooth_vals_A = bf_fitres_A[j, i][1:]
                vel_array[i, :] = bf_vel_A
                bf_array[i, :] = bf_vals_A
                bf_smooth_array[i, :] = smooth_vals_A
                model_array[i, :] = model_vals_A
            np.savetxt(location+filename_bulk+'_order_'+str(j)+'_velocities_A.txt', vel_array, fmt='%.6f', header='spec 1 \t spec 2 \t ...')
            np.savetxt(location + filename_bulk +'_order_'+str(j) + '_bfvals_A.txt', bf_array, fmt='%.6f',
                       header='spec 1 \t spec 2 \t ...')
            np.savetxt(location + filename_bulk +'_order_'+str(j) + '_bfsmooth_A.txt', bf_smooth_array, fmt='%.6f',
                       header='spec 1 \t spec 2 \t ...')
            np.savetxt(location + filename_bulk +'_order_'+str(j) + '_models_A.txt', model_array, fmt='%.6f',
                       header='spec 1 \t spec 2 \t ...')
    else:
        vel_array = np.empty((bf_fitres_A.size, bf_fitres_A[0][1].size))
        bf_array = np.empty((bf_fitres_A.size, bf_fitres_A[0][1].size))
        bf_smooth_array = np.empty((bf_fitres_A.size, bf_fitres_A[0][1].size))
        model_array = np.empty((bf_fitres_A.size, bf_fitres_A[0][1].size))
        for i in range(0, bf_fitres_A.size):
            model_vals_A, bf_velocity_A, bf_vals_A, bf_smooth_vals_A = bf_fitres_A[i][1:]
            vel_array[i, :] = bf_velocity_A
            bf_array[i, :] = bf_vals_A
            bf_smooth_array[i, :] = bf_smooth_vals_A
            model_array[i, :] = model_vals_A
        np.savetxt(location + filename_bulk + '_velocities_A.txt', vel_array, fmt='%.6f', header='spec 1 \t spec 2 \t ...')
        np.savetxt(location + filename_bulk + '_bfvals_A.txt', bf_array, fmt='%.6f', header='spec 1 \t spec 2 \t ...')
        np.savetxt(location + filename_bulk + '_bfsmooth_A.txt', bf_smooth_array, fmt='%.6f',
               header='spec 1 \t spec 2 \t ...')
        np.savetxt(location + filename_bulk + '_models_A.txt', model_array, fmt='%.6f', header='spec 1 \t spec 2 \t ...')

    if bf_fitres_B.ndim == 2:
        for j in range(0, bf_fitres_B[:, 0].size):
            vel_array = np.empty((bf_fitres_B[j, :].size, bf_fitres_B[j, 0][1].size))
            bf_array = np.empty((bf_fitres_B[j, :].size, bf_fitres_B[j, 0][1].size))
            bf_smooth_array = np.empty((bf_fitres_B[j, :].size, bf_fitres_B[j, 0][1].size))
            model_array = np.empty((bf_fitres_B[j, :].size, bf_fitres_B[j, 0][1].size))
            for i in range(0, bf_fitres_B[j, :].size):
                model_vals_B, bf_vel_B, bf_vals_B, smooth_vals_B = bf_fitres_B[j, i][1:]
                vel_array[i, :] = bf_vel_B
                bf_array[i, :] = bf_vals_B
                bf_smooth_array[i, :] = smooth_vals_B
                model_array[i, :] = model_vals_B
            np.savetxt(location+filename_bulk+'_order_'+str(j)+'_velocities_B.txt', vel_array, fmt='%.6f', header='spec 1 \t spec 2 \t ...')
            np.savetxt(location + filename_bulk +'_order_'+str(j) + '_bfvals_B.txt', bf_array, fmt='%.6f',
                       header='spec 1 \t spec 2 \t ...')
            np.savetxt(location + filename_bulk +'_order_'+str(j) + '_bfsmooth_B.txt', bf_smooth_array, fmt='%.6f',
                       header='spec 1 \t spec 2 \t ...')
            np.savetxt(location + filename_bulk +'_order_'+str(j) + '_models_B.txt', model_array, fmt='%.6f',
                       header='spec 1 \t spec 2 \t ...')
    else:
        vel_array = np.empty((bf_fitres_B.size, bf_fitres_B[0][1].size))
        bf_array = np.empty((bf_fitres_B.size, bf_fitres_B[0][1].size))
        bf_smooth_array = np.empty((bf_fitres_B.size, bf_fitres_B[0][1].size))
        model_array = np.empty((bf_fitres_B.size, bf_fitres_B[0][1].size))
        for i in range(0, bf_fitres_B.size):
            model_vals_B, bf_velocity_B, bf_vals_B, bf_smooth_vals_B = bf_fitres_B[i][1:]
            vel_array[i, :] = bf_velocity_B
            bf_array[i, :] = bf_vals_B
            bf_smooth_array[i, :] = bf_smooth_vals_B
            model_array[i, :] = model_vals_B
        np.savetxt(location + filename_bulk + '_velocities_B.txt', vel_array, fmt='%.6f', header='spec 1 \t spec 2 \t ...')
        np.savetxt(location + filename_bulk + '_bfvals_B.txt', bf_array, fmt='%.6f', header='spec 1 \t spec 2 \t ...')
        np.savetxt(location + filename_bulk + '_bfsmooth_B.txt', bf_smooth_array, fmt='%.6f',
               header='spec 1 \t spec 2 \t ...')
        np.savetxt(location + filename_bulk + '_models_B.txt', model_array, fmt='%.6f', header='spec 1 \t spec 2 \t ...')


def _create_wavelength_intervals(
        wavelength, wavelength_intervals: float or list, flux_collection, flux_templateA, flux_templateB,
        wavelength_buffer_size, separated_flux_A: np.ndarray = None, separated_flux_B: np.ndarray = None
):
    wavelength_interval_collection = []
    flux_interval_collection = []
    templateA_interval_collection = []
    templateB_interval_collection = []
    interval_buffer_mask = []
    separated_A_interval_collection = []
    separated_B_interval_collection = []
    w_interval_start = wavelength[0] + wavelength_buffer_size
    i = 0
    while True:
        if isinstance(wavelength_intervals, float):
            if w_interval_start + wavelength_intervals > wavelength[-1] - wavelength_buffer_size:
                w_interval_end = wavelength[-1] - wavelength_buffer_size
            else:
                w_interval_end = w_interval_start + wavelength_intervals

            if w_interval_end - w_interval_start < wavelength_intervals // 2:
                break
        else:
            if i >= len(wavelength_intervals):
                break
            w_interval_start = wavelength_intervals[i][0]
            w_interval_end = wavelength_intervals[i][1]

        w_interval = (w_interval_start, w_interval_end)

        if separated_flux_A is not None and separated_flux_B is not None:
            _, _, wl_buffered, flux_buffered_list, buffer_mask = spf.limit_wavelength_interval_multiple_spectra(
                w_interval, wavelength, flux_collection, flux_templateA, flux_templateB, separated_flux_A,
                separated_flux_B, buffer_size=wavelength_buffer_size, even_length=True
            )
            [fl_buffered, flA_buffered, flB_buffered, sflA_buffered, sflB_buffered] = flux_buffered_list
            separated_A_interval_collection.append(sflA_buffered)
            separated_B_interval_collection.append(sflB_buffered)
        else:
            _, _, wl_buffered, flux_buffered_list, buffer_mask = spf.limit_wavelength_interval_multiple_spectra(
                w_interval, wavelength, flux_collection, flux_templateA, flux_templateB,
                buffer_size=wavelength_buffer_size, even_length=True
            )
            [fl_buffered, flA_buffered, flB_buffered] = flux_buffered_list

        wavelength_interval_collection.append(wl_buffered)
        flux_interval_collection.append(fl_buffered)
        templateA_interval_collection.append(flA_buffered)
        templateB_interval_collection.append(flB_buffered)
        interval_buffer_mask.append(buffer_mask)
        i += 1
        if isinstance(wavelength_intervals, float):
            w_interval_start = w_interval_end

    if separated_flux_A is not None and separated_flux_B is not None:
        return (wavelength_interval_collection, flux_interval_collection, templateA_interval_collection,
                templateB_interval_collection, separated_A_interval_collection, separated_B_interval_collection,
                interval_buffer_mask)
    else:
        return (wavelength_interval_collection, flux_interval_collection, templateA_interval_collection,
                templateB_interval_collection, interval_buffer_mask)


def _combine_intervals(combine_intervals, wl_intervals, fl_intervals, templA_intervals, templB_intervals,
                       interval_buffer):
    for arg in reversed(combine_intervals):
        first_idx, last_idx = arg[0], arg[1]
        wl_intervals[first_idx] = np.append(wl_intervals[first_idx], wl_intervals[last_idx])
        fl_intervals[first_idx] = np.append(fl_intervals[first_idx], fl_intervals[last_idx],
                                                  axis=0)
        templA_intervals[first_idx] = np.append(templA_intervals[first_idx], templA_intervals[last_idx])
        templB_intervals[first_idx] = np.append(templB_intervals[first_idx], templB_intervals[last_idx])
        interval_buffer[first_idx] = np.append(interval_buffer[first_idx], interval_buffer[last_idx])

        del wl_intervals[last_idx]
        del fl_intervals[last_idx]
        del templA_intervals[last_idx]
        del templB_intervals[last_idx]
        del interval_buffer[last_idx]

    return wl_intervals, fl_intervals, templA_intervals, templB_intervals, interval_buffer


def spectral_separation_routine_multiple_intervals(
        wavelength: np.ndarray,
        wavelength_intervals: list,
        flux_collection: np.ndarray,
        flux_templateA: np.ndarray, flux_templateB: np.ndarray,
        RV_guess_collection: np.ndarray,
        options: RoutineOptions,
        sep_comp_options: SeparateComponentsOptions,
        rv_options: RadialVelocityOptions,
        buffer_size: float = None,
        time_values: np.ndarray = None,
        period: float = None
):
    (wl_interval_coll, flux_interval_coll, templA_interval_coll, templB_interval_coll,
     interval_buffer_mask) = _create_wavelength_intervals(
        wavelength, wavelength_intervals, flux_collection, flux_templateA, flux_templateB, buffer_size
    )

    results = []
    for i in range(0, len(wl_interval_coll)):
        wavelength_ = wl_interval_coll[i]
        flux_collection_ = flux_interval_coll[i]
        templateA_ = templA_interval_coll[i]
        templateB_ = templB_interval_coll[i]
        buffer_mask_ = interval_buffer_mask[i]

        RV_A, RV_B, sep_flux_A, sep_flux_B, wl_temp, RVb_flags = spectral_separation_routine(
            flux_collection_, templateA_, templateB_,
            wavelength_, options, sep_comp_options, rv_options, RV_guess_collection, time_values=time_values,
            period=period, buffer_mask=buffer_mask_
        )
        results.append([RV_A, RV_B, sep_flux_A, sep_flux_B, wl_temp, RVb_flags])
    return results

