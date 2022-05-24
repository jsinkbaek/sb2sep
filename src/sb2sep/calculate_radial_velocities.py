from sb2sep.broadening_function_svd import *
import sb2sep.rotational_broadening_function_fitting as rbff
import sb2sep.gaussian_fitting as gf
from copy import copy
from joblib import Parallel, delayed
from sb2sep.storage_classes import FitParameters, RadialVelocityOptions
import numpy as np


def radial_velocity_2_components(
        inv_flux, broadening_function_template:BroadeningFunction,
        options: RadialVelocityOptions
):
    BFsvd = copy(broadening_function_template)
    BFsvd.spectrum = inv_flux
    BFsvd.smooth_sigma = options.bf_smooth_sigma_A

    # Create Broadening Function for star A
    BFsvd.solve()
    BFsvd.smooth()

    # Fit rotational broadening function profile to Giant peak
    fitparams_A = FitParameters(
        options.vsini_A, options.spectral_resolution, options.velocity_fit_width_A, options.limbd_coef_A,
        options.bf_smooth_sigma_A, options.bf_velocity_span, options.vary_vsini_A, options.vsini_vary_limit_A,
        options.vary_limbd_coef_A, options.RV_A
    )
    fit_A, model_values_A = BFsvd.fit_rotational_profile(fitparams_A)

    # Create Broadening Function for star B
    bf, bf_smooth = BFsvd.bf, BFsvd.bf_smooth
    BFsvd.smooth_sigma = options.bf_smooth_sigma_B
    BFsvd.bf = BFsvd.bf - model_values_A        # subtract model for giant
    BFsvd.smooth()

    # Fit rotational broadening function profile for MS peak
    if options.limbd_coef_B is None:
        options.limbd_coef_B = options.limbd_coef_A
    fitparams_B = FitParameters(
        options.vsini_B, options.spectral_resolution, options.velocity_fit_width_B, options.limbd_coef_B,
        options.bf_smooth_sigma_B, options.bf_velocity_span, options.vary_vsini_B, options.vsini_vary_limit_B,
        options.vary_limbd_coef_B, options.RV_B
    )
    fit_B, model_values_B = BFsvd.fit_rotational_profile(fitparams_B)

    _, RV_A, _, _, _, _ = get_fit_parameter_values(fit_A.params)
    _, RV_B, _, _, _, _ = get_fit_parameter_values(fit_B.params)
    return (RV_A, RV_B), (model_values_A, fit_A, model_values_B, fit_B), (bf, bf_smooth)


def _pull_results_mspectra_2comp(broadening_function_template: BroadeningFunction, res_par, n_spectra, plot):
    RVs_A = np.empty((n_spectra,))
    RVs_B = np.empty((n_spectra,))
    broadening_function_vals = np.empty((broadening_function_template.velocity.size, n_spectra))
    broadening_function_vals_smoothed = np.empty((broadening_function_template.velocity.size, n_spectra))
    model_values_A = np.empty((broadening_function_template.velocity.size, n_spectra))
    model_values_B = np.empty((broadening_function_template.velocity.size, n_spectra))
    for i in range(0, n_spectra):
        RV_values = res_par[i][0]
        models, (bf, bf_smooth) = res_par[i][1], res_par[i][2]
        broadening_function_vals[:, i] = bf
        broadening_function_vals_smoothed[:, i] = bf_smooth
        model_values_A[:, i] = models[0]
        model_values_B[:, i] = models[2]
        if plot:
            plt.figure()
            plt.plot(broadening_function_template.velocity, bf_smooth)
            plt.plot(broadening_function_template.velocity, models[0], 'k--')
            plt.plot(broadening_function_template.velocity, models[2], 'k--')
            plt.show(block=False)
        RVs_A[i], RVs_B[i] = RV_values[0], RV_values[1]
    if plot:
        plt.show(block=True)
    extra_results = (broadening_function_template.velocity, broadening_function_vals, broadening_function_vals_smoothed,
                     model_values_A, model_values_B)
    return RVs_A, RVs_B, extra_results


def _pull_results_mspectra_1comp(broadening_function_template: BroadeningFunction, res_par, n_spectra, plot):
    RVs = np.empty((n_spectra, ))
    broadening_function_vals = np.empty((broadening_function_template.velocity.size, n_spectra))
    broadening_function_vals_smoothed = np.empty((broadening_function_template.velocity.size, n_spectra))
    model_values = np.empty((broadening_function_template.velocity.size, n_spectra))
    for i in range(0, n_spectra):
        RVs[i] = res_par[i][0]
        fit_res = res_par[i][1]
        model_values[:, i], broadening_function_vals[:, i] = fit_res[1], fit_res[3]
        broadening_function_vals_smoothed[:, i] = fit_res[4]
        if plot:
            plt.figure()
            plt.plot(broadening_function_template.velocity, broadening_function_vals_smoothed[:, i])
            plt.plot(broadening_function_template.velocity, model_values, 'k--')
            plt.show(block=False)
    if plot:
        plt.show(block=True)
    extra_results = (broadening_function_template.velocity, broadening_function_vals, broadening_function_vals_smoothed,
                     model_values)
    return RVs, extra_results


def radial_velocities_of_multiple_spectra(
        inv_flux_collection: np.ndarray,
        inv_flux_template: np.ndarray,
        options: RadialVelocityOptions,
        fit_two_components=True,
        number_of_parallel_jobs=4,
        plot=False
):
    """
    Calculates radial velocities for two components, for multiple spectra, by fitting two rotational broadening function
    profiles successively to calculated broadening functions. Uses joblib to parallelize the calculations. Calls
    radial_velocity_from_broadening_function() for each spectrum.

    Assumes that both components are well-described by the single template spectrum provided for the broadening function
    calculation.

    :param inv_flux_collection:     np.ndarray shape (:, n_spectra). Collection of inverted fluxes for each spectrum.
    :param inv_flux_template:       np.ndarray shape (:, ). Inverted flux of a template spectrum.
    :param options:                 fit options passed on from above.
    :param fit_two_components:      bool. Indicates if 1 or 2 components should be fitted for.
    :param number_of_parallel_jobs: int. Indicates the number of separate processes to spawn with joblib.
    :param plot:                    bool. Indicates whether results should be plotted, with separate figures for each
                                        spectrum.
    :return:    RVs_A, RVs_B, extra_results.
                RVs_A:  np.ndarray shape (n_spectra, ). RV values of component A.
                RVs_B:  np.ndarray shape (n_spectra, ). RV values of component B.
                extra_results: (broadening function velocity values, broadening function values,
                                smoothed broadening function values, fit model values for component A,
                                fit model values for component B).
    """
    n_spectra = inv_flux_collection[0, :].size
    broadening_function_template = BroadeningFunction(inv_flux_collection[:, 0], inv_flux_template,
                                                      options.bf_velocity_span, options.delta_v)
    broadening_function_template.smooth_sigma = options.bf_smooth_sigma_A

    # Arguments for parallel job
    if fit_two_components is True:
        arguments = (broadening_function_template, options)
        calc_function = radial_velocity_2_components
    else:
        fitparams = FitParameters(
            options.vsini_A, options.spectral_resolution, options.velocity_fit_width_A, options.limbd_coef_A,
            options.bf_smooth_sigma_A, options.bf_velocity_span, options.vary_vsini_A, options.vsini_vary_limit_A,
            options.vary_limbd_coef_A, options.RV_A
        )
        arguments = (broadening_function_template, fitparams)
        calc_function = radial_velocity_single_component

    # Create parallel call to calculate radial velocities
    res_par = Parallel(n_jobs=number_of_parallel_jobs)(
        delayed(calc_function)(inv_flux_collection[:, i], *arguments) for i in range(0, n_spectra)
    )

    # Pull results from call
    if fit_two_components is True:
        RVs_A, RVs_B, extra_results = _pull_results_mspectra_2comp(
            broadening_function_template, res_par, n_spectra, plot
        )
        bf_velocity, bf_vals, bf_vals_smooth, model_vals_A, model_vals_B = extra_results
        return RVs_A, RVs_B, (bf_velocity, bf_vals, bf_vals_smooth, model_vals_A, model_vals_B)
    else:
        RVs, (bf_velocity, bf_vals, bf_vals_smooth, model_vals) = _pull_results_mspectra_1comp(
            broadening_function_template, res_par, n_spectra, plot
        )
        return RVs, (bf_velocity, bf_vals, bf_vals_smooth, model_vals)


def radial_velocity_single_component(
        inv_flux: np.ndarray,
        broadening_function_template: BroadeningFunction,
        fitparams: FitParameters
):
    """
    Calculates the broadening function of a spectrum and fits a single rotational broadening function profile to it.
    Needs a template object of the BroadeningFunction with the correct parameters and template spectrum already set. 
    Setup to be of convenient use during the spectral separation routine (see spectral_separation_routine.py).
    
    :param inv_flux:                     np.ndarray. Inverted flux of the program spectrum (e.g. 1-normalized_flux)
    :param broadening_function_template: BroadeningFunction. The template used to calculate the broadening function.
    :param fitparams:                   InitialFitParameters. Object that stores the fitting parameters needed.
    :return:    RV, (fit, model_values, BFsvd.velocity, BFsvd.bf, BFsvd.bf_smooth)
                RV:                 float, the fitted RV value
                fit:                lmfit.MinimizerResult. The object storing the fit parameters.
                model_values:       np.ndarray. Broadening function values according to the fit.
                BFsvd.velocity:     np.ndarray. Velocity values for the broadening function and model values.
                BFsvd.bf:           np.ndarray. Broadening function values calculated.
                BFsvd.bf_smooth:    np.ndarray. Smoothed broadening function values.
    """
    BFsvd = copy(broadening_function_template)
    BFsvd.spectrum = inv_flux

    # Create Broadening Function
    BFsvd.solve()
    BFsvd.smooth()

    # Fit rotational broadening function profile
    if fitparams.fitting_profile == 'RotBF':
        fitting_routine = rbff.fitting_routine_rotational_broadening_profile
        get_fit_parameter_values = rbff.get_fit_parameter_values
    elif fitparams.fitting_profile == 'Gaussian':
        fitting_routine = gf.fitting_routine_gaussian_profile
        get_fit_parameter_values = gf.get_fit_parameter_values
    else:
        raise ValueError('Unrecognised fitting profile selected.')
    fit, model_values = BFsvd.fit_rotational_profile(fitparams, fitting_routine)

    res = get_fit_parameter_values(fit.params)
    RV = res[1]

    return RV, (fit, model_values, BFsvd.velocity, BFsvd.bf, BFsvd.bf_smooth)
