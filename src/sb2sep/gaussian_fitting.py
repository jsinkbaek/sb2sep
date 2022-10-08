
import numpy as np
from scipy.signal import fftconvolve
import lmfit
from lmfit.models import GaussianModel, ConstantModel
import scipy.constants as scc
from sb2sep.storage_classes import FitParameters


def gaussian_profile(
        velocities: np.ndarray, amplitude: float, radial_velocity_cm: float, gaussian_sigma: float,
        continuum_constant: float, spectrum_broadening_sigma: float
):
    n = velocities.size
    profile_values = np.ones(n) * continuum_constant

    profile_values += amplitude * np.exp(-(velocities-radial_velocity_cm)**2 / (2*gaussian_sigma**2))

    # Broadening by resolution and smoothing factor
    scaled_width = np.sqrt(2*np.pi) * spectrum_broadening_sigma
    broadening_vals = np.exp(-0.5 * (velocities/spectrum_broadening_sigma)**2) / scaled_width

    return fftconvolve(profile_values, broadening_vals, mode='same')


def weight_function(
        velocities: np.ndarray, broadening_function_values: np.ndarray, velocity_fit_half_width: float,
        radial_velocity_guess: float
):
    if radial_velocity_guess is None:
        peak_idx = np.argmax(broadening_function_values)
        mask = (velocities > velocities[peak_idx] - velocity_fit_half_width) & \
               (velocities < velocities[peak_idx] + velocity_fit_half_width)
    else:
        mask = (velocities > radial_velocity_guess - velocity_fit_half_width) & \
               (velocities < radial_velocity_guess + velocity_fit_half_width)

    weight_function_values = np.zeros(broadening_function_values.size)
    weight_function_values[mask] = 1.0
    return weight_function_values


def get_fit_parameter_values_old(parameters: lmfit.Parameters):
    amplitude = parameters['amplitude'].value
    radial_velocity_cm = parameters['radial_velocity_cm'].value
    gaussian_sigma = parameters['gaussian_sigma'].value
    continuum_constant = parameters['continuum_constant'].value
    spectrum_broadening_sigma = parameters['spectrum_broadening_sigma'].value
    return amplitude, radial_velocity_cm, gaussian_sigma, continuum_constant, spectrum_broadening_sigma


def get_fit_parameter_values(parameters):
    amplitude = parameters['amplitude'].value
    radial_velocity_cm = parameters['center'].value
    gaussian_sigma = parameters['sigma'].value
    continuum_constant = parameters['c'].value
    return amplitude, radial_velocity_cm, gaussian_sigma, continuum_constant


def compare_broadening_function_with_profile(
        parameters: lmfit.Parameters, velocities: np.ndarray, broadening_function_values: np.ndarray,
        weight_function_values: np.ndarray
):
    parameter_vals = get_fit_parameter_values(parameters)
    comparison = broadening_function_values - gaussian_profile(velocities, *parameter_vals)
    return weight_function_values * comparison


def fitting_routine_gaussian_profile_old(
        velocities: np.ndarray, broadening_function_values: np.ndarray, fitparams: FitParameters,
        smooth_sigma: float, dv: float, print_report=False, compare_func=compare_broadening_function_with_profile
):
    speed_of_light = scc.c / 1000  # in km/s
    spectrum_broadening_sigma = np.sqrt(
        ((speed_of_light/fitparams.spectral_resolution)/(2.354*dv))**2 + (smooth_sigma/dv)**2
    )   # contribution just from spectral resolution and smoothing of broadening function
    params = lmfit.Parameters()

    weight_function_values = weight_function(
        velocities, broadening_function_values, fitparams.velocity_fit_width, fitparams.RV
    )
    peak_idx = np.argmax(broadening_function_values*weight_function_values)
    params.add('amplitude', value=broadening_function_values[peak_idx], min=0.0)
    if fitparams.RV is None:
        params.add(
            'radial_velocity_cm', value=velocities[peak_idx], min=velocities[peak_idx]-fitparams.velocity_fit_width,
            max=velocities[peak_idx]+fitparams.velocity_fit_width
        )
    else:
        params.add(
            'radial_velocity_cm', value=fitparams.RV, min=fitparams.RV-fitparams.velocity_fit_width,
            max=fitparams.RV+fitparams.velocity_fit_width
        )
    if fitparams.vsini_vary_limit is not None:
        params.add(
            'gaussian_sigma', value=fitparams.vsini, vary=fitparams.vary_vsini,
            min=fitparams.vsini - fitparams.vsini*fitparams.vsini_vary_limit,
            max=fitparams.vsini + fitparams.vsini*fitparams.vsini_vary_limit
        )
    else:
        params.add('gaussian_sigma', value=fitparams.vsini, vary=fitparams.vary_vsini)
    params.add(
        'continuum_constant', value=fitparams.continuum, min=np.min(broadening_function_values),
        max=np.max(broadening_function_values), vary=fitparams.vary_continuum
    )
    params.add('spectrum_broadening_sigma', value=spectrum_broadening_sigma, vary=False)

    fit = lmfit.minimize(
        compare_func, params, args=(velocities, broadening_function_values, weight_function_values),
        xtol=1E-8, ftol=1E-8, max_nfev=500
    )

    if print_report:
        print(lmfit.fit_report(fit, show_correl=False))

    parameter_vals = get_fit_parameter_values(fit.params)
    model = gaussian_profile(velocities, *parameter_vals)

    return fit, model


def fitting_routine_gaussian_profile(
        velocities: np.ndarray, broadening_function_values: np.ndarray, fitparams: FitParameters,
        smooth_sigma: float, dv: float, print_report=False
):
    model = GaussianModel() + ConstantModel()
    params = model.make_params()
    weight_function_values = weight_function(
        velocities, broadening_function_values, fitparams.velocity_fit_width, fitparams.RV
    )
    peak_idx = np.argmax(broadening_function_values * weight_function_values)
    params['amplitude'].set(value=broadening_function_values[peak_idx], min=0.0)
    if fitparams.RV is None:
        peak_idx = np.argmax(broadening_function_values)
        params['center'].set(
            value=velocities[peak_idx], min=velocities[peak_idx]-fitparams.velocity_fit_width,
            max=velocities[peak_idx]+fitparams.velocity_fit_width, vary=True
        )
    else:
        params['center'].set(
            value=fitparams.RV, min=fitparams.RV - fitparams.velocity_fit_width,
            max=fitparams.RV + fitparams.velocity_fit_width, vary=True
        )

    if fitparams.vsini_vary_limit is not None:
        params['sigma'].set(
            value=fitparams.vsini, vary=fitparams.vary_vsini,
            min=fitparams.vsini - fitparams.vsini*fitparams.vsini_vary_limit,
            max=fitparams.vsini + fitparams.vsini*fitparams.vsini_vary_limit
        )
    else:
        params['sigma'].set(value=fitparams.vsini, vary=fitparams.vary_vsini)
    params['c'].set(
        value=fitparams.continuum, min=np.min(broadening_function_values),
        max=np.max(broadening_function_values), vary=fitparams.vary_continuum
    )

    fit = model.fit(broadening_function_values, params, x=velocities, xtol=1E-8, ftol=1E-8, max_nfev=500)

    if print_report:
        print(fit.fit_report(show_correl=False))

    model = fit.best_fit

    return fit, model
