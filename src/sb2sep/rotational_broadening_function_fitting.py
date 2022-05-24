"""
First edition on May 01 2021.
@author Jeppe Sinkbæk Thomsen, Master's student in astronomy at Aarhus University.
Supervisor: Assistant Professor Karsten Frank Brogaard.

A rotational broadening function fitting routine, with the model profile used, and all the relevant steps until the
lmfit minimizer method is called. Code is adapted from the shazam.py library for the SONG telescope
(written by Emil Knudstrup and Jens Jessen-Hansen)
"""

import numpy as np
from scipy.signal import fftconvolve
import lmfit
import scipy.constants as scc
from sb2sep.storage_classes import FitParameters


def rotational_broadening_function_profile(
        velocities: np.ndarray, amplitude: float, radial_velocity_cm: float, vsini: float, gaussian_width: float,
        continuum_constant: float, limbd_coef: float
):
    """
    Calculates a theoretical broadening function profile based on the one described in
    Kaluzny 2006: Eclipsing Binaries in the Open Cluster NGC 2243 II. Absolute Properties of NV CMa.
    Convolves it with a gaussian function to create a rotational broadening function profile.
    :param velocities:          np.ndarray, velocities to calculate profile for.
    :param amplitude:           float, normalization constant
    :param radial_velocity_cm:  float, radial velocity of the centre of mass of the star
    :param vsini:               float, linear velocity of the equator of the rotating star times sin(inclination)
    :param gaussian_width:      float, width of the gaussian broadening function that the profile will be folded with
    :param continuum_constant:  float, the continuum level
    :param limbd_coef:          float, linear limb darkening coefficient of the star.
    :return rot_bf_profile:     np.ndarray, the profile at the given velocities
    """
    n = velocities.size
    broadening_function_values = np.ones(n) * continuum_constant

    # The "a" coefficient of the profile
    a = (velocities - radial_velocity_cm) / vsini

    # Create bf function values
    mask = (np.abs(a) < 1.0)        # Select only near-peak values
    broadening_function_values[mask] += amplitude*((1-limbd_coef)*np.sqrt(1.0-a[mask]**2) + 0.25*np.pi*(1-a[mask]**2))

    # Create gs function values
    scaled_width = np.sqrt(2*np.pi) * gaussian_width
    gaussian_function_values = np.exp(-0.5 * (velocities/gaussian_width)**2) / scaled_width

    # Convolve rotational broadening function profile to get smoothed version
    rot_bf_profile = fftconvolve(broadening_function_values, gaussian_function_values, mode='same')
    return rot_bf_profile


def weight_function(
        velocities: np.ndarray, broadening_function_values: np.ndarray, velocity_fit_half_width: float,
        radial_velocity_guess: float
):
    """
    Weight function for the fit. Finds the peak value, and limits the data set to velocities within a certain distance
    of it.
    :param velocities:                  np.ndarray, the velocities of the broadening function
    :param broadening_function_values:  np.ndarray, the broadening function values
    :param velocity_fit_half_width:     float, the distance to each side within to include data for fit.
    :param radial_velocity_guess:       float, guess for the radial velocity. Used to limit weight function
    :return weight_function_values:     np.ndarray, set of 1s and 0s to weigh each data point with
    """
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


def get_fit_parameter_values(parameters: lmfit.Parameters):
    """
    Convenience function to pull parameter values from an lmfit.Parameters object.
    """
    amplitude = parameters['amplitude'].value
    radial_velocity_cm = parameters['radial_velocity_cm'].value
    vsini = parameters['vsini'].value
    gaussian_width = parameters['gaussian_width'].value
    continuum_constant = parameters['continuum_constant'].value
    limbd_coef = parameters['limbd_coef'].value

    return amplitude, radial_velocity_cm, vsini, gaussian_width, continuum_constant, limbd_coef


def compare_broadening_function_with_profile(
        parameters: lmfit.Parameters, velocities: np.ndarray, broadening_function_values: np.ndarray,
        weight_function_values: np.ndarray
):
    """
    Evaluates fit using rotational broadening function profile and fit parameters, and compares with the observed
    SVD broadening function. Returns the difference to the lmfit minimizer routine with weight function applied.
    :param parameters:                  lmfit.Parameters object of fit parameters
    :param velocities:                  np.ndarray, velocity values of the broadening function
    :param broadening_function_values:  np.ndarray, broadening function values
    :param weight_function_values:      np.ndarray, weight function values to apply. See weight_function() for details
    """
    parameter_vals = get_fit_parameter_values(parameters)

    comparison = broadening_function_values - rotational_broadening_function_profile(velocities, *parameter_vals)
    return weight_function_values * comparison


def fitting_routine_rotational_broadening_profile(
        velocities: np.ndarray, broadening_function_values: np.ndarray, fitparams: FitParameters,
        smooth_sigma: float, dv: float, print_report=False, compare_func=compare_broadening_function_with_profile
):
    """
    The fitting routine utilizing lmfit. Sets up the initial guesses to the fit, adds parameters, creates weight
    function, and calls the lmfit.minimize routine to fit a rotational broadening profile to match the observed
    broadening function.
    :param velocities:                  np.ndarray, velocity values of the broadening function.
    :param broadening_function_values:  np.ndarray, observed broadening function values.
    :param fitparams:                  an object holding the initial fit parameters:
            vsini_guess:                float, guess for the v sin(i) fit parameter for the model.
            limbd_coef:                 float, a linear limb darkening coefficient for the profile. This routine will
                                        not fit this parameter.
            velocity_fit_width:         float, how far out the fitting routine should include data points for the fit.
            spectral_resolution:        float, the resolution of the spectrograph used for the program spectrum.
    :param smooth_sigma:                float, the sigma value used for smoothing the broadening function values using a
                                        gaussian function.
    :param dv:                          float, the dv resolution of the interpolated spectrum in velocity space.
    :param print_report:                bool, whether to print the lmfit report after fitting. Default is False.
    :param compare_func:                function, the function used to compare the fit with the broadening function
                                        values.

    :return (fit, model):               fit: lmfit.MinimizerResult of the fit result.
                                        model: np.ndarray, model values of the broadening function according to the fit.
    """
    speed_of_light = scc.c / 1000  # in km/s
    gaussian_width = np.sqrt(((speed_of_light/fitparams.spectral_resolution)/(2.354 * dv))**2 + (smooth_sigma/dv)**2)
    params = lmfit.Parameters()

    weight_function_values = weight_function(
        velocities, broadening_function_values, fitparams.velocity_fit_width, fitparams.RV
    )
    peak_idx = np.argmax(broadening_function_values*weight_function_values)
    params.add('amplitude', value=broadening_function_values[peak_idx], min=0.0)
    if fitparams.RV is None:
        params.add(
            'radial_velocity_cm', value=velocities[peak_idx],
            min=velocities[peak_idx]-fitparams.velocity_fit_width,
            max=velocities[peak_idx]+fitparams.velocity_fit_width
        )
    else:
        params.add(
            'radial_velocity_cm', value=fitparams.RV, min=fitparams.RV-fitparams.velocity_fit_width,
            max=fitparams.RV+fitparams.velocity_fit_width
        )
    if fitparams.vsini_vary_limit is not None:
        params.add(
            'vsini', value=fitparams.vsini, vary=fitparams.vary_vsini,
            min=fitparams.vsini - fitparams.vsini*fitparams.vsini_vary_limit,
            max=fitparams.vsini + fitparams.vsini*fitparams.vsini_vary_limit
        )
    else:
        params.add('vsini', value=fitparams.vsini, vary=fitparams.vary_vsini)
    params.add('gaussian_width', value=gaussian_width, vary=False)
    params.add(
        'continuum_constant', value=fitparams.continuum, min=np.min(broadening_function_values),
        max=np.max(broadening_function_values), vary=fitparams.vary_continuum
    )
    params.add('limbd_coef', value=fitparams.limbd_coef, vary=fitparams.vary_limbd_coef)

    fit = lmfit.minimize(
        compare_func, params, args=(velocities, broadening_function_values, weight_function_values),
        xtol=1E-8, ftol=1E-8, max_nfev=500
    )
    if print_report:
        print(lmfit.fit_report(fit, show_correl=False))

    parameter_vals = get_fit_parameter_values(fit.params)
    model = rotational_broadening_function_profile(velocities, *parameter_vals)

    return fit, model
