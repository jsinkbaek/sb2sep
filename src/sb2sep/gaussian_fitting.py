import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import fftconvolve
import lmfit
from lmfit.models import GaussianModel, ConstantModel
import scipy.constants as scc
from sb2sep.storage_classes import FitParameters
from copy import copy


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
        radial_velocity_guess: float, limit_left=None, limit_right=None
):
    if radial_velocity_guess is None:
        peak_idx = np.argmax(broadening_function_values)
        mask = (velocities > velocities[peak_idx] - velocity_fit_half_width) & \
               (velocities < velocities[peak_idx] + velocity_fit_half_width)
        if limit_left is not None and limit_right is not None:
            mask = (velocities[peak_idx] > limit_left) & (velocities[peak_idx] < limit_right)
    else:
        mask = (velocities > radial_velocity_guess - velocity_fit_half_width) & \
               (velocities < radial_velocity_guess + velocity_fit_half_width)
        if limit_left is not None and limit_right is not None:
            mask = (radial_velocity_guess > limit_left) & (radial_velocity_guess < limit_right)

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
    if fitparams.data_limits is not None:
        weight_function_values = weight_function(
            velocities, broadening_function_values, fitparams.velocity_fit_width, fitparams.RV,
            limit_left=fitparams.data_limits[0], limit_right=fitparams.data_limits[1]
        )
    else:
        weight_function_values = weight_function(
            velocities, broadening_function_values, fitparams.velocity_fit_width, fitparams.RV
        )
    peak_idx = np.argmax(broadening_function_values * weight_function_values)
    params['amplitude'].set(value=broadening_function_values[peak_idx], min=0.0)
    if fitparams.amplitude is not None:
        params['amplitude'].set(value=fitparams.amplitude)
    if fitparams.amplitude_limits is not None:
        params['amplitude'].set(min=fitparams.amplitude_limits[0], max=fitparams.amplitude_limits[1])
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
    if fitparams.RV_limits is not None:
        params['center'].set(
            min=fitparams.RV_limits[0], max=fitparams.RV_limits[1]
        )

    if fitparams.vsini_limits is not None:
        params['sigma'].set(
            value=fitparams.vsini, vary=fitparams.vary_vsini,
            min=fitparams.vsini_limits[0],
            max=fitparams.vsini_limits[1]
        )
    elif fitparams.vsini_vary_limit is not None:
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
    if fitparams.continuum_limits is not None:
        params['c'].set(min=fitparams.continuum_limits[0], max=fitparams.continuum_limits[1])

    if fitparams.data_limits is not None:
        mask_data = (velocities > fitparams.data_limits[0]) & (velocities < fitparams.data_limits[1])
    elif fitparams.RV is not None:
        mask_data = (velocities > fitparams.RV - fitparams.velocity_fit_width) & \
                    (velocities < fitparams.RV + fitparams.velocity_fit_width)
    else:
        mask_data = (velocities > velocities[peak_idx]-fitparams.velocity_fit_width) & \
                    (velocities < velocities[peak_idx]+fitparams.velocity_fit_width)

    fit = model.fit(
        broadening_function_values[mask_data], params, x=velocities[mask_data], xtol=1E-8, ftol=1E-8, max_nfev=500
    )

    if print_report:
        print(fit.fit_report(show_correl=False))

    model = fit.eval(x=velocities)

    return fit, model


def fitting_routine_gaussian_gui(
    velocities: np.ndarray, broadening_function_values: np.ndarray, fitparams: FitParameters,
    smooth_sigma: float, dv: float
):
    fitparams_copy = FitParameters(
        copy(fitparams.vsini), copy(fitparams.spectral_resolution), copy(fitparams.velocity_fit_width),
        copy(fitparams.limbd_coef), copy(fitparams.bf_smooth_sigma), copy(fitparams.bf_velocity_span),
        copy(fitparams.vary_vsini), copy(fitparams.vsini_vary_limit), copy(fitparams.vary_limbd_coef),
        copy(fitparams.RV), copy(fitparams.continuum), copy(fitparams.vary_continuum),
        copy(fitparams.fitting_profile)
    )
    fit, model = fitting_routine_gaussian_profile(
        velocities, broadening_function_values, fitparams_copy, dv=dv, smooth_sigma=smooth_sigma
    )
    fig = plt.figure()
    line_bf = plt.plot(velocities, broadening_function_values, 'k-')
    line_fit = plt.plot(velocities, model, 'r-')[0]
    plt.draw()
    plt.pause(0.05)
    while True:
        print('Current fit parameters:')
        print(f"1. sigma       {fit.best_values['sigma']:.3f}")
        print(f"2. center      {fit.best_values['center']:.3f}")
        print(f"3. amplitude   {fit.best_values['amplitude']:.3f}")
        print(f"4. constant    {fit.best_values['c']:.3f}")
        print('Accept and iterate (y), accept and move to next spectrum (y!), ')
        print('impose parameter limits (l1, l2, l3, 4), change initial value (i1, i2, i3, i4),')
        print('or select central value and data limits by gui (g).')
        while True:
            inpt = input()
            if inpt == 'y':
                plt.close(fig)
                return fit, model
            elif inpt == 'y!':
                plt.close(fig)
                return fit, model, 0
            elif 'l' in inpt or 'i' in inpt or inpt == 'g':
                break
            else:
                print(f'Unrecognized input {inpt}. Try again.')
        if 'l' in inpt:
            idx = int(inpt[1])-1
            list_keys = ['sigma', 'center', 'amplitude', 'c']
            print(f'Provide limits (with space between min and max) for parameter {list_keys[idx]}')
            inpt_2 = input()
            inpt_2 = inpt_2.split()
            min_, max_ = float(inpt_2[0]), float(inpt_2[1])
            if idx == 0:
                fitparams_copy.vsini_limits = (min_, max_)
            elif idx == 1:
                fitparams_copy.RV_limits = (min_, max_)
            elif idx == 2:
                fitparams_copy.amplitude_limits = (min_, max_)
            elif idx == 3:
                fitparams_copy.continuum_limits = (min_, max_)
            else:
                raise ValueError(f'Unknown input number in {inpt}. Acceptable values [l1, l2, l3, l4].')
        elif 'i' in inpt:
            idx = int(inpt[1]) - 1
            list_keys = ['sigma', 'center', 'amplitude', 'c']
            print(f'Provide new initial value for {list_keys[idx]}')
            inpt_2 = float(input())
            if idx == 0:
                fitparams_copy.vsini = inpt_2
            elif idx == 1:
                fitparams_copy.RV = inpt_2
            elif idx == 2:
                fitparams_copy.amplitude = inpt_2
            elif idx == 3:
                fitparams_copy.continuum = inpt_2
            else:
                raise ValueError(f'Unknown input number in {inpt}. Acceptable values [i1, i2, i3, i4].')
        elif inpt == 'g':
            print('First two button presses marks data limit, last is RV and height')
            selects = plt.ginput(n=3, timeout=-1)
            x1, x2, x3, y3 = selects[0][0], selects[1][0], selects[2][0], selects[2][1]
            fitparams_copy.RV = x3
            fitparams_copy.amplitude = y3 * fit.best_values['sigma']*np.sqrt(2*np.pi)
            fitparams_copy.data_limits = (x1, x2)
        else:
            print(f'Unknown input {inpt}. Try again.')
            continue
        fit, model = fitting_routine_gaussian_profile(
            velocities, broadening_function_values, fitparams_copy, dv=dv, smooth_sigma=smooth_sigma
        )
        line_fit.set_ydata(model)
        fig.canvas.draw()
        fig.canvas.flush_events()
        plt.pause(0.05)
