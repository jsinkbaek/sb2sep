import numpy as np
import lmfit
from scipy.signal import fftconvolve
import scipy.constants as scc


def rotational_bf_model(vel, amp, rv, vsini, gauss_width, continuum, limbd):
    """
    Calculates a theoretical broadening function profile based on the one described in
    Kaluzny 2006: Eclipsing Binaries in the Open Cluster NGC 2243 II. Absolute Properties of NV CMa.
    Convolves it with a gaussian function to create a rotational broadening function profile.
    :param vel:          np.ndarray, velocities to calculate profile for.
    :param amp:           float, normalization constant
    :param rv:  float, radial velocity of the centre of mass of the star
    :param vsini:               float, linear velocity of the equator of the rotating star times sin(inclination)
    :param gauss_width:      float, width of the gaussian broadening function that the profile will be folded with
    :param continuum:  float, the continuum level
    :param limbd:          float, linear limb darkening coefficient of the star.
    :return rot_bf_profile:     np.ndarray, the profile at the given velocities
    """
    n = vel.size
    bf = np.ones(n)*continuum

    a = (vel - rv) / vsini
    mask = (np.abs(a) < 1.0)        # Select only near-peak values
    bf[mask] += amp*((1-limbd)*np.sqrt(1.0-a[mask]**2) + 0.25*np.pi*(1-a[mask]**2))
    gauss = np.exp(-0.5 * (vel / gauss) ** 2) / (np.sqrt(2 * np.pi) * gauss_width)
    rot_bf_profile = fftconvolve(bf, gauss, mode='same')
    return rot_bf_profile


def weight_function(vel, bfvals, vel_fit_hwidth, rv):
    mask = (vel > rv - vel_fit_hwidth) & (vel < rv + vel_fit_hwidth)
    weight = np.zeros(bfvals.size)
    weight[mask] = 1.0
    return weight


def get_values(pars):
    amp, rv, vsini = pars["amp"].value, pars["rv"].value, pars["vsini"].value
    gwidth, continuum, limbd = pars["gauss_width"].value, pars["continuum"].value, pars["limbd"].value
    return amp, rv, vsini, gwidth, continuum, limbd


def residual_func(params, vel, bfvals, weights):
    pvals = get_values(params)
    return weights * (bfvals - rotational_bf_model(vel, *pvals))


def fitting_routine(vel, bfvals, mask, config, dv, resolution, rv):
    speed_c = scc.c / 1000.     # in km/s
    smooth_sigma = config["SmoothSigma"]
    vel_width = config["VelocityFitWidth"]
    gaussian_width = np.sqrt(((speed_c/resolution)/(2.354 * dv))**2 + (smooth_sigma/dv)**2)
    pars = lmfit.Parameters()

    weights = weight_function(vel, bfvals, vel_width, rv)
    peak = np.argmax(bfvals * weights)
    if config["Amplitude"][0] == 0:
        params.add('amp', value=bfvals[peak], min=0.0, max=99 * bfvals[peak], vary=config["Amplitude"][1])
    else:
        params.add(
            'amp', value=config["Amplitude"][0], min=0.0, max=99 * config["Amplitude"][0], vary=config["Amplitude"][1]
        )
    params.add("rv", value=rv, min=rv-vel_width, max=vel+vel_width, vary=True)
    params.add("vsini", value=config["Vsini"][0], min=config["Vsini"][2],max=config["Vsini"][3],vary=config["Vsini"][1])
    params.add(
        "continuum", value=config["Continuum"][0], min=np.min(bfvals), max=np.max(bfvals), vary=config["Continuum"][1]
    )
    params.add("gauss_width", value=gaussian_width, vary=False)
    params.add(
        "limbd", value=config["LimbD"][0], min=config["LimbD"][2], max=config["LimbD"][3], vary=config["LimbD"][1]
    )

    # Estimate uncertainty as STD of BF outside of RV region of interest
    maxmin_hwidth = (np.max(vel)-np.min(vel))/2
    dif = np.max([maxmin_hwidth/3, vel_width])
    mask_err = (vel < rv - dif) | (vel > rv+dif)
    weights = weights / np.std(bfvals[mask_err])

    fit = lmfit.minimize(
        residual_func, params, args=(vel[mask], bfvals[mask], weights[mask]), xtol=1E-10, ftol=1E-10, max_nfev=500
    )

    pvals = get_values(fit.params)
    model = rotational_bf_model(vel, *pvals)
    return fit, model, pvals

